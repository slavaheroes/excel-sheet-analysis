import argparse
import os
from typing import Dict, Any, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from loguru import logger
from utils import build_object, load_yaml, set_seed


# ---------------------------
# Dataset
# ---------------------------
class TextClassificationDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        text_col: str,
        label_col: str,
        label2id: Dict[str, int],
        tokenizer,
        padding: str = "max_length",
        truncation: bool = True,
        max_length: int = 256,
    ):
        self.texts = df[text_col].astype(str).tolist()
        self.labels = [label2id[label] for label in df[label_col].tolist()]
        self.tokenizer = tokenizer
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------------------------
# Lightning Module
# ---------------------------
class BertTextClassifier(L.LightningModule):
    def __init__(self, cfg: Dict[str, Any], num_classes: int, id2label: Dict[int, str]):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.id2label = id2label

        # backbone
        model_name = cfg["model"]["model_name"]
        backbone_params = cfg["model"].get("params", {})
        self.backbone = AutoModel.from_pretrained(model_name, **backbone_params)
        hidden_size = self.backbone.config.hidden_size

        # pooling
        head_cfg = cfg["model"].get("head", {}) or {}
        self.pooling = head_cfg.get("pooling", "cls")
        dropout = float(head_cfg.get("dropout", 0.1))

        # classification head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

        # loss
        loss_cfg = cfg["loss"]
        loss_cls = build_object(loss_cfg["type"], loss_cfg["module"])
        self.criterion = loss_cls(**loss_cfg.get("params", {}))

        # sklearn buffers
        self._train_preds, self._train_labels = [], []
        self._val_preds, self._val_labels = [], []
        self._test_preds, self._test_labels = [], []

    # ---- pooling implementations ----
    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1)  # [B, T, 1]
            summed = (last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            return summed / counts
        elif self.pooling == "max":
            # set masked positions to -inf before max
            masked = last_hidden_state.masked_fill(attention_mask.unsqueeze(-1) == 0, float("-inf"))
            return masked.max(dim=1).values
        
        # CLS
        return last_hidden_state[:, 0, :]

    # ---- step logic ----
    def forward(self, **batch):
        outputs = self.backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        pooled = self._pool(outputs.last_hidden_state, batch["attention_mask"])
        logits = self.head(pooled)
        return logits

    def _step(self, batch, stage: str):
        logits = self(**{k: v for k, v in batch.items() if k != "labels"})
        loss = self.criterion(logits, batch["labels"])
        preds = torch.argmax(logits, dim=-1)

        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        # store for metrics
        if stage == "train":
            self._train_preds.append(preds.detach().cpu())
            self._train_labels.append(batch["labels"].detach().cpu())
        elif stage == "val":
            self._val_preds.append(preds.detach().cpu())
            self._val_labels.append(batch["labels"].detach().cpu())
        else:
            self._test_preds.append(preds.detach().cpu())
            self._test_labels.append(batch["labels"].detach().cpu())

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def on_train_epoch_end(self):
        if self._train_preds:
            y_pred = torch.cat(self._train_preds).numpy()
            y_true = torch.cat(self._train_labels).numpy()
            self.log("train_acc", accuracy_score(y_true, y_pred), prog_bar=True, on_epoch=True)
            self.log("train_f1_macro", f1_score(y_true, y_pred, average="macro"), prog_bar=True, on_epoch=True)

    def on_validation_epoch_end(self):
        if self._val_preds:
            y_pred = torch.cat(self._val_preds).numpy()
            y_true = torch.cat(self._val_labels).numpy()
            self.log("val_acc", accuracy_score(y_true, y_pred), prog_bar=True, on_epoch=True)
            self.log("val_f1_macro", f1_score(y_true, y_pred, average="macro"), prog_bar=True, on_epoch=True)

    def on_test_epoch_end(self):
        if self._test_preds:
            y_pred = torch.cat(self._test_preds).numpy()
            y_true = torch.cat(self._test_labels).numpy()
            self.log("test_acc", accuracy_score(y_true, y_pred), prog_bar=True, on_epoch=True)
            self.log("test_f1_macro", f1_score(y_true, y_pred, average="macro"), prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        opt_cfg = self.cfg["optimizer"]
        opt_cls = build_object(opt_cfg["type"], opt_cfg["module"])
        optimizer = opt_cls(self.parameters(), **opt_cfg.get("params", {}))

        if "lr_scheduler" in self.cfg and self.cfg["lr_scheduler"]:
            sch_cfg = self.cfg["lr_scheduler"]
            sch_fn = build_object(sch_cfg["type"], sch_cfg["module"])
            scheduler = sch_fn(optimizer, **sch_cfg.get("params", {}))
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": sch_cfg.get("interval", "epoch")}}
        return optimizer

# ---------------------------
# Utilities for callbacks, trainer
# ---------------------------

def build_callbacks(cb_cfg: Dict[str, Any]) -> List[Any]:
    cbs = []
    for _, spec in cb_cfg.items():
        cls = build_object(spec["type"], spec["module"])
        params = spec.get("params", {})
        cbs.append(cls(**params))
    return cbs


def build_trainer(trainer_cfg: Dict[str, Any], callbacks: List[Any], logger_obj=None) -> L.Trainer:
    tr_cls = build_object(trainer_cfg["type"], trainer_cfg["module"])
    params = dict(trainer_cfg.get("params", {}))
    if logger_obj is not None:
        params["logger"] = logger_obj
    return tr_cls(callbacks=callbacks, **params)

# ---------------------------
# Data preparation
# ---------------------------
def prepare_splits(
    csv_path: str,
    text_col: str,
    label_col: str,
    categories: List[str],
    val_size: float = 0.5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int], Dict[int, str]]:
    df = pd.read_csv(csv_path)

    label2id = {c: i for i, c in enumerate(categories)}
    id2label = {i: c for c, i in label2id.items()}

    # Stratified splits: train / val
    train_df, val_df = train_test_split(
        df, test_size=val_size, stratify=df[label_col], random_state=random_state
    )
    
    return train_df, val_df, label2id, id2label


# ---------------------------
# Main
# ---------------------------
def main(cfg_path: str):
    cfg = load_yaml(cfg_path) 
    set_seed(42)

    # Build tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["model_name"],
        use_fast=True,
    )

    # Prepare data splits
    ds_cfg = cfg["dataset"]
    train_df, val_df, label2id, id2label = prepare_splits(
        csv_path=ds_cfg["data_path"],
        text_col=ds_cfg["text_column"],
        label_col=ds_cfg["label_column"],
        categories=ds_cfg["categories"],
        random_state=42,
    )

    # Datasets
    train_dataset = TextClassificationDataset(
        train_df,
        text_col=ds_cfg["text_column"],
        label_col=ds_cfg["label_column"],
        label2id=label2id,
        tokenizer=tokenizer,
        padding='max_length',
        truncation=True,
        max_length= ds_cfg.get("max_length", 512),
    )
    
    val_dataset = TextClassificationDataset(
        val_df,
        text_col=ds_cfg["text_column"],
        label_col=ds_cfg["label_column"],
        label2id=label2id,
        tokenizer=tokenizer,
        padding='max_length',
        truncation=True,
        max_length=ds_cfg.get("max_length", 512),
    )

    # Dataloaders
    dl_cfg = cfg["dataloader"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=dl_cfg["batch_size"],
        num_workers=dl_cfg["num_workers"],
        pin_memory=dl_cfg["pin_memory"],
        drop_last=False,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=dl_cfg["batch_size"],
        num_workers=dl_cfg["num_workers"],
        pin_memory=dl_cfg["pin_memory"],
        drop_last=False,
        shuffle=False,
    )
    
    # Check dataloaders
    for loader in [train_loader, val_loader]:
        for batch in loader:
            logger.info(f"Batch keys: {batch.keys()}")
            logger.info(f"Batch sizes: {[v.size() for v in batch.values()]}")
            logger.info(f"Batch labels: {batch['labels']}")
            break

    # LightningModule
    lit_model = BertTextClassifier(cfg=cfg, num_classes=len(ds_cfg["categories"]), id2label=id2label)

    # Callbacks
    callbacks = build_callbacks(cfg.callbacks)
    tb_logger = TensorBoardLogger(save_dir="tb_logs", name=cfg.exp_name)

    # Trainer
    trainer = build_trainer(cfg.trainer, callbacks=callbacks, logger_obj=tb_logger)

    # Fit
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    logger.info("Training completed successfully!")
    trainer.test(lit_model, dataloaders=val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    main(args.config)
