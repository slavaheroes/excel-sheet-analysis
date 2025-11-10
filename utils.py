import pandas as pd
import numpy as np
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')


def _dedup(names):
    seen = {}
    out = []
    for n in names:
        base = n if n else "col"
        k = seen.get(base, 0)
        out.append(base if k == 0 else f"{base}.{k}")
        seen[base] = k + 1
    return out

def _nonempty_mask(df):
    # True if cell is a non-empty string OR a non-NaN non-string
    return df.map(lambda x: (isinstance(x, str) and x.strip() != "") or pd.notna(x))

def load_excel_tables(path, sheet_name=0, min_table_rows=2):
    """
    Read an Excel sheet and return a list of DataFrames, each representing a table.
    A 'table' is defined as a contiguous block of non-empty rows, separated by one
    or more empty rows. Within each block, the header row is chosen via your heuristic.
    """
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None, engine="openpyxl")
    cell = _nonempty_mask(raw)

    rows = np.where(cell.any(axis=1))[0]
    cols = np.where(cell.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return []

    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()
    block = raw.iloc[r0:r1+1, c0:c1+1].copy()
    cell_block = cell.iloc[r0:r1+1, c0:c1+1].copy()

    # Identify segments of consecutive non-empty rows (tables) separated by empty rows
    row_has_data = cell_block.any(axis=1).to_numpy()
    segments = []
    start = None
    for i, has in enumerate(row_has_data):
        if has and start is None:
            start = i
        elif not has and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(row_has_data) - 1))

    tables = []
    for (rs, re) in segments:
        sub = block.iloc[rs:re+1].copy()

        # If the sub-block is too small, skip (optional guard)
        if len(sub) < min_table_rows:
            continue

        # Heuristic to pick header row inside this sub-block (same as your logic)
        candidates = []
        for i in range(len(sub)):
            row = sub.iloc[i].astype(str).str.strip()
            score = (
                row.ne("").sum(),
                -row.str.len().replace([np.inf, -np.inf], np.nan).fillna(0).mean()
            )
            candidates.append((score, i))
        _, hdr_i = max(candidates)

        header = _dedup(
            sub.iloc[hdr_i].astype(str).str.strip().replace("", "col").tolist()
        )
        df = sub.iloc[hdr_i+1:].copy()
        df.columns = header

        # Drop all-empty rows/cols and normalize index
        df = df.dropna(how="all").dropna(how="all", axis=1).reset_index(drop=True)

        # Only keep non-empty results
        if df.shape[0] > 0 and df.shape[1] > 0:
            tables.append(df)

    return tables

def read_sheet(path, sheet=0, min_table_rows=2):
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path, dtype=str)
        df = df.rename(columns=lambda c: str(c).strip())
    elif path.lower().endswith((".xls", ".xlsx")):
        df = load_excel_tables(path, sheet_name=sheet, min_table_rows=min_table_rows)
        if not df:
            return pd.DataFrame()
    else:
        raise ValueError("Unsupported file format. Only .csv, .xls, .xlsx are supported.")
        
    return df

def infer_col_type(series: pd.Series):
    s = series.dropna().astype(str).str.strip()
    if s.empty:
        return "empty"

    # percent
    if (s.str.contains(r"%", regex=True).mean() > 0.5):
        return "percent"

    # currency (USD, EUR only)
    # Detect formats like "$100", "USD 200", "€300", "EUR 400"
    if (s.str.contains(r"^\$|USD|€|EUR", regex=True, case=False).mean() > 0.3):
        return "currency"

    # numeric (no currency symbols, mostly numbers)
    # Remove commas and spaces before numeric check
    s_num = s.str.replace(r"[^\d\.\-]", "", regex=True)
    numeric_ratio = pd.to_numeric(s_num, errors="coerce").notna().mean()
    if numeric_ratio > 0.5:
        return "numeric"

    # date
    parsed = pd.to_datetime(s, errors="coerce", utc=True)
    if parsed.notna().mean() > 0.5:
        return "date"

    # long text
    avg_len = s.str.len().mean()
    if avg_len > 40:
        return "text"

    # categorical (low unique ratio)
    unique_ratio = s.nunique(dropna=True) / max(len(s), 1)
    if unique_ratio < 0.2:
        return "categorical"

    # default → text
    return "text"

def top_values(series, k=3):
    s = series.dropna().astype(str)
    counts = s.value_counts().head(k)
    return [f"{v}" for v,_ in counts.items()]


def keyword_bag(df, max_kw=20, lang="english"):
    """
    Simple keyword extractor using NLTK:
    - tokenization
    - lowercase normalization
    - stopword removal (removes pronouns, conjunctions, fillers)
    - stemming
    - header boosting (simple)
    - frequency-based ranking
    """
    text = " ".join(
        [" ".join(df[c].dropna().astype(str)) for c in df.columns]
    )
    if not text.strip():
        return ""

    # 1) tokenize
    tokens = word_tokenize(text.lower())

    # 2) clean tokens: keep alphanumeric only
    tokens = [re.sub(r"[^a-z0-9\-']", "", t) for t in tokens]
    tokens = [t for t in tokens if len(t) > 3]          # drop tiny tokens
    tokens = [t for t in tokens if not t.isnumeric()]  # drop pure numbers

    # 3) stopword removal (this removes pronouns, conjunctions, etc.)
    sw = set(stopwords.words(lang))
    tokens = [t for t in tokens if t not in sw]

    # 4) stemming
    stemmer = PorterStemmer()
    stems = [stemmer.stem(t) for t in tokens]

    # 5) header boosting
    header_tokens = " ".join(df.columns).lower()
    header_tokens = [stemmer.stem(t) for t in word_tokenize(header_tokens)]

    header_set = set(header_tokens)

    # 6) frequency counting + boost
    freq = {}
    for s in stems:
        freq[s] = freq.get(s, 0) + 1
        if s in header_set:
            freq[s] += 3   # small boost for header-derived terms

    # 7) rank by score
    ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)

    # 8) return top K
    keywords = [w for w, _ in ranked[:max_kw]]
    return ", ".join(keywords)

def representative_rows(df, k=3, max_cols=8, max_cell_len=60):
    '''
    Score each row by number of unique tokens 
    '''
    row_tokens = []
    for i in range(len(df)):
        cells = df.iloc[i].fillna("").astype(str).tolist()
        joined = " ".join(cells).lower()
        toks = re.findall(r"[a-z0-9]+", joined)
        toks = [t for t in toks if len(t) >= 3]
        row_tokens.append(toks)
        
    scores = []
    for i, toks in enumerate(row_tokens):
        uniq = set(toks)
        score = len(uniq)
        scores.append((score, i))

    top_idx = [i for _, i in sorted(scores, reverse=True)[:k]]

    lines = []
    for i in top_idx:
        cells = df.loc[i].fillna("").astype(str).tolist()[:max_cols]
        clipped = [c if len(c) <= max_cell_len else c[:max_cell_len-1] + "…" for c in cells]
        lines.append(f"-" + " | ".join(clipped))
    return "\n".join(lines)
    

def sheet_to_text(path, sheet=0, max_cols=8, min_table_rows=2, max_keywords=15, max_rows=3):
    tables = read_sheet(path, sheet=sheet, min_table_rows=min_table_rows)
    if isinstance(tables, pd.DataFrame):
        tables = [tables]
    
    parts = []
    
    for table_i, table in enumerate(tables, start=1):
        col_info = []
        for c in table.columns[:max_cols]:
            col_type = infer_col_type(table[c])
            top_vals = top_values(table[c], k=2) if col_type in ("categorical", "text") else []
            col_info.append(
                f"[ {c}:{col_type}" + (f" top:{'; '.join(top_vals)}" if top_vals else "") + " ]"
            )
        
        block = []
        block.append(f"Table {table_i}.")
        block.append("Columns:")
        block.append(",\n".join(col_info))
        block.append("")
        block.append("Keywords:")
        block.append(keyword_bag(table, max_kw=max_keywords))
        
        parts.append("\n".join(block))
    
    # --- Final Samples section (after all tables) ---
    samples = []
    samples.append("Samples:")
    for table_i, table in enumerate(tables, start=1):
        samples.append(f"\nFrom Table {table_i}:")
        samples.append(representative_rows(table, k=max_rows))
    
    # Return all sections
    return "\n\n".join(parts) + "\n\n" + "\n".join(samples)
    
    
        
