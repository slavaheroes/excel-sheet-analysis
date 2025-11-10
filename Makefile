NAME?=slava-docker-generative
DATA_PATH?=/

build:
	docker build -t $(NAME) --network=host .

attach:
	docker attach $(NAME)

exec:
	docker exec -it slava-docker-seg /bin/bash

run:
	docker run --gpus all --rm -it \
	--net=host \
	--ipc=host \
	-v $(DATA_PATH):/workspace/ext_data \
	-v $(PWD):/workspace \
	-p 8080:8080 \
	--name=slava-docker-seg \
	$(NAME)

build-run: build run
