.PHONY: setup config up status smoke model down install unit integration worker parser-tests vision-model vision-tests embedding-model

# Ollama runs natively on a Mac (GPU); on Linux use `docker compose --profile ollama up -d` and exec into it.
vision-model:
	ollama pull qwen3-vl:4b-instruct

# BGE-M3 dense weights + tokenizer only (~2.3 GB, MIT). The worker never downloads at runtime.
embedding-model:
	.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3', revision='5617a9f61b028005a4858fdac845db406aefb181', cache_dir='.cache/huggingface/hub', allow_patterns=['config.json', 'pytorch_model.bin', 'sentencepiece.bpe.model', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'])"

vision-tests:
	RAG_RUN_VISION=1 .venv/bin/python -m pytest tests/vision -q

worker:
	.venv/bin/python -m pyconsumers.start

parser-tests:
	RAG_RUN_PARSER=1 .venv/bin/python -m pytest tests/parsing -q

# Python environment is independent from the microservices repository.
install:
	python3 -m venv .venv
	.venv/bin/python -m pip install -r pyconsumers/requirements-dev.lock

unit:
	.venv/bin/python -m pytest tests/unit -q

integration:
	RAG_RUN_INTEGRATION=1 .venv/bin/python -m pytest tests/integration -q

setup:
	python3 scripts/setup_env.py

config:
	docker compose config --quiet

up: setup config
	docker compose up -d --wait --wait-timeout 240

status:
	docker compose ps -a

smoke:
	python3 scripts/smoke_infra.py

# Explicit opt-in: downloads model weights into the persistent Ollama volume.
model:
	docker compose exec ollama ollama pull qwen3:4b
	python3 scripts/smoke_infra.py --check-model

# Stops services; preserves all named data volumes.
down:
	docker compose down
