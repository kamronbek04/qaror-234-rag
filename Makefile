# Unix shortcuts; every target is a plain command documented in README.md.
PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: install ingest run test test-ollama lint eval-retrieval eval-e2e up up-gpu down logs

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -e ".[dev]"
	[ -f .env ] || cp .env.example .env

ingest:
	$(BIN)/python -m app.cli ingest

run:
	$(BIN)/uvicorn app.main:app --host 0.0.0.0 --port 8000

test:
	$(BIN)/pytest

test-ollama:
	$(BIN)/pytest -m ollama

lint:
	$(BIN)/ruff check . && $(BIN)/ruff format --check .

eval-retrieval:
	$(BIN)/python -m app.cli eval retrieval --compare-chunking

eval-e2e:
	$(BIN)/python -m app.cli eval e2e

up:
	docker compose up -d

up-gpu:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

down:
	docker compose down

logs:
	docker compose logs -f api
