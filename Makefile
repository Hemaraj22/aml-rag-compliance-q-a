.PHONY: install ingest serve eval test docker-up docker-ingest docker-eval

install:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

ingest:
	.venv/bin/python -m app.ingest

serve:
	.venv/bin/uvicorn app.main:app --reload --port 8000

eval:
	.venv/bin/python -m eval.run_eval

test:
	.venv/bin/pytest -q

docker-up:
	docker compose up --build api

docker-ingest:
	docker compose --profile jobs run --rm ingest

docker-eval:
	docker compose --profile jobs run --rm eval
