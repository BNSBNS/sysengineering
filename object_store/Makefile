.PHONY: install dev lint format typecheck test test-unit test-integration test-chaos benchmark clean help

install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src/ tests/
	black --check src/ tests/

format:
	ruff check --fix src/ tests/
	black src/ tests/

typecheck:
	mypy src/

test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit/ -v -m unit

test-integration:
	pytest tests/integration/ -v -m integration

test-chaos:
	pytest tests/chaos/ -v -m chaos

benchmark:
	pytest tests/benchmarks/ -v --benchmark-json=benchmark_results.json

docker-build:
	docker build -t object_store:latest .

docker-run:
	docker-compose up -d

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

help:
	@echo "Available targets: install dev lint format typecheck test clean"
