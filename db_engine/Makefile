.PHONY: all install dev lint format typecheck test test-unit test-integration test-chaos test-property test-security benchmark clean docker-build docker-run docker-stop help

# Default target
all: lint typecheck test

# Installation
install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	pre-commit install

# Code quality
lint:
	ruff check src/ tests/
	black --check src/ tests/

format:
	ruff check --fix src/ tests/
	black src/ tests/

typecheck:
	mypy src/

# Testing
test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit/ -v --tb=short -m unit

test-integration:
	pytest tests/integration/ -v --tb=short -m integration

test-chaos:
	pytest tests/chaos/ -v --tb=short -m chaos

test-property:
	pytest tests/property/ -v --tb=short -m property

test-security:
	pytest tests/security/ -v --tb=short -m security

benchmark:
	pytest tests/benchmarks/ -v --benchmark-json=benchmark_results.json

test-coverage:
	pytest tests/ --cov=src/db_engine --cov-report=html --cov-report=term-missing

# Docker
docker-build:
	docker build -t db_engine:latest .

docker-run:
	docker-compose up -d db_engine

docker-stop:
	docker-compose down

docker-test:
	docker-compose --profile test run --rm db_engine_test

docker-logs:
	docker-compose logs -f db_engine

# Cleanup
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage coverage.xml
	rm -rf benchmark_results.json
	rm -rf dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Help
help:
	@echo "Available targets:"
	@echo "  install        - Install the package"
	@echo "  dev            - Install with development dependencies"
	@echo "  lint           - Run linters (ruff, black)"
	@echo "  format         - Auto-format code"
	@echo "  typecheck      - Run mypy type checking"
	@echo "  test           - Run all tests"
	@echo "  test-unit      - Run unit tests only"
	@echo "  test-integration - Run integration tests"
	@echo "  test-chaos     - Run chaos/fault injection tests"
	@echo "  test-property  - Run property-based tests"
	@echo "  test-security  - Run security tests"
	@echo "  benchmark      - Run performance benchmarks"
	@echo "  test-coverage  - Run tests with coverage report"
	@echo "  docker-build   - Build Docker image"
	@echo "  docker-run     - Run in Docker"
	@echo "  docker-stop    - Stop Docker container"
	@echo "  docker-test    - Run tests in Docker"
	@echo "  clean          - Clean build artifacts"
