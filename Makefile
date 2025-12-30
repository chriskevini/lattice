.PHONY: help install test lint format type-check security clean run

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies and setup pre-commit hooks
	poetry install
	poetry run pre-commit install
	poetry run pre-commit install --hook-type commit-msg

test: ## Run all tests with coverage
	poetry run pytest --cov --cov-report=term-missing

test-fast: ## Run tests without coverage (faster)
	poetry run pytest -x

lint: ## Run linting checks
	poetry run ruff check .

lint-fix: ## Run linting checks and auto-fix issues
	poetry run ruff check --fix .

format: ## Format code with ruff
	poetry run ruff format .

format-check: ## Check if code is formatted correctly
	poetry run ruff format --check .

type-check: ## Run type checking with mypy
	poetry run mypy lattice

security: ## Run security checks with bandit
	poetry run bandit -r lattice

pre-commit: ## Run all pre-commit hooks on all files
	poetry run pre-commit run --all-files

check-all: lint type-check security test ## Run all quality checks

clean: ## Clean up cache and build files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ htmlcov/ .coverage

run: ## Run the Discord bot
	poetry run python -m lattice

commit: ## Interactive conventional commit helper
	poetry run cz commit

bump-version: ## Bump version using commitizen
	poetry run cz bump

init-db: ## Initialize database schema
	poetry run python scripts/init_db.py

update: ## Update dependencies
	poetry update
	poetry run pre-commit autoupdate
