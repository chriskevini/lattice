.PHONY: help install test lint format type-check security clean run docker-up docker-down docker-logs docker-rebuild

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ============================================================================
# Docker Commands
# ============================================================================

docker-up: ## Start all services with Docker Compose
	docker compose up -d

docker-down: ## Stop all services
	docker compose down

docker-logs: ## Show logs from all services
	docker compose logs -f

docker-logs-bot: ## Show logs from bot service only
	docker compose logs -f bot

docker-logs-db: ## Show logs from database service only
	docker compose logs -f postgres

docker-rebuild: ## Rebuild and restart services
	docker compose down
	docker compose build --no-cache
	docker compose up -d

docker-shell: ## Open shell in bot container
	docker compose exec bot /bin/bash

docker-db-shell: ## Open PostgreSQL shell
	docker compose exec postgres psql -U lattice -d lattice

docker-clean: ## Remove all containers, volumes, and images
	docker compose down -v
	docker system prune -f

# ============================================================================
# Development Commands
# ============================================================================

install: ## Install dependencies and setup pre-commit hooks
	uv sync
	uv run pre-commit install
	uv run pre-commit install --hook-type commit-msg

# ============================================================================
# Testing & Quality
# ============================================================================

test: ## Run all tests with coverage
	uv run pytest --cov --cov-report=term-missing

test-fast: ## Run tests without coverage (faster)
	uv run pytest -x

lint: ## Run linting checks
	uv run ruff check .

lint-fix: ## Run linting checks and auto-fix issues
	uv run ruff check --fix .

format: ## Format code with ruff
	uv run ruff format .

format-check: ## Check if code is formatted correctly
	uv run ruff format --check .

type-check: ## Run type checking with mypy
	uv run mypy lattice

security: ## Run security checks with ruff
	uv run ruff check --select S .

pre-commit: ## Run all pre-commit hooks on all files
	uv run pre-commit run --all-files

check-all: lint type-check security test ## Run all quality checks

# ============================================================================
# Application Commands
# ============================================================================

run: ## Run the Discord bot (local development)
	uv run python -m lattice

clean: ## Clean up cache and build files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".uv" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ htmlcov/ .coverage

# ============================================================================
# Git Helpers
# ============================================================================

commit: ## Interactive conventional commit helper
	uv run cz commit

bump-version: ## Bump version using commitizen
	uv run cz bump

# ============================================================================
# Database Management
# ============================================================================

init-db: ## Initialize database schema
	uv run python scripts/init_db.py

update: ## Update dependencies
	uv update
	uv run pre-commit autoupdate
