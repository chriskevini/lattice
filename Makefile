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

docker-rebuild: ## Rebuild and restart services (use --no-cache for clean rebuild)
	docker compose down
	docker compose build
	docker compose up -d

docker-restart: ## Quick restart bot (for code changes, no rebuild)
	docker compose restart bot
	@echo "Bot restarted. View logs with: make docker-logs-bot"

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

pre-commit: ## Run all pre-commit hooks on all files
	uv run pre-commit run --all-files

check-discord-v2: ## Check that Discord UI components use V2 APIs
	python scripts/check_discord_v2.py

check-all: lint type-check check-discord-v2 ## Run all quality checks

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
	$(MAKE) migrate

migrate: ## Run database migrations
	uv run python scripts/migrate.py

# ============================================================================
# Local Extraction Model (Optional)
# ============================================================================

setup-local-model: ## Download and setup FunctionGemma-270M for local extraction
	@echo "Setting up local extraction model..."
	@mkdir -p models
	@if [ -f models/functiongemma-270m-q4_k_m.gguf ] && [ -s models/functiongemma-270m-q4_k_m.gguf ]; then \
		echo "✓ Model already exists at models/functiongemma-270m-q4_k_m.gguf"; \
	else \
		echo "Downloading FunctionGemma-270M (4-bit quantized, ~240MB)..."; \
		rm -f models/functiongemma-270m-q4_k_m.gguf; \
		wget --progress=bar:force \
			https://huggingface.co/unsloth/functiongemma-270m-it-GGUF/resolve/main/functiongemma-270m-it-Q4_K_M.gguf \
			-O models/functiongemma-270m-q4_k_m.gguf || { echo "✗ Download failed"; rm -f models/functiongemma-270m-q4_k_m.gguf; exit 1; }; \
		if [ ! -s models/functiongemma-270m-q4_k_m.gguf ]; then \
			echo "✗ Downloaded file is empty"; \
			rm -f models/functiongemma-270m-q4_k_m.gguf; \
			exit 1; \
		fi; \
		echo "✓ Model downloaded successfully"; \
	fi
	@echo ""
	@echo "Installing optional dependencies..."
	@if [ -n "$$VIRTUAL_ENV" ] || uv venv --help >/dev/null 2>&1 && [ -d .venv ]; then \
		uv pip install -e ".[local-extraction]"; \
	else \
		echo "⚠ No virtual environment detected. Skipping Python package installation."; \
		echo "   For Docker deployments, dependencies will be installed during image build."; \
		echo "   For local development, run 'make install' first to create a venv."; \
	fi
	@echo ""
	@echo "Configuring .env..."
	@if [ ! -f .env ]; then \
		echo "✗ .env file not found. Creating from .env.example..."; \
		cp .env.example .env; \
		echo "⚠ WARNING: Please update .env with your API keys and tokens!"; \
	fi
	@if grep -q "^LOCAL_EXTRACTION_MODEL_PATH=" .env; then \
		echo "✓ LOCAL_EXTRACTION_MODEL_PATH already configured in .env"; \
	elif grep -q "^# LOCAL_EXTRACTION_MODEL_PATH=" .env; then \
		sed -i 's|^# LOCAL_EXTRACTION_MODEL_PATH=.*|LOCAL_EXTRACTION_MODEL_PATH=./models/functiongemma-270m-q4_k_m.gguf|' .env; \
		echo "✓ Enabled LOCAL_EXTRACTION_MODEL_PATH in .env"; \
	else \
		echo "" >> .env; \
		echo "# Local extraction model (added by make setup-local-model)" >> .env; \
		echo "LOCAL_EXTRACTION_MODEL_PATH=./models/functiongemma-270m-q4_k_m.gguf" >> .env; \
		echo "✓ Added LOCAL_EXTRACTION_MODEL_PATH to .env"; \
	fi
	@if grep -q "^INCLUDE_LOCAL_EXTRACTION=" .env; then \
		sed -i 's|^INCLUDE_LOCAL_EXTRACTION=.*|INCLUDE_LOCAL_EXTRACTION=true|' .env; \
	elif grep -q "^# INCLUDE_LOCAL_EXTRACTION=" .env; then \
		sed -i 's|^# INCLUDE_LOCAL_EXTRACTION=.*|INCLUDE_LOCAL_EXTRACTION=true|' .env; \
	else \
		echo "INCLUDE_LOCAL_EXTRACTION=true" >> .env; \
	fi
	@echo "✓ Enabled INCLUDE_LOCAL_EXTRACTION for Docker builds"
	@echo ""
	@echo "✅ Setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  For Docker deployment: make docker-rebuild"
	@echo "  (This rebuilds the image with local-extraction support)"
	@echo ""
	@echo "Benefits:"
	@echo "  - Lower API costs (extraction runs locally)"
	@echo "  - Faster extraction (< 200ms vs ~500ms API)"
	@echo "  - Privacy (on-device processing)"
	@echo "  - Automatic fallback to API if unavailable"

check-local-model: ## Check if local extraction model is configured and available
	@echo "Checking local extraction model setup..."
	@if [ -f models/functiongemma-270m-q4_k_m.gguf ]; then \
		echo "✓ Model file exists: models/functiongemma-270m-q4_k_m.gguf"; \
	else \
		echo "✗ Model file not found. Run: make setup-local-model"; \
	fi
	@if grep -q "^LOCAL_EXTRACTION_MODEL_PATH=" .env 2>/dev/null; then \
		echo "✓ LOCAL_EXTRACTION_MODEL_PATH configured in .env"; \
	else \
		echo "✗ LOCAL_EXTRACTION_MODEL_PATH not set in .env"; \
	fi
	@if uv pip show llama-cpp-python >/dev/null 2>&1; then \
		echo "✓ llama-cpp-python installed"; \
	else \
		echo "✗ llama-cpp-python not installed. Run: make setup-local-model"; \
	fi

update: ## Update dependencies
	uv update
	uv run pre-commit autoupdate
