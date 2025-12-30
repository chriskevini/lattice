# Development Guide

## Setup

### Prerequisites
- Python 3.11+
- Poetry (recommended) or pip
- PostgreSQL 16+ with pgvector extension

### Installation

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Or with pip
pip install -e ".[dev]"

# Install pre-commit hooks
poetry run pre-commit install
poetry run pre-commit install --hook-type commit-msg
```

## Code Quality Standards

This project enforces strict code quality standards using automated tooling:

### Linting & Formatting (Ruff)

Ruff is an ultra-fast Python linter and formatter that replaces multiple tools (flake8, black, isort, etc.):

```bash
# Check code
poetry run ruff check .

# Auto-fix issues
poetry run ruff check --fix .

# Format code
poetry run ruff format .
```

### Type Checking (mypy)

Strict type checking is enforced:

```bash
poetry run mypy lattice
```

### Security Checks (Bandit)

Automated security vulnerability scanning:

```bash
poetry run bandit -r lattice
```

### Testing (pytest)

```bash
# Run all tests
poetry run pytest

# With coverage report
poetry run pytest --cov

# Run specific test file
poetry run pytest tests/unit/test_memory.py

# Run with verbose output
poetry run pytest -v
```

## Git Workflow

### Conventional Commits

This project uses [Conventional Commits](https://www.conventionalcommits.org/) enforced via Commitizen:

**Format:** `<type>(<scope>): <description>`

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `build`: Build system changes
- `ci`: CI/CD changes
- `chore`: Maintenance tasks

**Examples:**
```bash
git commit -m "feat(memory): add ENGRAM triple extraction"
git commit -m "fix(discord): handle rate limiting correctly"
git commit -m "docs: update setup instructions"
```

**Interactive commit helper:**
```bash
poetry run cz commit
```

### Pre-commit Hooks

Hooks automatically run on `git commit`:

1. **Ruff linting & formatting** - Auto-fixes code style issues
2. **mypy type checking** - Catches type errors
3. **Conventional commit validation** - Ensures proper commit messages
4. **General checks** - File size, merge conflicts, YAML/JSON syntax, etc.

**Run manually on all files:**
```bash
poetry run pre-commit run --all-files
```

**Skip hooks (not recommended):**
```bash
git commit --no-verify
```

## Project Structure

```
lattice/
├── core/           # Core orchestration and pipeline logic
├── memory/         # ENGRAM memory system (episodic, semantic, procedural)
├── discord_client/ # Discord bot interface
├── prompts/        # Prompt registry and templates
└── utils/          # Shared utilities

tests/
├── unit/           # Unit tests
└── integration/    # Integration tests

scripts/            # Deployment and utility scripts
```

## Development Workflow

1. **Create a branch:**
   ```bash
   git checkout -b feat/your-feature
   ```

2. **Write code with type hints:**
   ```python
   from typing import List
   
   def process_message(content: str, user_id: int) -> List[str]:
       """Process a Discord message.
       
       Args:
           content: The message content
           user_id: The Discord user ID
           
       Returns:
           List of extracted facts
       """
       ...
   ```

3. **Test locally:**
   ```bash
   poetry run pytest
   poetry run mypy lattice
   poetry run ruff check .
   ```

4. **Commit with conventional commits:**
   ```bash
   git add .
   git commit -m "feat(memory): add semantic triple extraction"
   ```

5. **Pre-commit hooks run automatically** and may auto-fix issues. If fixes are made, review and commit again.

## Performance Considerations (2GB RAM / 1vCPU)

### Memory Optimization
- Use lightweight embedding models: `all-MiniLM-L6-v2` (384-dim, ~80MB)
- Batch operations to reduce overhead
- Stream large datasets instead of loading into memory
- Use `asyncpg` connection pooling efficiently

### Database Optimization
- Limit vector search results
- Use HNSW index parameters tuned for low resources:
  ```sql
  WITH (m = 16, ef_construction = 64)
  ```
- Avoid large JOINs, prefer multiple queries

### Discord Bot Optimization
- Use lightweight discord.py client
- Implement exponential backoff for rate limiting
- Cache frequently accessed data

## Environment Variables

Create a `.env` file:

```bash
# Discord
DISCORD_TOKEN=your_bot_token
DISCORD_CONTROL_CHANNEL_ID=123456789

# Database
DATABASE_URL=postgresql://user:pass@localhost/lattice

# AI/ML
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_CACHE_DIR=./models

# Logging
LOG_LEVEL=INFO
```

## Troubleshooting

### Pre-commit hooks failing
```bash
# Update hooks to latest versions
poetry run pre-commit autoupdate

# Clear cache and retry
poetry run pre-commit clean
poetry run pre-commit run --all-files
```

### Type checking errors
```bash
# Reveal type of expression
reveal_type(my_variable)

# Ignore specific line (use sparingly)
my_variable = some_function()  # type: ignore[arg-type]
```

### Import errors
```bash
# Ensure proper installation
poetry install

# Check PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Resources

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [mypy Documentation](https://mypy.readthedocs.io/)
- [pytest Documentation](https://docs.pytest.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [discord.py Documentation](https://discordpy.readthedocs.io/)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
