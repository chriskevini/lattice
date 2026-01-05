# Development Guide

## Quick Start with Docker (Recommended)

The fastest way to get started is using Docker Compose:

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Edit .env with your credentials
# - Add your OPENROUTER_API_KEY
# - Add your DISCORD_TOKEN
# - Add your DISCORD_MAIN_CHANNEL_ID and DISCORD_DREAM_CHANNEL_ID

# 3. Start all services (PostgreSQL + Bot)
make docker-up

# 4. View logs
make docker-logs

# 5. Stop services
make docker-down
```

### Docker Commands

```bash
make docker-up          # Start all services
make docker-down        # Stop all services
make docker-logs        # Show all logs
make docker-logs-bot    # Show bot logs only
make docker-logs-db     # Show database logs only
make docker-rebuild     # Rebuild and restart
make docker-shell       # Open shell in bot container
make docker-db-shell    # Open PostgreSQL shell
make docker-clean       # Remove all containers and volumes
```

## Local Development Setup

### Prerequisites
- Python 3.12+
- [UV](https://docs.astral.sh/uv/) (modern Python package manager)
- PostgreSQL 16+ (or use Docker)
- Docker & Docker Compose (for containerized development)

### Installation

```bash
# Install UV (if not already installed)
pip install uv

# Install dependencies and dev tools
uv sync --extra dev

# Install pre-commit hooks
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg
```

## Code Quality Standards

This project enforces strict code quality standards using automated tooling:

### Linting & Formatting (Ruff)

Ruff is an ultra-fast Python linter and formatter that replaces multiple tools (flake8, black, isort, etc.):

```bash
# Check code
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

### Type Checking (mypy)

Strict type checking is enforced:

```bash
uv run mypy lattice
```

### Security Checks (Ruff)

Security checks are integrated into Ruff with the `S` rule set:

```bash
uv run ruff check --select S .
```

### Testing (pytest)

```bash
# Run all tests
uv run pytest

# With coverage report
uv run pytest --cov

# Run specific test file
uv run pytest tests/unit/test_memory.py

# Run with verbose output
uv run pytest -v
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
uv run cz commit
```

### Pre-commit Hooks

Hooks automatically run on `git commit`:

1. **Ruff linting & formatting** - Auto-fixes code style issues
2. **mypy type checking** - Catches type errors
3. **Conventional commit validation** - Ensures proper commit messages
4. **General checks** - File size, merge conflicts, YAML/JSON syntax, etc.

**Run manually on all files:**
```bash
uv run pre-commit run --all-files
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
    uv run pytest
    uv run mypy lattice
    uv run ruff check .
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
uv run pre-commit autoupdate

# Clear cache and retry
uv run pre-commit clean
uv run pre-commit run --all-files
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
uv sync
```

### High Memory Usage
- Check connection pool size (should be 2-5)
- Verify vector search limits (max 15 results)
- Profile with `memory_profiler` or `tracemalloc`
- Consider reducing batch sizes for processing

### Graph Traversal Performance
- Check entity table structure: `\d entities` in psql
- Verify graph indexes exist on semantic_triples
- Analyze query plan: `EXPLAIN ANALYZE SELECT ...`
- Consider increasing `effective_cache_size` in PostgreSQL

### Discord Rate Limiting
- Implement exponential backoff (see Discord patterns below)
- Add request queuing with `asyncio.Queue`
- Monitor rate limit headers in responses
- Cache frequently accessed data

### Type Errors After Dependency Update
```bash
make type-check  # Run mypy
# Fix or add type: ignore comments judiciously
# Update stubs if needed: uv add types-xxx
```

## Discord-Specific Patterns

```python
# ✅ Invisible feedback detection
if message.reference and message.reference.resolved.author.bot:
    await handle_invisible_feedback(message)
    await message.add_reaction("🫡")
    return  # Don't log to raw_messages

# ✅ Rate limiting with exponential backoff
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
async def send_message(content: str) -> None:
    await channel.send(content)

# ✅ North Star detection
if is_north_star_declaration(message.content):
    await upsert_north_star_fact(message)
    await message.reply("Noted 🌟")
    return
```

## Testing Patterns

### Unit Tests

Test isolated components without external dependencies:

```python
# tests/unit/test_memory.py
async def test_extract_semantic_triple():
    triple = extract_triple("User prefers dark mode")
    assert triple.predicate == "prefers"
    assert "dark mode" in triple.object_text
```

### Integration Tests

Test full pipeline with test database:

```python
# tests/integration/test_pipeline.py
async def test_full_ingestion_pipeline(test_db):
    message = create_test_message("I love Python")
    await ingest_message(message)

    facts = await query_facts_by_embedding("Python programming")
    assert len(facts) > 0
```

### Performance Tests

Validate memory/CPU constraints:

```python
@pytest.mark.performance
async def test_memory_usage_under_2gb():
    with memory_profiler():
        await process_large_batch(1000)
        assert peak_memory_mb < 2000
```

## Debugging Queries

### Graph Query Debugging

```sql
-- Check entity relationships
SELECT s.name AS subject, t.predicate, o.name AS object
FROM semantic_triples t
JOIN entities s ON t.subject_id = s.id
JOIN entities o ON t.object_id = o.id
ORDER BY t.created_at DESC
LIMIT 10;

-- Analyze index stats
SELECT * FROM pg_indexes WHERE tablename IN ('entities', 'semantic_triples');
```

### Temporal Chain Debugging

```sql
-- Traverse conversation chain
WITH RECURSIVE conversation_chain AS (
    SELECT id, content, is_bot, prev_turn_id, timestamp, 0 as depth
    FROM raw_messages
    WHERE id = $1  -- Current message

    UNION ALL

    SELECT rm.id, rm.content, rm.is_bot, rm.prev_turn_id, rm.timestamp, cc.depth + 1
    FROM raw_messages rm
    INNER JOIN conversation_chain cc ON rm.id = cc.prev_turn_id
    WHERE cc.depth < 10
)
SELECT * FROM conversation_chain ORDER BY depth DESC;
```

## Deployment

### Environment Setup

- **Platform**: 2GB RAM / 1vCPU VPS (Oracle Cloud, DigitalOcean, Hetzner, etc.)
- **PostgreSQL**: Version 16+
- **Process Manager**: systemd or supervisord (handle crashes gracefully)
- **Logging**: Structured logs (structlog) to file + stdout
- **Monitoring**: Track memory usage, response times, error rates
- **Backups**: Daily database dumps before dreaming cycle

### Systemd Service Example

```ini
[Unit]
Description=Lattice Discord Bot
After=network.target postgresql.service

[Service]
Type=simple
User=lattice
WorkingDirectory=/home/lattice/app
Environment="PATH=/home/lattice/.local/bin:/usr/bin"
ExecStart=/home/lattice/.local/bin/uv run python -m lattice
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## Resources

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [mypy Documentation](https://mypy.readthedocs.io/)
- [pytest Documentation](https://docs.pytest.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [discord.py Documentation](https://discordpy.readthedocs.io/)
