## 📌 Project Overview
**Lattice** is a single-user Adaptive Memory Orchestrator using the **ENGRAM** neuro-symbolic memory framework.
- **Target**: Personal use, high-fidelity context tracking for a single human operator.
- **Hardware**: Optimized for 2GB RAM / 1vCPU (e.g., small VPS or Raspberry Pi).
- **Interface**: Discord (via `py-cord`).
- **Goal**: Total evolvability via metadata-driven logic.

## 🏗️ Core Architecture (ENGRAM)
1. **Episodic** (`raw_messages`): Immutable conversation log.
2. **Semantic** (`entities`, `semantic_memories`): Text-based knowledge graph. Relationships are stored as `(subject, predicate, object)` strings without rigid IDs, facilitating natural language evolution and iterative BFS traversal.
3. **Procedural** (`prompt_registry`): Versioned LLM logic.

Details in [lattice/core/memory_orchestrator.py](lattice/core/memory_orchestrator.py).

## 🔄 Core Pipeline
1. **Ingest**: Message stored in episodic memory.
2. **Analyze**: Context Strategy identifies entities and retrieval needs using `CONTEXT_STRATEGY`.
3. **Retrieve**: Iterative BFS traversal across semantic graph.
4. **Generate**: Response produced with grounded context using `UNIFIED_RESPONSE`.
5. **Consolidate**: Async extraction of new entities and memories using `MEMORY_CONSOLIDATION`.

## 📂 Project Structure
- `lattice/core/`: Pipeline, ingestion, and extraction logic.
- `lattice/memory/`: ENGRAM implementations.
    - `graph.py`: Iterative BFS traversal for text-based relationships.
    - `canonical.py`: Deterministic normalization logic.
- `lattice/dreaming/`: Autonomous self-optimization loop (proposers/approvers).
- `lattice/scheduler/`: Task orchestration (triggers and adaptive runners).
- `lattice/discord_client/`: Bot interface and UI handlers.
- `lattice/prompts/`: Template management.
 - `lattice/utils/`: LLM client, auditing, database utilities, and placeholder registry/injector.
- `scripts/`: Database schema, seeding, and migration tools.
- `docs/`: Deep-dive guides for development and testing.
- `tests/`: Unit and integration test suites.
- `Makefile`: Central automation for installation, testing, and execution.

## 🛠️ Development Workflow
Refer to the [Makefile](Makefile) for all available automation.

```bash
make install       # Deps + pre-commit hooks
make init-db       # Initialize database
make nuke-db       # Delete all data
make run           # Run bot locally
make restart       # Restart all services
make test          # Run test suite
make check-all     # Lint, type-check, and test
make view-logs     # View last 100 lines of bot
make view-logs SERVICE=postgres  # View postgres logs
make view-logs TAIL=500         # View 500 lines
```

### Standards
- **Strict Typing**: Mandatory for all functions.
- **Docs**: Google-style docstrings (focus on "why").
- **Quality**: Enforced via Ruff and Mypy.
- **LLM**: All calls must use `AuditingLLMClient` for observability.
- **PRs**: Follow pull request template when creating PRs.

## ⚙️ Key Concepts
- **Dream Channel**: Meta-discussion and prompt approval. Messages here are never ingested.
- **Dreaming Cycle**: Autonomous self-optimization loop ([lattice/dreaming/](lattice/dreaming/)).
- **Canonicalization**: Deterministic normalization of entities and predicates.
- **Context Strategy**: Dynamic retrieval planning based on detected entities.
- **Audit View**: In-Discord UI for inspecting the system's "thought process".
- **Source Links**: Transparent attribution back to episodic memory for all generated memories.

## 📚 Resources
- **[README.md](README.md)**: Technical specs and setup.
- **[DEVELOPMENT.md](docs/DEVELOPMENT.md)**: Environment setup.
- **[TESTING.md](docs/TESTING.md)**: Testing guidelines.
