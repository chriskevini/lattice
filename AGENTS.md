# Agent Onboarding Guide

## 📌 Project Overview
**Lattice** is a single-user Adaptive Memory Orchestrator using the **ENGRAM** neuro-symbolic memory framework.
- **Target**: Personal use, high-fidelity context tracking for a single human operator.
- **Hardware**: Optimized for 2GB RAM / 1vCPU (e.g., small VPS or Raspberry Pi).
- **Stack**: Python 3.12+, PostgreSQL 15+, py-cord 2.7+.
- **Goal**: Total evolvability via metadata-driven logic.

## 🏗️ Core Architecture (ENGRAM)
1. **Episodic** (`raw_messages`): Immutable conversation log.
2. **Semantic** (`entities`, `semantic_memories`): Graph-first knowledge with entity resolution.
3. **Procedural** (`prompt_registry`): Versioned LLM logic.

Details in [lattice/core/memory_orchestrator.py](lattice/core/memory_orchestrator.py).

## 📂 Project Structure
- `lattice/core/`: Pipeline, ingestion, and extraction logic.
- `lattice/memory/`: ENGRAM implementations (episodic, graph, etc.).
- `lattice/discord_client/`: Bot interface and UI handlers.
- `lattice/prompts/`: Template management.
- `tests/`: Unit and integration test suites.

## 🛠️ Development Workflow
Refer to the [Makefile](Makefile) for all available automation.

```bash
make install    # Deps + pre-commit hooks
make test       # Run test suite
make check-all  # Lint, type-check, and test
make run        # Run bot locally
```

### Standards
- **Strict Typing**: Mandatory for all functions.
- **Docs**: Google-style docstrings (focus on "why").
- **Quality**: Enforced via Ruff and Mypy.
- **Commits**: Conventional Commits required.
- **LLM**: All calls must use `AuditingLLMClient` for observability.

## ⚙️ Key Concepts
- **Dream Channel**: Meta-discussion and prompt approval. Messages here are never ingested.
- **Dreaming Cycle**: Autonomous self-optimization loop ([lattice/dreaming/](lattice/dreaming/)).
- **Canonicalization**: Deterministic normalization of entities and predicates.
- **Context Strategy**: Dynamic retrieval planning based on detected entities.
- **Audit View**: In-Discord UI for inspecting the system's "thought process" for any response.
- **Source Links**: Transparent attribution back to episodic memory for all generated facts.

## 📚 Resources
- **[README.md](README.md)**: Technical specs, placeholders, and setup.
- **[DEVELOPMENT.md](docs/DEVELOPMENT.md)**: Environment setup.
- **[TESTING.md](docs/TESTING.md)**: Testing guidelines.
