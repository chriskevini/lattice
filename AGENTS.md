# Agent Onboarding Guide

## 📌 Project Overview
**Lattice** is an Adaptive Memory Orchestrator—a self-evolving Discord companion using the **ENGRAM** neuro-symbolic memory framework.
- **Constraints**: 2GB RAM / 1vCPU.
- **Stack**: Python 3.12+, PostgreSQL + pgvector, discord.py.
- **Core Goal**: Total evolvability via metadata-driven logic.

---

## 🏗️ [Core Architecture](#core-architecture)
### [Three-Tier Memory (ENGRAM)](#three-tier-memory-engram)
1. **[Episodic](#episodic-memory)** (`raw_messages`): Immutable conversation log.
2. **[Semantic](#semantic-memory)** (`stable_facts` + `semantic_triples`): Vector-embedded knowledge + graph relationships.
3. **[Procedural](#procedural-memory)** (`prompt_registry`): Evolving templates via the [Dreaming Cycle](#dreaming-cycle).

### [Key Design Principles](#key-design-principles)
- **Canonical Integrity**: No internal thoughts in public channels.
- **Unified Pipeline**: Same flow for reactive and proactive inputs.
- **Strict Channel Separation**:
  - `DISCORD_MAIN_CHANNEL_ID`: Stored episodic memory.
  - `DISCORD_DREAM_CHANNEL_ID`: Meta-discussion & approvals (never stored).

---

## 📂 [Project Structure](#project-structure)
- `lattice/core/`: Pipeline & ingestion logic.
- `lattice/memory/`: [ENGRAM](#three-tier-memory-engram) implementations.
- `lattice/discord_client/`: Bot interface & [Dream Channel](#strict-channel-separation) UI.
- `lattice/prompts/`: Templates & extraction strategies.
- `tests/`: [Unit and Integration tests](#development-workflow).

---

## 🛠️ [Development Workflow](#development-workflow)
### Quick Start
```bash
make install        # Install deps + hooks
make test           # Run tests
make check-all      # Lint, type-check, and test
```

### Standards
- **Strict Typing**: All functions must have type annotations.
- **Documentation**: Google-style docstrings (focus on "why").
- **Quality**: Enforced via Ruff (linting/formatting) and Mypy.
- **Commits**: [Conventional Commits](https://www.conventionalcommits.org/) required.

---

## ⚙️ [Key Implementation Details](#key-implementation-details)

### [Pipeline Flow](#pipeline-flow)
1. **Ingestion**: Message or proactive trigger.
2. **Short-Circuit**: North Star or feedback detection.
3. **Logging**: Episodic storage.
4. **[Context Analysis](#context-archetype-system)**: Dynamic scaling based on semantic archetypes.
5. **Retrieval**: Hybrid (Recency + Relevance + Relationships).
6. **Generation**: `prompt_registry` template execution.
7. **Consolidation**: Async extraction of facts and triples.

### [Context Archetype System](#context-archetype-system)
Automatically optimizes retrieval parameters (`CONTEXT_TURNS`, `VECTOR_LIMIT`, etc.) by matching message embeddings against centroids.
- **Validation**: Limits enforced via DB constraints to respect 2GB RAM.
- **Evolvability**: AI proposes archetypes → Human approves → Hot-reload.

### [Memory Optimization](#memory-optimization)
- **Streaming**: Use async generators for large message sets.
- **Pooling**: Strict connection limits (`min_size=2, max_size=5`).

---

## 📚 [Resources](#resources)
- **[README.md](README.md)**: Installation, [Database Schema](README.md#database-schema), and Config.
- **[DEVELOPMENT.md](DEVELOPMENT.md)**: Setup and troubleshooting.

---

## 🎯 Philosophy
1. **Logic as Data**: Prompts and archetypes live in the DB.
2. **Respect Constraints**: 2GB RAM is a hard wall.
3. **Document "Why"**: The "what" is in the code.
