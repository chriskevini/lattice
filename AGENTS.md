# Agent Onboarding Guide

## 📌 Project Overview
**Lattice** is an Adaptive Memory Orchestrator—a self-evolving Discord companion using the **ENGRAM** neuro-symbolic memory framework.
- **Constraints**: 2GB RAM / 1vCPU.
- **Stack**: Python 3.12+, PostgreSQL 15+, py-cord 2.7+.
- **Core Goal**: Total evolvability via metadata-driven logic.

---

## 🏗️ [Core Architecture](#core-architecture)
### [Three-Tier Memory (ENGRAM)](#three-tier-memory-engram)
1. **[Episodic](#episodic-memory)** (`raw_messages`): Immutable conversation log.
2. **[Semantic](#semantic-memory)** (`entities` + `semantic_triples`): Graph-first knowledge with entity extraction (replacing vector embeddings).
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
- **Rendered Prompt Storage**: Always store the fully rendered prompt sent to LLM (not just ingredients). Enables auditability and Dreaming Cycle optimization.
- **Testing**: Write tests for all new features. See [TESTING.md](docs/TESTING.md) for guidelines on mock completeness, type correctness, and common pitfalls.

---

## ⚙️ [Key Implementation Details](#key-implementation-details)

### [Pipeline Flow](#pipeline-flow)
1. **Ingestion**: Message or proactive trigger.
2. **Short-Circuit**: North Star or feedback detection.
3. **Logging**: Episodic storage.
4. **Query Extraction**: Simplified 2-field extraction (message type, entities).
5. **Retrieval**: Entity-driven adaptive context (see [Context Strategy](#context-strategy)).
6. **Generation**: `prompt_registry` template execution.
7. **Consolidation**: Async extraction of entities, triples, and activities.

### [Entity Extraction System](#entity-extraction-system)
Extracts entity mentions from messages for graph traversal:
- **Entities**: Array of named entities referenced in the message
  - Used for entity-driven graph traversal (see [Context Strategy](#context-strategy))
  - Example: `["lattice project", "Friday", "PostgreSQL"]`

**Design Philosophy**:
- Message intent (questions, goals, activities) is inferred naturally by the UNIFIED_RESPONSE template
- Simpler extraction = more reliable, faster, easier to maintain

### [Context Strategy](#context-strategy)
**Entity-Driven Adaptive Retrieval**:

Always retrieves **15 recent messages** (generous conversation history), but graph traversal is adaptive:
- **No entities**: `triple_depth=0` (self-contained messages like greetings, simple activities)
- **Has entities**: `triple_depth=2` (deep graph traversal for multi-hop relationships)

**Rationale**:
- Conversation history is cheap and always useful
- Graph traversal is expensive; only do it when entities provide starting points
- Depth=2 finds multi-hop relationships (e.g., project → deadline → date)

### [Memory Optimization](#memory-optimization)
- **Streaming**: Use async generators for large message sets.
- **Pooling**: Strict connection limits (`min_size=2, max_size=5`).

---

## 📚 [Resources](#resources)
- **[README.md](README.md)**: Installation, [Database Schema](README.md#database-schema), and Config.
- **[DEVELOPMENT.md](docs/DEVELOPMENT.md)**: Setup and troubleshooting.
- **[TESTING.md](docs/TESTING.md)**: Testing guidelines and best practices.

---

## 🎯 Philosophy
1. **Logic as Data**: Prompts and archetypes live in the DB.
2. **Respect Constraints**: 2GB RAM is a hard wall.
3. **Document "Why"**: The "what" is in the code.
