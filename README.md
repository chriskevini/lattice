# Lattice: Adaptive Memory Orchestrator

**Lattice** is a self-evolving Discord companion powered by the **ENGRAM** neuro-symbolic memory framework. It is specifically engineered to achieve high-order cognitive capabilities within extreme hardware constraints (2GB RAM / 1vCPU).

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## 🌟 Key Features

- **ENGRAM Memory Framework**: Three-tier neuro-symbolic system (Episodic, Semantic, Procedural).
- **Query Extraction**: API-based structured message analysis (Google Gemini Flash 1.5).
- **Dreaming Cycle**: Self-optimization loop where AI proposes logic updates for human approval.
- **Resource-First Design**: Native support for low-memory environments via streaming and pooling.
- **Graph-First Retrieval**: Unified search across temporal logs and relationship graphs.
- **Invisible Alignment**: Non-intrusive feedback loops using Discord's native UI elements.

---

## 🏗️ Architecture

Lattice operates on the principle that **logic is data**. By moving prompts and configuration into the database, the system achieves total evolvability without code changes.

### Memory Tiers
1.  **Episodic ([`raw_messages`](#1-episodic-memory-raw_messages))**: Immutable, time-ordered interaction log.
2.  **Semantic ([`entities`](#2-semantic-memory-entities) + [`semantic_triples`](#3-semantic-relationships-semantic_triples))**: Graph-based knowledge using entity extraction.
3.  **Procedural ([`prompt_registry`](#4-procedural-memory-prompt_registry))**: Versioned templates and behavioral strategies.

### Unified Pipeline
`Ingestion → Short-Circuit → Query Extraction → Graph Retrieval → Generation → Consolidation`

For a deep dive into the technical implementation, see **[AGENTS.md](AGENTS.md)**.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- PostgreSQL 15+
- [UV](https://github.com/astral-sh/uv) (recommended package manager)

### Setup
```bash
make install        # Install dependencies and pre-commit hooks
cp .env.example .env # Configure your tokens and DB
make init-db        # Initialize schema and base templates
make run            # Start the orchestrator
```

---

## ⚙️ Configuration

### Hardware Limits
Lattice enforces strict bounds to stay within 2GB RAM:
```env
MAX_EPISODIC_CONTEXT_TURNS=20
MAX_TRIPLE_DEPTH=3
```

### Discord Channels
- **`DISCORD_MAIN_CHANNEL_ID`**: The public face. Conversations are stored here.
- **`DISCORD_DREAM_CHANNEL_ID`**: The "subconscious". Meta-discussion and approval UI. **Never stored.**

---

## 🗄️ Database Schema

See [scripts/schema.sql](scripts/schema.sql) for the canonical schema.

### Quick Reference
| Table | Purpose |
|-------|---------|
| `raw_messages` | Stored Discord messages |
| `message_extractions` | Query extraction output |
| `entities` | Entity registry |
| `semantic_triples` | Graph relationships |
| `objectives` | User goals |
| `prompt_registry` | Prompt templates |
| `prompt_audits` | LLM call tracking |
| `dreaming_proposals` | Prompt optimization |
| `user_feedback` | User feedback |
| `system_health` | Configuration |

### Setup
```bash
make install        # Install dependencies and pre-commit hooks
cp .env.example .env # Configure your tokens and DB
make init-db        # Initialize schema and seed data
make run            # Start the bot
```

**Note:** No migrations needed for fresh setup. All prompt templates are in `scripts/seed.sql`.

**Lattice**: Building adaptive AI through conversation. 🧠✨
