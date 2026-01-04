# Lattice: Adaptive Memory Orchestrator

**Lattice** is a self-evolving Discord companion powered by the **ENGRAM** neuro-symbolic memory framework. It is specifically engineered to achieve high-order cognitive capabilities within extreme hardware constraints (2GB RAM / 1vCPU).

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## 🌟 Key Features

- **ENGRAM Memory Framework**: Three-tier neuro-symbolic system (Episodic, Semantic, Procedural).
- **Context Archetypes**: Dynamic retrieval scaling based on real-time semantic classification.
- **Dreaming Cycle**: Self-optimization loop where AI proposes logic updates for human approval.
- **Resource-First Design**: Native support for low-memory environments via streaming and pooling.
- **Hybrid Retrieval**: Unified search across temporal logs, vector embeddings, and relationship graphs.
- **Invisible Alignment**: Non-intrusive feedback loops using Discord's native UI elements.

---

## 🏗️ Architecture

Lattice operates on the principle that **logic is data**. By moving prompts and configuration into the database, the system achieves total evolvability without code changes.

### Memory Tiers
1.  **Episodic ([`raw_messages`](#1-episodic-memory-raw_messages))**: Immutable, time-ordered interaction log.
2.  **Semantic ([`stable_facts`](#2-semantic-memory-stable_facts) + [`semantic_triples`](#3-semantic-relationships-semantic_triples))**: Vectorized knowledge and Subject-Predicate-Object graphs.
3.  **Procedural ([`prompt_registry`](#4-procedural-memory-prompt_registry))**: Versioned templates and behavioral strategies.

### Unified Pipeline
`Ingestion → Short-Circuit → Archetype Analysis → Hybrid Retrieval → Generation → Consolidation`

For a deep dive into the technical implementation, see **[AGENTS.md](AGENTS.md)**.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- PostgreSQL 15+ with `pgvector`
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
Lattice enforces strict bounds to stay within 2GB RAM. These are controlled via `context_archetypes` in the DB, but capped by environment variables:
```env
MAX_EPISODIC_CONTEXT_TURNS=20
MAX_VECTOR_SEARCH_LIMIT=15
MAX_TRIPLE_DEPTH=3
```

### Discord Channels
- **`DISCORD_MAIN_CHANNEL_ID`**: The public face. Conversations are stored here.
- **`DISCORD_DREAM_CHANNEL_ID`**: The "subconscious". Meta-discussion and approval UI. **Never stored.**

---

## 🗄️ Database Schema

### 1. Episodic Memory: `raw_messages`
Stores the raw stream of consciousness.
```sql
CREATE TABLE raw_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    discord_message_id BIGINT UNIQUE NOT NULL,
    content TEXT NOT NULL,
    is_bot BOOLEAN DEFAULT false,
    timestamp TIMESTAMPTZ DEFAULT now()
);
```

### 2. Semantic Memory: `stable_facts`
Knowledge with 384-dimensional vector embeddings.
```sql
CREATE TABLE stable_facts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    embedding VECTOR(384),
    origin_id UUID REFERENCES raw_messages(id)
);
-- HNSW index optimized for 2GB RAM
CREATE INDEX ON stable_facts USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

### 3. Semantic Relationships: `semantic_triples`
The graph layer connecting facts.
```sql
CREATE TABLE semantic_triples (
    subject_id UUID REFERENCES stable_facts(id) ON DELETE CASCADE,
    predicate TEXT NOT NULL,
    object_id UUID REFERENCES stable_facts(id) ON DELETE CASCADE
);
```

### 4. Procedural Memory: `prompt_registry`
The behavioral engine.
```sql
CREATE TABLE prompt_registry (
    prompt_key TEXT PRIMARY KEY,
    template TEXT NOT NULL,
    version INT DEFAULT 1,
    active BOOLEAN DEFAULT true
);
```

---

## 🛠️ Development

### Standards
- **Strict Typing**: Enforced by Mypy.
- **Linting**: Enforced by Ruff.
- **Commits**: [Conventional Commits](https://www.conventionalcommits.org/) required.

### Commands
```bash
make test           # Run suite
make check-all      # Lint + Type Check + Test
make commit         # Trigger guided commit flow
```

---

## 🎯 Philosophy

1.  **Radical Simplicity**: Prefer metadata over complex code branches.
2.  **Hardware Respect**: 2GB RAM is a hard wall; build for it, not around it.
3.  **Total Transparency**: The bot's "thought process" is always visible in the Dream Channel.

---

**Lattice**: Building adaptive AI through conversation. 🧠✨
