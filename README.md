# Lattice: Adaptive Memory Orchestrator

**Lattice** is a self-evolving Discord companion powered by the **ENGRAM** neuro-symbolic memory framework. It is specifically engineered to achieve high-order cognitive capabilities within extreme hardware constraints (2GB RAM / 1vCPU).

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> **⚠️ NOTE**: Semantic memory architecture is being rewritten. See [Issue #61](https://github.com/chriskevini/lattice/issues/61) for the graph-first approach replacing vector embeddings with query extraction.

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
2.  **Semantic ([`entities`](#2-semantic-memory-entities) + [`semantic_triples`](#3-semantic-relationships-semantic_triples))**: Graph-first knowledge with query extraction (replacing vector-based facts).
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
_(Note: Vector search limits removed in Issue #61's graph-first architecture)_

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

### 2. Semantic Memory: `entities`
_(New in Issue #61)_ Named entities without embeddings, using keyword search.
```sql
CREATE TABLE entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT UNIQUE NOT NULL,
    entity_type TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    first_mentioned TIMESTAMPTZ DEFAULT now()
);
```

### 3. Semantic Relationships: `semantic_triples`
The graph layer connecting entities with temporal validity.
```sql
CREATE TABLE semantic_triples (
    subject_id UUID REFERENCES entities(id) ON DELETE CASCADE,
    predicate TEXT NOT NULL,
    object_id UUID REFERENCES entities(id) ON DELETE CASCADE,
    valid_from TIMESTAMPTZ DEFAULT now(),
    valid_until TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'::jsonb
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
