# Lattice: Adaptive Memory Orchestrator

**Lattice** is a self-evolving Discord companion powered by the **ENGRAM** neuro-symbolic memory framework. It is specifically engineered to achieve high-order cognitive capabilities within extreme hardware constraints (2GB RAM / 1vCPU).

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## 🌟 Key Features

- **ENGRAM Memory Framework**: Three-tier neuro-symbolic system (Episodic, Semantic, Procedural).
- **Context Strategy**: Structured message analysis and retrieval planning.
- **Dreaming Cycle**: Self-optimization loop where the system proposes logic updates for human approval.
- **Audit View**: In-Discord UI for inspecting the system's "thought process" and memory source attribution.
- **Resource-First Design**: Optimized for low-memory environments via streaming and connection pooling.
- **Graph-First Retrieval**: Unified search across temporal logs and relationship graphs.
- **Invisible Alignment**: Non-intrusive feedback loops using Discord's native UI elements.
- **Unified Pipeline**: Streamlined reactive flow using a single response template.

---

## 🏗️ Architecture

Lattice operates on the principle that **logic is data**. By moving prompts and configuration into the database, the system achieves total evolvability without code changes.

### Memory Tiers
1. **Episodic** ([`raw_messages`](#1-episodic-memory-raw_messages)): Immutable, time-ordered interaction log.
2. **Semantic** ([`entities`](#2-semantic-memory-entities) + [`semantic_memories`](#3-semantic-relationships-semantic_memories)): Text-based knowledge graph with iterative BFS traversal.
3. **Procedural** ([`prompt_registry`](#4-procedural-memory-prompt_registry)): Versioned templates and behavioral strategies.

### Unified Pipeline
`Ingestion → Short-Circuit → Context Strategy → Semantic Retrieval → Generation → Memory Consolidation`

For a deep dive into the technical implementation, see the [Technical Implementation](#-technical-implementation) section below.

---

## ⚙️ Technical Implementation

### Pipeline Flow
1. **Ingestion**: Store message in episodic memory.
2. **Context Strategy**: Analyze recent messages for entities, context flags, and unresolved entities.
3. **Retrieval**: Fetch relevant context from `semantic_memories`.
4. **Generation**: Produce response using `UNIFIED_RESPONSE` template, proactively clarifying unresolved entities.
5. **Consolidation**: Async extraction of new entities, memories, and activities.
6. **Canonicalization**: Deterministic storage of new entities/predicates.

### Entity Extraction System
- **Context Strategy (Step 2)**: Analyzes small window (10 msgs). Outputs canonical entities, context flags (`goal_context`, `activity_context`), and unresolved entities for clarification.
- **Memory Consolidation (Step 5)**: Deeper extraction on larger window (20 msgs). Canonicalizes new entities/predicates into the `entities` and `predicates` tables.

### Context Strategy
Adaptive retrieval based on entities and flags:
- **Entity-Based**: retrieve with `memory_depth=2` (multi-hop relationships).
- **Context Flags**:
    - `goal_context`: Fetch memories with `has goal` predicate.
    - `activity_context`: Fetch memories with `did activity` predicate.

### Canonical Placeholders
Managed by `PlaceholderRegistry` for automatic resolution and validation:
| Placeholder | Contains |
|-------------|----------|
| `{episodic_context}` | 14 recent messages (excluding current) |
| `{semantic_context}` | Relevant memories from knowledge graph |
| `{bigger_episodic_context}` | 20 messages for extraction (includes current) |
| `{smaller_episodic_context}` | 10 messages for Retrieval Planning (includes current) |
| `{user_message}` | The user's current message |
| `{unresolved_entities}` | Unresolved entities for clarification |
| `{goal_context}` | Active goals from knowledge graph |
| `{local_date}` / `{local_time}` | Current date/time with day/week info |
| `{date_resolution_hints}` | Resolved relative dates (e.g., "Friday → 2026-01-10") |
| `{canonical_entities}` | List from `entities` table |
| `{canonical_predicates}` | List from `predicates` table |
| `{scheduler_current_interval}` | Current proactive check-in interval (minutes) |

### 🧠 Dreaming Cycle
Autonomous prompt optimization using feedback and metrics.
1. **Analyze**: Priority score = `negative_rate × usage`.
2. **Propose**: LLM generates optimized templates from feedback samples.
3. **Review**: Human approval in Discord Dream Channel.
4. **Apply**: Approved proposals create new `prompt_registry` versions.

---

## ⚙️ Configuration

### Hardware Limits
Lattice enforces strict bounds to stay within 2GB RAM:
```env
MAX_EPISODIC_CONTEXT_TURNS=20
MAX_MEMORY_DEPTH=3
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
| `entities` | Entity registry |
| `semantic_memories` | Graph relationships |
| `objectives` | User goals |
| `prompt_registry` | Prompt templates |
| `prompt_audits` | LLM call tracking |
| `dreaming_proposals` | Prompt optimization |
| `user_feedback` | User feedback |
| `system_metrics` | Metrics and scheduler state |

### Setup
```bash
make install        # Install dependencies and pre-commit hooks
cp .env.example .env # Configure your tokens and DB
make init-db        # Initialize fresh schema and seed data
make run            # Start the bot
```

**Docker Compose Structure:**

Development builds from source with hot reload:
```bash
# Uses: docker-compose.base.yml + docker-compose.dev.yml
docker compose -f docker-compose.base.yml -f docker-compose.dev.yml up
# or simply:
make run
```

Production uses pre-built images (deployed via CI/CD):
```bash
# Uses: docker-compose.base.yml + docker-compose.prod.yml
docker compose -f docker-compose.base.yml -f docker-compose.prod.yml up -d
```

**Note:** No migrations needed for fresh setup. This branch uses a fresh database schema. All prompt templates are in `scripts/seed.sql`.

**Lattice**: Building adaptive AI through conversation. 🧠✨
