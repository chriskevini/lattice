# Agent Onboarding Guide - Lattice Project

## Project Overview

**Lattice** is an Adaptive Memory Orchestrator - a self-evolving Discord companion bot using the ENGRAM neuro-symbolic memory framework. It runs on constrained hardware (2GB RAM / 1vCPU) with PostgreSQL + pgvector.

## Core Architecture

### Three-Tier Memory System (ENGRAM Framework)

1. **Episodic Memory** (`raw_messages`)
   - Immutable conversation log
   - Temporal chaining via `prev_turn_id`
   - Source of truth for all interactions

2. **Semantic Memory** (`stable_facts` + `semantic_triples`)
   - `stable_facts`: Vector-embedded knowledge (384-dim)
   - `semantic_triples`: Explicit Subject-Predicate-Object relationships
   - Enables hybrid retrieval (vector + graph traversal)

3. **Procedural Memory** (`prompt_registry`)
   - Evolving templates and strategies
   - Self-modifying behavior system
   - Approval-gated updates via "dreaming cycle"

### Key Design Principles

1. **Canonical Integrity**: Never pollute visible conversation with internal thoughts
2. **Unified Pipeline**: Reactive (user input) and proactive (ghost messages) flow through same pipeline
3. **Invisible Alignment**: Feedback via 🫡 emoji reactions, North Star goals stored silently
4. **Total Evolvability**: All logic stored as data, not hardcoded
5. **Resource Constraints**: Optimize for 2GB RAM / 1vCPU throughout
6. **Transparent Constraints**: AI is aware of system limits and receives feedback when requests are clamped

### Discord Channels

- **Main Channel**: Primary user interaction, conversation, proactive check-ins
- **Dream Channel**: Meta activities, prompt optimization proposals, self-reflection, human approval gateway

## Project Structure

```
lattice/
├── core/           # Pipeline orchestration, unified ingestion, short-circuit logic
├── memory/         # ENGRAM implementation (episodic, semantic, procedural)
│   ├── episodic.py     # raw_messages handling, temporal chaining
│   ├── semantic.py     # stable_facts, semantic_triples, vector ops
│   └── procedural.py   # prompt_registry, template management
├── discord_client/ # Discord bot interface, reaction handling, rate limiting
├── prompts/        # Prompt templates and extraction strategies
└── utils/          # Embeddings, database pooling, logging

tests/
├── unit/           # Isolated component tests
└── integration/    # Full pipeline tests with test DB
```

## Technical Stack

- **Language**: Python 3.12+ (chosen for ML/AI ecosystem superiority)
- **Discord**: discord.py (async, lightweight)
- **Database**: asyncpg + pgvector
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2, 384-dim, ~80MB)
- **Type Safety**: mypy strict mode
- **Package Manager**: UV (10-100x faster than Poetry)
- **Linting**: Ruff (replaces flake8, black, isort, pyupgrade)
- **Testing**: pytest with async support

## Development Workflow

### Quick Start
```bash
make install        # Install deps + setup hooks
make test           # Run tests
make check-all      # Run all quality checks
make commit         # Interactive conventional commit
```

### Code Standards (Auto-Enforced)

- **Type hints required**: All functions must have type annotations
- **Docstrings required**: Google style, explain "why" not "what"
- **Line length**: 100 characters max
- **Import order**: stdlib → third-party → first-party (auto-sorted)
- **Complexity limit**: Max McCabe complexity of 10

Pre-commit hooks will auto-fix most issues. Trust the linters.

### Commit Messages

Use conventional commits (enforced):
```bash
feat(memory): add semantic triple extraction
fix(discord): handle rate limiting edge case
refactor(core): simplify pipeline short-circuit logic
perf(memory): optimize vector search for low RAM
docs: update ENGRAM architecture diagram
```

## Key Implementation Details

### Database Schema

See `README.md` lines 29-150 for complete DDL. Key tables:

- `raw_messages`: Episodic log with `prev_turn_id` chaining
- `stable_facts`: Vector-embedded facts (HNSW index: m=16, ef_construction=64)
- `semantic_triples`: Subject-Predicate-Object relationships
- `prompt_registry`: Evolvable templates with version control
- `context_archetypes`: Semantic archetypes for context classification
- `objectives`: User goals with saliency scoring
- `user_feedback`: Out-of-band feedback via emoji reactions
- `system_health`: Proactive scheduling metadata

### Pipeline Flow

1. **Ingestion**: Discord message or ghost signal (`<PROACTIVE_EVAL>`)
2. **Short-Circuit Logic**:
   - North Star declaration → upsert to `stable_facts` → ack → exit
   - Invisible feedback (reply to bot) → insert `user_feedback` → 🫡 → exit
   - Feedback undo (🗑️ on 🫡) → delete feedback → exit
3. **Episodic Logging**: Insert to `raw_messages` with temporal chaining
4. **Context Analysis**: Semantic archetype matching determines optimal context configuration
   - Embedding-based classification using `context_archetypes` table
   - Outputs: `CONTEXT_TURNS`, `VECTOR_LIMIT`, `SIMILARITY_THRESHOLD`, `TRIPLE_DEPTH`
   - Evolvable: AI can propose new archetypes via Dream Channel
5. **Hybrid Retrieval**:
   - Vector: Cosine similarity on `stable_facts.embedding`
   - Graph: Traverse `semantic_triples` for relational context
   - Episodic: Recent N turns via `prev_turn_id` chain
6. **Generation**: Route to `prompt_registry` template
7. **Async Consolidation**: Extract facts/triples/objectives (background)

**Details**: See `docs/message-data-flow.md` for complete step-by-step walkthrough

### Resource Constraint System

**Philosophy**: AI's goal is to generate the best response. System's goal is to enforce hardware constraints.

The system uses **Context Archetype Classification** to automatically determine optimal context configuration:
1. Incoming message → Generate embedding (~20ms)
2. Match against archetype centroids in `context_archetypes` table
3. Apply archetype's context settings (pre-validated against hardware limits)

**Resource Dimensions** (configured per archetype):

- **CONTEXT_TURNS** (1-20): Sequential conversation history
- **VECTOR_LIMIT** (0-15): Semantic search results
- **SIMILARITY_THRESHOLD** (0.5-0.9): Semantic matching strictness
- **TRIPLE_DEPTH** (0-3): Relationship graph hops

All archetype configurations (human or AI-proposed) are validated against MIN/MAX limits via database CHECK constraints, preventing configurations that would exceed hardware capacity.

### Context Archetype System

**Purpose**: Automatically determine optimal context configuration based on message type using semantic similarity matching.

**How It Works**:
1. Message arrives → Generate embedding (~20ms)
2. Match to archetype → Compare with pre-computed centroids in `context_archetypes` table
3. Apply configuration → Use archetype's context settings for retrieval

**Evolvability**: AI can propose new archetypes via Dream Channel. Human approves → auto-inserted into database → hot-reloaded within 60 seconds → immediately active.

**Details**: See `docs/context-archetype-system.md` for implementation and examples

### Memory Optimization Patterns

**Critical for 2GB RAM:**

```python
# ✅ Good: Stream results, limit queries
async def get_recent_context(n: int = 10) -> AsyncIterator[Message]:
    async for msg in fetch_messages_streaming(limit=n):
        yield msg

# ❌ Bad: Load everything into memory
messages = await fetch_all_messages()  # Could be 10k+ messages
```

```python
# ✅ Good: Connection pooling with limits
pool = await asyncpg.create_pool(min_size=2, max_size=5)

# ❌ Bad: Unlimited connections
pool = await asyncpg.create_pool(min_size=10, max_size=50)
```

## Common Tasks

### Adding a New Memory Type

1. Add table to schema (see `README.md` section 2.2)
2. Create model in `lattice/memory/`
3. Add extraction logic to consolidation pipeline
4. Update relevant `prompt_registry` templates
5. Add tests in `tests/unit/test_memory.py`

### Adding a New Prompt Template

```python
await db.execute("""
    INSERT INTO prompt_registry (prompt_key, template, temperature)
    VALUES ($1, $2, $3)
""", "NEW_TEMPLATE", template_text, 0.7)
```

### Modifying Short-Circuit Logic

Edit `lattice/core/pipeline.py`:
```python
async def should_short_circuit(message: Message) -> bool:
    if is_north_star(message):
        await handle_north_star(message)
        return True

    if is_invisible_feedback(message):
        await handle_feedback(message)
        return True

    return False
```

## Resources

### Documentation
- **README.md**: Complete system design specification
- **DEVELOPMENT.md**: Setup, code quality, workflow, troubleshooting
- **docs/message-data-flow.md**: Detailed message processing walkthrough
- **docs/context-archetype-system.md**: Context classification implementation

### Key Reference Points
- Database Schema: `README.md` lines 29-150
- Pipeline Operations: `README.md` lines 152-178
- Proactive Scheduling: `README.md` lines 180-202

## Philosophy

This project embodies **radical simplicity** through **metadata-driven evolution**. When in doubt:

1. **Store logic as data** (prompt_registry, context_archetypes, system_health)
2. **Optimize for constraints** (2GB RAM is real, respect it)
3. **Trust the linters** (they enforce consistency)
4. **Test async code** (bugs hide in concurrency)
5. **Document "why"** (the "what" is in the code)

Welcome to Lattice. Let's build something adaptive.
