# Agent Onboarding Guide

## 📌 Project Overview
**Lattice** is an Adaptive Memory Orchestrator—a self-evolving Discord companion using the **ENGRAM** neuro-symbolic memory framework.
- **Constraints**: 2GB RAM / 1vCPU.
- **Stack**: Python 3.12+, PostgreSQL 15+, py-cord 2.7+.
- **Core Goal**: Total evolvability via metadata-driven logic.

## 🏗️ Core Architecture

### Three-Tier Memory (ENGRAM)
1. **Episodic** (`raw_messages`): Immutable conversation log.
2. **Semantic** (`semantic_triples` + `canonical registries`): Graph-first knowledge with entity extraction. Includes `entities` and `predicates` tables for canonical form normalization.
3. **Procedural** (`prompt_registry`): Evolving templates via the Dreaming Cycle.

## 📂 Project Structure
- `lattice/core/`: Pipeline & ingestion logic.
- `lattice/memory/`: ENGRAM implementations.
- `lattice/discord_client/`: Bot interface & UI.
- `lattice/prompts/`: Templates & extraction strategies.
- `tests/`: Unit and integration tests.

### Discord Channel Separation
- **Main channels**: Normal user conversations (ingested into episodic/semantic memory).
- **Dream Channel**: Meta-discussions about prompts, feedback, and Dreaming Cycle output. Messages here are processed separately and do not affect user memory.

## 🛠️ Development Workflow

### Quick Start
```bash
make install # Install deps + hooks
make test # Run tests
make check-all # Lint, type-check, and test
```

### Standards
- **Strict Typing**: All functions must have type annotations.
- **Documentation**: Google-style docstrings (focus on "why").
- **Quality**: Enforced via Ruff (linting/formatting) and Mypy.
- **Commits**: Conventional Commits required.
- **Auditable**: A unified `AuditingLLMClient` ensures all LLM generations are auditable and optimizable.
- **Testing**: Write tests for all new features. See [TESTING.md](docs/TESTING.md) for guidelines.

## ⚙️ Key Implementation Details

### Pipeline Flow
1. **Ingestion + Logging**: Store message in episodic memory.
2. **Retrieval Planning**: Analysis of recent messages (10 including current) for entities, context flags, and unknown entities.
3. **Context Retrieval**: Fetch context from `semantic_triples` (see Context Strategy).
4. **Response Generation**: Using UNIFIED_RESPONSE template. May proactively clarify unknown entities.
5. **Batch Memory Extraction**: Async extraction of entities, triples, and activities.
6. **Canonicalization**: Deterministic storage of new entities/predicates in canonical registries.

**Notes**:
- Dates are resolved and available via `{date_resolution_hints}`.

### Entity Extraction System
Entity extraction occurs in two distinct pipeline steps:

- **Step 3: Retrieval Planning**  
  Analyzes the smaller conversation window (10 messages including current).  
  Outputs:  
  • Regular entities (canonical or previously seen; used for graph traversal)  
  • Context flags (e.g., `goal_context`, `activity_context`)  
  • Unknown entities (new abbreviations or unclear references, e.g., "bf", "lkea") — passed to Response Generation for clarification.

- **Step 6: Batch Memory Extraction**  
  Performs deeper extraction on the larger window (20 messages including current) and canonicalizes new entities/predicates into the `entities` and `predicates` tables.

### Context Strategy
Adaptive retrieval based on entities and flags from Retrieval Planning:

**Entity-Based Retrieval**:
- No entities detected → no semantic context (response relies on episodic history)
- Entities present → retrieve with triple_depth=2 (multi-hop relationships, e.g., User → Vancouver → Canada)

**Context Flags** (from Retrieval Planning):

| Flag | Trigger | Retrieval |
|------|---------|-----------|
| `goal_context` | User mentions goals, todos, deadlines | Fetch triples with `has goal` predicate |
| `activity_context` | User asks "what did I do" | Fetch all triples with `did activity` predicate |

Flags are passed to Response Generation which handles them appropriately.

### Canonical Placeholders
Use these placeholder names consistently across all prompts:

| Placeholder                  | Contains                                                                 | Example                                                                 |
|------------------------------|--------------------------------------------------------------------------|-------------------------------------------------------------------------|
| `{episodic_context}`         | 14 recent messages (current message NOT included)                        | "User: ...\nBot: ..."                |
| `{semantic_context}`         | Relevant facts from knowledge graph                                      | "User lives in Vancouver"  |
| `{bigger_episodic_context}`  | 20 messages for extraction (includes current)                            | Full batch of new messages for Batch Memory Extraction                   |
| `{smaller_episodic_context}` | 10 messages including current for Retrieval Planning analysis            | Small batch of new messages for Retrieval Planning          |
| `{user_message}`             | The user's current message                                               | "How's the project going?"                                              |
| `{goal_context}`             | Active goals from knowledge graph                                        | "User has goal complete project\ncomplete project due by: 2026-01-15" |
| `{local_date}`               | Current date with day of week                                            | "2026/01/08, Thursday"                                                  |
| `{local_time}`               | Current time                                                             | "14:30"                                                                 |
| `{date_resolution_hints}`    | Resolved relative dates to ISO format                                    | "Friday → 2026-01-10, tomorrow → 2026-01-09"                             |
| `{scheduler_current_interval}` | Scheduler check interval (minutes)                                      | 15                                                                      |
| `{feedback_samples}`         | Feedback samples for prompt optimization                                 | See PROMPT_OPTIMIZATION template                                        |
| `{metrics}`                  | Performance metrics string                                               | "95% success rate (15 positive, 5 negative, 80 neutral)."               |
| `{canonical_entities}`       | Direct list from canonical `entities` table                              | "Mother, Boyfriend, marathon, IKEA"                                     |
| `{canonical_predicates}`     | Direct list from canonical `predicates` table                            | "likes, works at, did activity, has goal"                               |
| `{unknown_entities}`         | Detected in Retrieval Planning; intended for clarification before canonicalization (e.g., "bf", "lkea") | "bf, lkea"                                                              |

**Placeholder Consistency Rules**:
- Analysis tasks (Retrieval Planning, Batch Memory Extraction) include current message in context.
- Response tasks separate current message as `{user_message}` for emphasis.
- New placeholders must be documented here.
- Use existing placeholders before creating new ones.
- Placeholders are validated at `response_generator.py:validate_template_placeholders`.

## 🧠 Dreaming Cycle
Autonomous prompt optimization using feedback and metrics.

### Flow
1. **Analyze** (`lattice/dreaming/analyzer.py`): Queries audits + feedback for templates with 10+ uses. Calculates priority score (negative_rate × usage).
2. **Propose** (`lattice/dreaming/proposer.py`): LLM generates optimized templates from feedback samples. Validates placeholders.
3. **Review**: Proposals stored in `dreaming_proposals` (status: pending). Human approves/rejects in Dream Channel.
4. **Apply**: Approved proposals create new `prompt_registry` version (append-only).

### Scheduling
- Daily at 3:00 AM UTC (configurable)
- Manual trigger: `!dream` command
- Controlled by: `dreaming_enabled`, `dreaming_min_uses` in `system_health`

## 📚 Resources
- **[README.md](README.md)**: Installation, Database Schema, and Config.
- **[DEVELOPMENT.md](docs/DEVELOPMENT.md)**: Setup and troubleshooting.
- **[TESTING.md](docs/TESTING.md)**: Testing guidelines and best practices.
