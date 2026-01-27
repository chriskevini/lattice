# Architecture Evolution Timeline

This document summarizes the observable architecture evolution from the repository's git history. The repository currently contains a **grafted root commit**, so the timeline is necessarily short and reflects only what is visible in this repo.

## High Line-Count Commits

| Date (UTC) | Commit | Lines | Summary |
| --- | --- | ---: | --- |
| 2026-01-21 | `0cdae26454` | 45,456 | Initial import and large feature drop: dual-agent response system, ENGRAM memory tiers, Discord UI, scheduler, migrations, tests, and docs. |

## Timeline of Architectural Evolution

### 2026-01-21 — Foundation + Dual-Agent Architecture (Initial Import)
**Primary patterns introduced**
- **ENGRAM memory tiers**: episodic (`raw_messages`), semantic (`entities`, `semantic_memories`), procedural (`prompt_registry`) stored in DB.
- **Unified pipeline**: `Ingestion → Context Strategy → Semantic Retrieval → Generation → Consolidation`, implemented in `lattice/core`.
- **Repository + Service boundaries**: `lattice/memory/repositories.py` encapsulates DB I/O; `core` contains pipeline orchestration and context strategy.
- **Discord-first interface**: `lattice/discord_client` provides handlers, audit mirror, threads, and UI controls.
- **Dependency injection**: `db_pool` and `llm_client` are passed explicitly through core components.
- **Dual-agent response system**: simultaneous LLM responses for semantic vs. embedding memory via webhooks.

**Milestones & pivots**
- **Single commit establishing full system**: The codebase, docs, migrations, tests, and operational scripts were introduced in one large commit. This defines the baseline architecture and patterns.
- **Dual-agent UX**: The “Lattice” vs “Vector” agent personas (webhook-based) are part of the initial architecture, signaling a deliberate multi-perspective response model.

### 2026-01-27 — Planning Update
**Commit**: `f3c4554` (“Initial plan”)  
This is a documentation/planning placeholder in the current branch; no architectural changes are visible.

## Major Milestones, Pivots, and Refactors (Observed)
Because the repository history is short and the root commit is grafted, only the initial import is observable. The initial commit already includes:
- Full ENGRAM memory architecture (episodic/semantic/procedural).
- Unified pipeline with context strategy and memory consolidation.
- Discord-specific UX (audit view, dreaming channel).
- Scheduler subsystem for proactive nudges and dreaming cycles.
- Extensive tests and documentation from day one.

No subsequent architectural refactors or pivots are visible in the git history provided.

## If Building This Again: Early Decisions to Reduce Refactors
Based on the current architecture, the following decisions should be made **at the very beginning** to reduce future refactors:

1. **Lock the boundaries early**
   - Keep `core` orchestration, `memory` storage, and `discord_client` UI isolated and stable.
   - Maintain strict interfaces (e.g., repository pattern) to avoid ripple refactors.

2. **Define the memory model upfront**
   - Commit to the ENGRAM tiers (episodic/semantic/procedural) and table schema early.
   - Plan migrations and prompt registry versioning before features expand.

3. **Decide on single vs. multi-agent UX**
   - Dual-agent mode affects message storage (`sender`), UI, and LLM call volume.
   - Choosing this upfront avoids later retrofitting of sender semantics and webhooks.

4. **Codify dependency injection**
   - Continue passing `db_pool` and `llm_client` explicitly; avoid globals.
   - Makes testing and refactoring safer and reduces hidden coupling.

5. **Establish prompt evolution rules**
   - Treat prompts as data with explicit versioning and migration strategy.
   - This prevents future rework around configuration drift or prompt history.

6. **Enforce operational constraints early**
   - The system is optimized for 2GB RAM/1vCPU; keep this invariant to avoid late-stage performance rewrites.

If additional historical commits are added later, this timeline should be expanded to document the exact refactor points and architectural shifts.
