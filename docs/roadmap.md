# ⚠️ DEPRECATED: See Issue #61

**This roadmap has been superseded.**

The current development direction is defined in **[Issue #61: Rewrite Semantic Memory - Graph-First Architecture with Query Extraction](https://github.com/chriskevini/lattice/issues/61)**.

## Why the Change?

The old roadmap (archived at [`docs/archive/roadmap-v1.md`](archive/roadmap-v1.md)) focused on incremental improvements to a fundamentally broken semantic memory architecture:

- **Phase 2** (Graph-Based Retrieval): Trying to "actually use the graph" built on broken foundations
- **Phase 5** (Memory Quality): Band-aid fixes like deduplication don't solve the root problem

**The root problem:** Embedding bare entity names (`"alice"`, `"mom"`) provides zero retrieval value.

## The New Direction (Issue #61)

Issue #61 proposes a complete rewrite that:

1. **Replaces vector embeddings** with **FunctionGemma-270M query extraction** (same RAM budget, better utility)
2. **Graph-first architecture**: Entities without embeddings, relationships with temporal validity
3. **Structured extraction**: JSONB message analysis for routing and context building
4. **Schema evolution**: Extraction templates evolve via Dreaming Cycle (append-only, backward compatible)

Read the full specification in **[Issue #61](https://github.com/chriskevini/lattice/issues/61)**.

---

## What Remains Valid from V1

The following completed phases are still relevant:

### Phase 1: Deployment & Infrastructure ✅
- Auto-deploy pipeline (GitHub Actions)
- Health checks and rollback workflow
- Quality gates (lint, type-check, test)

### Phase 3: Human-in-the-loop & Feedback ✅
- Dream Channel UI with mirrors
- Modal-based feedback collection
- Prompt audit trail
- Approval workflow for proposals

### Phase 4: Dreaming Cycle ✅
- Autonomous prompt optimization
- Context-aware sampling
- Version safety and transparency

**These systems remain operational and will integrate with the new semantic memory architecture.**

---

## Quick Reference

- **Active Roadmap**: [Issue #61](https://github.com/chriskevini/lattice/issues/61)
- **Archived V1 Roadmap**: [`docs/archive/roadmap-v1.md`](archive/roadmap-v1.md)
- **Project Philosophy**: [`AGENTS.md`](../AGENTS.md)
- **Development Setup**: [`DEVELOPMENT.md`](../DEVELOPMENT.md)
