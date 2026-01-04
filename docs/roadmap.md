# Lattice Project Roadmap

This roadmap outlines the strategic direction for Lattice, an Adaptive Memory Orchestrator. The priorities are balanced between operational stability and the "Total Evolvability" mission of the system.

## Phase 1: Deployment & Infrastructure ✅ COMPLETE
**Issues: #25, #17**
*Goal: Move from manual "make" commands to a professional, automated pipeline.*

- [x] **Unified DevOps Pipeline**: GitHub Actions deploy workflow builds Docker image on push to `main` and deploys via SSH
- [x] **Health Verification**: HTTP health check endpoint at `/health` with 10-retry verification in deploy pipeline
- [x] **Self-Hosted Deploy**: Server-side Docker build with automated migration runs and health verification
- [x] **Quality Gates**: Automated lint, type-check, and test validation on all PRs
- [x] **Rollback Workflow**: Manual rollback action for quick recovery from bad deploys

**Status**: Production-ready auto-deploy system operational. Issues #25 and #17 can be closed.

---

## Phase 2: Context Archetype System
**Priority: High (Architectural Foundation)**
*Goal: Give the AI "knobs" to control how much it remembers based on the conversation type.*

- [ ] **Archetype Implementation**: Create the `context_archetypes` DB table and classification logic. This is the most critical technical gap remaining in the memory architecture.
- [ ] **Dynamic Retrieval**: Switch the bot from static context variables to dynamic parameters (`CONTEXT_TURNS`, `VECTOR_LIMIT`) determined by the conversation's semantic match (e.g., "debugging" vs. "casual chat").

**Note**: Archetype system design has been documented (see `docs/archive/context-archetype-system.md`), but implementation is pending.

---

## Phase 3: Human-in-the-loop & Feedback ✅ MOSTLY COMPLETE
**Issues: #41, #15**
*Goal: Make the "Dream Channel" a place where humans can guide the bot's growth.*

- [x] **Dream Channel UI (#32)**: Unified mirror system with embeds showing reactive/proactive messages (PRs #37, #38, #39)
- [x] **Feedback System**: Modal-based feedback collection with sentiment tracking and prompt audit linkage
- [x] **Prompt Auditing (#27)**: Full audit trail of prompts, responses, and user feedback stored in `prompt_audits` table
- [x] **Extraction Mirroring (#13)**: Display extracted semantic triples and objectives in Dream Channel for visibility
- [x] **Approval Workflow**: Discord button-based approval/rejection for dreaming proposals with V2 components
- [x] **Transparency Features**: View rendered prompts button for both regular responses and optimizer prompts (#49, #54)
- [ ] **Threaded Discussions (#41)**: Add Discord threads on proposals so human discussion doesn't clutter the main channel
- [ ] **Extraction Corrections (#15)**: Allow humans to "reply" to a bad extraction in a thread; the bot stores this correction to refine its extraction prompts during the next Dreaming Cycle

**Status**: Core feedback loop operational. Dreaming Cycle fully functional with human-in-the-loop approval. Threading and extraction corrections remain as UX enhancements.

---

## Phase 4: Dreaming Cycle ✅ COMPLETE
**Issues: #27, #28, #33, #34, #49**
*Goal: Autonomous prompt optimization through feedback analysis.*

- [x] **Prompt Auditing (#27)**: Store all prompts, responses, and feedback with version tracking (PR #31)
- [x] **Analysis Queries (#33)**: Calculate prompt effectiveness metrics from audit data (PR #40)
- [x] **Optimization Proposals (#34)**: LLM-based prompt improvement generation with confidence scoring (PR #40)
- [x] **Scheduler Integration (#28)**: Daily 3 AM UTC runs with manual `!dream` trigger (PR #40)
- [x] **Approval Workflow**: Discord UI for human review/approve/reject of proposals (PR #40)
- [x] **Context-Aware Optimization (#49)**: Hybrid sampling with rendered prompts and experience cases (PRs #53, #54)
- [x] **Version Safety**: Optimistic locking and stale proposal auto-rejection (PR #52)
- [x] **Transparency**: View rendered optimization prompts in Dream Channel (PR #54)

**Status**: Fully operational autonomous prompt evolution system with human oversight. All core dreaming cycle features complete.

---

## Phase 5: Advanced Memory & Evolution
**Goal: Improve knowledge quality and unify growth systems.**

### Memory Quality (#50)
- [ ] **Vector-Based Deduplication**: Use pgvector similarity (>0.9) to detect duplicate facts before insertion
- [ ] **Triple Reconciliation**: Merge logically identical SPO triples and handle conflicting beliefs
- [ ] **Triples-First Retrieval**: Prioritize graph relationships over raw fact strings in context building

### Unified Evolution (#43)
- [ ] **Unified Proposals**: Merge Template optimization and Archetype optimization into a single decision engine
- [ ] **Coherent Growth Packages**: AI proposes both new prompt *and* context configuration together
- [ ] **Cross-System Validation**: Ensure prompt changes align with archetype expectations

### Knowledge Graph (#11)
- [ ] **Predicate Synonyms**: Store synonym mappings in DB to allow refinement through dream cycles
- [ ] **Semantic Normalization**: Standardize similar predicates during extraction

**Status**: Next phase of evolution. Phase 2 (Archetypes) is prerequisite for unified evolution (#43).

---

## Recently Completed (2026-01-04)

### Dreaming Cycle Enhancements
- **PR #54**: View rendered optimization prompt button for proposal transparency
- **PR #53**: Hybrid sampling with experience cases for better optimizer context
- **PR #52**: Operational improvements (stale proposal rejection, version safety)
- **PR #51**: Documentation cleanup and archiving
- **PR #48**: Schema fixes and threshold tuning

### Infrastructure & Operations
- **PRs #44-47**: Server-side Docker build, health checks, and auto-deploy pipeline
- **PR #40**: Full Dreaming Cycle implementation (analyzer, proposer, scheduler)
- **PRs #37-39**: Dream Channel UI phases 1-3 (reactive, proactive, extraction mirrors)

### Memory & Core Features
- **PR #36**: Bot.py refactoring for better separation of concerns
- **PR #31**: Prompt audit table implementation
- **PR #29**: Lightweight database migration system
- **PR #19**: Proactive scheduler with ghost messages
- **PR #16**: Objective extraction from conversations
- **PR #10**: Semantic triples and graph traversal

---

## Issue Status Summary

### 🟢 Completed / Can Close
- **#25**: Auto-deploy from GitHub Actions ✅ (PRs #44-47)
- **#17**: Automated Docker rebuilds ✅ (Merged into #25)
- **#27**: Prompt Audit Table ✅ (PR #31)
- **#28**: Dreaming Cycle Scheduler ✅ (PR #40)
- **#32**: Dream Channel Mirror UI ✅ (PRs #37-39)
- **#33**: Analysis Queries ✅ (PR #40)
- **#34**: Optimization Proposals ✅ (PR #40)
- **#49**: Optimizer Context ✅ (PRs #53-54)
- **#13**: Extraction Mirroring ✅ (PR #39)
- **#18**: Proactive Scheduler ✅ (PR #19)
- **#21**: Bot.py Refactoring ✅ (PR #36)
- **#26**: Migration System ✅ (PR #29)

### 🟡 Active / In Progress
- **#43**: Unified Template+Archetype optimization (blocked by Phase 2 - Context Archetypes)
- **#41**: Threaded discussions for proposals (UX enhancement)
- **#15**: User corrections for extraction quality (UX enhancement)
- **#50**: Semantic deduplication in consolidation phase (quality improvement)

### 🔵 Not Started / Future
- **Context Archetype System**: Core architectural feature (Phase 2)
- **#20**: Proactive scheduler personalization enhancements
- **#11**: Predicate synonyms in DB
- **#24**: Consolidating memory modules (technical debt)

---

## Recommended Next Steps

1. **Close Completed Issues**: #13, #17, #18, #21, #25, #26, #27, #28, #32, #33, #34, #49
2. **Phase 2 Priority**: Implement Context Archetype System (architectural foundation for unified evolution)
3. **Quality Improvements**: Address #50 (semantic deduplication) to prevent knowledge bloat
4. **UX Enhancements**: #41 (threaded discussions) and #15 (extraction corrections) for better human feedback
