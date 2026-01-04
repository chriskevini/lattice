# Lattice Project Roadmap

This roadmap outlines the strategic direction for Lattice, an Adaptive Memory Orchestrator. The priorities are balanced between operational stability and the "Total Evolvability" mission of the system.

## Phase 1: Deployment & Infrastructure
**Issues: #25, #17**
*Goal: Move from manual "make" commands to a professional, automated pipeline.*

- [ ] **Unified DevOps Pipeline**: Implement a GitHub Action that builds the Docker image on push to `main`, runs integration tests, and pushes to GHCR (merging #17).
- [ ] **Health Verification**: Add a health check mechanism (internal HTTP or Discord signal) to the bot so the pipeline can verify "Successful Start" before completing the deploy (addressing #25).
- [ ] **Self-Hosted Deploy**: Configure SSH-based deployment to pull the new image and restart the bot automatically on the production server.

## Phase 2: Context Archetype System
**Priority: High (Architectural Foundation)**
*Goal: Give the AI "knobs" to control how much it remembers based on the conversation type.*

- [ ] **Archetype Implementation**: Create the `context_archetypes` DB table and classification logic. This is the most critical technical gap remaining in the memory architecture.
- [ ] **Dynamic Retrieval**: Switch the bot from static context variables to dynamic parameters (`CONTEXT_TURNS`, `VECTOR_LIMIT`) determined by the conversation's semantic match (e.g., "debugging" vs. "casual chat").

## Phase 3: Human-in-the-loop & Feedback
**Issues: #41, #15**
*Goal: Make the "Dream Channel" a place where humans can guide the bot's growth.*

- [ ] **Threaded Discussions (#41)**: Add the Discord logic to create threads on proposals so human discussion doesn't clutter the main channel.
- [ ] **Extraction Corrections (#15)**: Allow humans to "reply" to a bad extraction in a thread; the bot stores this correction to refine its extraction prompts during the next Dreaming Cycle.

## Phase 4: Unified Evolution
**Issue: #43**
*Goal: Prevent the AI's different growth systems from fighting.*

- [ ] **Unified Proposals**: Merge the Template optimization and Archetype optimization into a single decision engine. The AI proposes *both* a new prompt and a new context configuration as a single coherent "growth package."

---

## Issue Status Summary

### 🟢 Active / Priority
- **#25**: Auto-deploy from GitHub Actions (Priority 1)
- **#41**: Threaded discussions for proposals (Priority 3)
- **#43**: Unified Template+Archetype optimization (Priority 4)
- **#15**: User corrections for extraction quality (Priority 3)

### 🟡 Medium / Future
- **#20**: Proactive scheduler future enhancements (Personalization)
- **#17**: Automated Docker rebuilds (Merging into #25)

### ⚪ Low / Deferred
- **#11**: Predicate synonyms in DB (Incremental improvement)
- **#24**: Consolidating memory modules (Technical debt)

### 🔴 Closed / Superceded
- **#18**: Proactive Scheduler Implementation (Merged in #19)
- **#27, #28, #32, #33, #34**: Dreaming Cycle Phase 1-4 (Merged in #40)
