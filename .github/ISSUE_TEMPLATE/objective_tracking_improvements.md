---
name: Objective Tracking System Analysis
about: Analysis of current implementation and proposed improvements
title: 'Improve objective tracking with semantic matching and activity updates'
labels: enhancement, memory
assignees: ''
---

## Summary

Analyze and improve the objective tracking system to handle natural language updates (e.g., "finished my homework") and reduce false negatives from exact string matching.

## Current Implementation

### Schema (`scripts/init_db.py:234-242`)
```sql
CREATE TABLE objectives (
    id UUID PRIMARY KEY,
    description TEXT NOT NULL,
    saliency_score FLOAT DEFAULT 0.5,
    status TEXT CHECK (status IN ('pending', 'completed', 'archived')) DEFAULT 'pending',
    origin_id UUID REFERENCES raw_messages(id),  -- ✅ Links to source message
    last_updated TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_objectives_description ON objectives (LOWER(description));
```

### Extraction (`lattice/memory/episodic.py:266-336`)
- **Trigger**: Async consolidation after every message
- **Prompt**: `OBJECTIVE_EXTRACTION` template (migration `002:137-196`)
- **Output**: JSON array with `{description, saliency, status}`
- **Model**: Uses LLM with temperature=0.1 for consistency

**Prompt Instructions** (line 159):
```
- **For goal updates:** If user mentions progress on an existing goal,
  mark it as "completed" and include original description
```

### Storage & Updates (`lattice/memory/episodic.py:339-418`)
```python
# Normalization via LOWER() for case-insensitive matching
normalized = description.lower().strip()

# Check for existing objective
existing = await conn.fetchrow(
    "SELECT id, status, saliency_score FROM objectives WHERE LOWER(description) = $1",
    normalized
)

if existing:
    # Update if status or saliency changed
    if status_changed or saliency_changed:
        await conn.execute(
            "UPDATE objectives SET status = $1, saliency_score = $2, last_updated = now() WHERE id = $3",
            status, saliency, existing["id"]
        )
else:
    # Insert new objective
    await conn.execute(
        "INSERT INTO objectives (description, saliency_score, status, origin_id) VALUES ($1, $2, $3, $4)",
        description, saliency, status, message_id
    )
```

## Current Limitations

### 1. **Exact String Matching**

**Problem**: Objectives only match via `LOWER(description)`, requiring exact wording.

**Example Failure Cases**:
```
Original: "Finish my homework by Friday"
Update:   "finished homework"
Result:   ❌ No match → Creates duplicate objective
```

```
Original: "Complete the Lattice refactor"
Update:   "Done with the refactor"
Result:   ❌ No match → Both objectives stay pending
```

**Root Cause**: Line 368-372 uses exact string comparison after normalization.

### 2. **LLM Reliability on Updates**

**Problem**: LLM must output the **exact original description** for updates to work.

**Prompt Instruction** (line 159):
> "If user mentions progress on an existing goal, mark it as 'completed' and **include original description**"

**Reality**: LLM often paraphrases, causing mismatches:
```
Original objective: "Register to courses by tonight"
User: "just registered for my courses!"
LLM extraction: {"description": "Register for courses", "status": "completed"}
Result: ❌ No match → Duplicate objective created
```

### 3. **No Entity Overlap Detection**

**Problem**: System doesn't use entity extraction to find related objectives.

**Example**:
```
Extraction from "finished homework":
{
  "message_type": "activity_update",
  "entities": ["homework"]  // From query extraction (Design D)
}

Current behavior: Extraction ignored, only description string used
Desired behavior: Use entity overlap to find matching objectives
```

### 4. **Limited Activity Update Intelligence**

**Problem**: Simple messages like "finished my homework" may not extract objectives at all.

**Test Case**:
```
User: "finished my homework"

Current: Depends entirely on LLM extraction quality
- If LLM extracts: {"description": "Finish homework", "status": "completed"}
  → May or may not match existing objective
- If LLM skips extraction (too vague):
  → No update happens at all
```

## Proposed Improvements

### Option A: Semantic Similarity Matching (Embeddings)

**Approach**: Compare objective descriptions using embeddings.

```python
async def find_matching_objective(
    description: str,
    entities: list[str] | None = None,
    threshold: float = 0.85
) -> UUID | None:
    """Find objective by semantic similarity + entity overlap.

    Args:
        description: New objective description
        entities: Extracted entities from query extraction
        threshold: Minimum similarity score (0.0-1.0)

    Returns:
        UUID of matching objective, or None
    """
    # 1. Get all pending objectives from DB
    pending_objectives = await get_pending_objectives()

    # 2. Compute embedding for new description
    new_embedding = await embed_text(description)

    # 3. Find best match by cosine similarity
    best_match = None
    best_score = 0.0

    for obj in pending_objectives:
        obj_embedding = await embed_text(obj["description"])
        similarity = cosine_similarity(new_embedding, obj_embedding)

        # Boost score if entities overlap
        if entities:
            entity_overlap = compute_entity_overlap(obj["description"], entities)
            similarity += (entity_overlap * 0.1)  # 10% boost per overlapping entity

        if similarity > best_score and similarity >= threshold:
            best_score = similarity
            best_match = obj["id"]

    return best_match
```

**Pros**:
- ✅ Handles paraphrasing naturally
- ✅ Works across different wordings
- ✅ Can leverage existing entity extraction

**Cons**:
- ❌ Requires embedding model (200-300MB RAM)
- ❌ Additional latency for embedding computation
- ❌ May match unrelated objectives with similar wording

### Option B: Entity-Based Matching (Graph-Native)

**Approach**: Use entity overlap from query extraction to find related objectives.

```python
async def find_matching_objective_by_entities(
    entities: list[str],
    message_type: str
) -> UUID | None:
    """Find objective by entity overlap.

    Args:
        entities: Extracted entities from query extraction
        message_type: "activity_update" or "goal" (from extraction)

    Returns:
        UUID of matching objective, or None
    """
    if not entities:
        return None

    # Query objectives that mention any of these entities
    query = """
        SELECT id, description, status
        FROM objectives
        WHERE status = 'pending'
        AND (
            LOWER(description) LIKE ANY($1)
        )
        ORDER BY last_updated DESC
        LIMIT 5
    """

    # Build LIKE patterns: ["homework", "courses"] → ["%homework%", "%courses%"]
    patterns = [f"%{entity.lower()}%" for entity in entities]

    matches = await conn.fetch(query, patterns)

    # Return most recent match
    return matches[0]["id"] if matches else None
```

**Usage**:
```python
# During consolidation
extraction = query_extraction.extract_query_structure(message)

if extraction.message_type == "activity_update":
    # Check for matching objective by entity overlap
    matching_obj = await find_matching_objective_by_entities(
        entities=extraction.entities,
        message_type=extraction.message_type
    )

    if matching_obj:
        # Update existing objective
        await update_objective_status(matching_obj, status="completed")
    else:
        # Extract and create new objective
        objectives = await extract_objectives(message_id, content, context)
        await store_objectives(message_id, objectives)
```

**Pros**:
- ✅ Zero additional RAM (uses existing entity extraction)
- ✅ Fast (simple SQL query)
- ✅ Leverages Design D entity-driven architecture
- ✅ No embedding model needed

**Cons**:
- ⚠️ May match multiple objectives (need disambiguation)
- ⚠️ Relies on entity extraction quality

### Option C: Hybrid Approach (Entity + Fuzzy String)

**Approach**: Combine entity overlap with fuzzy string matching.

```python
from difflib import SequenceMatcher

async def find_matching_objective_hybrid(
    description: str,
    entities: list[str] | None = None,
    entity_threshold: float = 0.5,
    string_threshold: float = 0.7
) -> UUID | None:
    """Find objective using entity overlap + fuzzy string matching.

    Args:
        description: New objective description
        entities: Extracted entities (optional boost)
        entity_threshold: Minimum entity overlap ratio (0.0-1.0)
        string_threshold: Minimum string similarity ratio (0.0-1.0)

    Returns:
        UUID of best matching objective, or None
    """
    pending = await get_pending_objectives()

    best_match = None
    best_score = 0.0

    for obj in pending:
        # Compute fuzzy string similarity
        string_sim = SequenceMatcher(
            None,
            description.lower(),
            obj["description"].lower()
        ).ratio()

        # Compute entity overlap if entities provided
        entity_boost = 0.0
        if entities:
            obj_entities = extract_entities_from_text(obj["description"])
            overlap = len(set(entities) & set(obj_entities))
            total = max(len(entities), len(obj_entities))
            entity_boost = (overlap / total) if total > 0 else 0.0

        # Combined score (70% string, 30% entities)
        combined_score = (string_sim * 0.7) + (entity_boost * 0.3)

        if combined_score > best_score:
            best_score = combined_score
            best_match = obj["id"]

    # Return match if above threshold
    if best_score >= max(string_threshold, entity_threshold):
        return best_match

    return None
```

**Pros**:
- ✅ No embeddings needed (lightweight)
- ✅ Handles typos and paraphrasing
- ✅ Uses entity extraction from Design D
- ✅ Tunable thresholds for precision/recall

**Cons**:
- ⚠️ `difflib.SequenceMatcher` may be slow for many objectives
- ⚠️ Still requires tuning thresholds

## Recommended Solution

**Implement Option B (Entity-Based) first**, then optionally add Option C (Hybrid) if false negatives remain.

### Rationale
1. **Aligns with Design D**: Leverages existing entity extraction from #87
2. **Zero RAM overhead**: No embedding model needed (2GB constraint)
3. **Fast iteration**: Simple SQL query, easy to test
4. **Natural extension**: Can upgrade to hybrid later without major refactor

### Implementation Plan

#### Phase 1: Entity-Based Matching
- [ ] Add `find_matching_objective_by_entities()` to `episodic.py`
- [ ] Update `consolidate_message()` to check entity matches before extracting
- [ ] Add logging for match/no-match cases (analyze false positives/negatives)
- [ ] Unit tests for matching logic

#### Phase 2: Integration with Query Extraction
- [ ] Pass `extraction.entities` to consolidation
- [ ] Use entity overlap for objective matching
- [ ] Handle disambiguation (multiple matches → pick most recent)
- [ ] Update `OBJECTIVE_EXTRACTION` prompt to clarify update behavior

#### Phase 3: Optional Hybrid Enhancement
- [ ] Implement `find_matching_objective_hybrid()` with `difflib`
- [ ] A/B test: entity-only vs hybrid
- [ ] Tune thresholds based on production data
- [ ] Roll out if improvement > 10% match rate

## Testing Strategy

### Unit Tests
```python
def test_entity_matching_homework():
    """Test: 'finished homework' matches 'Finish homework by Friday'"""
    assert find_matching_objective_by_entities(["homework"], "activity_update")

def test_entity_matching_multiple_entities():
    """Test: 'registered for courses' matches 'Register to courses by tonight'"""
    assert find_matching_objective_by_entities(["courses", "register"], "activity_update")

def test_no_match_different_entities():
    """Test: 'finished project' does NOT match 'Finish homework'"""
    assert not find_matching_objective_by_entities(["project"], "activity_update")
```

### Integration Tests
```python
async def test_activity_update_completes_objective():
    """Test full flow: goal creation → activity update → status change"""
    # 1. User sets goal
    await simulate_message("I need to finish my homework by Friday")
    objectives = await get_pending_objectives()
    assert len(objectives) == 1
    assert objectives[0]["status"] == "pending"

    # 2. User reports completion
    await simulate_message("finished my homework")
    objectives = await get_pending_objectives()
    assert len(objectives) == 0  # Should be marked completed

    completed = await get_completed_objectives()
    assert len(completed) == 1
    assert completed[0]["status"] == "completed"
```

## Success Metrics

**Before Implementation** (Baseline):
- Track objective update success rate
- Measure false negatives (unmatched updates)
- Measure false positives (wrong matches)

**After Implementation** (Target):
- ✅ 80%+ activity updates match existing objectives
- ✅ < 5% false positive rate
- ✅ No duplicate objectives for same goal

## Open Questions

1. **Disambiguation**: If multiple objectives match, should we:
   - Pick most recent? (default behavior)
   - Ask user to clarify?
   - Update all matches?

2. **Partial completion**: Should we support status like "in_progress"?
   ```
   User: "halfway done with homework"
   → Update status to "in_progress" instead of "completed"
   ```

3. **Saliency decay**: Should pending objectives lose saliency over time?
   ```sql
   -- Auto-archive stale objectives after 30 days?
   UPDATE objectives
   SET status = 'archived'
   WHERE status = 'pending'
   AND last_updated < now() - interval '30 days'
   ```

4. **Multi-objective updates**: How to handle batch updates?
   ```
   User: "finished homework and registered for courses"
   → Should update 2 separate objectives
   ```

## Related Issues

- #61 (Graph-first architecture with query extraction) - CLOSED, implemented in #87
- #87 (Entity-driven context optimization) - MERGED
- #15 (User corrections for extraction quality)

## References

- Current implementation: `lattice/memory/episodic.py:266-418`
- Objective parsing: `lattice/utils/objective_parsing.py`
- Extraction prompt: `scripts/migrations/002_insert_initial_prompt_templates.sql:137-196`
- Design D architecture: `docs/context_optimization_exploration.md`
