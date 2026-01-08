# Activity Tracking and Query Support

## Problem Statement

Currently, users cannot ask natural questions about their past activities like:
- "What did I do last week?"
- "Summarize my activities from last month"
- "How did I spend my time this week?"

These queries have no actionable response because:
1. Activities are not extracted as durable facts (BATCH_MEMORY_EXTRACTION ignores them)
2. No predicate-based graph queries exist (graph traversal only starts from entities)
3. No mechanism to detect activity query intent

## Solution Overview

Implement a complete activity tracking system using ENGRAM's existing infrastructure:

1. **Add activity predicates** to BATCH_MEMORY_EXTRACTION
2. **Add predicate-based query** to graph traversal
3. **Add predicate extraction** for query intent detection
4. **Reuse semantic_context** for activity results (no template changes)

## Architecture

```
User Message: "What did I do last week?"
    │
    ├─ Entity Extraction: [] (no entities mentioned)
    ├─ Predicate Extraction: ["performed_activity"]
    │
    └─ Pipeline detects predicate query
         │
         ├─ Query: find_by_predicate("performed_activity", last_week)
         │
         └─ Format results into semantic_context:
            "user performed_activity: coding | coding has_duration: 3 hours
             user performed_activity: running | running has_duration: 30 mins"
            │
            └─ Unified Response generates summary
```

## Implementation Details

### 1. Add Activity Predicates to BATCH_MEMORY_EXTRACTION

Add to `scripts/seed.sql`:

```sql
### Activity-related (new)
performed_activity, has_duration (e.g., "3 hours", "30 minutes")
```

Add extraction examples:

```sql
**New Messages:**
User: Spent 3 hours coding today
**Output:**
[
  {"subject": "user", "predicate": "performed_activity", "object": "coding"},
  {"subject": "coding", "predicate": "has_duration", "object": "3 hours"}
]
```

### 2. Add find_by_predicate() to graph.py

```python
async def find_by_predicate(
    self,
    predicate: str,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Find all triples with a specific predicate within a date range."""
```

### 3. Add Predicate Extraction

Create `extract_predicates()` in `lattice/core/entity_extraction.py`:

```python
ACTIVITY_QUERY_PATTERNS = [
    r"\bwhat did i (do|work|build)\b",
    r"\bhow did i spend my time\b",
    r"\bsummarize my activit(y|ies)\b",
    r"\bwhat have i been up to\b",
    r"\bwhat have i been doing\b",
]

async def extract_predicates(
    message: str,
) -> list[str]:
    """Detect predicate query intent from user message."""
    # Returns: ["performed_activity"] for activity queries
```

### 4. Pipeline Integration

Modify `lattice/core/pipeline.py`:

```python
async def handle_message(message):
    # ... existing entity extraction ...

    # Check for predicate queries
    predicates = await extract_predicates(message.content)
    if "performed_activity" in predicates:
        date_range = parse_relative_date_range(message.content)  # "last week" → dates
        activities = await graph.find_by_predicate(
            "performed_activity",
            start_date=date_range.start,
            end_date=date_range.end,
        )
        # Format as semantic triples
        semantic_context += format_activities_as_triples(activities)
```

### 5. Reuse semantic_context

No template changes needed. Activity results are formatted as:

```
**Relevant facts from past conversations:**
user performed_activity: coding
user performed_activity: running
coding has_duration: 3 hours
running has_duration: 30 minutes
```

The existing `UNIFIED_RESPONSE` template with `{semantic_context}` handles this naturally.

## Files to Modify

| File | Changes |
|------|---------|
| `scripts/seed.sql` | Add activity predicates + examples to BATCH_MEMORY_EXTRACTION |
| `lattice/memory/graph.py` | Add `find_by_predicate()` method |
| `lattice/core/entity_extraction.py` | Add `extract_predicates()` function |
| `lattice/core/pipeline.py` | Detect predicate queries + inject into semantic_context |
| `lattice/utils/date_resolution.py` | Add `parse_relative_date_range()` helper |

## Backward Compatibility

- ✅ Existing entity extraction unchanged
- ✅ Existing templates work as-is (semantic_context format unchanged)
- ✅ Existing graph queries work as-is (find_by_predicate is additive)
- ⚠️ Requires running extraction on historical messages to populate activities

## Future Enhancements (Out of Scope)

- Activity duration filtering/sorting
- Activity frequency analysis
- Cross-activity relationship discovery
- Proactive activity summaries
- Activity-based goal progress tracking

## Test Cases

```python
# Entity/predicate extraction
extract_predicates("What did I do last week?") → ["performed_activity"]
extract_predicates("Summarize my activities") → ["performed_activity"]
extract_predicates("Meeting tomorrow") → []  # No activity query

# Graph query
find_by_predicate("performed_activity", last_monday, sunday)
→ [{"subject": "user", "predicate": "performed_activity", "object": "coding", "created_at": ...}]

# Integration
handle_message("What did I do last week?")
→ semantic_context includes formatted activity triples
→ Response summarizes activities
```
