# Testing Guidelines for Lattice

This document provides comprehensive guidelines for writing and maintaining tests in the Lattice project. It captures lessons learned from the Test Suite Audit and establishes best practices.

---

## 📁 Test Organization

### Directory Structure

```
tests/
├── unit/                    # Fast, mocked tests
│   ├── test_*.py           # Individual component tests
│   └── ...
└── integration/             # Tests requiring real database
    ├── test_*.py
    └── ...
```

### When to Use Which

| Type | Use When | Database | Speed |
|------|----------|----------|-------|
| Unit | Testing individual functions/classes | Mocked | Fast |
| Integration | Testing database operations | Real (skipped in CI) | Slow |

### CI Configuration

Integration tests are automatically skipped in CI when `DATABASE_URL` is not set:

```python
pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="Requires real database connection",
)
```

---

## 🎯 Common Patterns

### 1. Mock Completeness

**Always include all fields that SQL queries return.** Incomplete mocks can hide bugs and cause confusing test failures.

```python
# ❌ Incomplete mock (missing origin_id and created_at):
{
    "subject": "Alice",
    "predicate": "works_at",
    "object": "Acme Corp"
}

# ✅ Complete mock (matches actual SQL query):
{
    "id": uuid4(),
    "subject": "Alice",
    "predicate": "works_at",
    "object": "Acme Corp",
    "origin_id": message_uuid,
    "created_at": "2024-01-01T00:00:00Z"
}
```

**Why this matters**: The actual SQL query returns specific fields. If your mock is missing fields, the test passes but doesn't reflect reality.

---

### 2. Type Correctness in Mocks

**Match expected types exactly**, especially for JSON fields that the implementation validates.

```python
# ❌ Wrong type (dict instead of string):
mock_llm_result.content = """{
    "proposed_template": "New template",
    "expected_improvements": {},  # Should be string!
    "confidence": 0.8
}"""

# ✅ Correct type:
mock_llm_result.content = """{
    "proposed_template": "New template",
    "expected_improvements": "Improved performance expected",
    "confidence": 0.8
}"""
```

**Common type mismatches to avoid**:
- `{}` (dict) vs `""` (string) for optional fields
- `0` (int) vs `0.0` (float) for numeric fields
- `"0"` (string) vs `0` (int) for ID fields

---

### 3. Truthy vs None Checks

**Use explicit `is not None` for numeric values that could be 0 or 0.0.**

```python
# ❌ Temperature 0.0 incorrectly falls back to 0.7:
temperature = prompt_template.temperature if prompt_template.temperature else 0.7
# Result: temperature=0.7 when template has temperature=0.0

# ✅ Explicit None check preserves 0.0:
temperature = prompt_template.temperature if prompt_template.temperature is not None else 0.7
# Result: temperature=0.0 when template has temperature=0.0
```

**When to use which**:
| Pattern | Use When |
|---------|----------|
| `if value:` | Checking for truthiness (empty string, empty list, None, 0 all falsy) |
| `if value is not None:` | Preserving 0, 0.0, False as valid values |

---

### 4. Async Mock Sequencing

**Use `side_effect` lists for multiple async calls with different return values.**

```python
async def test_entity_resolution_tier3_creates_new():
    """Test when no match found, LLM normalizes, then creates new entity."""

    # Mock returns different values for each call:
    # 1. No direct match for "john"
    # 2. No match for "John" (LLM normalized)
    # 3. Entity creation succeeds
    mock_conn.fetchrow.side_effect = [
        None,  # First call: no direct match
        None,  # Second call: LLM normalization found nothing
        {"id": new_entity_id},  # Third call: creation result
    ]

    result = await resolve_entity("john")
    assert result["id"] == new_entity_id
```

---

### 5. Asserting Log Messages

**Check that multiple fields appear in the SAME log record, not just anywhere.**

```python
# ❌ Checks strings in ANY log message:
assert any("Channel not found" in rec.message for rec in caplog.records)
assert any("999999999" in rec.message for rec in caplog.records)

# ✅ Checks both in SAME record:
assert any(
    "Channel not found" in rec.message and "999999999" in rec.message
    for rec in caplog.records
)
```

---

## ⚠️ Common Pitfalls

### 1. Separator Order in String Splitting

**Check longer separators first when they contain shorter ones.**

```python
# ❌ "->" splits "-->" incorrectly:
"alice --> works_at --> Acme Corp".split("->")
# Result: ["alice -", " works_at -", " Acme Corp"]

# ✅ Check "-->" first:
for separator in ["-->", "->", "→"]:  # Longest first!
    if separator in line:
        parts = line.split(separator)
```

**Separators in Lattice code**:
- `-->`: Long dash-arrow (used in test_triple_extraction.py)
- `->`: Standard arrow
- `→`: Unicode arrow

---

### 2. Chunk Length Calculation

**The last line in a chunk doesn't have a trailing newline.**

```python
# ❌ Incorrect: Adds +1 for every line
for line in lines:
    line_length = len(line) + 1  # Wrong for last line!

# ✅ Correct: First line doesn't need newline, others do
for line in lines:
    if current_length == 0:  # First line
        line_length = len(line)
    elif len(line) > 0:  # Subsequent lines
        line_length = len(line) + 1
```

---

### 3. Pipeline Bypassing in Tests

**Tests should mirror the actual data flow, not bypass pipeline stages.**

```python
# ❌ Bypasses parse_triples normalization:
triples = [{"subject": "Bob", "predicate": "loves", "object": "pizza"}]
await store_semantic_triples(message_id, triples)
# Expects "loves" → "likes" but storage doesn't normalize!

# ✅ Uses same pipeline as production:
raw_output = '[{"subject": "Bob", "predicate": "loves", "object": "pizza"}]'
triples = parse_triples(raw_output)  # Normalization happens here!
assert triples[0]["predicate"] == "likes"  # Verified
await store_semantic_triples(message_id, triples)
```

**Key insight**: In production, `parse_triples()` normalizes predicates before `store_semantic_triples()` receives them. Tests should do the same.

---

### 4. Count Assertions with Unique IDs

**When each test uses a unique ID, use exact equality, not inequalities.**

```python
# ❌ Over-lenient (passes for any count >= 2):
assert rows[0]["count"] >= 2  # "May have more from previous tests"

# ✅ Exact (each test uses unique message_id):
assert rows[0]["count"] == 2  # Exactly 2 triples stored
```

**Exception**: Only use `>=` when tests share data or IDs are not unique.

---

### 5. Missing Edge Case Tests

**Add tests for None returns and error paths.**

```python
async def test_scheduler_loop_handles_none_next_check():
    """Verify graceful handling when get_next_check_at returns None."""

    async def get_next_check_side_effect() -> datetime | None:
        return None  # Simulates edge case

    with patch("...get_next_check_at", side_effect=get_next_check_side_effect):
        # Should not raise, handle gracefully
        result = await scheduler._scheduler_loop()
```

**Common edge cases to test**:
- `None` returns from database queries
- Empty lists from `fetch()` calls
- Empty strings in content fields
- Zero values for counters/rates
- Past dates for time-sensitive logic

---

## 🛠️ Running Tests

### Quick Reference

```bash
make test                    # Run full test suite
make test-fast              # Without coverage (faster)
make check-all              # Lint + Type Check + Test

pytest tests/unit/ -v       # Unit tests only
pytest tests/integration/ -v  # Integration tests only
pytest -k "pattern"         # Filter by name
pytest --tb=short           # Short tracebacks
pytest -x                   # Stop on first failure
```

### Integration Tests

```bash
# Set DATABASE_URL to run integration tests
export DATABASE_URL="postgresql://user:pass@localhost:5432/lattice"
pytest tests/integration/ -v
```

Without `DATABASE_URL`, integration tests are automatically skipped.

---

## 📋 Checklist for New Tests

- [ ] Mock includes all fields that the implementation expects
- [ ] Mock types match expected types (especially JSON fields)
- [ ] Truthy checks use `is not None` for numeric 0/0.0 values
- [ ] Async calls use `side_effect` for multiple return values
- [ ] Log assertions check fields in same record
- [ ] Test follows actual pipeline flow (no bypasses)
- [ ] Count assertions use exact values when IDs are unique
- [ ] Edge cases (None, empty, zero) are tested
- [ ] Test name clearly describes what's being tested
- [ ] Comments explain non-obvious test logic

---

## 📚 Related Documents

- [README.md](../README.md): Project overview and quick start
- [AGENTS.md](AGENTS.md): Development workflow and architecture
- [DEVELOPMENT.md](DEVELOPMENT.md): Setup and troubleshooting

---

*Last updated: 2026-01-06*
*Based on lessons from Test Suite Audit (GitHub Issue #113)*
