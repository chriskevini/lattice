# User Feedback Analysis

**Generated:** 2026-01-07

## Summary

| Metric | Count |
|--------|-------|
| Total Feedback | 13 |
| Positive | 3 |
| Negative | 9 |
| Neutral | 0 |

## Feedback Entries with Context

| Sentiment | User Message | Extracted | Feedback |
|-----------|--------------|-----------|----------|
| **negative** | (empty - message not stored) | N/A | (Quick negative feedback) |
| **positive** | (empty) | N/A | (Quick positive feedback) |
| **negative** | `what else do you know about me?` | `[]` | there should be no objectives extracted |
| **positive** | `what's my name?` | `[]` | (Quick positive feedback) |
| **positive** | (empty - proactive message) | N/A | (Quick positive feedback) |
| **negative** | `what's my name` | `[]` | (Quick negative feedback) |
| **negative** | `I really need to register to courses by tonight` | `[{"subject":"User","predicate":"needs_to_register","object":"courses"}]` | only extract as an objective. not as a triple. we need better instructions to separate these. |
| **negative** | `we're not hunting for names. we are establishing base lines` | `[{"subject":"User","predicate":"establishes","object":"base lines"}]` | way too eager to extract. nothing should be extracted here. |
| **negative** | `what is your name?` | `[]` | nothing should be extracted |
| **positive** | `i live in richmond canada, right next to vancouver, canada` | 3 triples (lives_in, is_near, is_in) | (Quick positive feedback) |
| **negative** | `wrong. your name is Lattice` | `[{"subject":"User","predicate":"has_name","object":"Lattice"}]` | wrong subject. should be: assistant -> has_name -> Lattice. no objective |
| **negative** | `what is your name?` | `[]` | same problem. questions should not be a source of extractions |
| **negative** | `what is my name` | `[]` | that is not an objective. this should be classified as a question and nothing extracted |

## Key Patterns in Feedback

### Positive Cases (3)
- Correct handling of questions (no extraction expected)
- Good triple extraction for factual statements (e.g., location information)

### Negative Cases (9)
Issues identified:

1. **Over-extraction** (3 cases)
   - System extracting from metacommentary that shouldn't be captured
   - Example: "we're not hunting for names. we are establishing base lines" → extracted `{"subject":"User","predicate":"establishes","object":"base lines"}`
   - Example: "I really need to register to courses by tonight" → extracted as triple instead of objective

2. **Wrong Entity Subjects** (1 case)
   - User corrected: "User has_name Lattice" → should be "Assistant has_name Lattice"
   - Bot incorrectly attributed the name to the user

3. **Triple vs Objective Confusion** (1 case)
   - Feedback explicitly states: "only extract as an objective. not as a triple"
   - Need better instructions to separate extraction types

4. **Questions Triggering Extraction** (2 cases)
   - Questions like "what is your name?" and "what is my name" are triggering extraction attempts
   - Should be classified as questions with no extraction

5. **Empty Messages/Quick Feedback** (2 cases)
   - Some messages not stored in raw_messages table
   - Quick feedback buttons used without detailed comments

## Recommendations

1. Add explicit rules to skip extraction for:
   - Questions (interrogative sentences)
   - Metacommentary about the system itself
   - Self-referential statements about the conversation

2. Distinguish between triples and objectives in extraction prompts:
   - Triples: factual relationships between entities
   - Objectives: user-stated goals/intentions (salient statements of intent)

3. Fix entity resolution:
   - Assistant's name should be attributed to Assistant entity, not User

4. Consider adding minimum salience thresholds for extraction
