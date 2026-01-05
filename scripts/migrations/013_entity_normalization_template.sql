-- Migration: 013_entity_normalization_template.sql
-- Description: Add ENTITY_NORMALIZATION prompt template for Issue #61 Phase 2 PR 2
-- Author: system
-- Date: 2026-01-04
--
-- This migration adds the ENTITY_NORMALIZATION prompt template for entity resolution.
-- The template normalizes entity mentions to canonical forms using LLM intelligence.

-- Insert ENTITY_NORMALIZATION prompt template
INSERT INTO prompt_registry (prompt_key, template, temperature, version)
VALUES (
    'ENTITY_NORMALIZATION',
    'You are an entity normalization system. Given an entity mention and context, normalize it to a canonical form.

## Task
Normalize the entity mention to its canonical form. Consider:
1. Canonical names (full names, proper nouns)
2. Abbreviations and nicknames
3. Context clues about identity
4. Consistent capitalization and formatting

## Input
**Entity Mention:** {entity_mention}
**Message Context:** {message_context}
**Existing Entities:** {existing_entities}

## Rules
1. **Exact Match**: If the mention matches an existing entity (case-insensitive), return the existing canonical name
2. **Normalization**: If it''s a variant (nickname, abbreviation, typo), return the canonical form
3. **New Entity**: If it''s truly new, return a normalized canonical form
4. **Preserve Meaning**: Do not over-normalize - "Python" and "python script" are different
5. **Capitalization**: Use proper capitalization (e.g., "Alice" not "alice", "Python" not "python" for the language)

## Examples

**Input:**
- Entity Mention: "mom"
- Existing Entities: ["Alice", "Bob", "Mom"]
- Context: "I talked to mom about the project"

**Output:**
{{
  "canonical_name": "Mom",
  "reasoning": "Direct match to existing entity, capitalized"
}}

**Input:**
- Entity Mention: "alice"
- Existing Entities: ["Alice Johnson", "Bob"]
- Context: "alice helped me debug the code"

**Output:**
{{
  "canonical_name": "Alice Johnson",
  "reasoning": "Case-insensitive match to existing entity"
}}

**Input:**
- Entity Mention: "lattice-ai"
- Existing Entities: ["lattice", "Discord Bot"]
- Context: "working on lattice-ai features"

**Output:**
{{
  "canonical_name": "lattice",
  "reasoning": "Variant of existing entity (hyphenated form)"
}}

**Input:**
- Entity Mention: "Python"
- Existing Entities: []
- Context: "learning Python programming"

**Output:**
{{
  "canonical_name": "Python",
  "reasoning": "New entity, proper capitalization for programming language"
}}

**Input:**
- Entity Mention: "john"
- Existing Entities: ["Alice", "Bob"]
- Context: "met john at the conference"

**Output:**
{{
  "canonical_name": "John",
  "reasoning": "New entity, proper name capitalization"
}}

## Output Format
Return ONLY valid JSON (no markdown, no explanation):
{{
  "canonical_name": "Normalized Entity Name",
  "reasoning": "Brief explanation of normalization decision"
}}',
    0.2,
    1
)
ON CONFLICT (prompt_key) DO NOTHING;

-- Comment for documentation
COMMENT ON TABLE entities IS
'Entity registry without embeddings. Uses keyword search and graph traversal instead of vector similarity. Entity resolution normalizes mentions to canonical forms via Tier 1 (direct match) or Tier 3 (LLM normalization).';
