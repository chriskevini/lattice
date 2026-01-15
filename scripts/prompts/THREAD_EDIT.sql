-- =============================================================================
-- Canonical Prompt Template
-- Convention: Keep canonical prompts at v1. Git handles version history.
-- User customizations (via dream cycle) get v2, v3, etc.
-- =============================================================================
-- Template: THREAD_EDIT
-- Description: Generates prompt edits based on user feedback in audit threads
-- Temperature: 0.3

INSERT INTO prompt_registry (prompt_key, version, template, temperature)
VALUES ('THREAD_EDIT', 1, $TPL$
You are editing a prompt template based on user feedback.

=== ORIGINAL TEMPLATE ===
{original_template}

=== AUDIT CONTEXT ===
{rendered_prompt}

{raw_output}

=== RECENT DISCUSSION ===
{context}

=== USER REQUEST ===
{message}

Your task:
1. Understand what change the user wants based on the audit context and discussion
2. Propose a minimal, sensible modification to the template
3. Explain your reasoning

Respond with JSON:
{"modified_template": "complete updated template", "explanation": "what changed and why"}
$TPL$, 0.3)
ON CONFLICT (prompt_key, version) DO NOTHING;
