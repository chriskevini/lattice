-- Migration: Update PROMPT_OPTIMIZATION to use Hybrid Experience Cases
-- Description: Updates the optimizer prompt to use hybrid sampling strategy.
-- Shows 3 detailed cases with rendered prompt excerpts for deep diagnosis,
-- plus lightweight cases (user message + response + feedback) for breadth.
-- This balances diagnostic value with token efficiency under 2GB RAM constraints.

INSERT INTO prompt_registry (prompt_key, template, temperature)
VALUES (
    'PROMPT_OPTIMIZATION',
    'CURRENT TEMPLATE:
{current_template}

PERFORMANCE METRICS:
- Total uses: {total_uses}
- Feedback rate: {feedback_rate}
- Success rate: {success_rate} ({positive_feedback} positive, {negative_feedback} negative)

EXPERIENCE CASES (Real-world examples with context):
{experience_cases}

TASK: Optimize the current prompt template.
1. Analyze the DETAILED experiences first—inspect the rendered prompts to see what episodic/semantic/graph context was provided.
2. Check if the context quality contributed to poor responses (e.g., irrelevant facts, missing information, wrong archetype).
3. Cross-reference the bot''s response with the user''s feedback and the "CURRENT TEMPLATE" provided above.
4. Identify specific instructions or lack thereof in the current template that caused poor responses.
4. Propose one improved template that fixes the main issues with minimal, targeted changes—preserve everything that already works.
5. Explain each change and how it ties to the specific experiences provided (especially the detailed ones).
6. Estimate realistic improvements.

Respond ONLY with this simplified JSON (no extra text):
{{
  "proposed_template": "full improved template here",
  "changes": [
    {{"issue": "brief description of problem from feedback", "fix": "what you changed", "why": "expected benefit"}}
  ],
  "expected_improvements": "short paragraph on likely impact (e.g., clarity, consistency, success rate, context relevance)",
  "confidence": 0.XX
}}

Prioritize clarity, minimal changes, and direct ties to the actual experiences (especially rendered prompt quality).',
    0.7
)
ON CONFLICT (prompt_key) DO UPDATE SET
    template = EXCLUDED.template,
    temperature = EXCLUDED.temperature,
    updated_at = CURRENT_TIMESTAMP;
