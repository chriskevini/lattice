-- Migration: Add PROMPT_OPTIMIZATION template
-- Description: Moves hardcoded optimization prompt to database for evolvability

INSERT INTO prompt_registry (prompt_key, template, temperature)
VALUES (
    'PROMPT_OPTIMIZATION',
    'CURRENT TEMPLATE:
{current_template}

PERFORMANCE METRICS:
- Total uses: {total_uses}
- Feedback rate: {feedback_rate}
- Success rate: {success_rate} ({positive_feedback} positive, {negative_feedback} negative)

NEGATIVE FEEDBACK SAMPLES (key excerpts indicating issues):
{negative_feedback_samples}

POSITIVE FEEDBACK SAMPLES (key excerpts showing what''s working well):
{positive_feedback_samples}

TASK: Optimize the current prompt template.
1. Analyze the negative feedback for common patterns and root causes.
2. Identify what works well from the positive feedback.
3. Propose one improved template that fixes the main issues with minimal, targeted changes—preserve everything that already works.
4. Explain each change and how it ties to the feedback.
5. Estimate realistic improvements.

Respond ONLY with this simplified JSON (no extra text):
{{
  "proposed_template": "full improved template here",
  "changes": [
    {{"issue": "brief description of problem from feedback", "fix": "what you changed", "why": "expected benefit"}}
  ],
  "expected_improvements": "short paragraph on likely impact (e.g., clarity, consistency, success rate)",
  "confidence": 0.XX
}}

Prioritize clarity, minimal changes, and direct ties to feedback.',
    0.7
)
ON CONFLICT (prompt_key) DO UPDATE SET
    template = EXCLUDED.template,
    temperature = EXCLUDED.temperature,
    updated_at = CURRENT_TIMESTAMP;

