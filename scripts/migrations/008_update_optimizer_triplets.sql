-- Migration: Update PROMPT_OPTIMIZATION to use Optimized Experience Triplets
-- Description: Updates the optimizer prompt to use optimized triplets (User Message -> Bot Response -> Feedback)
-- This avoids repeating the full rendered prompt which is redundant since the template is already provided.

INSERT INTO prompt_registry (prompt_key, template, temperature)
VALUES (
    'PROMPT_OPTIMIZATION',
    'CURRENT TEMPLATE:
{current_template}

PERFORMANCE METRICS:
- Total uses: {total_uses}
- Feedback rate: {feedback_rate}
- Success rate: {success_rate} ({positive_feedback} positive, {negative_feedback} negative)

EXPERIENCE TRIPLETS (Actual examples of User Message -> Bot Response -> Feedback):
{experience_triplets}

TASK: Optimize the current prompt template.
1. Analyze the experience triplets to understand why the current template produced the bot''s response given the user message.
2. Cross-reference thebot''s response with the user''s feedback and the "CURRENT TEMPLATE" provided above.
3. Identify specific instructions or lack thereof in the current template that caused poor responses.
4. Propose one improved template that fixes the main issues with minimal, targeted changes—preserve everything that already works.
5. Explain each change and how it ties to the specific experiences provided.
6. Estimate realistic improvements.

Respond ONLY with this simplified JSON (no extra text):
{{
  "proposed_template": "full improved template here",
  "changes": [
    {{"issue": "brief description of problem from feedback", "fix": "what you changed", "why": "expected benefit"}}
  ],
  "expected_improvements": "short paragraph on likely impact (e.g., clarity, consistency, success rate)",
  "confidence": 0.XX
}}

Prioritize clarity, minimal changes, and direct ties to the actual experiences.',
    0.7
)
ON CONFLICT (prompt_key) DO UPDATE SET
    template = EXCLUDED.template,
    temperature = EXCLUDED.temperature,
    updated_at = CURRENT_TIMESTAMP;
