-- Template: PROMPT_OPTIMIZATION (v1)
-- Description: Propose prompt improvements based on feedback
-- Temperature: 0.7

INSERT INTO prompt_registry (prompt_key, version, template, temperature)
VALUES ('PROMPT_OPTIMIZATION', 1, $TPL$## Feedback Samples
{feedback_samples}

## Metrics
{metrics}

## Current Template
```
{current_template}
```

## Task
Synthesize a recurring pain point from feedback and propose ONE minimal fix.

## Output Format
Return ONLY valid JSON:
{
  "pain_point": "1 sentence describing the recurring issue",
  "proposed_change": "1 line change OR 1 example demonstrating the fix",
   "justification": "brief explanation of why this change addresses the pain point"
}$TPL$, 0.7)
ON CONFLICT (prompt_key, version) DO NOTHING;
