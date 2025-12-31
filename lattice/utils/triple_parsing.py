"""Triple parsing utilities for semantic triple extraction."""

import json
import logging


logger = logging.getLogger(__name__)


def parse_triples(raw_output: str) -> list[dict[str, str]]:
    """Parse LLM output into structured triples.

    Args:
        raw_output: Raw string output from LLM

    Returns:
        List of validated triples with subject, predicate, object
    """
    try:
        cleaned = raw_output.strip()

        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if len(lines) >= 2:
                cleaned = "\n".join(lines[1:-1])
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].strip()

        triples = json.loads(cleaned)

        if not isinstance(triples, list):
            logger.warning("parse_triples: expected list, got %s", type(triples))
            return []

        validated = []
        for item in triples:
            if (
                isinstance(item, dict)
                and "subject" in item
                and "predicate" in item
                and "object" in item
                and all(isinstance(item[k], str) for k in ("subject", "predicate", "object"))
            ):
                validated.append(
                    {
                        "subject": item["subject"].strip(),
                        "predicate": item["predicate"].strip().lower(),
                        "object": item["object"].strip(),
                    }
                )
            else:
                logger.warning("parse_triples: invalid triple format: %s", item)

        return validated

    except json.JSONDecodeError as e:
        logger.warning("parse_triples: JSON decode error: %s", e)
        return []
    except Exception as e:
        logger.error("parse_triples: unexpected error: %s", e)
        return []
