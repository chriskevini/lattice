"""Triple parsing utilities for semantic triple extraction."""

import json
import logging


logger = logging.getLogger(__name__)

PREDICATE_SYNONYMS: dict[str, list[str]] = {
    "likes": ["likes", "enjoys", "loves", "fond_of", "appreciates"],
    "works_at": ["works_at", "employed_at", "job_at", "position_at", "works_for"],
    "created": ["created", "built", "made", "constructed", "developed", "authored"],
    "located_in": ["located_in", "based_in", "situated_in", "in"],
    "knows": ["knows", "acquainted_with", "friend_of", "familiar_with"],
    "member_of": ["member_of", "part_of", "belongs_to", "affiliated_with"],
}


def normalize_predicate(predicate: str) -> str:
    """Normalize predicate to canonical form.

    Maps synonym predicates to their canonical form for consistency.

    Args:
        predicate: Raw predicate string from triple

    Returns:
        Normalized predicate string
    """
    normalized = predicate.lower().strip()
    for canonical, synonyms in PREDICATE_SYNONYMS.items():
        if normalized in synonyms:
            return canonical
    return normalized


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
                        "predicate": normalize_predicate(item["predicate"]),
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
