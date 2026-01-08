"""Memory extraction parsing utilities for batch triple extraction."""

import logging
from typing import TypedDict


logger = logging.getLogger(__name__)


class ParsedTriple(TypedDict):
    """Parsed semantic triple."""

    subject: str
    predicate: str
    object: str


PREDICATE_SYNONYMS: dict[str, list[str]] = {
    "likes": ["likes", "enjoys", "loves", "fond_of", "appreciates"],
    "works_at": ["works_at", "employed_at", "job_at", "position_at", "works_for"],
    "created": ["created", "built", "made", "constructed", "developed", "authored"],
    "located_in": ["located_in", "based_in", "situated_in", "in"],
    "knows": ["knows", "acquainted_with", "friend_of", "familiar_with"],
    "member_of": ["member_of", "part_of", "belongs_to", "affiliated_with"],
}

MIN_CODE_BLOCK_LINES = 2


def _normalize_predicate(predicate: str) -> str:
    """Normalize predicate to canonical form.

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


def _parse_triples_from_list(triples: list) -> list[ParsedTriple]:
    """Parse triples from a list, validating structure.

    Args:
        triples: List of triple dicts from JSON

    Returns:
        List of validated triples
    """
    validated: list[ParsedTriple] = []
    for item in triples:
        if not isinstance(item, dict):
            logger.warning(
                "parse_triples: expected dict for triple, got %s", type(item)
            )
            continue

        if (
            "subject" in item
            and "predicate" in item
            and "object" in item
            and all(
                isinstance(item[k], str) for k in ("subject", "predicate", "object")
            )
        ):
            validated.append(
                {
                    "subject": item["subject"].strip(),
                    "predicate": _normalize_predicate(item["predicate"]),
                    "object": item["object"].strip(),
                }
            )
        else:
            logger.warning("parse_triples: invalid triple format: %s", item)

    return validated
