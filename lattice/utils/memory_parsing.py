"""Memory extraction parsing utilities for batch triple extraction."""

import logging
from typing import TypedDict


logger = logging.getLogger(__name__)


class ParsedTriple(TypedDict):
    """Parsed semantic triple."""

    subject: str
    predicate: str
    object: str


def _normalize_predicate(predicate: str) -> str:
    """Normalize predicate to canonical form.

    Args:
        predicate: Raw predicate string from triple

    Returns:
        Normalized predicate string
    """
    return predicate.lower().strip()


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
