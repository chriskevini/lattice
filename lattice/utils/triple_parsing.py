"""Triple parsing utilities for semantic triple extraction."""

import json
import logging


logger = logging.getLogger(__name__)


TRIPLE_PART_COUNT = 3
MIN_CODE_BLOCK_LINES = 2


def normalize_predicate(predicate: str) -> str:
    """Normalize predicate to canonical form.

    Args:
        predicate: Raw predicate string from triple

    Returns:
        Normalized predicate string
    """
    return predicate.lower().strip()


def _parse_text_format(text: str) -> list[dict[str, str]] | None:
    """Parse text format triples (subject -> predicate -> object).

    Handles formats like:
        Triples:
        alice -> knows -> bob
        bob -> works_at -> company

    Args:
        text: Text to parse

    Returns:
        List of triples or None if format not recognized
    """
    lines = text.strip().split("\n")
    triples: list[dict[str, str]] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.lower().startswith("triple"):
            continue

        for separator in ["-->", "->", "→"]:
            if separator in line:
                parts = line.split(separator)
                if len(parts) == TRIPLE_PART_COUNT:
                    subject = parts[0].strip()
                    predicate = parts[1].strip()
                    obj = parts[2].strip()

                    if subject and predicate and obj:
                        triples.append(
                            {
                                "subject": subject,
                                "predicate": normalize_predicate(predicate),
                                "object": obj,
                            }
                        )
                break

    return triples if triples else None


def parse_triples(raw_output: str) -> list[dict[str, str]]:
    """Parse LLM output into structured triples.

    Args:
        raw_output: Raw string output from LLM

    Returns:
        List of validated triples with subject, predicate, object
    """
    cleaned = raw_output.strip()

    try:
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if len(lines) >= MIN_CODE_BLOCK_LINES:
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
                and all(
                    isinstance(item[k], str) for k in ("subject", "predicate", "object")
                )
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
        logger.warning("parse_triples: JSON decode error: %s, trying text format", e)
        text_triples = _parse_text_format(cleaned)
        if text_triples is not None:
            return text_triples
        return []
    except Exception:
        logger.exception("parse_triples: unexpected error")
        return []
