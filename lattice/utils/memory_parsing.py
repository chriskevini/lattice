"""Memory parsing utilities for semantic memory extraction."""

import json
import logging


logger = logging.getLogger(__name__)


MEMORY_PART_COUNT = 3
MIN_CODE_BLOCK_LINES = 2


def normalize_predicate(predicate: str) -> str:
    """Normalize predicate to canonical form.

    Args:
        predicate: Raw predicate string from memory

    Returns:
        Normalized predicate string
    """
    return predicate.lower().strip()


def _parse_text_format(text: str) -> list[dict[str, str]] | None:
    """Parse text format memories (subject -> predicate -> object).

    Handles formats like:
        Memories:
        alice -> knows -> bob
        bob -> works_at -> company

    Args:
        text: Text to parse

    Returns:
        List of memories or None if format not recognized
    """
    lines = text.strip().split("\n")
    memories: list[dict[str, str]] = []

    for raw_line in lines:
        line = raw_line.strip()
        if (
            not line
            or line.lower().startswith("memory")
            or line.lower().startswith("triple")
        ):
            continue

        for separator in ["-->", "->", "→"]:
            if separator in line:
                parts = line.split(separator)
                if len(parts) == MEMORY_PART_COUNT:
                    subject = parts[0].strip()
                    predicate = parts[1].strip()
                    obj = parts[2].strip()

                    if subject and predicate and obj:
                        memories.append(
                            {
                                "subject": subject,
                                "predicate": normalize_predicate(predicate),
                                "object": obj,
                            }
                        )
                break

    return memories if memories else None


def parse_semantic_memories(raw_output: str) -> list[dict[str, str]]:
    """Parse LLM output into structured semantic memories.

    Args:
        raw_output: Raw string output from LLM

    Returns:
        List of validated memories with subject, predicate, object
    """
    cleaned = raw_output.strip()

    try:
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if len(lines) >= MIN_CODE_BLOCK_LINES:
                cleaned = "\n".join(lines[1:-1])
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].strip()

        memories = json.loads(cleaned)

        if not isinstance(memories, list):
            logger.warning(
                "parse_semantic_memories: expected list, got %s", type(memories)
            )
            return []

        validated = []
        for item in memories:
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
                logger.warning(
                    "parse_semantic_memories: invalid memory format: %s", item
                )

        return validated

    except json.JSONDecodeError as e:
        logger.warning(
            "parse_semantic_memories: JSON decode error: %s, trying text format", e
        )
        text_memories = _parse_text_format(cleaned)
        if text_memories is not None:
            return text_memories
        return []
    except Exception:
        logger.exception("parse_semantic_memories: unexpected error")
        return []


def parse_triples(raw_output: str) -> list[dict[str, str]]:
    """Backwards compatibility wrapper for parse_semantic_memories."""
    return parse_semantic_memories(raw_output)
