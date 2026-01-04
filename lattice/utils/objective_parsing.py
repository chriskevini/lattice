"""Objective parsing utilities for objective extraction."""

import json
import logging


logger = logging.getLogger(__name__)

MIN_SALIENCY = 0.0
MAX_SALIENCY = 1.0
VALID_STATUSES = frozenset({"pending", "completed", "archived"})
MIN_CODE_BLOCK_LINES = 2
MIN_SALIENCY_DELTA = 0.01


def parse_objectives(raw_output: str) -> list[dict[str, str | float]]:
    """Parse LLM output into structured objectives.

    Args:
        raw_output: Raw string output from LLM

    Returns:
        List of validated objectives with description, saliency, and status
    """
    cleaned = raw_output.strip()

    try:
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if len(lines) >= MIN_CODE_BLOCK_LINES:
                cleaned = "\n".join(lines[1:-1])
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].strip()

        objectives = json.loads(cleaned)

        if not isinstance(objectives, list):
            logger.warning("parse_objectives: expected list, got %s", type(objectives))
            return []

        validated = []
        for item in objectives:
            if not isinstance(item, dict):
                logger.warning("parse_objectives: expected dict, got %s", type(item))
                continue

            description = item.get("description")
            if (
                not description
                or not isinstance(description, str)
                or not description.strip()
            ):
                logger.warning(
                    "parse_objectives: missing or invalid description: %s", item
                )
                continue

            saliency = item.get("saliency", 0.5)
            if not isinstance(saliency, (int, float)):
                logger.warning(
                    "parse_objectives: invalid saliency, using default: %s",
                    item,
                )
                saliency = 0.5
            saliency = float(saliency)
            saliency = max(MIN_SALIENCY, min(MAX_SALIENCY, saliency))

            status = item.get("status", "pending")
            status = (
                "pending" if not isinstance(status, str) else status.lower().strip()
            )
            if status not in VALID_STATUSES:
                logger.warning(
                    "parse_objectives: invalid status '%s', using 'pending'",
                    status,
                )
                status = "pending"

            validated.append(
                {
                    "description": description.strip(),
                    "saliency": saliency,
                    "status": status,
                }
            )

        return validated

    except json.JSONDecodeError as e:
        logger.warning("parse_objectives: JSON decode error: %s, trying text format", e)
        return []
    except Exception:
        logger.exception("parse_objectives: unexpected error")
        return []
