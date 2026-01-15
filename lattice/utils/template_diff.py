"""Template diff utilities for prompt management."""

import difflib


def generate_diff(
    old: str, new: str, from_version: str = "current", to_version: str = "preview"
) -> str:
    """Generate a unified diff between two template strings.

    Args:
        old: The original template content
        new: The new template content
        from_version: Label for the old version (default: "current")
        to_version: Label for the new version (default: "preview")

    Returns:
        Unified diff string suitable for Discord code block
    """
    old_lines = old.splitlines(keepends=True) if old else []
    new_lines = new.splitlines(keepends=True) if new else []

    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=from_version,
        tofile=to_version,
    )

    return (
        "".join(diff) if diff else f"No changes (from {from_version} to {to_version})"
    )
