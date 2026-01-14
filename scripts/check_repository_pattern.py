#!/usr/bin/env python3
"""Check for violations of the repository pattern.

This script ensures that direct database access (db_pool.pool.acquire) is only
used within repository implementations, not in business logic.
"""

import re
import sys
from pathlib import Path

# Files allowed to use direct database access
ALLOWED_FILES = {
    "lattice/memory/repositories.py",
    "lattice/memory/context.py",
    "lattice/utils/database.py",
}

# Pattern to detect direct database access
DB_ACCESS_PATTERN = re.compile(r"db_pool\.pool\.acquire|conn\.fetch|conn\.execute")

# Patterns to detect MagicMock anti-patterns in production code
MAGICMOCK_PATTERN = re.compile(
    r"from unittest\.mock import.*MagicMock|isinstance.*MagicMock"
)


def check_file(filepath: Path) -> list[str]:
    """Check a single file for repository pattern violations.

    Args:
        filepath: Path to the file to check

    Returns:
        List of error messages (empty if no violations)
    """
    errors = []

    # Skip test files and __pycache__
    if "test" in str(filepath) or "__pycache__" in str(filepath):
        return errors

    # Check if file is in allowed list
    relative_path = str(filepath).replace(str(Path.cwd()) + "/", "")
    is_allowed = any(relative_path.endswith(allowed) for allowed in ALLOWED_FILES)

    try:
        content = filepath.read_text()
        lines = content.split("\n")

        # Check for direct database access
        if not is_allowed:
            for i, line in enumerate(lines, 1):
                if DB_ACCESS_PATTERN.search(line) and "# repository-ok" not in line:
                    errors.append(
                        f"{filepath}:{i}: Direct database access detected. "
                        "Use repository pattern instead."
                    )

        # Check for MagicMock usage in production code
        for i, line in enumerate(lines, 1):
            if MAGICMOCK_PATTERN.search(line) and "# mock-ok" not in line:
                errors.append(
                    f"{filepath}:{i}: MagicMock detected in production code. "
                    "Use proper type checking or repository pattern instead."
                )

    except Exception as e:
        errors.append(f"{filepath}: Error reading file: {e}")

    return errors


def main() -> int:
    """Main entry point.

    Returns:
        0 if no violations, 1 if violations found
    """
    lattice_dir = Path("lattice")
    if not lattice_dir.exists():
        print("Error: lattice directory not found", file=sys.stderr)
        return 1

    all_errors = []

    # Check all Python files in lattice/
    for filepath in lattice_dir.rglob("*.py"):
        errors = check_file(filepath)
        all_errors.extend(errors)

    if all_errors:
        print("Repository pattern violations found:", file=sys.stderr)
        for error in all_errors:
            print(f"  {error}", file=sys.stderr)
        return 1

    print("✓ Repository pattern check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
