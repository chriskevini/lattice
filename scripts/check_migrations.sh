#!/usr/bin/env bash
set -euo pipefail

echo "Auditing migration changes against main..."

CHANGED_FILES=$(git diff --name-only main...HEAD -- 'scripts/migrations/' 2>/dev/null | xargs || true)

if [ -z "$CHANGED_FILES" ]; then
    echo "No existing migrations were modified. Proceeding."
    exit 0
fi

FORBIDDEN_CHANGES=()
for FILE in $CHANGED_FILES; do
    if git ls-tree -r main --name-only | grep -q "^$FILE$"; then
        FORBIDDEN_CHANGES+=("$FILE")
    fi
done

if [ ${#FORBIDDEN_CHANGES[@]} -ne 0 ]; then
    echo "ERROR: The following immutable migrations were modified or renamed:"
    for FILE in "${FORBIDDEN_CHANGES[@]}"; do
        echo "  - $FILE"
    done
    echo "Directly modifying merged migrations causes schema desync. Create a new version instead."
    exit 1
fi

echo "No immutable migration changes detected."
