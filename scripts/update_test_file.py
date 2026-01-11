#!/usr/bin/env python3
"""Script to update test file for DI refactoring."""

import re

with open("/home/chris/Work/lattice/tests/unit/test_batch_consolidation.py", "r") as f:
    lines = f.readlines()

result = []
i = 0
while i < len(lines):
    line = lines[i]

    # Check for start of db_pool patch (handle both 4 and 8 space indentation)
    if 'with patch("lattice.utils.database.db_pool") as mock_db_pool:' in line:
        # Count the indentation
        indent = len(line) - len(line.lstrip())
        # This line should be removed
        i += 1
        # Next line should be mock_db_pool.pool = mock_pool - remove it
        if i < len(lines) and "mock_db_pool.pool = mock_pool" in lines[i]:
            i += 1
        # Check for double patch
        if (
            i < len(lines)
            and 'with patch("lattice.utils.database.db_pool", mock_db_pool):'
            in lines[i]
        ):
            # This is the double-patch case - remove it and dedent following content by 8
            i += 1
            dedent_amount = 8
        else:
            # Simple case - just remove the outer with, dedent following content by 4
            dedent_amount = 4

        # Now dedent the following content
        while i < len(lines):
            next_line = lines[i]
            stripped = next_line.lstrip()
            current_indent = len(next_line) - len(stripped)
            # If this line has content and is indented less than or equal to original indent, stop
            if stripped and current_indent <= indent:
                break
            # Dedent
            if stripped:
                new_indent = current_indent - dedent_amount
                result.append(" " * new_indent + stripped + "\n")
            else:
                result.append(next_line)
            i += 1
        continue

    result.append(line)
    i += 1

content = "".join(result)

# Fix function calls
content = re.sub(
    r"(\s+)await run_batch_consolidation\(\)",
    r"\1await run_batch_consolidation(db_pool=mock_pool)",
    content,
)
content = re.sub(
    r"(\s+)await check_and_run_batch\(\)",
    r"\1await check_and_run_batch(db_pool=mock_pool)",
    content,
)

with open("/home/chris/Work/lattice/tests/unit/test_batch_consolidation.py", "w") as f:
    f.write(content)

print("Done")
