#!/usr/bin/env python3
"""Enforce Discord Components V2 usage.

This script checks that all Discord UI components use V2 APIs (DesignerView + ActionRow)
instead of deprecated V1 APIs (View + @discord.ui.button decorator).

Exit codes:
    0: All checks passed
    1: V1 component usage detected
"""

import re
import sys
from pathlib import Path


# Patterns that indicate V1 usage (banned)
V1_PATTERNS = [
    (
        r"class\s+\w+\(discord\.ui\.View\)",
        "Use discord.ui.DesignerView instead of discord.ui.View for V2 components",
    ),
    (
        r"@discord\.ui\.button",
        "Use ActionRow with Button instead of @discord.ui.button decorator (V2 pattern)",
    ),
    (
        r"@discord\.ui\.select",
        "Use ActionRow with Select instead of @discord.ui.select decorator (V2 pattern)",
    ),
]

# Files to check
DISCORD_CLIENT_FILES = [
    "lattice/discord_client/dream.py",
    "lattice/discord_client/bot.py",
    "lattice/dreaming/approval.py",
]

# Exceptions (modals are not V2 components, they're a separate feature)
ALLOWED_PATTERNS = [
    r"class\s+\w+Modal\(discord\.ui\.Modal\)",  # Modals are allowed
]


def check_file(filepath: Path) -> list[tuple[int, str, str]]:
    """Check a file for V1 component usage.

    Args:
        filepath: Path to the file to check

    Returns:
        List of (line_number, line_content, error_message) tuples
    """
    if not filepath.exists():
        return []

    errors = []
    content = filepath.read_text()
    lines = content.splitlines()

    for line_num, line in enumerate(lines, 1):
        # Skip lines that match allowed patterns
        if any(re.search(pattern, line) for pattern in ALLOWED_PATTERNS):
            continue

        # Check for V1 patterns
        for pattern, message in V1_PATTERNS:
            if re.search(pattern, line):
                errors.append((line_num, line.strip(), message))

    return errors


def main() -> int:
    """Run V2 component checks."""
    project_root = Path(__file__).parent.parent
    all_errors = []

    print("🔍 Checking Discord UI component usage...")
    print()

    for file_path in DISCORD_CLIENT_FILES:
        full_path = project_root / file_path
        errors = check_file(full_path)

        if errors:
            all_errors.extend(errors)
            print(f"❌ {file_path}:")
            for line_num, line_content, error_message in errors:
                print(f"   Line {line_num}: {error_message}")
                print(f"   > {line_content}")
                print()

    if all_errors:
        print("=" * 80)
        print("⚠️  V1 Discord components detected!")
        print()
        print("MIGRATION GUIDE:")
        print()
        print("1. Replace discord.ui.View with discord.ui.DesignerView:")
        print("   ❌ class MyView(discord.ui.View):")
        print("   ✅ class MyView(discord.ui.DesignerView):")
        print()
        print("2. Replace @discord.ui.button with ActionRow + Button:")
        print(
            "   ❌ @discord.ui.button(label='Click', style=discord.ButtonStyle.primary)"
        )
        print("      async def my_button(self, interaction, button):")
        print()
        print(
            "   ✅ button = discord.ui.Button(label='Click', style=discord.ButtonStyle.primary)"
        )
        print("      button.callback = self._make_callback()")
        print("      action_row = discord.ui.ActionRow(button)")
        print("      self.add_item(action_row)")
        print()
        print("3. Modals are NOT V2 components - they stay as discord.ui.Modal")
        print()
        print("See lattice/discord_client/dream.py for complete examples.")
        print("=" * 80)
        return 1

    print("✅ All Discord UI components are using V2 APIs!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
