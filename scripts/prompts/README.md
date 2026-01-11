# Prompt Templates

Canonical prompt templates for Lattice.

## Versioning Convention

| Version | Who controls | How to update |
|---------|--------------|---------------|
| v1 | Maintainer (you) | Edit this file, commit, run migrations |
| v2+ | User (dream cycle) | User modifies via Discord, never touched by maintainer |

**Rule:** Always keep canonical prompts at v1. Git handles version history. User customizations get v2, v3, etc.

## Updating a Canonical Prompt

1. Edit the `.sql` file
2. Commit the change
3. Run `python scripts/migrate.py` - users who haven't customized will get the update automatically

## Adding a New Prompt

1. Create `NEW_PROMPT.sql` in this directory
2. Follow the header format in existing files
3. Add to migrations if it needs schema changes
4. Run migrations to load it
