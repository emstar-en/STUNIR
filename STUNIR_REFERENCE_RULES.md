# STUNIR_REFERENCE_RULES

This file describes how repository documents should reference other repository documents.
It is intended to reduce navigation ambiguity for humans and AI agents.

## Rules

1. Use repo-relative paths in markdown links.
2. Do not link to per-run output artifacts under `build/` and `receipts/` unless referencing fixtures.
3. If a directory is referenced, prefer an explicit `README.md` target.
4. Avoid placeholder READMEs that have no outbound references; they should point back to `ENTRYPOINT.md` and `AI_START_HERE.md`.
5. Keep at least one stable root entrypoint (`ENTRYPOINT.md`) and one stable navigation index (`AI_START_HERE.md`).

## Recommended minimum cross-links

- `ENTRYPOINT.md` should link to `AI_START_HERE.md`.
- Placeholder files should link back to `ENTRYPOINT.md`.
