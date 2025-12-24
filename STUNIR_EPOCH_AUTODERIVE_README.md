# STUNIR epoch auto-derive overlay

This overlay updates STUNIR so the build does **not** require the user to supply an epoch.

## What changes

- `tools/epoch.py`
  - Removes implicit wall-clock fallback by default.
  - Adds deterministic epoch derivation from the `spec/` directory digest ("DERIVED_SPEC_DIGEST_V1").
  - Keeps overrides (`STUNIR_BUILD_EPOCH`, `SOURCE_DATE_EPOCH`) as the highest priority.

- `tools/gen_provenance.py`
  - Fixes the script (the upstream file is truncated) and emits both:
    - `build/provenance.json`
    - `build/provenance.h` (consumed by `tools/prov_emit.c`)

- `scripts/build.sh`
  - Updates messaging to reflect that a deterministic default exists.

- `README.md`
  - Updates the documented epoch selection semantics.

## How to apply

Extract this zip at the repository root. It contains replacement files at their correct paths.

## Notes

- The derived epoch is deterministic per spec digest. It is not intended to represent real time.
- If you truly want wall-clock behavior for a non-reproducible run, call:
  - `python3 tools/epoch.py --allow-current-time ...`
