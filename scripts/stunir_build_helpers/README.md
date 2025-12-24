# STUNIR build helpers (release-layout)

This bundle is **language-agnostic**: it provides one generic wrapper that sets `STUNIR_OUTPUT_TARGETS` and runs `scripts/build.sh`.

It is packaged in a **release-like layout** so that extracting at the repository root creates:

- `scripts/stunir_build_helpers/...`

and does **not** drop helper files directly into the repo root.

## Primary entrypoints

- `scripts/stunir_build_helpers/build_targets.sh`
- `scripts/stunir_build_helpers/build_targets.ps1`

They:

- verify you are running from the STUNIR repo root (by requiring `scripts/build.sh`)
- set `STUNIR_OUTPUT_TARGETS` to the targets you request
- optionally set `STUNIR_REQUIRE_DEPS` to fail-fast when runtime/toolchain dependencies are missing

## Presets

Convenience wrappers live under:

- `scripts/stunir_build_helpers/presets/`

These call `build_targets.*` with common target sets.

## Notes

- Target names + aliases and required contracts are defined in `contracts/target_requirements.json`.
- When `STUNIR_OUTPUT_TARGETS` is set, `scripts/build.sh` will run `scripts/ensure_deps.sh` to resolve requirements and probe dependencies.
- For runtime-backed targets (e.g. `lisp_sbcl`, `python_cpython`, `smt_z3`), you typically want `--require-deps` so the build fails instead of skipping.

## Examples (conceptual)

This README intentionally does not include copy/paste commands; use the presets or call `build_targets` with your target list.
