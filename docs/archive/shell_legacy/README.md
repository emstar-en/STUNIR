# Archived: Shell-Centric Pipeline Scripts

> **Archive Date:** 2026-02-20
> **Reason:** Shell offloading pattern deprecated due to host contamination and portability risks

---

## What Was Archived

### Native Shell Pipeline (`tools/native/shell/`)
- `stunir_core.sh` — Shell-based compiler and emitter using jq for JSON manipulation

This implementation used shell commands and `jq` to process JSON, which:
- Invites host contamination (environment-specific behavior)
- Has poor portability across different shell implementations
- Bypasses structured receipt generation

### Router Script (`tools/native/router.sh`)
- Simple router to Haskell native binary

### Pipeline Scripts (`scripts/`)
- `build_haskell_first.sh`
- `build_haskell_pipeline.sh`
- `build_rust_pipeline.sh`
- `haskell_pipeline_complete.sh`
- `rust_pipeline_complete.sh`
- `pipeline_haskell.sh`
- `pipeline_python.sh`
- `pipeline_rust.sh`
- `run_all_pipelines.sh`
- `test_haskell_pipeline.sh`
- `test_rust_pipeline.sh`
- `test_haskell_rust_confluence.sh`
- `emit_shell.sh`

These scripts implemented parallel pipelines that diverged from the canonical SPARK pipeline.

---

## Current Approach

The canonical pipeline is now:

1. **SPARK tools** (`tools/spark/`) — Primary implementation with formal verification
2. **`scripts/build.sh`** — Unified entry point (SPARK-first)
3. **`scripts/verify.sh`** — Receipt verification

See `tools/spark/ARCHITECTURE.md` for the canonical architecture.

---

## Policy Reference

See `docs/archive/ARCHIVE_POLICY.md` for the full archival policy, including the
rationale for deprecating shell offloading patterns.
