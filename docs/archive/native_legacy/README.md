# Archived: Native Legacy Code

> **Archive Date:** 2026-02-20
> **Reason:** Consolidation into language buckets (`tools/spark/`, `tools/python/`, `tools/rust/`, `tools/haskell/`)

---

## What Was Archived

### rust_root/ (from `src/*.rs`)

Root-level Rust pipeline implementation that lived outside `tools/`. Archived because:
1. SPARK is the primary pipeline
2. Similar functionality exists in `tools/rust/`
3. Pipeline code should live under `tools/`

### native_tools/ (from `tools/native/`)

Native tools directory containing mixed-language implementations:
- C code
- Rust code
- Python code
- Haskell code
- Shell scripts

Archived because:
1. Mixed-language structure violated the language-bucket policy
2. Shell offloading is deprecated
3. SPARK is the primary pipeline

---

## Current Implementation

For active implementations, see:
- `tools/spark/` — Ada SPARK primary implementation
- `tools/python/` — Python reference implementation
- `tools/rust/` — Rust implementation
- `tools/haskell/` — Haskell implementation

---

## Policy Reference

See `docs/archive/ARCHIVE_POLICY.md` for:
- Shell offloading deprecation rationale
- SPARK-first policy
- Language bucket consolidation policy
