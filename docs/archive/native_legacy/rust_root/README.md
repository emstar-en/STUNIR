# Archived: Root Rust Pipeline Implementation

> **Archive Date:** 2026-02-20
> **Original Location:** `src/*.rs`, `src/emit/`
> **Reason:** Consolidation into `tools/` hierarchy; SPARK is the primary pipeline

---

## What Was Archived

This directory contains the root-level Rust pipeline implementation that previously lived in `src/` at the repository root. These files have been archived because:

1. **SPARK is the primary pipeline** — The Ada SPARK implementation in `tools/spark/` is the canonical implementation.
2. **Consolidation** — Pipeline code should live under `tools/` for consistency.
3. **Duplication** — Similar functionality exists in `tools/rust/` and `tools/native/rust/`.

### Files Archived

| File | Purpose |
|------|---------|
| `main.rs` | CLI entry point for `stunir-native` |
| `spec_to_ir.rs` | Spec to IR conversion |
| `emit.rs` | Code emission orchestration |
| `ir_v1.rs` | IR v1 format definitions |
| `canonical.rs` | Canonicalization utilities |
| `validate.rs` | Validation logic |
| `import.rs` | Import handling |
| `toolchain.rs` | Toolchain verification |
| `emit/*.rs` | Language-specific emitters (bash, js, python, powershell, wat) |

---

## Current Implementation

For active Rust pipeline code, see:
- `tools/rust/` — Rust implementation of STUNIR tools
- `tools/native/rust/` — Native Rust verifier

For the canonical pipeline, see:
- `tools/spark/` — Ada SPARK primary implementation
- `tools/spark/ARCHITECTURE.md` — Canonical architecture reference

---

## Policy Reference

See `docs/archive/ARCHIVE_POLICY.md` for the full archival policy.
