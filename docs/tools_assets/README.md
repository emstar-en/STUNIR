# STUNIR Tools Assets

> **Archive Date:** 2026-02-20
> **Purpose:** Tool-agnostic assets moved from `tools/` during consolidation

---

## What's Here

This directory contains non-code assets that were moved from `tools/` to enforce the language-only bucket structure (`tools/spark/`, `tools/python/`, `tools/rust/`, `tools/haskell/`).

### Directories

| Directory | Contents | Original Location |
|-----------|----------|-------------------|
| `emitter_generator/` | Emitter specs, templates, YAML configs | `tools/emitter_generator/` |
| `analysis/` | Analysis JSON configs | `tools/analysis/` |
| `confluence/` | Confluence test scripts and docs | `tools/confluence/` |
| `debuggers/` | Debugger configs | `tools/debuggers/` |
| `inspectors/` | Inspector manifests | `tools/inspectors/` |
| `profilers/` | Profiler configs | `tools/profilers/` |

### Files

| File | Purpose |
|------|---------|
| `discover_toolchain.sh` | Toolchain discovery script |
| `dispatch_config.json` | Dispatch configuration |
| `INDEX.json` | Tools index |
| `index.machine.json` | Machine index |
| `manifest.machine.json` | Machine manifest |
| `prov_emit.c` | Provenance emitter (C) |

---

## Policy Reference

See `docs/archive/ARCHIVE_POLICY.md` for the full archival policy.

For active code, see:
- `tools/spark/` — Ada SPARK primary implementation
- `tools/python/` — Python reference implementation
- `tools/rust/` — Rust implementation
- `tools/haskell/` — Haskell implementation
