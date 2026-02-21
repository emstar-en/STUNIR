# Archived: Deprecated SPARK Sources

> **Archive Date:** 2026-02-20
> **Reason:** Superseded by SPARK core refactor (`tools/spark/src/core/`) or consolidated into language buckets

---

## What Was Archived

### Original Deprecated Sources (from `tools/spark/src/deprecated/`)

These files were the original monolithic implementations of the STUNIR SPARK tools.
They have been replaced by the modular pipeline in `tools/spark/src/core/`.

### Deprecated Tools

| Tool | Deprecated | Removal | Replacement |
|------|-----------|---------|-------------|
| `stunir_spec_to_ir_main` | 2026-01-15 | 2026-06-01 | `ir_converter_main` |
| `stunir_ir_to_code_main` | 2026-01-20 | 2026-06-01 | `code_emitter_main` |
| `stunir_code_index_main` | 2026-01-20 | 2026-06-01 | (integrated into pipeline) |

### Files in This Archive

- `stunir_spec_to_ir.adb` / `stunir_spec_to_ir.ads` — Original spec-to-IR converter
- `stunir_ir_to_code.adb` / `stunir_ir_to_code.ads` — Original IR-to-code emitter
- `stunir_spec_to_ir_main.adb` — Original main entry point
- `stunir_ir_to_code_main.adb` — Original main entry point
- `stunir_spec_to_ir_v2.adb` — Intermediate refactor attempt
- `stunir_json_utils.adb.backup` — Backup of JSON utilities
- `*.backup` files — Various backups

### Additional Archived SPARK Code (2026-02-20)

| Directory | Original Location | Reason |
|-----------|-------------------|--------|
| `targets_spark/` | `tools/python/targets/spark/` | Outdated SPARK targets; use `tools/spark/` instead |
| `python_core_spark/` | `tools/python/core/` | SPARK/Ada code in Python bucket; use `tools/spark/` instead |

---

## Current Implementation

The current SPARK implementation is in:

- **Pipeline Driver:** `tools/spark/src/core/pipeline_driver_main.adb`
- **IR Converter:** `tools/spark/src/core/ir_converter_main.adb`
- **Code Emitter:** `tools/spark/src/core/code_emitter_main.adb`
- **Spec Assembler:** `tools/spark/src/core/spec_assembler_main.adb`

See `tools/spark/ARCHITECTURE.md` for the canonical architecture reference.

---

## Governance

Per `tools/spark/CONTRIBUTING.md`:

> `src/deprecated/` is NOT in `Source_Dirs`. This is deliberate. Files there are
> preserved for historical reference only. Do not add new files to `src/deprecated/`.

These files are now archived here and should not be restored to active development.
