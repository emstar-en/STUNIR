<!--
{
  "schema": "stunir_tombstone_v1",
  "ir_version": "v1",
  "module_name": "stunir_deprecated_manifest",
  "generated_epoch": 0,
  "status": "deprecated",
  "removal_date": "2026-06-01",
  "governance": "Files in this directory are EXCLUDED from stunir_tools.gpr Source_Dirs. They are preserved for historical reference only. Do not add new files here; use src/ subdirectories for active development."
}
-->

# `src/deprecated/` — Deprecated Legacy Monoliths

> **GOVERNANCE:** This directory is **intentionally excluded** from `stunir_tools.gpr`'s
> `Source_Dirs`. None of these files are compiled. They are preserved for historical
> reference and diff archaeology only.
>
> **Scheduled removal:** 2026-06-01 (per `ARCHITECTURE.core.json` deprecation schedule).
>
> **Do NOT add new files here.** If you are deprecating an active tool, move it here,
> update `stunir_tools.gpr` to remove it from `Source_Dirs`, and add an entry to this
> file. See `CONTRIBUTING.md` for the full deprecation checklist.

---

## File Inventory

### `stunir_spec_to_ir.ads` / `stunir_spec_to_ir.adb` (475 lines)

| Field | Value |
|-------|-------|
| **Package** | `STUNIR_Spec_To_IR` |
| **Deprecated** | 2026-01-15 |
| **Removal** | 2026-06-01 |
| **Replaced by** | `src/spec/extraction_to_spec.adb` + `src/spec/spec_validate.adb` + `src/core/ir_converter.adb` |
| **Reason** | Original monolithic spec→IR converter. Manifest-focused (file hashes only); does not produce Semantic IR AST. Uses `GNAT.SHA256` directly instead of the `src/verification/hash_compute.adb` tool. `Max_Manifest_Entries=500` hardcoded limit. |
| **SPARK Mode** | On (but 36 compile errors against current type system) |

### `stunir_spec_to_ir_v2.adb` (370 lines)

| Field | Value |
|-------|-------|
| **Package** | `STUNIR_Spec_To_IR` (same unit name — collision with above) |
| **Deprecated** | 2026-01-20 |
| **Removal** | 2026-06-01 |
| **Replaced by** | `src/core/ir_converter.adb` + `src/semantic_ir/` hierarchy |
| **Reason** | V2 rewrite that converts specs to Semantic IR (not file manifests). Superseded by the modular `src/core/ir_converter.adb` + `src/semantic_ir/` hierarchy. Same package name as V1 causes duplicate-unit error if both were compiled. |
| **SPARK Mode** | On |

### `stunir_spec_to_ir.adb.manifest_version` (342 lines)

| Field | Value |
|-------|-------|
| **Type** | Backup / alternate variant |
| **Deprecated** | 2026-01-15 |
| **Removal** | 2026-06-01 |
| **Replaced by** | `src/core/ir_converter.adb` |
| **Reason** | Alternate version of `stunir_spec_to_ir.adb` focused solely on manifest generation. Superseded by the modular pipeline. Not a valid Ada source file name (contains `.manifest_version` suffix). |

### `stunir_spec_to_ir_main.adb` (539 bytes)

| Field | Value |
|-------|-------|
| **Package** | *(main entry point)* |
| **Deprecated** | 2026-01-15 |
| **Removal** | 2026-06-01 |
| **Replaced by** | `src/core/ir_converter_main.adb` |
| **Reason** | Entry point for the legacy `stunir_spec_to_ir` monolith. Replaced by `ir_converter_main.adb` which drives the modular pipeline. |

### `stunir_ir_to_code.ads` / `stunir_ir_to_code.adb` (1,971 lines)

| Field | Value |
|-------|-------|
| **Package** | `STUNIR_IR_To_Code` |
| **Deprecated** | 2026-01-20 |
| **Removal** | 2026-06-01 |
| **Replaced by** | `src/emitters/stunir-emitters.ads` + `src/emitters/stunir-emitters-*.ads` family |
| **Reason** | Original monolithic IR→code emitter. 1,971 lines; 12 target languages hardcoded. Uses `Lang_*` enum names (should be `Target_*` per current `stunir_types.ads`). 36 compile errors against current type system. Replaced by the modular `STUNIR.Emitters` hierarchy with per-language child packages. |
| **SPARK Mode** | On (but 36 compile errors) |

### `stunir_ir_to_code.adb.backup` (930 lines)

| Field | Value |
|-------|-------|
| **Type** | Backup snapshot |
| **Deprecated** | 2026-01-10 |
| **Removal** | 2026-06-01 |
| **Replaced by** | `src/emitters/` hierarchy |
| **Reason** | Pre-refactor snapshot of `stunir_ir_to_code.adb`. `pragma SPARK_Mode (On)`. Preserved for diff reference only. |

### `stunir_ir_to_code_main.adb` (539 bytes)

| Field | Value |
|-------|-------|
| **Package** | *(main entry point)* |
| **Deprecated** | 2026-01-20 |
| **Removal** | 2026-06-01 |
| **Replaced by** | `src/core/code_emitter_main.adb` |
| **Reason** | Entry point for the legacy `stunir_ir_to_code` monolith. Replaced by `code_emitter_main.adb`. |

### `stunir_json_utils.adb.backup`

| Field | Value |
|-------|-------|
| **Type** | Backup snapshot |
| **Deprecated** | 2026-01-15 |
| **Removal** | 2026-06-01 |
| **Replaced by** | `src/json/stunir_json_utils.adb` |
| **Reason** | Backup of the JSON utilities implementation. Active version is in `src/json/`. Not a valid Ada source file name (`.backup` suffix). |

---

## Why These Files Are Kept (Not Deleted)

1. **Diff archaeology** — The monoliths encode design decisions that informed the modular
   refactor. Keeping them allows future contributors to understand *why* the current
   structure exists.

2. **Removal schedule** — The `ARCHITECTURE.core.json` deprecation schedule sets
   2026-06-01 as the removal date. Premature deletion before that date would violate
   the published contract.

3. **GPR exclusion is the enforcement mechanism** — Since `src/deprecated/` is absent
   from `stunir_tools.gpr`'s `Source_Dirs`, these files have zero effect on the build.
   Keeping them costs nothing at compile time.

---

## Replacement Map

```
stunir_spec_to_ir_main     →  ir_converter_main
stunir_spec_to_ir          →  extraction_to_spec + spec_validate + ir_converter
stunir_spec_to_ir_v2       →  ir_converter + src/semantic_ir/ hierarchy
stunir_ir_to_code_main     →  code_emitter_main
stunir_ir_to_code          →  src/emitters/stunir-emitters-*.ads family
stunir_json_utils.backup   →  src/json/stunir_json_utils.adb
```
