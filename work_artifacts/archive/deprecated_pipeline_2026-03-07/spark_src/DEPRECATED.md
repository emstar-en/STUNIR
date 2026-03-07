# Deprecated SPARK Tools

This directory contains deprecated SPARK tools that have been replaced by
canonical implementations. These files are preserved for historical reference
only and are **excluded from the build** (see `stunir_tools.gpr` RULE 3).

**Scheduled removal: 2026-06-01**

---

## Deprecated Files (moved 2026-02-20)

| File | Reason | Replacement |
|------|--------|-------------|
| `json_validator.adb` | Duplicates `json_validate.adb` | `src/json/json_validate.adb` |
| `json_read.adb` | Duplicates `json_validate.adb` | `src/json/json_validate.adb` |
| `json_write.adb` | Empty file (no implementation) | N/A |
| `file_hash.adb` | Duplicates `hash_compute.adb` | `src/verification/hash_compute.adb` |
| `type_map_cpp.adb` | Duplicates `type_map.adb --lang=cpp` | `src/types/type_map.adb` |
| `type_resolve.adb` | Duplicates `type_resolver.adb` | `src/types/type_resolver.adb` |
| `spec_validate.adb` | Stub implementation | `src/spec/spec_validate_schema.adb` |

---

## Monolithic Tools Decomposed (moved 2026-01-04)

These monolithic tools were decomposed into smaller micro-tools following the
single-responsibility principle. Each micro-tool has a library package and a
thin CLI wrapper.

### Phase 1: Spec Assembly

| Old Tool | New Micro-Tools | Status |
|----------|-----------------|--------|
| `spec_assembler.ads/.adb` | `src/spec/spec_parse.ads/.adb`, `src/spec/extract_parse.ads/.adb`, `src/spec/assemble_spec.ads/.adb` | DEPRECATED |
| `extraction_parser.ads/.adb` | `src/spec/extract_parse.ads/.adb` | DEPRECATED |

### Phase 2: IR Conversion

| Old Tool | New Micro-Tools | Status |
|----------|-----------------|--------|
| `ir_converter.ads/.adb` | `src/ir/ir_parse.ads/.adb`, `src/ir/spec_to_ir.ads/.adb` | DEPRECATED |

### Phase 3: Code Emission

| Old Tool | New Micro-Tools | Status |
|----------|-----------------|--------|
| `code_emitter.ads/.adb` | `src/emitters/emit_target.ads/.adb` | DEPRECATED |

### Migration Guide

**Old**: `ir_converter_main input_spec.json output_ir.json`
**New**: `spec_to_ir_main input_spec.json output_ir.json`

**Old**: `code_emitter_main input_ir.json --target python output.py`
**New**: `emit_target_main input_ir.json python output.py`

---

## Source

These files were moved from their original locations on 2026-02-20 as part of
the SPARK toolchain semantic collision cleanup. See:
- `docs/archive/TOOLCHAIN_FINAL_DUPLICATION_REPORT.md` for the original analysis
- `ARCHITECTURE.md` for canonical tool documentation
- `stunir_tools.gpr` for the authoritative build manifest
