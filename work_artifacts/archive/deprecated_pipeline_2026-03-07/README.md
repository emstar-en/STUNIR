# Deprecated Pipeline Items Archive

**Archive Date**: 2026-03-07  
**Reason**: Cleanup of deprecated IR/pipeline items to avoid model confusion

## Archived Items

### SPARK Source Files (spark_src/)
Moved from `tools/spark/src/deprecated/`:
- `code_emitter.adb/ads` - Old code emitter (replaced by `emit_target.adb`)
- `extraction_parser.adb/ads` - Old extraction parser
- `file_hash.adb` - Moved to `src/files/`
- `ir_converter.adb/ads` - Old IR converter (replaced by `ir_parse.adb`)
- `json_read.adb`, `json_validator.adb`, `json_write.adb` - Old JSON utilities
- `spec_assembler.adb/ads` - Old spec assembler (replaced by `assemble_spec.adb`)
- `spec_validate.adb` - Old spec validator (replaced by `spec_validate_schema.adb`)
- `type_map_cpp.adb` - Old C++ type mapper
- `type_resolve.adb` - Old type resolver (replaced by `type_resolver.adb`)
- `DEPRECATED.md` - Original deprecation documentation

### SPARK Emitters (spark_emitters/)
Moved from `tools/spark/src/emitters/`:
- `stunir-emitters-lisp.ads` - Deprecated Lisp family emitter
- `stunir-emitters-prolog_family.ads` - Deprecated Prolog family emitter

These were marked as deprecated with scheduled removal 2026-06-01. The canonical emitter path is now `emit_target.adb`.

### Python Emitters (python_emitters/)
Moved from `tools/python/ir/emitters/language_families/`:
- `lisp.py` - Deprecated Python Lisp emitter
- `prolog.py` - Deprecated Python Prolog emitter

These were marked as deprecated with scheduled removal 2026-06-01.

## Updated References

### Tool Name Changes
| Old Name | New Name | Notes |
|----------|----------|-------|
| `stunir_spec_to_ir_main` | `spec_to_ir_main` or `ir_converter_main` | Spec to IR conversion |
| `stunir_ir_to_code_main` | `code_emitter_main` | IR to code emission |
| `stunir_code_index_main` | `file_indexer`, `hash_compute` | Integrated into pipeline |

### Files Updated
- `scripts/build.sh` - Removed deprecated tool checks
- `scripts/test_confluence.sh` - Updated to current tool names
- `test_stunir_spark.sh` - Updated to current tool names
- `README.md` - Updated tool references
- `VERSION_STATUS.md` - Updated tool references
- `precompiled/README.md` - Updated tool references
- `tools/spark/README.md` - Updated deprecated tools list
- `tools/spark/ARCHITECTURE.md` - Updated deprecation schedule
- `tools/spark/ARCHITECTURE.core.json` - Updated legacy compatibility
- `tools/spark/ARCHITECTURE.workflows.json` - Updated workflow patterns
- `tools/spark/local_toolchain.lock.json` - Updated tool names
- `tools/spark/AI_TOOL_CONTRACTS.md` - Updated tool names
- `tools/python/spec_to_ir.py` - Updated primary implementation reference
- `tools/python/ir_to_code.py` - Updated primary implementation reference
- `tools/python/scripts/stunir_executor.py` - Updated tool names

## Current Tool Names

Use these tool names for all new work:

### Primary Pipeline Tools
- `spec_to_ir_main` - Spec to IR conversion
- `ir_converter_main` - Alternative spec to IR conversion
- `code_emitter_main` - IR to code emission
- `pipeline_driver_main` - Full pipeline orchestration

### Utility Tools
- `file_indexer` - File indexing
- `hash_compute` - Hash computation
- `lang_detect` - Language detection
- `format_detect` - Format detection
- `spec_assembler_main` - Spec assembly
- `assemble_spec_main` - Alternative spec assembly

## Notes

- All deprecated items have been moved to this archive directory
- No stubs were left in original locations
- Active pipeline code has been updated to use current tool names
- Archive preserves history for reference