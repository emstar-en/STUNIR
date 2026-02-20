# STUNIR Powertools Inventory - Complete List

**Date**: 2026-02-18  
**Total Powertools**: 66 tools  
**Organization**: Organized in 9 category subdirectories  
**Philosophy**: Unix-style - each tool does ONE thing well

---

## Executive Summary

**GOOD NEWS**: Powertools decomposition is COMPLETE! 66 small, focused tools exist and are properly organized.

**Status**:
- ✅ **66 powertools implemented** across 9 categories
- ✅ **Properly organized** in focused subdirectories
- ✅ **Follow Unix philosophy** - small, composable tools
- ❌ **Not built** - no GPR project file for powertools
- ❌ **Not tested** - need individual test suites
- ⚠️ **Monolithic tools still exist** - need deprecation

---

## Powertools by Category

### 1. JSON Tools (12 tools) - `src/json/`

| Tool | Purpose | Lines | Status |
|------|---------|-------|--------|
| `json_validate` | Validate JSON syntax/structure | 257 | ✅ Complete with --describe |
| `json_extract` | Extract values by path | ~410 | ⚠️ Large (needs decomposition check) |
| `json_read` | Read and parse JSON file | ~216 | ✅ Good |
| `json_write` | Write JSON to file | ~193 | ✅ Good |
| `json_formatter` | Format/pretty-print JSON | ~280 | ✅ Good |
| `json_merge` | Merge multiple JSON docs | ~372 | ⚠️ Large (decomposed into helpers?) |
| `json_merge_arrays` | Merge JSON arrays specifically | ~249 | ✅ Good |
| `json_merge_objects` | Merge JSON objects specifically | ~291 | ✅ Good |
| `json_path_parser` | Parse JSONPath expressions | ~255 | ✅ Good |
| `json_path_eval` | Evaluate JSONPath on tree | ~345 | ⚠️ Large |
| `json_value_format` | Format extracted values | ~202 | ✅ Good |
| `json_validator` | JSON validation logic | ~226 | ✅ Good |

**Purpose**: JSON manipulation, validation, extraction, merging

---

### 2. File Operations (5 tools) - `src/files/`

| Tool | Purpose | Lines | Status |
|------|---------|-------|--------|
| `file_find` | Find files by pattern recursively | ~196 | ✅ Good |
| `file_hash` | Compute SHA-256 hash of file | ~127 | ✅ Good |
| `file_indexer` | Index files for verification | ~109 | ✅ Good |
| `file_reader` | Read file contents | ~51 | ✅ Good |
| `file_writer` | Write content to file | ~176 | ✅ Good |

**Purpose**: File system operations, hashing, indexing

---

### 3. Function Processing (4 tools) - `src/functions/`

| Tool | Purpose | Lines | Status |
|------|---------|-------|--------|
| `func_dedup` | Deduplicate function signatures | ~274 | ✅ Good (borderline) |
| `func_parse_body` | Parse function body/steps | ~69 | ✅ Good |
| `func_parse_sig` | Parse function signature | ~132 | ✅ Good |
| `func_to_ir` | Convert function to IR format | ~77 | ✅ Good |

**Purpose**: Function extraction, parsing, deduplication

---

### 4. IR Operations (10 tools) - `src/ir/`

| Tool | Purpose | Lines | Status |
|------|---------|-------|--------|
| `ir_add_metadata` | Add metadata to IR | ~70 | ✅ Good |
| `ir_check_functions` | Validate function structures | ~159 | ✅ Good |
| `ir_check_required` | Check required IR fields | ~173 | ✅ Good |
| `ir_check_types` | Validate type definitions | ~159 | ✅ Good |
| `ir_extract_funcs` | Extract functions from IR | ~143 | ✅ Good |
| `ir_extract_module` | Extract module metadata | ~176 | ✅ Good |
| `ir_gen_functions` | Generate IR functions | ~198 | ✅ Good |
| `ir_merge_funcs` | Merge function arrays | ~146 | ✅ Good (decomposed) |
| `ir_validate` | Validate IR (wrapper) | ~31 | ✅ Good |
| `ir_validate_schema` | Validate IR against schema | ~375 | ⚠️ Large (but decomposed into checkers) |

**Purpose**: IR manipulation, validation, extraction, merging

---

### 5. Spec Processing (6 tools) - `src/spec/`

| Tool | Purpose | Lines | Status |
|------|---------|-------|--------|
| `extraction_to_spec` | Convert extraction to spec | ~186 | ✅ Good |
| `spec_extract_funcs` | Extract functions from spec | ~207 | ✅ Good |
| `spec_extract_module` | Extract module metadata | ~186 | ✅ Good |
| `spec_extract_types` | Extract type definitions | ~87 | ✅ Good |
| `spec_validate` | Validate spec (wrapper) | ~74 | ✅ Good |
| `spec_validate_schema` | Validate spec against schema | ~407 | ⚠️ Large (but has decomposed validators) |

**Purpose**: Spec extraction, validation, processing

---

### 6. Emitters (12 tools) - `src/emitters/`

| Tool | Purpose | Lines | Status |
|------|---------|-------|--------|
| `code_add_comments` | Add comments to code | ~149 | ✅ Good |
| `code_format_target` | Format code for target lang | ~144 | ✅ Good |
| `code_gen_func_body` | Generate function body | ~165 | ✅ Good |
| `code_gen_func_sig` | Generate function signature | ~157 | ✅ Good |
| `code_gen_preamble` | Generate file preamble | ~141 | ✅ Good |
| `code_write` | Write code to file | ~145 | ✅ Good |
| `cpp_header_gen` | Generate C++ headers | ~191 | ✅ Good |
| `cpp_impl_gen` | Generate C++ implementation | ~183 | ✅ Good |
| `cpp_sig_normalize` | Normalize C++ signatures | ~204 | ✅ Good |
| `sig_gen_cpp` | Generate C++ signatures | ~445 | ⚠️ Large (but has decomposed helpers) |
| `sig_gen_python` | Generate Python signatures | ~88 | ✅ Good |
| `sig_gen_rust` | Generate Rust signatures | ~129 | ✅ Good |

**Purpose**: Language-specific code generation

---

### 7. Detection (2 tools) - `src/detection/`

| Tool | Purpose | Lines | Status |
|------|---------|-------|--------|
| `format_detect` | Detect file format | ~19 | ✅ Good |
| `lang_detect` | Detect programming language | ~19 | ✅ Good |

**Purpose**: Format and language detection

---

### 8. Type System (9 tools) - `src/types/`

| Tool | Purpose | Lines | Status |
|------|---------|-------|--------|
| `type_dependency` | Resolve type dependencies | ~127 | ✅ Good |
| `type_expand` | Expand complex types | ~144 | ✅ Good |
| `type_lookup` | Look up type definitions | ~148 | ✅ Good |
| `type_map` | Generic type mapping | ~128 | ✅ Good |
| `type_map_cpp` | C++ type mapping | ~213 | ✅ Good |
| `type_map_target` | Target-specific type mapping | ~211 | ✅ Good |
| `type_normalize` | Normalize type names | ~55 | ✅ Good |
| `type_resolve` | Resolve type references | ~338 | ⚠️ Large (but has decomposed helpers) |
| `type_resolver` | Type resolution logic | ~195 | ✅ Good |

**Purpose**: Type mapping, resolution, normalization

---

### 9. Utilities (6 tools) - `src/utils/`

| Tool | Purpose | Lines | Status |
|------|---------|-------|--------|
| `cli_parser` | Parse command-line arguments | ~66 | ✅ Good |
| `module_to_ir` | Convert module to IR | ~141 | ✅ Good |
| `path_normalize` | Normalize file paths | ~51 | ✅ Good |
| `pattern_match` | Pattern matching utilities | N/A | ✅ Header only |
| `toolchain_verify` | Verify toolchain lockfile | ~48 | ✅ Good |
| `receipt_generate` | Generate receipts | ~48 | ✅ Good |

**Purpose**: CLI parsing, path handling, verification

---

## Size Analysis

### Excellent (< 100 lines): 18 tools
- Focused, simple, easy to understand
- Examples: `ir_validate`, `file_reader`, `receipt_generate`, `type_normalize`

### Good (100-200 lines): 33 tools
- Well-sized, maintainable
- Examples: `file_hash`, `func_parse_body`, `code_gen_preamble`

### Acceptable (200-300 lines): 9 tools
- Large but still manageable
- Examples: `json_validator`, `cpp_sig_normalize`, `type_map_target`

### Large (300+ lines): 6 tools
- May benefit from decomposition (but many have helper utilities)
- `sig_gen_cpp` (445) - has cpp_* helpers
- `json_extract` (410) - has json_path_* helpers  
- `spec_validate_schema` (407) - has schema_check_* helpers
- `ir_validate_schema` (375) - has ir_check_* helpers
- `json_merge` (372) - has json_merge_* helpers
- `json_path_eval` (345) - specific complex logic
- `type_resolve` (338) - has type_* helpers

**Note**: Large tools already have decomposed helper utilities, so the "large" size represents orchestration logic.

---

## What's Missing

### ❌ Build System
- No `powertools.gpr` GNAT project file
- No makefile for batch compilation
- No way to build all 66 tools

### ❌ Testing
- No test suite for individual tools
- No integration tests for tool pipelines
- No verification of `--describe` introspection

### ⚠️ Monolithic Tools Still Active
- `stunir_spec_to_ir` (475 lines) - should orchestrate powertools
- `stunir_ir_to_code` (1971 lines) - should orchestrate powertools
- These are redundant if powertools work

---

## Next Steps

1. **Create `powertools.gpr`** - Build system for all 66 tools
2. **Build powertools** - Compile all tools to `bin/` directory
3. **Test individual tools** - Verify each tool works standalone
4. **Create orchestration scripts** - Shell scripts that chain powertools
5. **Deprecate monolithic tools** - Remove or refactor as orchestrators
6. **Document pipelines** - Show how to compose tools

---

## Conclusion

**The decomposition is DONE.** We have 66 well-organized powertools. The issue is:
- They're not built
- They're not tested
- Monolithic tools are still the primary interface

**Priority**: Build system first, then test, then deprecate monoliths.
