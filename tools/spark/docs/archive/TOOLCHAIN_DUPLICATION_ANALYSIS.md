# STUNIR SPARK Toolchain - Duplication Analysis

**Date**: 2025-02-17  
**Purpose**: Identify duplicate functionality to ensure orthogonal tool design  
**Principle**: Each tool must have a single, distinct, non-overlapping function

---

## Duplication Categories

### 1. JSON Validation - DUPLICATES FOUND

**Tools**:
- `json_validate.adb` - Full JSON validator using STUNIR_JSON_Parser
- `json_validator.adb` - Minimal JSON validator

**Analysis**:
- ❌ **DUPLICATE**: Both validate JSON syntax
- ✅ **KEEP**: `json_validate` (more complete, uses STUNIR_JSON_Parser, proper exit codes)
- 🗑️ **REMOVE**: `json_validator` (minimal, redundant)

**Justification**: json_validate is the proper implementation with full STUNIR integration.

---

### 2. File Hashing - DUPLICATES FOUND

**Tools**:
- `file_hash.adb` - Basic file hash computation with SHA256
- `hash_compute.adb` - Full hash tool with algorithm selection, verification, JSON output

**Analysis**:
- ❌ **DUPLICATE**: Both compute SHA256 hashes
- ✅ **KEEP**: `hash_compute` (feature-complete: --algorithm, --verify, --json, stdin support)
- 🗑️ **REMOVE**: `file_hash` (basic, subset of hash_compute)

**Justification**: hash_compute is the complete tool with all necessary features.

---

### 3. Type Mapping - POTENTIAL DUPLICATES

**Tools**:
- `type_map.adb` - Generic type mapper (C → C++/Python/Rust/Go)
- `type_map_target.adb` - STUNIR internal types → target language types
- `type_map_cpp.adb` - Specific C++ type mapper

**Analysis**:
- ⚠️ **OVERLAP**: Multiple tools handle type mapping
- ✅ **KEEP**: `type_map` (generic, multi-language)
- ❓ **EVALUATE**: `type_map_target` - May have distinct function for internal types
- 🗑️ **REMOVE**: `type_map_cpp` (redundant - type_map handles C++ with --lang=cpp)

**Justification**: type_map is the general solution. type_map_cpp duplicates C++ functionality.

---

### 4. Type Resolution - DUPLICATES FOUND

**Tools**:
- `type_resolve.adb` - Simple type alias resolver
- `type_resolver.adb` - Orchestrator calling type_lookup, type_expand, type_dependency

**Analysis**:
- ❌ **DUPLICATE**: Both resolve type references
- ✅ **KEEP**: `type_resolver` (complete orchestrator with proper architecture)
- 🗑️ **REMOVE**: `type_resolve` (simpler, subset of type_resolver)

**Justification**: type_resolver follows proper decomposition with orchestration.

---

### 5. Spec Validation - POTENTIAL DUPLICATES

**Tools**:
- `spec_validate.adb` - Stub version for spec validation
- `spec_validate_schema.adb` - Orchestrator for schema validation (calls schema_check_*)

**Analysis**:
- ⚠️ **DIFFERENT**: spec_validate is a stub (line 1: "Simplified stub version")
- ✅ **KEEP**: `spec_validate_schema` (proper implementation, orchestrates schema_check_*)
- 🗑️ **REMOVE or IMPLEMENT**: `spec_validate` (currently stub, should be removed or completed)

**Justification**: Only one spec validator needed; spec_validate_schema is the real implementation.

---

### 6. JSON I/O - NEED INVESTIGATION

**Tools**:
- `json_read.adb` - Read and validate JSON
- `json_write.adb` - Write JSON (empty/stub file - 3 blank lines!)

**Analysis**:
- ❓ **json_read**: Duplicates json_validate? Need to check if it adds distinct I/O functionality
- 🗑️ **REMOVE**: `json_write` (EMPTY FILE - literally 3 blank lines!)

**Action**: Check if json_read provides distinct functionality beyond json_validate.

---

### 7. IR Validation - POTENTIAL DUPLICATES

**Tools**:
- `ir_validate.adb`
- `ir_validate_schema.adb`

**Analysis**: Need to check for duplication similar to spec_validate pattern.

---

## Summary of Duplicates Found

### Confirmed Duplicates to Remove (5 tools):

1. ❌ `json_validator.adb` - Replaced by json_validate
2. ❌ `file_hash.adb` - Replaced by hash_compute
3. ❌ `type_map_cpp.adb` - Replaced by type_map --lang=cpp
4. ❌ `type_resolve.adb` - Replaced by type_resolver
5. ❌ `json_write.adb` - EMPTY FILE (3 blank lines)

### Potentially Duplicate (Need Investigation):

6. ❓ `json_read.adb` - May duplicate json_validate
7. ❓ `spec_validate.adb` - Stub version, may be replaced by spec_validate_schema
8. ❓ `type_map_target.adb` - May have distinct internal type mapping function
9. ❓ `ir_validate.adb` vs `ir_validate_schema.adb`

---

## Tools Still to Analyze for Duplication

### CLI & Command (3 tools)
- cli_parser
- command_utils  
- path_normalize

### Code Generation (6 tools)
- code_add_comments
- code_format_target
- code_gen_func_body
- code_gen_func_sig
- code_gen_preamble
- code_write

### Function Analysis (3 tools)
- func_parse_body
- func_parse_sig
- func_to_ir

### IR Pipeline (6 tools)
- ir_add_metadata
- ir_extract_funcs
- ir_extract_module
- ir_merge_funcs
- module_to_ir
- func_to_ir (overlap with function analysis?)

### File Operations (3 tools)
- file_find
- file_reader
- file_writer

### Spec Tools (2 tools)
- spec_extract_funcs
- spec_extract_types

### Other (2 tools)
- manifest_generate
- receipt_generate

### Signature Generators (3 tools)
- sig_gen_cpp
- sig_gen_rust
- sig_gen_python

### CPP Tools (3 tools)
- cpp_header_gen
- cpp_impl_gen
- cpp_sig_normalize

---

## Next Steps

1. ✅ Identified 5 confirmed duplicates
2. ⏳ Investigate 4 potential duplicates
3. ⏳ Analyze remaining 26 tools for duplication
4. ⏳ Create final list of orthogonal tools
5. ⏳ Define removal/consolidation plan

---

## Orthogonality Principles

1. **One Function Per Tool**: Each tool does exactly one thing
2. **No Overlap**: If tool A and tool B both do X, one must be removed
3. **Composition Over Duplication**: Build complex tools by orchestrating simple ones
4. **Clear Boundaries**: Each tool has a well-defined input/output contract

---

**Status**: In Progress - 5 duplicates identified, ~40 tools remaining to analyze
