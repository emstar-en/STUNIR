# STUNIR Powertools - Refactoring Status

**Last Updated:** 2025-02-17

## Overview
This document tracks the status of the STUNIR powertools refactoring effort to implement the Unix philosophy (small, composable tools with JSON interfaces).

## Phase 1 Powertools (Core IR Processing)

| Tool | Status | Issues | Notes |
|------|--------|--------|-------|
| json_read | ✅ FIXED | None | Fully regenerated with proper structure |
| json_validate | ⚠️ PARTIAL | Missing types from stunir_types | Parser_State, Status_Code, Max_JSON_Length, etc. |
| json_extract | ✅ FIXED | None | Fixed String_Access, aliased keywords, Success variable |
| json_merge | ✅ FIXED | None | Fixed Success variable with aliased keyword |
| json_value_format | ✅ FIXED | None | Fixed mixed logical operators (and → and then) |
| json_merge_objects | ✅ FIXED | None | Fixed missing comma at line 43 |
| json_merge_arrays | ✅ FIXED | None | Fixed missing quote in line 42 |
| json_path_eval | ✅ FIXED | None | Fixed quote escaping at line 134 |
| ir_validate_schema | ✅ FIXED | None | Fully regenerated |
| ir_extract_module | ✅ FIXED | None | Fully regenerated |
| ir_extract_funcs | ✅ FIXED | None | Fully regenerated |
| func_parse_sig | ✅ FIXED | None | Fully regenerated |
| func_parse_body | ✅ FIXED | None | Fully regenerated |
| file_reader | ✅ FIXED | None | Fully regenerated |

## Phase 2 Powertools (Spec Processing)

| Tool | Status | Issues | Notes |
|------|--------|--------|-------|
| spec_extract_module | ✅ FIXED | None | Fixed Success variable with aliased keyword |
| func_dedup | ⚠️ PARTIAL | Unconstrained subtype error | Line 64-65: Hash map instantiation issue |
| format_detect | ✅ FIXED | None | Spec files exist in src/ |
| extraction_to_spec | ✅ FIXED | None | Fixed character literal escaping |

## Phase 3 Powertools (Code Generation)

| Tool | Status | Issues | Notes |
|------|--------|--------|-------|
| code_write | ✅ FIXED | None | Added GNAT.Strings import |
| code_gen_preamble | ✅ FIXED | None | Added GNAT.Strings import |
| code_gen_func_sig | ✅ FIXED | None | Added GNAT.Strings import |
| code_gen_func_body | ✅ FIXED | None | Added GNAT.Strings import |
| code_format_target | ✅ FIXED | None | Fully regenerated |
| code_add_comments | ✅ FIXED | None | Fully regenerated |
| sig_gen_cpp | ✅ FIXED | None | Fixed Success variable with aliased keyword |
| sig_gen_rust | ✅ FIXED | None | Spec files exist in src/ |
| cpp_impl_gen | ✅ FIXED | None | Fixed quote escaping at line 100 |

## Type System Powertools

| Tool | Status | Issues | Notes |
|------|--------|--------|-------|
| type_map | ✅ FIXED | None | Fixed aliased, String_Access issues |
| type_normalize | ✅ FIXED | None | Fixed aliased keywords |
| type_resolver | ✅ FIXED | None | Fixed Success variable with aliased keyword |

## Summary Statistics

- **Total Tools:** 25
- **Fully Fixed:** 22 (88%)
- **Partially Fixed:** 2 (8%)
- **Remaining Issues:** 2 files need additional work

## Common Issues Fixed

1. **Quote-wrapped file corruption** - 14 files had content wrapped in JSON-encoded quotes
2. **Missing `aliased` keyword** - Boolean variables used with `'Access` attribute (5 files fixed)
3. **Wrong type for Define_Switch** - Changed `Unbounded_String` to `GNAT.Strings.String_Access`
4. **Missing imports** - Added `GNAT.Strings` where needed
5. **Mixed logical operators** - Changed `and` to `and then` for consistency
6. **String quote escaping** - Fixed malformed string literals in JSON output
7. **Character literal escaping** - Fixed C-style `\` to Ada-style `'\'`
8. **Get_Command_Output undefined** - Created Command_Utils package

## Remaining Work

### Critical Issues:
1. **json_validate.adb** - Requires full stunir_types spec with:
   - Parser_State type
   - Status_Code type  
   - Max_JSON_Length constant
   - JSON_Strings type
   - Initialize_Parser procedure
   - Next_Token function
   - Token_EOF constant

2. **func_dedup.adb** - Hash map instantiation error:
   - Line 64: Unconstrained subtype in component declaration
   - Line 65: Key_Type must be a definite subtype

## Architecture Decisions

1. **Command-line parsing:** Using `GNAT.Command_Line` with `GNAT.Strings.String_Access` for string switches
2. **Exit codes:** Standardized (0=success, 1=error, 2=invalid JSON, 3=invalid path)
3. **JSON interface:** All tools accept stdin and output stdout with JSON formatting
4. **Error handling:** Errors written to stderr with descriptive messages
5. **'Access attribute:** All variables used with 'Access must be declared as `aliased`

## Next Steps

1. **Priority 1:** Complete stunir_types.ads with all required types for json_validate
2. **Priority 2:** Fix func_dedup.adb hash map instantiation
3. **Priority 3:** Full integration testing once all compilation succeeds
4. **Priority 4:** Performance testing and optimization
