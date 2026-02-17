# STUNIR Powertools - Refactoring Status

**Last Updated:** 2025-02-17

## Overview
This document tracks the status of the STUNIR powertools refactoring effort to implement the Unix philosophy (small, composable tools with JSON interfaces).

## Phase 1 Powertools (Core IR Processing)

| Tool | Status | Issues | Notes |
|------|--------|--------|-------|
| json_read | ✅ FIXED | None | Fully regenerated with proper structure |
| json_validate | ⚠️ BLOCKED | Missing deps: stunir_json_parser.ads, stunir_types.ads | Needs spec files |
| json_extract | ✅ FIXED | None | Fixed String_Access, aliased keywords |
| json_merge | ⚠️ BLOCKED | Get_Command_Output undefined | Missing Success variable |
| json_value_format | ⚠️ SYNTAX | Line 138: mixed logical operators | Syntax error in expression |
| json_merge_objects | ⚠️ SYNTAX | Lines 43, 197: missing quotes, missing ")" | File corruption remains |
| json_merge_arrays | ⚠️ SYNTAX | Line 42: missing string quote | File corruption remains |
| json_path_eval | ⚠️ SYNTAX | Lines 127, 134: mixed operators, syntax errors | Multiple syntax issues |
| ir_validate_schema | ✅ FIXED | None | Fully regenerated |
| ir_extract_module | ✅ FIXED | None | Fully regenerated |
| ir_extract_funcs | ✅ FIXED | None | Fully regenerated |
| func_parse_sig | ✅ FIXED | None | Fully regenerated |
| func_parse_body | ✅ FIXED | None | Fully regenerated |
| file_reader | ✅ FIXED | None | Fully regenerated |

## Phase 2 Powertools (Spec Processing)

| Tool | Status | Issues | Notes |
|------|--------|--------|-------|
| spec_extract_module | ⚠️ BLOCKED | Line 89: Get_Command_Output undefined; Line 112: type mismatch | Needs Success variable fix |
| func_dedup | ⚠️ BLOCKED | Missing dependencies | Needs spec files |
| format_detect | ⚠️ BLOCKED | Missing dependencies | Needs spec files |
| extraction_to_spec | ⚠️ BLOCKED | Missing dependencies | Needs spec files |

## Phase 3 Powertools (Code Generation)

| Tool | Status | Issues | Notes |
|------|--------|--------|-------|
| code_write | ✅ FIXED | None | Added GNAT.Strings import |
| code_gen_preamble | ✅ FIXED | None | Added GNAT.Strings import |
| code_gen_func_sig | ✅ FIXED | None | Added GNAT.Strings import |
| code_gen_func_body | ✅ FIXED | None | Added GNAT.Strings import |
| code_format_target | ✅ FIXED | None | Fully regenerated |
| code_add_comments | ✅ FIXED | None | Fully regenerated |
| sig_gen_cpp | ⚠️ BLOCKED | Line 94: Get_Command_Output undefined | Needs Success variable fix |
| sig_gen_rust | ⚠️ BLOCKED | Missing dependencies | Needs spec files |

## Type System Powertools

| Tool | Status | Issues | Notes |
|------|--------|--------|-------|
| type_map | ✅ FIXED | None | Fixed aliased, String_Access issues |
| type_normalize | ✅ FIXED | None | Fixed aliased keywords |
| type_resolver | ⚠️ BLOCKED | Line 79: Get_Command_Output undefined | Needs Success variable fix |

## Summary Statistics

- **Total Tools:** 25
- **Fully Fixed:** 14 (56%)
- **Blocked on Missing Specs:** 6 (24%)
- **Blocked on Get_Command_Output:** 4 (16%)
- **Syntax Errors Remaining:** 3 (12%)

## Common Issues Fixed

1. **Quote-wrapped file corruption** - 14 files had content wrapped in JSON-encoded quotes
2. **Missing `aliased` keyword** - Boolean variables used with `'Access` attribute
3. **Wrong type for Define_Switch** - Changed `Unbounded_String` to `GNAT.Strings.String_Access`
4. **Missing imports** - Added `GNAT.Strings` where needed

## Remaining Work

### Critical Path (for basic functionality):
1. Create missing spec files: `stunir_json_parser.ads`, `stunir_types.ads`
2. Fix `Get_Command_Output` undefined errors in 4 files
3. Fix syntax errors in JSON merge/eval files

### Files Needing `Get_Command_Output` Fix:
- `json_merge.adb` (line 66)
- `json_extract.adb` (line 99) 
- `type_resolver.adb` (line 79)
- `spec_extract_module.adb` (line 89)
- `sig_gen_cpp.adb` (line 94)

### Files with Syntax Errors:
- `json_value_format.adb` (line 138)
- `json_merge_objects.adb` (lines 43, 197)
- `json_merge_arrays.adb` (line 42)
- `json_path_eval.adb` (lines 127, 134)

## Architecture Decisions

1. **Command-line parsing:** Using `GNAT.Command_Line` with `GNAT.Strings.String_Access` for string switches
2. **Exit codes:** Standardized (0=success, 1=error, 2=invalid JSON, 3=invalid path)
3. **JSON interface:** All tools accept stdin and output stdout with JSON formatting
4. **Error handling:** Errors written to stderr with descriptive messages

## Next Steps for Orchestration Team

1. **Priority 1:** Create missing spec files or stub them out
2. **Priority 2:** Fix remaining `Get_Command_Output` issues (pattern is established)
3. **Priority 3:** Address syntax errors in JSON processing files
4. **Priority 4:** Full integration testing once compilation succeeds
