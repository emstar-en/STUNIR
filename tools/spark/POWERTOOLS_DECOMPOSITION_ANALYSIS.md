# STUNIR Powertools: Decomposition Analysis & Strategy

**Date**: 2026-02-17  
**Purpose**: Analyze generated powertools and create decomposition strategy for oversized tools  
**Target**: Keep all tools under 200 lines following Unix philosophy

---

## Executive Summary

Generated 49 powertools from sequential guide. Analysis reveals:
- ✓ **27 core tools identified** in sequential guide
- ✓ **22 additional utilities** created during decomposition
- ⚠️ **8 tools exceed 300 lines** - require decomposition
- ⚠️ **15 tools are empty stubs** - need implementation
- ✓ **24 tools are properly sized** (< 250 lines)

---

## Tool Size Analysis (PowerShell results)

### Critical: Tools > 300 lines (MUST DECOMPOSE)
| Tool | Lines | Status | Decomposition Priority |
|------|-------|--------|------------------------|
| sig_gen_cpp | 445 | ❌ Too complex | HIGH - Type mapping + signature gen |
| json_extract | 410 | ❌ Too complex | HIGH - Path parsing + extraction |
| spec_validate_schema | 407 | ❌ Too complex | HIGH - Schema validation rules |
| json_merge | 372 | ❌ Too complex | MEDIUM - Merge logic + conflict resolution |
| ir_validate_schema | 375 | ❌ Too complex | MEDIUM - IR validation rules |
| type_resolve | 338 | ❌ Too complex | MEDIUM - Type resolution logic |
| hash_compute | 318 | ⚠️ Borderline | LOW - Already focused |
| func_dedup | 274 | ⚠️ Borderline | LOW - Already focused |

### Warning: Tools 200-300 lines (REVIEW)
| Tool | Lines | Status | Action |
|------|-------|--------|--------|
| json_validate | 256 | ⚠️ Large | Review - may be acceptable |
| json_validator | 226 | ⚠️ Large | Review - consolidate with json_validate? |
| spec_extract_funcs | 207 | ⚠️ Large | Review - extraction logic |
| file_find | 196 | ⚠️ Large | Review - already has decomposed utilities |
| json_read | 216 | ⚠️ Large | Refine using decomposed utilities |
| json_write | 193 | ⚠️ Large | Refine using decomposed utilities |
| manifest_generate | 182 | ✓ Acceptable | Keep as is |

### Good: Tools < 200 lines (KEEP AS IS)
| Tool | Lines | Status |
|------|-------|--------|
| ir_merge_funcs | 146 | ✓ Good |
| module_to_ir | 141 | ✓ Good |
| ir_extract_module | 131 | ✓ Good |
| sig_gen_rust | 129 | ✓ Good |
| type_map | 128 | ✓ Good |
| file_hash | 127 | ✓ Good |
| code_write | 123 | ✓ Good |
| file_indexer | 109 | ✓ Good |
| type_map_target | 107 | ✓ Good |
| sig_gen_python | 88 | ✓ Good |
| spec_validate | 74 | ✓ Good |
| json_validator | 72 | ✓ Good |
| func_parse_body | 69 | ✓ Good |
| type_normalize | 55 | ✓ Good |
| spec_extract_types | 51 | ✓ Good |
| receipt_generate | 48 | ✓ Good |
| code_gen_func_body | 48 | ✓ Good |
| file_reader | 51 | ✓ Good |
| cli_parser | 66 | ✓ Good |
| ir_validate | 31 | ✓ Good |
| lang_detect | 19 | ✓ Good |

### Empty: Stub files (NEED IMPLEMENTATION)
| Tool | Lines | Action |
|------|-------|--------|
| json_write | 0 | Implement using file_writer + json_formatter utilities |
| path_normalize | 0 | Implement path normalization logic |
| toolchain_verify | 1 | Implement lockfile verification |
| extraction_to_spec | 1 | Implement extraction logic |
| file_hash | 1 | Already has implementation (127 lines reported separately?) |
| code_gen_preamble | 1 | Implement preamble generation |
| code_add_comments | 1 | Implement comment addition |
| cli_parser | 1 | Already has implementation (66 lines reported separately?) |
| code_format_target | 1 | Implement formatter invocation |
| code_gen_func_sig | 1 | Implement signature generation |
| ir_extract_funcs | 1 | Implement IR function extraction |
| ir_add_metadata | 1 | Implement metadata addition |
| format_detect | 1 | Implement format detection |
| func_to_ir | 1 | Implement function→IR conversion |
| file_reader | 1 | Already has implementation (51 lines reported separately?) |

---

## Decomposition Strategy

### Phase 1: Critical Tools (>400 lines) - IMMEDIATE

#### 1. **sig_gen_cpp** (445 lines) → 4 utilities
Decompose into:
- `type_map_cpp.adb` (50 lines) - C++ type mapping only
- `cpp_signature_gen.adb` (80 lines) - Generate C++ function signatures
- `cpp_header_gen.adb` (60 lines) - Generate header guards, includes
- `cpp_namespace_wrap.adb` (40 lines) - Wrap code in namespace

**New sig_gen_cpp** (150 lines):
- CLI parsing
- Call type_map_cpp for type conversions
- Call cpp_signature_gen for function signatures
- Call cpp_header_gen for header structure
- Call cpp_namespace_wrap if --namespace flag
- Output result

#### 2. **json_extract** (410 lines) → 3 utilities
Decompose into:
- `json_path_parser.adb` (100 lines) - Parse dot-notation paths
- `json_path_eval.adb` (120 lines) - Evaluate path on JSON tree
- `json_value_format.adb` (50 lines) - Format extracted value (raw/quoted)

**New json_extract** (140 lines):
- CLI parsing
- Read JSON input
- Call json_path_parser to parse path
- Call json_path_eval to extract value
- Call json_value_format to format output
- Output result

#### 3. **spec_validate_schema** (407 lines) → 4 utilities
Decompose into:
- `schema_check_required.adb` (60 lines) - Check required fields exist
- `schema_check_types.adb` (80 lines) - Validate field types
- `schema_check_format.adb` (70 lines) - Validate format rules
- `validation_reporter.adb` (50 lines) - Format validation reports

**New spec_validate_schema** (150 lines):
- CLI parsing
- Read spec JSON
- Call schema_check_required
- Call schema_check_types
- Call schema_check_format
- Call validation_reporter to generate report
- Output result with proper exit code

### Phase 2: Large Tools (300-400 lines) - HIGH PRIORITY

#### 4. **json_merge** (372 lines) → 3 utilities
Decompose into:
- `json_merge_objects.adb` (100 lines) - Merge JSON objects
- `json_merge_arrays.adb` (80 lines) - Merge JSON arrays
- `json_conflict_resolver.adb` (70 lines) - Resolve merge conflicts

**New json_merge** (120 lines):
- CLI parsing with merge strategy options
- Read multiple JSON inputs
- Call appropriate merge utility based on type
- Call conflict resolver if needed
- Output merged result

#### 5. **ir_validate_schema** (375 lines) → 4 utilities
Decompose into:
- `ir_check_required.adb` (60 lines) - Check required IR fields
- `ir_check_functions.adb` (90 lines) - Validate function structures
- `ir_check_types.adb` (70 lines) - Validate type definitions
- `validation_reporter.adb` (50 lines) - Reuse from spec validation

**New ir_validate_schema** (150 lines):
- CLI parsing
- Read IR JSON
- Call ir_check_required
- Call ir_check_functions
- Call ir_check_types
- Call validation_reporter
- Output result

#### 6. **type_resolve** (338 lines) → 3 utilities
Decompose into:
- `type_lookup.adb` (80 lines) - Look up type definitions
- `type_expand.adb` (100 lines) - Expand complex types
- `type_dependency.adb` (70 lines) - Resolve type dependencies

**New type_resolve** (90 lines):
- CLI parsing
- Read type definitions
- Call type_lookup
- Call type_expand
- Call type_dependency
- Output resolved types

### Phase 3: Borderline Tools (250-300 lines) - REVIEW

#### 7. **hash_compute** (318 lines)
**Decision**: Keep as is - focused responsibility (hashing only)
- Streaming logic is essential
- Progress reporting is valuable
- Already well-structured

#### 8. **func_dedup** (274 lines)
**Decision**: Keep as is - focused responsibility (deduplication only)
- Comparison logic is complex but necessary
- Signature matching is core feature
- Already well-structured

### Phase 4: Medium Tools (200-250 lines) - REFINE

#### 9. **json_validate** (256 lines) + **json_validator** (226 lines)
**Action**: Consolidate into single tool
- Remove duplication
- json_validate should be the main tool
- json_validator appears to be older/duplicate implementation
- **New json_validate** (180 lines) after consolidation

#### 10. **json_read** (216 lines) + **json_write** (193 lines)
**Action**: Refine using decomposed utilities
- json_read: Use cli_parser + file_reader + json_validate
- json_write: Use cli_parser + json_validate + file_writer

**New json_read** (120 lines):
```ada
-- Use cli_parser for argument handling
-- Use file_reader for file input
-- Use json_validate for validation
-- Output validated JSON
```

**New json_write** (100 lines):
```ada
-- Use cli_parser for argument handling
-- Use json_validate for validation
-- Use json_formatter for pretty-printing
-- Use file_writer for output
```

---

## New Utilities Needed

### JSON Utilities
1. `json_formatter.adb` (60 lines) - Pretty-print JSON with indentation
2. `json_path_parser.adb` (100 lines) - Parse JSON path expressions
3. `json_path_eval.adb` (120 lines) - Evaluate paths on JSON
4. `json_value_format.adb` (50 lines) - Format extracted values
5. `json_merge_objects.adb` (100 lines) - Merge JSON objects
6. `json_merge_arrays.adb` (80 lines) - Merge JSON arrays
7. `json_conflict_resolver.adb` (70 lines) - Resolve merge conflicts

### C++ Generation Utilities
8. `type_map_cpp.adb` (50 lines) - STUNIR→C++ type mapping
9. `cpp_signature_gen.adb` (80 lines) - Generate C++ signatures
10. `cpp_header_gen.adb` (60 lines) - Generate C++ headers
11. `cpp_namespace_wrap.adb` (40 lines) - Namespace wrapper

### Validation Utilities
12. `schema_check_required.adb` (60 lines) - Check required fields
13. `schema_check_types.adb` (80 lines) - Validate types
14. `schema_check_format.adb` (70 lines) - Validate formats
15. `validation_reporter.adb` (50 lines) - Format validation reports
16. `ir_check_required.adb` (60 lines) - Check required IR fields
17. `ir_check_functions.adb` (90 lines) - Validate IR functions
18. `ir_check_types.adb` (70 lines) - Validate IR types

### Type Utilities
19. `type_lookup.adb` (80 lines) - Look up type definitions
20. `type_expand.adb` (100 lines) - Expand complex types
21. `type_dependency.adb` (70 lines) - Resolve dependencies

### File Utilities
22. `file_writer.adb` (50 lines) - Write content to file with error handling

---

## Implementation Priority

### **IMMEDIATE** (This week)
1. Create decomposition utilities for sig_gen_cpp
2. Create decomposition utilities for json_extract
3. Create decomposition utilities for spec_validate_schema
4. Refactor these 3 tools to use new utilities
5. Test and verify functionality preserved

### **HIGH** (Next week)
6. Create decomposition utilities for json_merge
7. Create decomposition utilities for ir_validate_schema
8. Create decomposition utilities for type_resolve
9. Refactor these 3 tools
10. Consolidate json_validate + json_validator
11. Refine json_read and json_write using utilities

### **MEDIUM** (Following week)
12. Review and refine spec_extract_funcs (207 lines)
13. Review and refine file_find (196 lines) - may already be good with decomp
14. Implement empty stub files
15. Full integration testing

### **LOW** (Ongoing)
16. Documentation updates
17. Performance optimization
18. Additional test coverage

---

## Success Criteria

✓ **No tool exceeds 200 lines** (except hash_compute, func_dedup with justification)  
✓ **All tools follow Unix philosophy** (single responsibility)  
✓ **Utilities are reusable** across multiple tools  
✓ **All tools compile** without errors  
✓ **All tools pass tests** with expected behavior  
✓ **Pipeline integration** works end-to-end  

---

## Tool Responsibility Matrix

### Core JSON Tools
| Tool | Responsibility | Uses Utilities |
|------|----------------|----------------|
| json_read | Read & validate JSON from file/stdin | cli_parser, file_reader, json_validate |
| json_write | Write JSON to file with formatting | cli_parser, json_formatter, file_writer |
| json_validate | Validate JSON structure & syntax | STUNIR_JSON_Parser |
| json_extract | Extract values by path | json_path_parser, json_path_eval, json_value_format |
| json_merge | Merge multiple JSON inputs | json_merge_objects, json_merge_arrays, json_conflict_resolver |

### Spec Processing Tools
| Tool | Responsibility | Uses Utilities |
|------|----------------|----------------|
| spec_extract_funcs | Extract functions array | json_extract |
| spec_extract_types | Extract types array | json_extract |
| spec_extract_module | Extract module metadata | json_extract |
| spec_validate_schema | Validate spec schema | schema_check_*, validation_reporter |
| type_normalize | Normalize type names | (standalone) |

### IR Tools
| Tool | Responsibility | Uses Utilities |
|------|----------------|----------------|
| func_to_ir | Convert function spec→IR | type_normalize |
| module_to_ir | Convert module metadata→IR | (standalone) |
| ir_merge_funcs | Merge IR function arrays | json_merge_arrays |
| ir_add_metadata | Add metadata to IR | (standalone) |
| ir_validate_schema | Validate IR schema | ir_check_*, validation_reporter |
| ir_extract_funcs | Extract functions from IR | json_extract |
| ir_extract_module | Extract module from IR | json_extract |

### Code Generation Tools
| Tool | Responsibility | Uses Utilities |
|------|----------------|----------------|
| type_map_target | Map types to target language | type_map_cpp, type_map_rust, etc. |
| code_gen_preamble | Generate language preamble | (standalone) |
| code_gen_func_sig | Generate function signature | type_map_target, cpp_signature_gen |
| code_gen_func_body | Generate function body | (standalone) |
| sig_gen_cpp | Generate C++ signatures | type_map_cpp, cpp_signature_gen, cpp_header_gen |
| sig_gen_rust | Generate Rust signatures | (standalone) |
| code_add_comments | Add metadata comments | (standalone) |
| code_format_target | Format with external formatter | (standalone) |
| code_write | Write code to file | file_writer |

---

## Next Steps

1. ✅ Review this decomposition strategy
2. ⬜ Create utility specifications for new decomposed tools
3. ⬜ Generate utility implementations
4. ⬜ Refactor oversized tools to use utilities
5. ⬜ Test all refactored tools
6. ⬜ Update POWERTOOLS_REFINE_TODO.md with new utilities
7. ⬜ Commit decomposed and refined tools

---

**End of Decomposition Analysis**
