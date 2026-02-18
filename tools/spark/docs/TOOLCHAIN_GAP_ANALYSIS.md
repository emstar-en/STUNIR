# STUNIR SPARK Toolchain - Gap Analysis for Alpha Release

**Date**: 2025-02-17  
**Status**: Critical architectural decision required  
**Issue**: 73 tools in codebase, only 47 in build system, no formal toolchain defined

---

## Executive Summary

The STUNIR SPARK toolchain has grown organically to **73 tools**, but lacks formal organization:

- ✅ **All 73 tools compile successfully** (100% SPARK compliant)
- ⚠️ **Only 47 tools in build file** (26 tools orphaned)
- ⚠️ **No formal toolchain manifest** (unclear what's production vs experimental)
- ⚠️ **All tools in single "powertools" directory** (no categorical organization)
- 🚫 **Alpha release blocked** until toolchain is formalized

---

## Current State

### Build System Status

**File**: `powertools.gpr`  
**Tools in Build**: 47  
**Tools in Codebase**: 73  
**Missing from Build**: 26 tools

### Tools Currently in Build (47)

Organized by phase in `powertools.gpr`:

#### Phase 1: Foundation Tools (13 tools)
- json_validate, json_extract, json_merge, json_formatter
- json_path_parser, json_value_format, json_merge_objects, json_merge_arrays
- json_path_eval
- type_normalize, type_map, type_resolver
- func_dedup

#### Phase 2: Code Analysis (4 tools)
- format_detect, lang_detect
- extraction_to_spec, spec_extract_module

#### Phase 3: Code Generation (9 tools)
- sig_gen_cpp, sig_gen_rust, sig_gen_python
- type_resolve, type_map_cpp
- cpp_sig_normalize, cpp_header_gen, cpp_impl_gen
- ir_gen_functions

#### Phase 4: Verification & Attestation (7 tools)
- hash_compute, receipt_generate, toolchain_verify
- spec_validate, spec_validate_schema
- ir_validate, file_indexer

#### Phase 5: Validation Utilities (7 tools)
- schema_check_required, schema_check_types, schema_check_format
- validation_reporter
- ir_check_required, ir_check_functions, ir_check_types

#### Phase 6: Type Utilities (3 tools)
- type_lookup, type_expand, type_dependency

#### Phase 7: File Utilities (1 tool)
- file_writer

#### Additional Working Tools (2 tools)
- json_validator, type_map_target

---

## Tools Missing from Build (26)

These tools exist in `src/powertools/` but are **NOT** in `powertools.gpr`:

### CLI & Command Utilities
1. **cli_parser** - CLI argument parsing
2. **command_utils** - Command execution utilities
3. **path_normalize** - Path normalization

### Code Generation Tools
4. **code_add_comments** - Add comments to generated code
5. **code_format_target** - Format code for target language
6. **code_gen_func_body** - Generate function bodies
7. **code_gen_func_sig** - Generate function signatures
8. **code_gen_preamble** - Generate file preambles
9. **code_write** - Write generated code to files

### Function Analysis Tools
10. **func_parse_body** - Parse function bodies
11. **func_parse_sig** - Parse function signatures
12. **func_to_ir** - Convert functions to IR

### IR Pipeline Tools
13. **ir_add_metadata** - Add metadata to IR
14. **ir_extract_funcs** - Extract functions from IR
15. **ir_extract_module** - Extract module from IR
16. **ir_merge_funcs** - Merge function definitions
17. **ir_validate_schema** - Validate IR schema
18. **module_to_ir** - Convert module to IR

### File Operations
19. **file_find** - Find files by pattern
20. **file_hash** - Hash file contents
21. **file_reader** - Read files with error handling

### Spec Tools
22. **spec_extract_funcs** - Extract functions from spec
23. **spec_extract_types** - Extract types from spec
24. **manifest_generate** - Generate manifest files

### JSON Operations  
25. **json_read** - Read JSON files
26. **json_write** - Write JSON files

### Core Parser
- **stunir_json_parser** - Core JSON parser (referenced by others, may be library not tool)

---

## Architectural Issues

### Issue 1: No Categorical Organization

**Current**: All 73 tools in `src/powertools/`  

**Problem**: Flat directory makes it hard to:
- Understand tool purpose
- Navigate codebase
- Define API boundaries
- Manage dependencies

**Proposed Structure** (based on existing directories in `src/`):

```
src/
├── core/           # Core types, utilities, JSON parser
├── parsing/        # Language parsers, extractors
├── analysis/       # Code analysis, function parsing
├── semantic_ir/    # IR manipulation and validation
├── emitters/       # Code generators for target languages
├── validation/     # Schema and data validation
├── utils/          # CLI, file, path utilities
└── powertools/     # Composable CLI tools (public API)
```

### Issue 2: No Toolchain Manifest

**Problem**: No clear definition of:
- Which tools are production-ready
- Which tools are experimental
- Which tools are core vs optional
- Tool dependencies and ordering
- API contracts and compatibility

**Solution**: Need `TOOLCHAIN_MANIFEST.json`

### Issue 3: Unclear Alpha Release Scope

**Questions**:
1. Which tools are **essential** for alpha?
2. Which tools are **nice-to-have**?
3. Which tools are **experimental**?
4. What are the **core workflows** alpha must support?

---

## Proposed Toolchain Tiers

### Tier 1: Core Toolchain (Alpha Release - Essential)

**Purpose**: Minimal viable toolchain for C → Spec → Code generation

**Tools** (15-20):
- **Phase 0**: file_indexer, hash_compute, lang_detect
- **Phase 1**: format_detect, extraction_to_spec, spec_validate, func_dedup
- **Phase 2**: type_normalize, type_map
- **Phase 3**: sig_gen_cpp, sig_gen_rust, sig_gen_python
- **Phase 4**: toolchain_verify, receipt_generate
- **Utilities**: json_validate, json_extract, json_merge

### Tier 2: Extended Toolchain (Post-Alpha)

**Purpose**: Advanced features, optimizations, additional languages

**Tools** (20-30):
- All IR manipulation tools
- Code generation utilities (comments, formatting)
- Advanced type resolution
- Schema validation suite
- Additional language generators

### Tier 3: Experimental (Development)

**Purpose**: Research, prototypes, future features

**Tools** (20-30):
- Experimental parsers
- Advanced analysis tools
- Optimization passes

---

## Recommended Actions for Alpha Release

### 1. Define Minimal Core Toolchain (URGENT)

**Decision needed**: Which 15-20 tools are absolutely required for alpha?

**Criteria**:
- Required for primary workflow (C → Spec → Multi-language bindings)
- Stable and tested
- Documented with examples
- Zero compilation errors

### 2. Create Toolchain Manifest (HIGH PRIORITY)

**File**: `TOOLCHAIN_MANIFEST.json`

**Contents**:
```json
{
  "version": "0.1.0-alpha",
  "schema_version": "1.0",
  "release_tier": "alpha",
  "core_tools": [...],
  "optional_tools": [...],
  "experimental_tools": [...],
  "workflows": {
    "c_to_bindings": {
      "tools": ["file_indexer", "extraction_to_spec", "sig_gen_cpp"],
      "description": "Extract C functions and generate bindings"
    }
  }
}
```

### 3. Reorganize Directory Structure (MEDIUM PRIORITY)

**Options**:

**Option A**: Keep powertools flat, but add categories to build system
- ✅ Minimal disruption
- ✅ Fast to implement
- ❌ Doesn't solve navigation issues

**Option B**: Redistribute tools to category directories
- ✅ Clear organization
- ✅ Enforces API boundaries
- ❌ More work, may break existing workflows

**Option C**: Hybrid - Move libraries to categories, keep tools in powertools
- ✅ Balanced approach
- ✅ Tools remain discoverable
- ✅ Libraries properly organized

**Recommendation**: **Option C** for alpha

### 4. Update Build System (HIGH PRIORITY)

**Tasks**:
- Add missing 26 tools to `powertools.gpr` (if production-ready)
- Or explicitly mark as experimental/excluded
- Create separate .gpr for core vs extended toolchain
- Add build targets: `make core`, `make extended`, `make all`

### 5. Document Toolchain (HIGH PRIORITY)

**Files needed**:
- `TOOLCHAIN_MANIFEST.json` - Formal tool listing
- `TOOLCHAIN_GUIDE.md` - How to use the toolchain
- `TOOL_REFERENCE.md` - Reference for each tool
- Update `ARCHITECTURE.tools.json` with actual state

---

## Questions for Decision

1. **What is the primary use case for alpha release?**
   - C library → Multi-language bindings?
   - Spec validation and IR manipulation?
   - Full pipeline including verification?

2. **What tools are absolutely required?**
   - Which workflows must work in alpha?
   - What can be deferred to beta?

3. **How should tools be organized?**
   - Keep flat structure?
   - Reorganize by category?
   - Hybrid approach?

4. **What's the release timeline?**
   - Does alpha need all 73 tools?
   - Or can it ship with core 15-20?

5. **What about the 26 missing tools?**
   - Add all to build?
   - Mark as experimental?
   - Defer to post-alpha?

---

## Next Steps

**Immediate** (blocking alpha):
1. ✅ Complete this gap analysis
2. ⏳ **Get user input on alpha scope and priorities**
3. ⏳ Create TOOLCHAIN_MANIFEST.json
4. ⏳ Update powertools.gpr with decision on 26 missing tools
5. ⏳ Document core workflow with examples

**Short-term** (alpha release):
1. Test core tools with real-world examples
2. Create integration test suite
3. Write user guide for alpha toolchain
4. Package and distribute

**Long-term** (post-alpha):
1. Reorganize directory structure
2. Add remaining tools incrementally
3. Expand documentation
4. Formal verification with gnatprove

---

## Summary Statistics

| Metric | Count |
|--------|-------|
| Total Tools in Codebase | 73 |
| Tools in Build File | 47 |
| Tools Missing from Build | 26 |
| Tools Compiling Successfully | 73 (100%) |
| SPARK Compliance | 100% |
| Documented in Architecture | ~20 |
| Ready for Alpha | ❓ (Decision needed) |

---

**Status**: ⚠️ **BLOCKED** - Awaiting architectural decisions before alpha release

**Critical Path**: Define core toolchain → Create manifest → Update build → Test workflows → Release
