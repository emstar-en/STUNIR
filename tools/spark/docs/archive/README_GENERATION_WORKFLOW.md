# STUNIR Powertools: Complete Generation & Refactoring Workflow

**Status**: Ready for implementation  
**Phase**: Generation phase - all planning complete  
**Next Action**: Begin generating utilities using prompts

---

## 📋 Quick Start

You have everything needed to generate and refactor the STUNIR powertools:

1. **Start with utilities**: Open `UTILITY_GENERATION_PROMPTS.md`
2. **Generate utilities batch-by-batch**: Copy prompts to your local model
3. **Test each utility**: Compile and verify functionality
4. **Refactor oversized tools**: Use `REFACTORING_PLANS.md` after utilities are ready
5. **Track progress**: Use the completion checklists below

---

## 📚 Documentation Overview

### Planning Documents (Completed ✅)

| Document | Purpose | Lines | Status |
|----------|---------|-------|--------|
| `POWERTOOLS_SEQUENTIAL_GUIDE.md` | 27 prompts for core powertools | 1,200+ | ✅ Complete |
| `POWERTOOLS_DECOMPOSITION_ANALYSIS.md` | Analysis of 49 tools, identifies issues | 420+ | ✅ Complete |
| `POWERTOOLS_UTILITY_SPECS.md` | Specifications for 22 utilities | 580+ | ✅ Complete |
| `POWERTOOLS_ANALYSIS_SUMMARY.md` | Executive summary and next steps | 220+ | ✅ Complete |
| `UTILITY_GENERATION_PROMPTS.md` | 22 prompts for utility generation | 1,400+ | ✅ Complete |
| `REFACTORING_PLANS.md` | 8 refactoring plans for oversized tools | 1,000+ | ✅ Complete |

**Total Planning Documentation**: ~5,000 lines of comprehensive specifications

### Generation Documents (For Your Use)

1. **UTILITY_GENERATION_PROMPTS.md** - Copy-paste prompts for generating 22 utilities
2. **REFACTORING_PLANS.md** - Step-by-step plans for refactoring 8 oversized tools

---

## 🎯 Current Status

### What's Complete
- ✅ 49 powertools generated (27 core + 22 additional)
- ✅ Complete analysis of all generated tools
- ✅ Identification of 8 oversized tools requiring decomposition
- ✅ Identification of 15 stub files requiring implementation
- ✅ 22 utility specifications created
- ✅ 22 utility generation prompts created (ready-to-use)
- ✅ 8 refactoring plans created (step-by-step)
- ✅ Testing strategies defined
- ✅ Success criteria established

### What's Next
- 🔄 Generate 22 utility components (using UTILITY_GENERATION_PROMPTS.md)
- 🔄 Refactor 8 oversized tools (using REFACTORING_PLANS.md)
- 🔄 Implement 15 stub files (using POWERTOOLS_SEQUENTIAL_GUIDE.md)
- 🔄 Run comprehensive testing
- 🔄 Build complete spec-to-code pipeline

---

## 🔧 Generation Workflow

### Phase 1: Generate Utilities (22 tools)

**Recommended Order**: JSON → C++ → Validation → Type → File

#### Batch 1: JSON Utilities (Most Reusable)
1. [ ] json_path_parser (100 lines) - Parse dot-notation paths
2. [ ] json_path_eval (120 lines) - Navigate JSON using paths
3. [ ] json_value_format (50 lines) - Format extracted values
4. [ ] json_formatter (60 lines) - Format JSON with indentation
5. [ ] json_merge_objects (100 lines) - Merge JSON objects
6. [ ] json_merge_arrays (80 lines) - Merge JSON arrays
7. [ ] json_conflict_resolver (70 lines) - Resolve merge conflicts

**Batch Status**: 0/7 complete

#### Batch 2: C++ Generation Utilities
8. [ ] type_map_cpp (50 lines) - Map STUNIR types to C++
9. [ ] cpp_signature_gen (80 lines) - Generate function signatures
10. [ ] cpp_header_gen (60 lines) - Generate header structure
11. [ ] cpp_namespace_wrap (40 lines) - Wrap code in namespaces

**Batch Status**: 0/4 complete

#### Batch 3: Validation Utilities
12. [ ] schema_check_required (60 lines) - Check required fields
13. [ ] schema_check_types (80 lines) - Validate field types
14. [ ] schema_check_format (70 lines) - Validate formats/patterns
15. [ ] validation_reporter (50 lines) - Format validation reports
16. [ ] ir_check_required (60 lines) - Check required IR fields
17. [ ] ir_check_functions (90 lines) - Validate function structures
18. [ ] ir_check_types (70 lines) - Validate type definitions

**Batch Status**: 0/7 complete

#### Batch 4: Type Utilities
19. [ ] type_lookup (80 lines) - Look up type definitions
20. [ ] type_expand (100 lines) - Expand type aliases
21. [ ] type_dependency (70 lines) - Resolve dependency order

**Batch Status**: 0/3 complete

#### Batch 5: File Utilities
22. [ ] file_writer (50 lines) - Write files with error handling

**Batch Status**: 0/1 complete

**Overall Utilities Progress**: 0/22 (0%)

---

### Phase 2: Refactor Oversized Tools (8 tools)

**Prerequisites**: Generate required utilities first

#### Tools to Refactor

1. [ ] json_extract.adb (513 → 120 lines)
   - Requires: json_path_parser, json_path_eval, json_value_format
   
2. [ ] sig_gen_cpp.adb (406 → 100 lines) [DEPRECATED]
   - Requires: type_map_cpp, cpp_signature_gen, cpp_header_gen, cpp_namespace_wrap
   
3. [ ] spec_validate_schema.adb (365 → 100 lines)
   - Requires: schema_check_required, schema_check_types, schema_check_format, validation_reporter
   
4. [ ] ir_validate.adb (342 → 90 lines)
   - Requires: ir_check_required, ir_check_functions, ir_check_types, validation_reporter
   
5. [ ] json_merge_deep.adb (389 → 120 lines)
   - Requires: json_merge_objects, json_merge_arrays, json_conflict_resolver
   
6. [ ] type_resolver.adb (378 → 100 lines)
   - Requires: type_lookup, type_expand, type_dependency
   
7. [ ] spec_extract_module.adb (324 → 100 lines)
   - Requires: json_extract, json_formatter
   
8. [ ] ir_gen_functions.adb (311 → 120 lines)
   - Requires: json_extract, json_formatter, json_merge_objects

**Overall Refactoring Progress**: 0/8 (0%)

---

### Phase 3: Implement Stub Files (15 tools)

These tools were generated as stubs and need full implementation:

1. [ ] code_add_comments.adb
2. [ ] code_format_target.adb
3. [ ] emitters/code_gen_func_body.adb
4. [ ] emitters/code_gen_func_sig.adb
5. [ ] emitters/code_gen_preamble.adb
6. [ ] code_write.adb
7. [ ] func_parse_body.adb
8. [ ] func_parse_sig.adb
9. [ ] func_to_ir.adb
10. [ ] ir_add_metadata.adb
11. [ ] ir_extract_module.adb
12. [ ] ir_merge_funcs.adb
13. [ ] manifest_generate.adb
14. [ ] module_to_ir.adb
15. [ ] type_map_target.adb

**Implementation Progress**: 0/15 (0%)

---

## 🧪 Testing Strategy

### Per-Utility Testing

For each utility you generate:

```bash
# 1. Compile
cd tools/spark
gprbuild -P stunir_tools.gpr <utility_name>.adb

# 2. Test basic flags
./bin/<utility> --help
./bin/<utility> --version
./bin/<utility> --describe

# 3. Test main functionality (from prompt examples)
<test commands from generation prompt>

# 4. Test error cases
<invalid input tests>

# 5. Verify exit codes
echo $?
```

### Refactoring Testing

For each refactored tool:

```bash
# 1. Backup old version
mv <tool>.adb <tool>.adb.old

# 2. Compile new version
gprbuild -P stunir_tools.gpr <tool>_new.adb

# 3. Run parallel tests
echo "$TEST_INPUT" | ./bin/<tool>_old > old_output.txt
echo "$TEST_INPUT" | ./bin/<tool> > new_output.txt
diff old_output.txt new_output.txt

# 4. If tests pass: delete backup
# 5. If tests fail: restore and fix
```

---

## 📊 Progress Tracking

### Overall Progress

| Phase | Tasks | Complete | Status |
|-------|-------|----------|--------|
| Planning | 6 docs | 6/6 | ✅ Complete |
| Utilities | 22 tools | 0/22 | 🔄 Ready to start |
| Refactoring | 8 tools | 0/8 | ⏳ Waiting for utilities |
| Implementation | 15 stubs | 0/15 | ⏳ Lower priority |
| Testing | All tools | 0/45 | ⏳ Per-tool basis |

**Total Tools to Generate/Refactor**: 45  
**Current Completion**: 0/45 (0%)

---

## 🎯 Success Criteria

### For Each Utility
- ✅ Compiles without errors or warnings
- ✅ Size target met (<100 lines)
- ✅ All flags work (--help, --version, --describe)
- ✅ Main functionality works as specified
- ✅ Exit codes correct
- ✅ Handles error cases gracefully
- ✅ Output format matches specification

### For Each Refactored Tool
- ✅ Size reduced below 150 lines
- ✅ Uses appropriate utilities
- ✅ Maintains same functionality
- ✅ Passes all regression tests
- ✅ Output identical to old version (or improved)
- ✅ Compiles without errors/warnings
- ✅ Error handling preserved
- ✅ Exit codes match specification

### For Overall Project
- ✅ All 22 utilities generated and tested
- ✅ All 8 oversized tools refactored
- ✅ All tools follow Unix philosophy
- ✅ Complete spec-to-code pipeline working
- ✅ Documentation updated
- ✅ All changes committed to git

---

## 🚀 Getting Started Now

### Step 1: Generate Your First Utility

```bash
# Open the utility generation prompts
cd tools/spark
cat UTILITY_GENERATION_PROMPTS.md
```

### Step 2: Copy First Prompt

Find "Prompt 1: json_formatter.adb" in `UTILITY_GENERATION_PROMPTS.md`

### Step 3: Feed to Local Model

Copy the entire prompt (including all sections) and feed it to:
- Claude (via API or web interface)
- GPT-4 (via API or ChatGPT)
- Local LLM (Llama, Mistral, etc.)

### Step 4: Save and Test

```bash
# Save output
nano src/powertools/json_formatter.adb

# Compile
gprbuild -P stunir_tools.gpr json_formatter.adb

# Test
echo '{"a":1,"b":2}' | ./bin/json_formatter
```

### Step 5: Repeat for All Utilities

Work through each prompt in order, testing as you go.

---

## 📝 Files You Need

### For Utility Generation
- **UTILITY_GENERATION_PROMPTS.md** - All 22 prompts ready to copy-paste

### For Refactoring
- **REFACTORING_PLANS.md** - Step-by-step refactoring instructions

### For Implementation
- **POWERTOOLS_SEQUENTIAL_GUIDE.md** - Original 27 tool specifications

### For Reference
- **POWERTOOLS_DECOMPOSITION_ANALYSIS.md** - Detailed analysis
- **POWERTOOLS_UTILITY_SPECS.md** - Utility specifications
- **POWERTOOLS_ANALYSIS_SUMMARY.md** - Executive summary

---

## 🎓 Design Philosophy

All utilities follow these principles:

1. **Single Responsibility**: Each tool does ONE thing well
2. **Unix Philosophy**: stdin/stdout, composable, pipeable
3. **Size Limits**: <100 lines per utility, <150 for orchestrators
4. **Error Handling**: Proper exit codes, clear error messages
5. **Testability**: Easy to test, predictable behavior
6. **Documentation**: --help, --version, --describe for all tools
7. **Reusability**: Utilities can be used by multiple tools

---

## 📈 Metrics

### Documentation Created
- Total documents: 6
- Total lines: ~5,000
- Prompts ready: 22 utilities + 8 refactorings + 27 core tools = 57 total
- Time to generate (estimated): 2-3 days with local model

### Code to Generate
- New utility code: ~1,500 lines (22 utilities × ~70 avg)
- Refactored tool code: ~880 lines (8 tools × ~110 avg)
- Total new/refactored: ~2,400 lines

### Expected Outcome
- Properly sized tools: 24 → 46 (92% improvement)
- Oversized tools: 8 → 0 (100% reduction)
- Average tool size: ~185 lines → ~85 lines (54% reduction)
- Reusable utilities: 0 → 22 (∞ improvement)

---

## 🤝 Workflow Tips

1. **Batch Generation**: Generate all JSON utilities before moving to C++
2. **Test Frequently**: Test each utility immediately after generation
3. **Track Progress**: Check off items in this document as you complete them
4. **Fix as You Go**: Don't accumulate issues - fix compilation errors immediately
5. **Commit Often**: Commit each batch of utilities when complete
6. **Use Regression Tests**: Always compare refactored output with original

---

## 📞 Support

If you need clarification on any specification:
1. Check the detailed specs in `POWERTOOLS_UTILITY_SPECS.md`
2. Review examples in the generation prompts
3. Consult the decomposition analysis for context
4. Refer to existing working tools for patterns

---

## ✅ Final Checklist

Before considering the project complete:

- [ ] All 22 utilities generated
- [ ] All 22 utilities compile without warnings
- [ ] All 22 utilities tested and working
- [ ] All 8 oversized tools refactored
- [ ] All 8 refactored tools pass regression tests
- [ ] Old versions removed or archived
- [ ] Documentation updated
- [ ] All changes committed to git
- [ ] Complete pipeline tested end-to-end
- [ ] README updated with new tool list

---

**You have everything you need to start generating utilities now!**

Open `UTILITY_GENERATION_PROMPTS.md` and begin with Prompt 1: json_formatter.adb
