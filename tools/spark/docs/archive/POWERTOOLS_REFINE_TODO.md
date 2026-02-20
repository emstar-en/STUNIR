# STUNIR Powertools Refinement TODO List

**Purpose**: Track refinement progress for all 27 powertools after initial generation and their supporting utilities

## Refinement Process Overview
1. **Review**: Examine generated code against requirements
2. **Test**: Verify functionality with test cases
3. **Fix**: Address compilation errors and edge cases
4. **Optimize**: Improve performance or structure where needed
5. **Document**: Add comments and improve --describe output

## Phase 1: Core JSON & File Tools (7 tools)

### Tool 0a: cli_parser - Command Line Argument Parser
- [x] Review CLI parsing logic (Code Review Only)
- [ ] Test with various argument combinations - Requires compilation

### Tool 0b: file_reader - File Reading Utility
- [x] Review file reading implementation (Code Review Only)
- [ ] Test with large files (>8KB) - Requires compilation

### Tool 1: json_read - Read and Validate JSON
- [x] Review JSON validation logic (Code Review Only)
- [ ] Test with malformed JSON inputs - Requires compilation
- [ ] Verify exit codes (0=success, 1=validation error) - Requires compilation
- [ ] Check --help, --version, --describe outputs - Requires compilation

### Tool 2: json_write - Write JSON to File
- [x] Review pretty-printing logic (Code Review Only)
- [ ] Test pretty-printing functionality - Requires compilation
- [ ] Verify file write error handling - Requires compilation
- [ ] Check directory creation for output paths - Requires compilation

### Tool 2b: json_validate - Validate JSON Structure
- [x] Review validation logic (Code Review Only)
- [ ] Test with malformed JSON inputs - Requires compilation

### Tool 3: file_find - Find Files by Pattern (DEPRECATED)
- [x] Marked for deprecation
- [ ] Update to use decomposed components

### Tool 3a: path_normalize - Normalize File Paths
- [ ] Review path normalization logic
- [ ] Test cross-platform compatibility
- [ ] Verify symlink resolution

### Tool 3b: pattern_match - Evaluate Patterns
- [ ] Review pattern matching algorithm
- [ ] Test complex patterns (nested wildcards)
- [ ] Verify pattern validation

### Tool 3c: dir_walker - Traverse Directories
- [ ] Review directory traversal logic
- [ ] Test recursive behavior
- [ ] Verify permission handling

### Tool 3d: file_filter - Filter Files
- [ ] Review filtering criteria implementation
- [ ] Test size-based and time-based filters
- [ ] Verify type filtering

### Tool 3e: result_formatter - Format Results
- [ ] Review output formatting logic
- [ ] Test JSON and CLI formats
- [ ] Verify sorting options

### Tool 4: file_hash - Compute SHA-256 Hash
- [x] Review progress reporting logic
- [x] Verify file size calculation
- [ ] Test with large files (>8KB) - Requires compilation
- [ ] Verify streaming implementation - Requires compilation
- [ ] Confirm hash output format (lowercase hex) - Requires compilation

### Tool 4b: file_utils - File Utility Functions
- [x] Review file operations logic (Code Review Only)
- [ ] Test with various file paths - Requires compilation

### Tool 5: toolchain_verify - Verify Toolchain Lockfile
- [ ] Review required field validation logic
- [ ] Verify error message structure and consistency
- [ ] Test with missing/invalid lockfiles - Requires compilation
- [ ] Full integration testing - Requires compilation

### Tool 6: manifest_generate - Generate File Manifest
- [ ] Test with multiple file inputs
- [ ] Verify hash computation for each file
- [ ] Check JSON array output format

## Phase 2: Spec Processing Tools (5 tools)

### Tool 7: spec_extract_funcs - Extract Functions from Spec
- [ ] Review function extraction logic (Code Review Only)
- [ ] Test with actual complex spec structures - Requires compilation
- [ ] Verify empty array handling (Code Review Only)

### Tool 8: spec_extract_types - Extract Types from Spec
- [ ] Review type extraction logic (Code Review Only)
- [ ] Test actual type extraction - Requires compilation

### Tool 9: spec_extract_module - Extract Module Metadata
- [ ] Review metadata field extraction logic (Code Review Only)
- [ ] Test actual metadata extraction - Requires compilation

### Tool 10: spec_validate_schema - Validate Spec Schema
- [ ] Review schema validation logic (Code Review Only)
- [ ] Verify actual error message structure - Requires compilation

### Tool 11: type_normalize - Normalize Type Names
- [ ] Review case-insensitive matching logic (Code Review Only)
- [ ] Verify all normalization rules - Requires compilation

## Phase 3: IR Generation Tools (4 tools)

### Tool 12: func_to_ir - Convert Function Spec to IR
- [ ] Review IR format compliance logic (Code Review Only)
- [ ] Verify type normalization consistency - Requires compilation

### Tool 13: module_to_ir - Convert Module Spec to IR
- [ ] Review IR header format logic (Code Review Only)
- [ ] Verify field presence - Requires compilation

### Tool 14: ir_merge_funcs - Merge IR Function Arrays
- [ ] Review line-delimited vs concatenated JSON handling (Code Review Only)
- [ ] Verify array output format - Requires compilation

### Tool 15: ir_add_metadata - Add IR Metadata
- [ ] Review timestamp format handling logic (Code Review Only)
- [ ] Verify default values - Requires compilation

## Phase 4: IR Processing Tools (5 tools)

### Tool 16: ir_validate_schema - Validate IR Schema
- [ ] Review IR schema validation logic (Code Review Only)
- [ ] Verify actual error message structure - Requires compilation

### Tool 17: ir_extract_module - Extract Module from IR
- [ ] Review IR module extraction logic (Code Review Only)

### Tool 18: ir_extract_funcs - Extract Functions from IR
- [ ] Review IR function array extraction logic (Code Review Only)

### Tool 19: func_parse_sig - Parse Function Signature
- [ ] Review function signature parsing logic (Code Review Only)

### Tool 20: func_parse_body - Parse Function Body
- [ ] Review function body steps parsing logic (Code Review Only)

## Phase 5: Code Generation Tools (7 tools)

### Tool 21: type_map_target - Map Type to Target Language
- [ ] Review target language mapping logic (Code Review Only)
- [ ] Verify error handling for unsupported targets - Requires compilation

### Tool 22: code_gen_preamble - Generate Code Preamble
- [ ] Review template-based generation logic (Code Review Only)
- [ ] Verify all target language preambles - Requires compilation

### Tool 23: code_gen_func_sig - Generate Function Signature
- [ ] Review signature generation logic (Code Review Only)
- [ ] Verify type mapping consistency - Requires compilation

### Tool 24: code_gen_func_body - Generate Function Body
- [ ] Review IR operation handling logic (Code Review Only)
- [ ] Verify stub generation - Requires compilation
- [ ] Test actual indentation behavior - Requires compilation

### Tool 25: code_add_comments - Add Comments to Code
- [ ] Review header comment format logic (Code Review Only)
- [ ] Verify actual metadata extraction - Requires compilation

### Tool 26: code_format_target - Format Code
- [ ] Review formatter availability logic (Code Review Only)
- [ ] Verify graceful fallback behavior - Requires compilation

### Tool 27: code_write - Write Code to File
- [ ] Review directory creation logic (Code Review Only)
- [ ] Verify actual write error handling - Requires compilation

## Refinement Workflow
1. **Select a tool** from the TODO list
2. **Review generated code** against requirements
3. **Identify issues** (compilation errors, logical flaws)
4. **Implement fixes** using edit tools
5. **Test thoroughly** with edge cases
6. **Update TODO** with [x] when complete
7. **Move to next tool**

## Testing Strategy
- Use provided test commands from POWERTOOLS_SEQUENTIAL_GUIDE.md
- Create additional edge case tests for each tool
- Verify integration between tools in pipeline

## Notes
- Focus on Ada SPARK compliance and correctness first
- Then optimize for performance and code quality
- Document any recurring issues or patterns

