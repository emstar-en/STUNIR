# STUNIR Powertools: Refactoring Plans

**Purpose**: Step-by-step refactoring plans for oversized powertools  
**Usage**: Copy each plan and feed to local model for refactoring  
**Target**: Break 300+ line tools into <150 line components

---

## Refactoring Strategy

Each oversized tool will be refactored using this approach:

1. **Generate new utilities first** (from UTILITY_GENERATION_PROMPTS.md)
2. **Create new refactored version** of oversized tool
3. **Test new version** alongside old version
4. **Replace old version** once tests pass
5. **Update documentation** and dependencies

---

## Tool 1: json_extract.adb (513 lines → ~120 lines)

### Current Issues
- 513 lines (4.3x over target)
- Combines path parsing, JSON navigation, value extraction, and formatting
- Multiple responsibilities in single file

### Refactoring Plan

**Step 1: Generate Required Utilities**
Generate these utilities first (see UTILITY_GENERATION_PROMPTS.md):
- json_path_parser (100 lines) - Parse dot-notation paths
- json_path_eval (120 lines) - Navigate JSON using parsed paths
- json_value_format (50 lines) - Format extracted values

**Step 2: Create Refactored json_extract.adb**

```
Generate a refactored Ada SPARK 2014 utility named `json_extract` that orchestrates JSON path extraction.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success, 1=path not found, 2=invalid JSON, 3=invalid path
- Target size: 120 lines maximum
- Uses: json_path_parser, json_path_eval, json_value_format

Behavior:
- Read JSON from stdin
- Parse path using json_path_parser subprocess
- Evaluate path using json_path_eval subprocess
- Format result using json_value_format subprocess
- Output formatted value to stdout
- Chain utilities via pipes

Architecture:
```
stdin (JSON) → json_extract
                   ↓
              [parse path] → json_path_parser
                   ↓
              [eval path] → json_path_eval
                   ↓
              [format value] → json_value_format
                   ↓
                stdout
```

Options:
- --help, -h        : Show usage
- --version, -v     : Show version 0.1.0-alpha
- --describe        : Output JSON metadata
- --path PATH       : JSON path to extract (required)
- --default VALUE   : Default if path not found
- --raw             : Output raw value (no quotes)
- --format FORMAT   : Output format (json/text/raw)

Exit Codes:
- 0: Success - value extracted
- 1: Path not found (no default)
- 2: Invalid JSON input
- 3: Invalid path syntax

Implementation Notes:
- Main logic: orchestrate subprocess calls
- Use Ada.Processes or system calls to invoke utilities
- Handle subprocess exit codes
- Pass data via pipes (stdin/stdout)
- Minimal JSON parsing in main tool
- Error handling for subprocess failures

Subprocess Invocation Examples:
```ada
-- Parse path
Path_Components := Run_Command("json_path_parser --path " & Path);

-- Evaluate path
Value := Run_Command("json_path_eval --path '" & Path_Components & "'", JSON_Input);

-- Format value
Result := Run_Command("json_value_format --raw", Value);
```

Testing:
```bash
echo '{"a":{"b":{"c":123}}}' | json_extract --path "a.b.c"
# Output: 123

echo '{"users":[{"name":"Alice"},{"name":"Bob"}]}' | json_extract --path "users.1.name"
# Output: "Bob"

echo '{"x":1}' | json_extract --path "y" --default "null"
# Output: null
```

Follow Unix philosophy: orchestrate specialized tools
Keep main tool focused on coordination, not implementation
Target 120 lines (subprocess management + error handling)
```

**Step 3: Testing Plan**

```bash
# Test basic extraction
echo '{"a":1}' | ./bin/json_extract --path "a"

# Test nested paths
echo '{"a":{"b":{"c":123}}}' | ./bin/json_extract --path "a.b.c"

# Test array access
echo '{"arr":[1,2,3]}' | ./bin/json_extract --path "arr.1"

# Test default values
echo '{"a":1}' | ./bin/json_extract --path "b" --default "null"

# Test raw format
echo '{"name":"test"}' | ./bin/json_extract --path "name" --raw

# Compare with old version
diff <(echo '{"a":1}' | ./bin/json_extract_old --path "a") \
     <(echo '{"a":1}' | ./bin/json_extract --path "a")
```

**Step 4: Replacement Steps**

1. Backup old version: `mv json_extract.adb json_extract.adb.old`
2. Install new version: `mv json_extract_new.adb json_extract.adb`
3. Recompile: `gprbuild -P stunir_tools.gpr json_extract.adb`
4. Run regression tests
5. If tests pass: delete old version
6. If tests fail: restore old version, fix issues

---

## Tool 2: sig_gen_cpp.adb (406 lines → ~100 lines) [DEPRECATED]

### Current Issues
- 406 lines (3.4x over target)
- Combines type mapping, signature generation, header formatting, namespace wrapping
- C++ code generation logic mixed with orchestration

### Refactoring Plan

**Step 1: Generate Required Utilities**
Generate these utilities first:
- type_map_cpp (50 lines) - Map STUNIR types to C++
- cpp_signature_gen (80 lines) - Generate function signatures
- cpp_header_gen (60 lines) - Generate header structure
- cpp_namespace_wrap (40 lines) - Wrap code in namespaces

**Step 2: Create Refactored sig_gen_cpp.adb** [DEPRECATED]

```
Generate a refactored Ada SPARK 2014 utility named `sig_gen_cpp` that orchestrates C++ signature file generation. [DEPRECATED]

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success, 1=invalid IR, 2=generation error
- Target size: 100 lines maximum
- Uses: type_map_cpp, cpp_signature_gen, cpp_header_gen, cpp_namespace_wrap

Behavior:
- Read STUNIR IR JSON from stdin
- Extract module name and functions
- Generate C++ header with function signatures
- Output to specified file or stdout

Architecture:
```
stdin (IR JSON) → sig_gen_cpp
                      ↓
                 [parse IR]
                      ↓
                 [generate header] → cpp_header_gen
                      ↓
                 [for each function]
                      ↓
                 [map types] → type_map_cpp
                      ↓
                 [generate sig] → cpp_signature_gen
                      ↓
                 [wrap namespace] → cpp_namespace_wrap
                      ↓
                   stdout
```

Options:
- --help, -h        : Show usage
- --version, -v     : Show version 0.1.0-alpha
- --describe        : Output JSON metadata
- --output FILE     : Output file (default: stdout)
- --namespace NS    : Wrap in namespace

Exit Codes:
- 0: Success - signatures generated
- 1: Invalid IR format
- 2: Generation error

Implementation Notes:
- Main logic: orchestrate utility invocations
- Extract module_name and functions from IR
- For each function:
  - Map argument types via type_map_cpp
  - Map return type via type_map_cpp
  - Generate signature via cpp_signature_gen
- Assemble header via cpp_header_gen
- Wrap in namespace via cpp_namespace_wrap
- Write to file or stdout

Subprocess Invocation Pattern:
```ada
-- Generate header structure
Header := Run_Command("cpp_header_gen --module " & Module_Name);

-- For each function
for Func of Functions loop
   -- Map return type
   CPP_Return := Run_Command("type_map_cpp --type " & Func.Return_Type);
   
   -- Map argument types
   for Arg of Func.Args loop
      CPP_Type := Run_Command("type_map_cpp --type " & Arg.Type);
      Arg.CPP_Type := CPP_Type;
   end loop;
   
   -- Generate signature
   Sig_JSON := Build_Signature_JSON(Func);
   Signature := Run_Command("cpp_signature_gen", Sig_JSON);
end loop;

-- Wrap in namespace
Result := Run_Command("cpp_namespace_wrap --namespace " & Namespace, All_Signatures);
```

Testing:
```bash
echo '{"module_name":"test","functions":[{"name":"add","return_type":"i32","args":[{"name":"a","type":"i32"}]}]}' | sig_gen_cpp
# Output: C++ header with signatures

echo '{"module_name":"math","functions":[...]}' | sig_gen_cpp --output math.hpp --namespace math
# Generates math.hpp with namespace math
```

Follow Unix philosophy: orchestrate specialized tools
Keep main tool focused on coordination
Target 100 lines (IR parsing + orchestration)
```

**Step 3: Testing Plan**

```bash
# Test basic signature generation
cat test_ir.json | ./bin/sig_gen_cpp

# Test namespace wrapping
cat test_ir.json | ./bin/sig_gen_cpp --namespace mylib

# Test file output
cat test_ir.json | ./bin/sig_gen_cpp --output test.hpp

# Test type mapping
cat ir_with_various_types.json | ./bin/sig_gen_cpp

# Compare with old version
diff <(cat test_ir.json | ./bin/sig_gen_cpp_old) \
     <(cat test_ir.json | ./bin/sig_gen_cpp)
```

---

## Tool 3: spec_validate_schema.adb (365 lines → ~100 lines)

### Current Issues
- 365 lines (3x over target)
- Combines required field checking, type validation, format validation, reporting
- All validation logic in single file

### Refactoring Plan

**Step 1: Generate Required Utilities**
Generate these utilities first:
- schema_check_required (60 lines) - Check required fields
- schema_check_types (80 lines) - Validate field types
- schema_check_format (70 lines) - Validate formats/patterns
- validation_reporter (50 lines) - Format validation reports

**Step 2: Create Refactored spec_validate_schema.adb**

```
Generate a refactored Ada SPARK 2014 utility named `spec_validate_schema` that orchestrates schema validation.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=valid, 1=validation error
- Target size: 100 lines maximum
- Uses: schema_check_required, schema_check_types, schema_check_format, validation_reporter

Behavior:
- Read spec JSON from stdin
- Run all validation checks
- Collect validation results
- Generate formatted report
- Output report to stdout

Architecture:
```
stdin (Spec JSON) → spec_validate_schema
                         ↓
                    [check required] → schema_check_required
                         ↓
                    [check types] → schema_check_types
                         ↓
                    [check formats] → schema_check_format
                         ↓
                    [collect results]
                         ↓
                    [format report] → validation_reporter
                         ↓
                      stdout
```

Options:
- --help, -h        : Show usage
- --version, -v     : Show version 0.1.0-alpha
- --describe        : Output JSON metadata
- --schema FILE     : Schema definition file
- --format FORMAT   : Report format (text/json)
- --verbose         : Show all checks (not just errors)

Exit Codes:
- 0: Valid - all checks passed
- 1: Validation error

Implementation Notes:
- Run validation checks in sequence
- Collect results in JSON array format
- Pass results to validation_reporter
- Each check returns JSON result object
- Main tool orchestrates and aggregates

Validation Flow:
```ada
-- Initialize results
Results := Empty_Array;

-- Check required fields
Required_Result := Run_Command("schema_check_required --fields '" & Required_Fields & "'", Spec_JSON);
Results.Append(Required_Result);

-- Check types
Type_Result := Run_Command("schema_check_types --schema '" & Type_Schema & "'", Spec_JSON);
Results.Append(Type_Result);

-- Check formats
Format_Result := Run_Command("schema_check_format --rules '" & Format_Rules & "'", Spec_JSON);
Results.Append(Format_Result);

-- Generate report
Report := Run_Command("validation_reporter --format text", Results);
Output(Report);
```

Testing:
```bash
# Test valid spec
cat valid_spec.json | spec_validate_schema --schema schema.json
# Output: VALID

# Test invalid spec
cat invalid_spec.json | spec_validate_schema --schema schema.json
# Output: INVALID: ...errors...

# Test verbose mode
cat spec.json | spec_validate_schema --schema schema.json --verbose
# Shows all checks

# Test JSON output
cat spec.json | spec_validate_schema --schema schema.json --format json
# Output: {"status":"valid",...}
```

Follow Unix philosophy: orchestrate validation checks
Keep main tool focused on coordination
Target 100 lines
```

---

## Tool 4: ir_validate.adb (342 lines → ~90 lines)

### Current Issues
- 342 lines (2.9x over target)
- Combines IR schema validation, function validation, type validation
- All IR validation logic in single file

### Refactoring Plan

**Step 1: Generate Required Utilities**
Generate these utilities first:
- ir_check_required (60 lines) - Check required IR fields
- ir_check_functions (90 lines) - Validate function structures
- ir_check_types (70 lines) - Validate type definitions
- validation_reporter (50 lines) - Format validation reports

**Step 2: Create Refactored ir_validate.adb**

```
Generate a refactored Ada SPARK 2014 utility named `ir_validate` that orchestrates IR validation.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=valid, 1=validation error
- Target size: 90 lines maximum
- Uses: ir_check_required, ir_check_functions, ir_check_types, validation_reporter

Behavior:
- Read IR JSON from stdin
- Run all IR validation checks
- Collect results
- Generate report
- Output report to stdout

Architecture:
```
stdin (IR JSON) → ir_validate
                      ↓
                 [check IR schema] → ir_check_required
                      ↓
                 [check functions] → ir_check_functions
                      ↓
                 [check types] → ir_check_types
                      ↓
                 [collect results]
                      ↓
                 [format report] → validation_reporter
                      ↓
                   stdout
```

Options:
- --help, -h        : Show usage
- --version, -v     : Show version 0.1.0-alpha
- --describe        : Output JSON metadata
- --format FORMAT   : Report format (text/json)
- --verbose         : Show all checks

Exit Codes:
- 0: Valid IR
- 1: Validation error

Implementation Notes:
- Similar to spec_validate_schema
- IR-specific validation checks
- Orchestrate utility calls
- Aggregate results
- Format report

Testing:
```bash
# Test valid IR
cat valid_ir.json | ir_validate
# Output: VALID

# Test invalid IR
cat invalid_ir.json | ir_validate
# Output: INVALID: ...errors...
```

Follow Unix philosophy: orchestrate validation checks
Target 90 lines
```

---

## Tool 5: json_merge_deep.adb (389 lines → ~120 lines)

### Current Issues
- 389 lines (3.2x over target)
- Combines object merging, array merging, conflict resolution, deep recursion
- Complex merge logic all in one file

### Refactoring Plan

**Step 1: Generate Required Utilities**
Generate these utilities first:
- json_merge_objects (100 lines) - Merge objects at single level
- json_merge_arrays (80 lines) - Merge arrays
- json_conflict_resolver (70 lines) - Resolve conflicts

**Step 2: Create Refactored json_merge_deep.adb**

```
Generate a refactored Ada SPARK 2014 utility named `json_merge_deep` that orchestrates deep JSON merging.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success, 1=merge conflict, 2=invalid input
- Target size: 120 lines maximum
- Uses: json_merge_objects, json_merge_arrays, json_conflict_resolver

Behavior:
- Read multiple JSON documents from stdin
- Recursively merge using appropriate utilities
- Handle nested structures
- Output merged result

Architecture:
```
stdin (JSON docs) → json_merge_deep
                         ↓
                    [identify type]
                         ↓
                    object? → json_merge_objects
                    array?  → json_merge_arrays
                    conflict? → json_conflict_resolver
                         ↓
                    [recurse on nested]
                         ↓
                      stdout
```

Options:
- --help, -h        : Show usage
- --version, -v     : Show version 0.1.0-alpha
- --describe        : Output JSON metadata
- --strategy STRAT  : Conflict strategy (last/first/error)
- --recursive       : Deep merge nested objects

Exit Codes:
- 0: Success
- 1: Merge conflict
- 2: Invalid input

Implementation Notes:
- Orchestrate shallow merge utilities
- Add recursion logic for nested structures
- Handle conflicts via json_conflict_resolver
- Main tool provides deep merge coordination

Testing:
```bash
echo '{"a":{"x":1},"b":2}
{"a":{"y":2},"c":3}' | json_merge_deep --recursive
# Output: {"a":{"x":1,"y":2},"b":2,"c":3}
```

Target 120 lines
```

---

## Tool 6: type_resolver.adb (378 lines → ~100 lines)

### Current Issues
- 378 lines (3.2x over target)
- Combines type lookup, expansion, dependency resolution
- All type resolution logic in one file

### Refactoring Plan

**Step 1: Generate Required Utilities**
Generate these utilities first:
- type_lookup (80 lines) - Look up type definitions
- type_expand (100 lines) - Expand type aliases
- type_dependency (70 lines) - Resolve dependency order

**Step 2: Create Refactored type_resolver.adb**

```
Generate a refactored Ada SPARK 2014 utility named `type_resolver` that orchestrates type resolution.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success, 1=resolution error
- Target size: 100 lines maximum
- Uses: type_lookup, type_expand, type_dependency

Behavior:
- Read type reference from stdin
- Look up definition
- Expand aliases
- Resolve dependencies
- Output resolved type

Architecture:
```
stdin (type ref) → type_resolver
                       ↓
                  [lookup] → type_lookup
                       ↓
                  [expand] → type_expand
                       ↓
                  [deps] → type_dependency
                       ↓
                    stdout
```

Options:
- --help, -h        : Show usage
- --version, -v     : Show version 0.1.0-alpha
- --describe        : Output JSON metadata
- --registry FILE   : Type registry
- --recursive       : Recursively resolve

Exit Codes:
- 0: Success
- 1: Resolution error

Testing:
```bash
echo "MyType" | type_resolver --registry types.json
# Output: resolved type definition
```

Target 100 lines
```

---

## Tool 7: spec_extract_module.adb (324 lines → ~100 lines)

### Current Issues
- 324 lines (2.7x over target)
- Combines JSON extraction, path navigation, module-specific extraction
- Can use json_extract utilities

### Refactoring Plan

**Step 1: Use Existing Utilities**
- json_extract (after refactoring)
- json_path_parser
- json_path_eval
- json_formatter

**Step 2: Create Refactored spec_extract_module.adb**

```
Generate a refactored Ada SPARK 2014 utility named `spec_extract_module` that extracts module information from spec.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success, 1=extraction error
- Target size: 100 lines maximum
- Uses: json_extract, json_formatter

Behavior:
- Read spec JSON from stdin
- Extract module fields
- Format output
- Output module info

Implementation Notes:
- Use json_extract for field extraction
- Module-specific orchestration
- Format via json_formatter

Testing:
```bash
cat spec.json | spec_extract_module
# Output: module info
```

Target 100 lines
```

---

## Tool 8: ir_gen_functions.adb (311 lines → ~120 lines)

### Current Issues
- 311 lines (2.6x over target)
- Combines function extraction, IR generation, formatting
- Can use extraction utilities

### Refactoring Plan

**Step 1: Use Existing Utilities**
- json_extract
- json_formatter
- json_merge_objects

**Step 2: Create Refactored ir_gen_functions.adb**

```
Generate a refactored Ada SPARK 2014 utility named `ir_gen_functions` that generates IR function representations.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success, 1=generation error
- Target size: 120 lines maximum
- Uses: json_extract, json_formatter, json_merge_objects

Behavior:
- Read spec JSON from stdin
- Extract function definitions
- Generate IR function structures
- Output IR functions array

Implementation Notes:
- Use json_extract to get functions
- Generate IR structure
- Format via json_formatter
- Orchestration logic in main tool

Testing:
```bash
cat spec.json | ir_gen_functions
# Output: IR functions array
```

Target 120 lines
```

---

## Refactoring Workflow

### For Each Oversized Tool:

**Phase 1: Preparation**
1. Generate all required utilities (see UTILITY_GENERATION_PROMPTS.md)
2. Test each utility independently
3. Ensure utilities compile and work correctly

**Phase 2: Refactoring**
1. Copy refactoring plan prompt for the tool
2. Feed to local model
3. Review generated refactored code
4. Save as `<tool>_new.adb`

**Phase 3: Testing**
1. Compile new version: `gprbuild -P stunir_tools.gpr <tool>_new.adb`
2. Run tests from refactoring plan
3. Compare output with old version
4. Fix any discrepancies

**Phase 4: Replacement**
1. Backup old: `mv <tool>.adb <tool>.adb.old`
2. Install new: `mv <tool>_new.adb <tool>.adb`
3. Recompile project
4. Run full regression tests
5. Delete old version if successful

**Phase 5: Documentation**
1. Update tool documentation
2. Update dependency list
3. Commit changes with descriptive message

---

## Regression Testing

Create test suite for each refactored tool:

```bash
#!/bin/bash
# test_<tool>.sh

TOOL="./bin/<tool>"
TOOL_OLD="./bin/<tool>_old"

echo "Testing <tool> refactoring..."

# Test 1: Basic functionality
TEST1_INPUT='<test input>'
EXPECTED1='<expected output>'
ACTUAL1=$(echo "$TEST1_INPUT" | $TOOL <args>)
if [ "$ACTUAL1" != "$EXPECTED1" ]; then
  echo "FAIL: Test 1"
  exit 1
fi

# Test 2: Error handling
# ...

# Compare with old version
echo "$TEST1_INPUT" | $TOOL <args> > new_output.txt
echo "$TEST1_INPUT" | $TOOL_OLD <args> > old_output.txt
if ! diff new_output.txt old_output.txt; then
  echo "FAIL: Output differs from old version"
  exit 1
fi

echo "PASS: All tests passed"
```

---

## Success Criteria

For each refactored tool:
- ✅ Size reduced to <150 lines
- ✅ Uses appropriate utilities
- ✅ Maintains same functionality
- ✅ Passes all regression tests
- ✅ Output identical to old version (or improved)
- ✅ Compiles without errors/warnings
- ✅ Error handling preserved
- ✅ Exit codes match specification
- ✅ Documentation updated

---

## Recommended Refactoring Order

**Batch 1: JSON Tools** (generate JSON utilities first)
1. json_extract (513 → 120 lines) - Most critical

**Batch 2: C++ Generation** (generate C++ utilities first)
2. sig_gen_cpp (406 → 100 lines)

**Batch 3: Validation Tools** (generate validation utilities first)
3. spec_validate_schema (365 → 100 lines)
4. ir_validate (342 → 90 lines)

**Batch 4: Merge & Type Tools**
5. json_merge_deep (389 → 120 lines)
6. type_resolver (378 → 100 lines)

**Batch 5: Extraction & Generation**
7. spec_extract_module (324 → 100 lines)
8. ir_gen_functions (311 → 120 lines)

---

**End of Refactoring Plans**
