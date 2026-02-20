# STUNIR Powertools: Utility Generation Prompts

**Purpose**: Ready-to-use prompts for generating 22 utility components  
**Usage**: Copy each prompt and feed to local model (Claude, GPT-4, etc.)  
**Target**: Each utility <100 lines, single responsibility, Ada SPARK 2014

---

## How to Use These Prompts

1. **Copy entire prompt** for desired utility (including all sections)
2. **Feed to local model** (Claude, GPT-4, local LLM)
3. **Save output** to `tools/spark/src/powertools/<utility_name>.adb`
4. **Test compilation**: `cd tools/spark && gprbuild -P stunir_tools.gpr <utility_name>.adb`
5. **Test functionality**: Use examples provided in prompt
6. **Move to next utility** once current one compiles and works

---

## JSON Utilities (7 tools)

### Prompt 1: json_formatter.adb

```
Generate an Ada SPARK 2014 utility named `json_formatter` that formats JSON with indentation and whitespace.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success, 1=invalid JSON, 2=processing error
- Target size: 60 lines maximum
- Single responsibility: JSON beautification only

Behavior:
- Read JSON from stdin
- Format with proper indentation (default: 2 spaces)
- Output formatted JSON to stdout
- Support --compact mode (remove all whitespace)
- Support --indent N flag (custom indent level)
- Support --sort-keys flag (alphabetically sort object keys)

Options:
- --help, -h        : Show usage
- --version, -v     : Show version 0.1.0-alpha
- --describe        : Output JSON metadata
- --indent N        : Indent level (default: 2)
- --compact         : Remove all whitespace
- --sort-keys       : Sort object keys alphabetically

Exit Codes:
- 0: Success - JSON formatted
- 1: Invalid JSON input
- 2: Processing error

Usage Examples:
```bash
echo '{"a":1,"b":2}' | json_formatter
# Output:
# {
#   "a": 1,
#   "b": 2
# }

echo '{"a": 1, "b": 2}' | json_formatter --compact
# Output: {"a":1,"b":2}

echo '{"z":1,"a":2}' | json_formatter --sort-keys
# Output:
# {
#   "a": 2,
#   "z": 1
# }
```

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded
- Ada.Characters.Handling

Implementation notes:
- Keep simple - basic formatting only
- No full JSON parsing required (use string manipulation)
- Count braces/brackets for indentation levels
- Handle nested objects/arrays with proper indentation
- Skip whitespace in compact mode

Follow Unix philosophy: do one thing well, stdin/stdout, composable
Keep code minimal and focused - target 60 lines
```

---

### Prompt 2: json_path_parser.adb

```
Generate an Ada SPARK 2014 utility named `json_path_parser` that parses dot-notation JSON paths into components.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success, 1=invalid path syntax
- Target size: 100 lines maximum
- Single responsibility: Path parsing only (no JSON evaluation)

Behavior:
- Read path from stdin or --path argument
- Parse dot-notation path into components
- Support array indices: both `array.0` and `array[0]` syntax
- Output path components as JSON array to stdout
- Each component is a string in the array

Path Syntax Supported:
- Dot notation: `a.b.c` → ["a", "b", "c"]
- Array indices: `a.0.b` → ["a", "0", "b"]
- Bracket notation: `a[0].b` → ["a", "0", "b"]
- Mixed: `a.b[1].c.d[2]` → ["a", "b", "1", "c", "d", "2"]

Options:
- --help, -h        : Show usage
- --version, -v     : Show version 0.1.0-alpha
- --describe        : Output JSON metadata
- --path PATH       : Path to parse (alternative to stdin)

Exit Codes:
- 0: Success - path parsed
- 1: Invalid path syntax

Usage Examples:
```bash
echo "functions.0.name" | json_path_parser
# Output: ["functions","0","name"]

json_path_parser --path "module.types[2].fields"
# Output: ["module","types","2","fields"]

echo "a.b.c" | json_path_parser
# Output: ["a","b","c"]
```

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded
- Ada.Strings.Fixed

Implementation notes:
- Split on '.' delimiter
- Handle '[' and ']' brackets (extract number, treat as component)
- Validate path syntax (no empty components, valid indices)
- Output valid JSON array format
- No actual JSON processing needed - just string parsing

Follow Unix philosophy: do one thing well, stdin/stdout, composable
Keep code minimal and focused - target 100 lines
```

---

### Prompt 3: json_path_eval.adb

```
Generate an Ada SPARK 2014 utility named `json_path_eval` that evaluates parsed JSON paths on a JSON document.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success, 1=path not found, 2=invalid JSON
- Target size: 120 lines maximum
- Single responsibility: Path evaluation only (assumes valid path components from json_path_parser)

Behavior:
- Read JSON document from stdin
- Read path components from --path argument (JSON array format)
- Traverse JSON document following path components
- Output extracted value to stdout
- Support --default VALUE flag (return default if path not found)

Options:
- --help, -h        : Show usage
- --version, -v     : Show version 0.1.0-alpha
- --describe        : Output JSON metadata
- --path ARRAY      : JSON array of path components (required)
- --default VALUE   : Default value if path not found

Exit Codes:
- 0: Success - value found and extracted
- 1: Path not found (no default provided)
- 2: Invalid JSON input

Usage Examples:
```bash
echo '{"a":{"b":{"c":123}}}' | json_path_eval --path '["a","b","c"]'
# Output: 123

echo '{"users":[{"name":"Alice"},{"name":"Bob"}]}' | json_path_eval --path '["users","1","name"]'
# Output: "Bob"

echo '{"x":1}' | json_path_eval --path '["y"]' --default "null"
# Output: null
```

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded
- Ada.Strings.Fixed

Implementation notes:
- Basic JSON navigation (find keys, handle arrays)
- For object access: find key, extract value
- For array access: parse index, access element
- Handle nested structures recursively
- Return raw value (string, number, object, array)
- Minimal JSON parsing - just enough to navigate

Follow Unix philosophy: do one thing well, stdin/stdout, composable
Keep code minimal and focused - target 120 lines
Designed to work with json_path_parser output
```

---

### Prompt 4: json_value_format.adb

```
Generate an Ada SPARK 2014 utility named `json_value_format` that formats extracted JSON values for output.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success
- Target size: 50 lines maximum
- Single responsibility: Value formatting only

Behavior:
- Read value from stdin
- Apply formatting based on flags
- Output formatted value to stdout
- Support raw mode (remove quotes from strings)
- Support type annotation mode
- Support compact mode

Options:
- --help, -h        : Show usage
- --version, -v     : Show version 0.1.0-alpha
- --describe        : Output JSON metadata
- --raw             : Remove quotes from strings
- --type            : Add type annotation
- --compact         : Minimize whitespace

Exit Codes:
- 0: Always success

Usage Examples:
```bash
echo '"hello"' | json_value_format --raw
# Output: hello

echo '123' | json_value_format --type
# Output: 123 (number)

echo '{"a":1}' | json_value_format --type
# Output: {"a":1} (object)

echo '  "value"  ' | json_value_format --compact
# Output: "value"
```

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded
- Ada.Strings.Fixed
- Ada.Characters.Handling

Implementation notes:
- Simple string manipulation
- Detect value type (string, number, boolean, null, object, array)
- For --raw: strip leading/trailing quotes if string
- For --type: append type label
- For --compact: trim whitespace
- No full JSON parsing needed

Follow Unix philosophy: do one thing well, stdin/stdout, composable
Keep code minimal and focused - target 50 lines
```

---

### Prompt 5: json_merge_objects.adb

```
Generate an Ada SPARK 2014 utility named `json_merge_objects` that merges two or more JSON objects.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success, 1=merge conflict (strategy=error), 2=invalid input
- Target size: 100 lines maximum
- Single responsibility: Object merging with conflict detection

Behavior:
- Read multiple JSON objects from stdin (one per line)
- Merge all objects into single result object
- Handle key conflicts based on strategy
- Output merged object to stdout

Merge Strategies:
- --strategy last   : Last value wins (default)
- --strategy first  : First value wins
- --strategy error  : Exit with error on conflict

Options:
- --help, -h        : Show usage
- --version, -v     : Show version 0.1.0-alpha
- --describe        : Output JSON metadata
- --strategy STRAT  : Conflict resolution (last/first/error)

Exit Codes:
- 0: Success - objects merged
- 1: Merge conflict (strategy=error)
- 2: Invalid JSON input

Usage Examples:
```bash
echo '{"a":1,"b":2}
{"b":3,"c":4}' | json_merge_objects
# Output: {"a":1,"b":3,"c":4}

echo '{"a":1,"b":2}
{"b":3,"c":4}' | json_merge_objects --strategy first
# Output: {"a":1,"b":2,"c":4}

echo '{"a":1}
{"a":2}' | json_merge_objects --strategy error
# Exit 1: ERROR: Key conflict: a
```

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded

Implementation notes:
- Parse objects line by line
- Extract key-value pairs from each object
- Build result object with conflict resolution
- Track seen keys for conflict detection
- Output valid JSON object format

Follow Unix philosophy: do one thing well, stdin/stdout, composable
Keep code minimal and focused - target 100 lines
```

---

### Prompt 6: json_merge_arrays.adb

```
Generate an Ada SPARK 2014 utility named `json_merge_arrays` that merges two or more JSON arrays.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success, 2=invalid input
- Target size: 80 lines maximum
- Single responsibility: Array merging (concatenation or union)

Behavior:
- Read multiple JSON arrays from stdin (one per line)
- Merge arrays by concatenation or union
- Output merged array to stdout
- Support --unique flag (remove duplicates)
- Support --sort flag (sort result)

Options:
- --help, -h        : Show usage
- --version, -v     : Show version 0.1.0-alpha
- --describe        : Output JSON metadata
- --unique          : Remove duplicate values
- --sort            : Sort resulting array

Exit Codes:
- 0: Success - arrays merged
- 2: Invalid JSON input

Usage Examples:
```bash
echo '[1,2,3]
[4,5,6]' | json_merge_arrays
# Output: [1,2,3,4,5,6]

echo '[1,2,3]
[2,3,4]' | json_merge_arrays --unique
# Output: [1,2,3,4]

echo '[3,1,2]
[6,4,5]' | json_merge_arrays --sort
# Output: [1,2,3,4,5,6]
```

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded

Implementation notes:
- Parse arrays line by line
- Extract elements from each array
- Concatenate all elements
- For --unique: track seen values, skip duplicates
- For --sort: sort elements (numeric or lexicographic)
- Output valid JSON array format

Follow Unix philosophy: do one thing well, stdin/stdout, composable
Keep code minimal and focused - target 80 lines
```

---

### Prompt 7: json_conflict_resolver.adb

```
Generate an Ada SPARK 2014 utility named `json_conflict_resolver` that applies conflict resolution strategies to merge operations.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success, 1=unresolved conflict
- Target size: 70 lines maximum
- Single responsibility: Conflict resolution logic only

Behavior:
- Read conflict report from stdin (JSON format)
- Apply resolution strategy
- Output resolution decision to stdout

Conflict Report Format:
```json
{
  "key": "field_name",
  "values": ["value1", "value2"],
  "sources": ["source1", "source2"]
}
```

Strategies:
- --strategy last    : Use last value
- --strategy first   : Use first value
- --strategy manual  : Prompt user (not implemented - exit error)
- --strategy error   : Report error and exit 1

Options:
- --help, -h        : Show usage
- --version, -v     : Show version 0.1.0-alpha
- --describe        : Output JSON metadata
- --strategy STRAT  : Resolution strategy (required)

Exit Codes:
- 0: Success - conflict resolved
- 1: Unresolved conflict

Usage Examples:
```bash
echo '{"key":"name","values":["Alice","Bob"],"sources":["file1","file2"]}' | json_conflict_resolver --strategy last
# Output: {"key":"name","value":"Bob"}

echo '{"key":"age","values":[25,30],"sources":["old","new"]}' | json_conflict_resolver --strategy first
# Output: {"key":"age","value":25}
```

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded

Implementation notes:
- Parse conflict report JSON
- Extract key and values array
- Apply strategy to select value
- Output resolution as JSON object
- Simple logic - no complex decision making

Follow Unix philosophy: do one thing well, stdin/stdout, composable
Keep code minimal and focused - target 70 lines
```

---

## C++ Generation Utilities (4 tools)

### Prompt 8: type_map_cpp.adb

```
Generate an Ada SPARK 2014 utility named `type_map_cpp` that maps STUNIR type names to C++ types.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success, 1=unknown type
- Target size: 50 lines maximum
- Single responsibility: Type mapping only

Behavior:
- Read STUNIR type name from stdin or --type argument
- Map to C++ type using predefined rules
- Output C++ type to stdout

Type Mapping Rules:
- i8 → int8_t
- i16 → int16_t
- i32 → int32_t
- i64 → int64_t
- u8 → uint8_t
- u16 → uint16_t
- u32 → uint32_t
- u64 → uint64_t
- f32 → float
- f64 → double
- bool → bool
- str → std::string
- void → void

Options:
- --help, -h        : Show usage
- --version, -v     : Show version 0.1.0-alpha
- --describe        : Output JSON metadata
- --type TYPE       : Type to map (alternative to stdin)

Exit Codes:
- 0: Success - type mapped
- 1: Unknown type

Usage Examples:
```bash
echo "i32" | type_map_cpp
# Output: int32_t

echo "str" | type_map_cpp
# Output: std::string

type_map_cpp --type f64
# Output: double

echo "unknown_type" | type_map_cpp
# Exit 1: ERROR: Unknown type: unknown_type
```

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded
- Ada.Characters.Handling (for case conversion)

Implementation notes:
- Simple lookup table / case statement
- Case-insensitive matching
- Output only the mapped type (no extra text)
- Exit with error for unknown types

Follow Unix philosophy: do one thing well, stdin/stdout, composable
Keep code minimal and focused - target 50 lines
```

---

### Prompt 9: cpp_signature_gen.adb

```
Generate an Ada SPARK 2014 utility named `cpp_signature_gen` that generates C++ function signatures from JSON.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success, 1=invalid signature JSON
- Target size: 80 lines maximum
- Single responsibility: C++ signature generation only

Behavior:
- Read function signature JSON from stdin
- Generate C++ function signature
- Output signature to stdout
- Support optional modifiers (inline, static, const)

Input JSON Format:
```json
{
  "name": "add",
  "return_type": "int32_t",
  "args": [
    {"name": "a", "type": "int32_t"},
    {"name": "b", "type": "int32_t"}
  ]
}
```

Output Format:
```cpp
int32_t add(int32_t a, int32_t b);
```

Options:
- --help, -h        : Show usage
- --version, -v     : Show version 0.1.0-alpha
- --describe        : Output JSON metadata
- --inline          : Add inline keyword
- --static          : Add static keyword
- --const           : Add const qualifier

Exit Codes:
- 0: Success - signature generated
- 1: Invalid signature JSON

Usage Examples:
```bash
echo '{"name":"add","return_type":"int32_t","args":[{"name":"a","type":"int32_t"},{"name":"b","type":"int32_t"}]}' | cpp_signature_gen
# Output: int32_t add(int32_t a, int32_t b);

echo '{"name":"get_value","return_type":"int32_t","args":[]}' | cpp_signature_gen --inline --const
# Output: inline int32_t get_value() const;
```

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded

Implementation notes:
- Parse JSON to extract name, return_type, args
- Build signature string: [modifiers] return_type name(param_list);
- Format parameter list: type1 name1, type2 name2, ...
- Handle empty parameter list: ()
- Add semicolon at end

Follow Unix philosophy: do one thing well, stdin/stdout, composable
Keep code minimal and focused - target 80 lines
```

---

### Prompt 10: cpp_header_gen.adb

```
Generate an Ada SPARK 2014 utility named `cpp_header_gen` that generates C++ header structure (guards, includes).

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success
- Target size: 60 lines maximum
- Single responsibility: Header structure generation only

Behavior:
- Read module name from stdin or --module argument
- Generate C++ header guard and includes
- Output header structure to stdout
- Support custom guard name
- Support #pragma once instead of guards

Output Structure:
```cpp
#ifndef MODULE_NAME_H
#define MODULE_NAME_H

#include <cstdint>
#include <string>

// Content goes here

#endif // MODULE_NAME_H
```

Options:
- --help, -h        : Show usage
- --version, -v     : Show version 0.1.0-alpha
- --describe        : Output JSON metadata
- --module NAME     : Module name (alternative to stdin)
- --guard NAME      : Custom header guard name
- --pragma-once     : Use #pragma once instead of guards

Exit Codes:
- 0: Always success

Usage Examples:
```bash
echo "MyModule" | cpp_header_gen
# Output:
# #ifndef MYMODULE_H
# #define MYMODULE_H
# ...
# #endif // MYMODULE_H

cpp_header_gen --module utils --pragma-once
# Output:
# #pragma once
# ...
```

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded
- Ada.Characters.Handling (for case conversion)

Implementation notes:
- Convert module name to uppercase for guard
- Add standard includes (cstdint, string)
- If --pragma-once: output #pragma once instead of guards
- Output structure with placeholder comment
- Simple template-based generation

Follow Unix philosophy: do one thing well, stdin/stdout, composable
Keep code minimal and focused - target 60 lines
```

---

### Prompt 11: cpp_namespace_wrap.adb

```
Generate an Ada SPARK 2014 utility named `cpp_namespace_wrap` that wraps code in C++ namespace declaration.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success
- Target size: 40 lines maximum
- Single responsibility: Namespace wrapping only

Behavior:
- Read code from stdin
- Wrap in namespace declaration
- Output wrapped code to stdout
- Support nested namespaces (a::b::c)

Options:
- --help, -h        : Show usage
- --version, -v     : Show version 0.1.0-alpha
- --describe        : Output JSON metadata
- --namespace NAME  : Namespace name (required)
- --nested          : Support nested namespaces (a::b::c)

Exit Codes:
- 0: Always success

Usage Examples:
```bash
echo "void foo();" | cpp_namespace_wrap --namespace mylib
# Output:
# namespace mylib {
#   void foo();
# }

echo "int bar();" | cpp_namespace_wrap --namespace company::product
# Output:
# namespace company {
# namespace product {
#   int bar();
# }
# }
```

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded

Implementation notes:
- Read all code from stdin
- If --nested: split namespace on "::", create nested structure
- Otherwise: single namespace wrapper
- Indent code inside namespace (2 spaces)
- Output opening/closing braces

Follow Unix philosophy: do one thing well, stdin/stdout, composable
Keep code minimal and focused - target 40 lines
```

---

## Validation Utilities (7 tools)

### Prompt 12: schema_check_required.adb

```
Generate an Ada SPARK 2014 utility named `schema_check_required` that verifies required fields exist in JSON.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=all present, 1=missing fields
- Target size: 60 lines maximum
- Single responsibility: Field existence checking only

Behavior:
- Read JSON from stdin
- Read required fields list from --fields argument (JSON array)
- Check all required fields exist
- Output validation result to stdout

Options:
- --help, -h        : Show usage
- --version, -v     : Show version 0.1.0-alpha
- --describe        : Output JSON metadata
- --fields ARRAY    : JSON array of required field paths

Exit Codes:
- 0: All required fields present
- 1: One or more fields missing

Usage Examples:
```bash
echo '{"a":1,"b":2}' | schema_check_required --fields '["a","b"]'
# Output: VALID

echo '{"a":1}' | schema_check_required --fields '["a","b"]'
# Output: INVALID: Missing field 'b'
# Exit 1

echo '{"module":{"name":"test"}}' | schema_check_required --fields '["module.name"]'
# Output: VALID
```

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded

Implementation notes:
- Parse required fields array
- For each field: check if present in JSON (simple string search for basic check)
- Support dot notation for nested fields (module.name)
- Report first missing field or VALID
- Keep validation simple - not full JSON parsing

Follow Unix philosophy: do one thing well, stdin/stdout, composable
Keep code minimal and focused - target 60 lines
```

---

### Prompt 13: schema_check_types.adb

```
Generate an Ada SPARK 2014 utility named `schema_check_types` that validates field types match schema.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=all valid, 1=type mismatch
- Target size: 80 lines maximum
- Single responsibility: Type validation only

Behavior:
- Read JSON from stdin
- Read type schema from --schema argument (JSON object)
- Validate each field's type matches schema
- Output validation result to stdout

Schema Format:
```json
{
  "name": "string",
  "version": "string",
  "count": "number",
  "active": "boolean",
  "functions": "array",
  "metadata": "object"
}
```

Options:
- --help, -h        : Show usage
- --version, -v     : Show version 0.1.0-alpha
- --describe        : Output JSON metadata
- --schema JSON     : Type schema as JSON object

Exit Codes:
- 0: All types valid
- 1: Type mismatch found

Usage Examples:
```bash
echo '{"name":"test","count":5}' | schema_check_types --schema '{"name":"string","count":"number"}'
# Output: VALID

echo '{"name":"test","count":"5"}' | schema_check_types --schema '{"name":"string","count":"number"}'
# Output: INVALID: Field 'count' expected number, got string
# Exit 1
```

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded

Implementation notes:
- Parse schema JSON
- For each field in schema: find field in input JSON
- Detect value type (string, number, boolean, array, object, null)
- Compare detected type with expected type
- Report first mismatch or VALID
- Simple type detection - check first/last chars, patterns

Follow Unix philosophy: do one thing well, stdin/stdout, composable
Keep code minimal and focused - target 80 lines
```

---

### Prompt 14: schema_check_format.adb

```
Generate an Ada SPARK 2014 utility named `schema_check_format` that validates format constraints (patterns, ranges).

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=all valid, 1=format violation
- Target size: 70 lines maximum
- Single responsibility: Format validation only

Behavior:
- Read JSON from stdin
- Read format rules from --rules argument (JSON object)
- Validate each field's format
- Output validation result to stdout

Format Rules Format:
```json
{
  "version": {"pattern": "^\\d+\\.\\d+\\.\\d+$"},
  "name": {"minLength": 1, "maxLength": 64},
  "age": {"min": 0, "max": 150}
}
```

Options:
- --help, -h        : Show usage
- --version, -v     : Show version 0.1.0-alpha
- --describe        : Output JSON metadata
- --rules JSON      : Format rules as JSON object

Exit Codes:
- 0: All formats valid
- 1: Format violation found

Usage Examples:
```bash
echo '{"version":"1.0.0"}' | schema_check_format --rules '{"version":{"pattern":"^\\d+\\.\\d+\\.\\d+$"}}'
# Output: VALID

echo '{"version":"1.0"}' | schema_check_format --rules '{"version":{"pattern":"^\\d+\\.\\d+\\.\\d+$"}}'
# Output: INVALID: Field 'version' does not match pattern
# Exit 1

echo '{"name":"test"}' | schema_check_format --rules '{"name":{"minLength":1,"maxLength":64}}'
# Output: VALID
```

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded
- GNAT.Regpat (for pattern matching)

Implementation notes:
- Parse format rules JSON
- For each field: extract value from input JSON
- Apply format constraints:
  - pattern: regex match
  - minLength/maxLength: string length check
  - min/max: numeric range check
- Report first violation or VALID
- Basic regex support - simple patterns

Follow Unix philosophy: do one thing well, stdin/stdout, composable
Keep code minimal and focused - target 70 lines
```

---

### Prompt 15: validation_reporter.adb

```
Generate an Ada SPARK 2014 utility named `validation_reporter` that generates consistent validation reports.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success
- Target size: 50 lines maximum
- Single responsibility: Report formatting only

Behavior:
- Read validation results from stdin (JSON array)
- Format as readable report
- Output formatted report to stdout
- Support text and JSON output formats

Input Format:
```json
[
  {"field": "schema", "status": "ok"},
  {"field": "module.name", "status": "error", "message": "required but missing"},
  {"field": "version", "status": "warning", "message": "non-standard format"}
]
```

Options:
- --help, -h        : Show usage
- --version, -v     : Show version 0.1.0-alpha
- --describe        : Output JSON metadata
- --format FORMAT   : Output format (text/json)
- --verbose         : Include all checks (not just errors)

Exit Codes:
- 0: Always success (report formatting never fails)

Usage Examples:
```bash
echo '[{"field":"schema","status":"ok"},{"field":"module.name","status":"error","message":"missing"}]' | validation_reporter
# Output:
# INVALID: 1 error(s)
#   ✗ module.name: missing

echo '[...]' | validation_reporter --format json
# Output: {"status":"invalid","errors":[{"field":"module.name","message":"missing"}]}

echo '[...]' | validation_reporter --verbose
# Output:
# INVALID: 1 error(s)
#   ✓ schema: ok
#   ✗ module.name: missing
```

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded

Implementation notes:
- Parse validation results array
- Count errors/warnings
- Format based on --format flag:
  - text: human-readable with ✓/✗ symbols
  - json: structured JSON output
- Show only errors by default (--verbose shows all)
- Simple formatting logic

Follow Unix philosophy: do one thing well, stdin/stdout, composable
Keep code minimal and focused - target 50 lines
```

---

### Prompt 16: ir_check_required.adb

```
Generate an Ada SPARK 2014 utility named `ir_check_required` that checks required fields in STUNIR IR JSON.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=all present, 1=missing fields
- Target size: 60 lines maximum
- Single responsibility: IR field existence checking

Behavior:
- Read IR JSON from stdin
- Check required IR-specific fields
- Output validation result to stdout

Required IR Fields:
- schema (must be "stunir_flat_ir_v1")
- ir_version
- module_name
- functions (array)

Options:
- --help, -h        : Show usage
- --version, -v     : Show version 0.1.0-alpha
- --describe        : Output JSON metadata

Exit Codes:
- 0: All required fields present
- 1: One or more fields missing

Usage Examples:
```bash
echo '{"schema":"stunir_flat_ir_v1","ir_version":"0.1.0","module_name":"test","functions":[]}' | ir_check_required
# Output: VALID

echo '{"schema":"stunir_flat_ir_v1"}' | ir_check_required
# Output: INVALID: Missing required IR fields: ir_version, module_name, functions
# Exit 1
```

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded

Implementation notes:
- Similar to schema_check_required but IR-specific
- Check presence of required fields
- Validate schema field value
- Report all missing fields in single message
- Simple string searching

Follow Unix philosophy: do one thing well, stdin/stdout, composable
Keep code minimal and focused - target 60 lines
```

---

### Prompt 17: ir_check_functions.adb

```
Generate an Ada SPARK 2014 utility named `ir_check_functions` that validates IR function structures.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=all valid, 1=validation error
- Target size: 90 lines maximum
- Single responsibility: Function structure validation

Behavior:
- Read IR JSON from stdin
- Extract functions array
- Validate each function structure
- Output validation result to stdout

Function Validation Checks:
- Function has "name" field (non-empty string)
- Function has "return_type" field
- Function has "args" array
- Each arg has "name" and "type" fields
- Function has "steps" array
- Each step is valid JSON object

Options:
- --help, -h        : Show usage
- --version, -v     : Show version 0.1.0-alpha
- --describe        : Output JSON metadata

Exit Codes:
- 0: All functions valid
- 1: Validation error found

Usage Examples:
```bash
echo '{"functions":[{"name":"add","return_type":"i32","args":[{"name":"a","type":"i32"}],"steps":[]}]}' | ir_check_functions
# Output: VALID

echo '{"functions":[{"name":"","return_type":"i32","args":[]}]}' | ir_check_functions
# Output: INVALID: Function 0: name is empty
# Exit 1
```

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded

Implementation notes:
- Parse functions array
- Iterate through each function
- Check required fields present and valid
- Validate args structure
- Validate steps structure (basic check)
- Report first error found with function index
- Keep validation simple - structural checks only

Follow Unix philosophy: do one thing well, stdin/stdout, composable
Keep code minimal and focused - target 90 lines
```

---

### Prompt 18: ir_check_types.adb

```
Generate an Ada SPARK 2014 utility named `ir_check_types` that validates IR type definitions.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=all valid, 1=validation error
- Target size: 70 lines maximum
- Single responsibility: Type definition validation

Behavior:
- Read IR JSON from stdin
- Extract types array (if present)
- Validate each type definition
- Output validation result to stdout

Type Validation Checks:
- Type has "name" field (non-empty)
- Type has "kind" field (struct/enum/alias)
- For struct: has "fields" array
- For enum: has "variants" array
- For alias: has "target" field
- No circular dependencies (basic check)

Options:
- --help, -h        : Show usage
- --version, -v     : Show version 0.1.0-alpha
- --describe        : Output JSON metadata

Exit Codes:
- 0: All types valid (or no types present)
- 1: Validation error found

Usage Examples:
```bash
echo '{"types":[{"name":"Point","kind":"struct","fields":[{"name":"x","type":"i32"}]}]}' | ir_check_types
# Output: VALID

echo '{"types":[{"name":"","kind":"struct"}]}' | ir_check_types
# Output: INVALID: Type 0: name is empty
# Exit 1

echo '{}' | ir_check_types
# Output: VALID (no types to check)
```

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded

Implementation notes:
- Parse types array (if exists)
- If no types: exit success
- For each type: validate structure based on kind
- Check for basic circular dependencies (type refers to itself)
- Report first error with type index
- Keep validation simple - structural checks

Follow Unix philosophy: do one thing well, stdin/stdout, composable
Keep code minimal and focused - target 70 lines
```

---

## Type Utilities (3 tools)

### Prompt 19: type_lookup.adb

```
Generate an Ada SPARK 2014 utility named `type_lookup` that looks up type definitions in a type registry.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=found, 1=not found
- Target size: 80 lines maximum
- Single responsibility: Type lookup only

Behavior:
- Read type name from stdin or --type argument
- Read type registry from --registry file
- Find type definition in registry
- Output type definition to stdout

Registry Format:
```json
{
  "types": {
    "MyList": {"type": "array", "element": "i32"},
    "Point": {"type": "struct", "fields": [{"name": "x", "type": "i32"}]},
    "Status": {"type": "enum", "variants": ["ok", "error"]}
  }
}
```

Options:
- --help, -h        : Show usage
- --version, -v     : Show version 0.1.0-alpha
- --describe        : Output JSON metadata
- --type NAME       : Type name to lookup (alternative to stdin)
- --registry FILE   : JSON file with type registry

Exit Codes:
- 0: Type found and output
- 1: Type not found in registry

Usage Examples:
```bash
echo "MyList" | type_lookup --registry types.json
# Output: {"type":"array","element":"i32"}

type_lookup --type Point --registry types.json
# Output: {"type":"struct","fields":[{"name":"x","type":"i32"}]}

echo "Unknown" | type_lookup --registry types.json
# Exit 1: ERROR: Type not found: Unknown
```

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded

Implementation notes:
- Read registry file
- Parse types object
- Search for type by name
- Extract and output definition
- Simple JSON object lookup

Follow Unix philosophy: do one thing well, stdin/stdout, composable
Keep code minimal and focused - target 80 lines
```

---

### Prompt 20: type_expand.adb

```
Generate an Ada SPARK 2014 utility named `type_expand` that expands type aliases and composite types.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success, 1=expansion error
- Target size: 100 lines maximum
- Single responsibility: Type expansion only

Behavior:
- Read type reference from stdin
- Read type registry from --registry file
- Recursively expand type if it's an alias
- Output expanded type definition to stdout

Options:
- --help, -h        : Show usage
- --version, -v     : Show version 0.1.0-alpha
- --describe        : Output JSON metadata
- --registry FILE   : JSON file with type registry
- --recursive       : Recursively expand nested types

Exit Codes:
- 0: Type expanded successfully
- 1: Expansion error (circular reference, not found)

Usage Examples:
```bash
echo "MyAlias" | type_expand --registry types.json
# Input registry: {"types":{"MyAlias":{"alias":"i32"}}}
# Output: i32

echo "MyList" | type_expand --registry types.json --recursive
# Input registry: {"types":{"MyList":{"type":"array","element":"MyInt"},"MyInt":{"alias":"i32"}}}
# Output: {"type":"array","element":"i32"}
```

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded

Implementation notes:
- Read registry and type reference
- If type is alias: look up target and expand
- If --recursive: expand nested type references
- Detect circular references (track expansion path)
- Output final expanded type
- Handle primitives (return as-is)

Follow Unix philosophy: do one thing well, stdin/stdout, composable
Keep code minimal and focused - target 100 lines
```

---

### Prompt 21: type_dependency.adb

```
Generate an Ada SPARK 2014 utility named `type_dependency` that resolves type dependency order.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success, 1=circular dependency
- Target size: 70 lines maximum
- Single responsibility: Dependency resolution only

Behavior:
- Read type definitions from stdin (JSON array)
- Analyze type dependencies
- Output dependency-ordered type names to stdout
- Detect circular dependencies

Options:
- --help, -h        : Show usage
- --version, -v     : Show version 0.1.0-alpha
- --describe        : Output JSON metadata
- --check-cycles    : Error on circular dependencies (default: true)

Exit Codes:
- 0: Dependencies resolved successfully
- 1: Circular dependency detected

Usage Examples:
```bash
echo '[{"name":"A","deps":[]},{"name":"B","deps":["A"]},{"name":"C","deps":["B","A"]}]' | type_dependency
# Output: ["A","B","C"]

echo '[{"name":"A","deps":["B"]},{"name":"B","deps":["A"]}]' | type_dependency
# Exit 1: ERROR: Circular dependency: A -> B -> A
```

Input Format:
```json
[
  {"name": "Point", "deps": []},
  {"name": "Line", "deps": ["Point"]},
  {"name": "Polygon", "deps": ["Point", "Line"]}
]
```

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded

Implementation notes:
- Parse type definitions array
- Build dependency graph
- Topological sort (Kahn's algorithm or DFS)
- Detect cycles during sort
- Output ordered array of type names
- Keep algorithm simple

Follow Unix philosophy: do one thing well, stdin/stdout, composable
Keep code minimal and focused - target 70 lines
```

---

## File Utilities (1 tool)

### Prompt 22: file_writer.adb

```
Generate an Ada SPARK 2014 utility named `file_writer` that writes content to a file with error handling.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success, 2=write error, 3=permission denied
- Target size: 50 lines maximum
- Single responsibility: File writing with error handling

Behavior:
- Read content from stdin
- Write to file specified as argument
- Create parent directories if needed (--create-dirs)
- Support append mode (--append)
- Support backup of existing file (--backup)

Options:
- --help, -h        : Show usage
- --version, -v     : Show version 0.1.0-alpha
- --describe        : Output JSON metadata
- --create-dirs     : Create parent directories
- --append          : Append instead of overwrite
- --backup          : Create backup if file exists (.bak)

Exit Codes:
- 0: Success - file written
- 2: Write error
- 3: Permission denied

Usage Examples:
```bash
echo "content" | file_writer output.txt
# Writes "content" to output.txt

echo "more" | file_writer --append output.txt
# Appends "more" to output.txt

echo "data" | file_writer --create-dirs path/to/file.txt
# Creates path/to/ if needed, writes file.txt

echo "new" | file_writer --backup existing.txt
# Creates existing.txt.bak, writes new content
```

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Directories
- Ada.IO_Exceptions

Implementation notes:
- Read all content from stdin
- If --create-dirs: create parent directories
- If --backup and file exists: copy to .bak
- Open file in appropriate mode (append/overwrite)
- Write content
- Handle IO exceptions with proper exit codes
- Close file properly

Follow Unix philosophy: do one thing well, stdin/stdout, composable
Keep code minimal and focused - target 50 lines
```

---

## Generation Workflow

### Step-by-Step Process

1. **Select utility to generate** (start with JSON utilities)
2. **Copy entire prompt** (including all sections)
3. **Feed to local model** (Claude, GPT-4, etc.)
4. **Review generated code** for:
   - Ada SPARK 2014 compliance
   - Size target met (<100 lines)
   - All requirements implemented
   - Clean compilation
5. **Save to file**: `tools/spark/src/powertools/<name>.adb`
6. **Test compilation**:
   ```bash
   cd tools/spark
   gprbuild -P stunir_tools.gpr <name>.adb
   ```
7. **Test functionality** with examples from prompt
8. **Fix any issues** and recompile
9. **Mark as complete** in tracking document
10. **Move to next utility**

### Recommended Generation Order

**Batch 1: JSON Path Utilities** (most critical for json_extract refactor)
1. json_path_parser
2. json_path_eval
3. json_value_format

**Batch 2: JSON Formatting & Merging**
4. json_formatter
5. json_merge_objects
6. json_merge_arrays
7. json_conflict_resolver

**Batch 3: C++ Generation Utilities** (for code_gen_func_sig and emitters)
8. type_map_cpp
9. cpp_signature_gen
10. cpp_header_gen
11. cpp_namespace_wrap

**Batch 4: Validation Utilities** (for validator refactors)
12. schema_check_required
13. schema_check_types
14. schema_check_format
15. validation_reporter
16. ir_check_required
17. ir_check_functions
18. ir_check_types

**Batch 5: Type & File Utilities**
19. type_lookup
20. type_expand
21. type_dependency
22. file_writer

---

## Testing Each Utility

After generating each utility, test with these commands:

```bash
# Test help
./bin/<utility> --help

# Test version
./bin/<utility> --version

# Test describe
./bin/<utility> --describe

# Test main functionality (use examples from prompt)
<example command from prompt>

# Test error cases
<invalid input test>

# Check exit code
echo $?  # Should match expected exit code
```

---

## Success Criteria

For each utility:
- ✅ Compiles without errors
- ✅ Compiles without warnings
- ✅ Size target met (<100 lines)
- ✅ All flags work (--help, --version, --describe)
- ✅ Main functionality works as specified
- ✅ Exit codes correct
- ✅ Handles error cases gracefully
- ✅ Output format matches specification

---

**End of Utility Generation Prompts**
