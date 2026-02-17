# STUNIR Powertools: Utility Component Specifications

**Date**: 2026-02-17  
**Purpose**: Specifications for utility components needed for tool decomposition  
**Target**: Each utility < 100 lines, single responsibility, reusable

---

## JSON Utilities

### 1. `json_formatter.adb` - Pretty-Print JSON

**Purpose**: Format JSON with indentation and whitespace  
**Size Target**: 60 lines  
**Responsibility**: JSON beautification only

**Interface**:
```ada
procedure JSON_Formatter is
   -- Reads JSON from stdin
   -- Outputs formatted JSON to stdout
   -- Options:
   --   --indent N    : Indent level (default: 2)
   --   --compact     : Remove all whitespace
   --   --sort-keys   : Sort object keys alphabetically
end JSON_Formatter;
```

**Exit Codes**:
- 0: Success
- 1: Invalid JSON
- 2: Processing error

**Usage Examples**:
```bash
echo '{"a":1,"b":2}' | json_formatter
# Output: {
#   "a": 1,
#   "b": 2
# }

echo '{"a": 1}' | json_formatter --compact
# Output: {"a":1}
```

---

### 2. `json_path_parser.adb` - Parse JSON Paths

**Purpose**: Parse dot-notation JSON paths into components  
**Size Target**: 100 lines  
**Responsibility**: Path parsing only (no evaluation)

**Interface**:
```ada
procedure JSON_Path_Parser is
   -- Reads path from stdin or argument
   -- Outputs path components as JSON array to stdout
   -- Examples:
   --   "functions.0.name" → ["functions", "0", "name"]
   --   "module.types[2]" → ["module", "types", "2"]
end JSON_Path_Parser;
```

**Exit Codes**:
- 0: Success
- 1: Invalid path syntax

**Path Syntax**:
- Dot notation: `a.b.c`
- Array indices: `a.0.b` or `a[0].b`
- Nested: `a.b[1].c.d`

**Usage Examples**:
```bash
echo "functions.0.name" | json_path_parser
# Output: ["functions", "0", "name"]

json_path_parser --path "module.types[2].fields"
# Output: ["module", "types", "2", "fields"]
```

---

### 3. `json_path_eval.adb` - Evaluate JSON Paths

**Purpose**: Evaluate parsed path on JSON document  
**Size Target**: 120 lines  
**Responsibility**: Path evaluation only (assumes valid path)

**Interface**:
```ada
procedure JSON_Path_Eval is
   -- Reads JSON from stdin
   -- Reads path components from --path argument
   -- Outputs extracted value to stdout
   -- Options:
   --   --path COMPONENTS : JSON array of path components
   --   --default VALUE   : Default if path not found
end JSON_Path_Eval;
```

**Exit Codes**:
- 0: Success (value found)
- 1: Path not found (no default)
- 2: Invalid JSON

**Usage Examples**:
```bash
echo '{"a":{"b":{"c":123}}}' | json_path_eval --path '["a","b","c"]'
# Output: 123

echo '{"x":1}' | json_path_eval --path '["y"]' --default "null"
# Output: null
```

---

### 4. `json_value_format.adb` - Format Extracted Values

**Purpose**: Format extracted JSON values for output  
**Size Target**: 50 lines  
**Responsibility**: Value formatting only

**Interface**:
```ada
procedure JSON_Value_Format is
   -- Reads value from stdin
   -- Outputs formatted value to stdout
   -- Options:
   --   --raw     : Remove quotes from strings
   --   --type    : Output type annotation
   --   --compact : Minimize whitespace
end JSON_Value_Format;
```

**Exit Codes**:
- 0: Success

**Usage Examples**:
```bash
echo '"hello"' | json_value_format --raw
# Output: hello

echo '123' | json_value_format --type
# Output: 123 (number)
```

---

### 5-7. JSON Merge Utilities

#### 5. `json_merge_objects.adb` - Merge JSON Objects

**Purpose**: Merge two or more JSON objects  
**Size Target**: 100 lines  
**Responsibility**: Object merging with conflict detection

**Interface**:
```ada
procedure JSON_Merge_Objects is
   -- Reads multiple JSON objects from stdin (one per line)
   -- Outputs merged object to stdout
   -- Options:
   --   --strategy {last,first,error} : Conflict resolution
end JSON_Merge_Objects;
```

**Exit Codes**:
- 0: Success
- 1: Merge conflict (strategy=error)
- 2: Invalid input

---

#### 6. `json_merge_arrays.adb` - Merge JSON Arrays

**Purpose**: Merge two or more JSON arrays  
**Size Target**: 80 lines  
**Responsibility**: Array merging (concatenation or union)

**Interface**:
```ada
procedure JSON_Merge_Arrays is
   -- Reads multiple JSON arrays from stdin
   -- Outputs merged array to stdout
   -- Options:
   --   --unique  : Remove duplicates
   --   --sort    : Sort result
end JSON_Merge_Arrays;
```

**Exit Codes**:
- 0: Success
- 2: Invalid input

---

#### 7. `json_conflict_resolver.adb` - Resolve Merge Conflicts

**Purpose**: Apply conflict resolution strategies  
**Size Target**: 70 lines  
**Responsibility**: Conflict resolution logic

**Interface**:
```ada
procedure JSON_Conflict_Resolver is
   -- Reads conflict report from stdin (JSON)
   -- Outputs resolution to stdout
   -- Options:
   --   --strategy {last,first,manual,error}
end JSON_Conflict_Resolver;
```

---

## C++ Generation Utilities

### 8. `type_map_cpp.adb` - STUNIR → C++ Type Mapping

**Purpose**: Map STUNIR type names to C++ types  
**Size Target**: 50 lines  
**Responsibility**: Type mapping only

**Interface**:
```ada
procedure Type_Map_CPP is
   -- Reads STUNIR type from stdin or argument
   -- Outputs C++ type to stdout
   -- Mapping:
   --   i8→int8_t, i16→int16_t, i32→int32_t, i64→int64_t
   --   u8→uint8_t, u16→uint16_t, u32→uint32_t, u64→uint64_t
   --   f32→float, f64→double
   --   bool→bool, str→std::string, void→void
end Type_Map_CPP;
```

**Exit Codes**:
- 0: Success
- 1: Unknown type

**Usage Examples**:
```bash
echo "i32" | type_map_cpp
# Output: int32_t

echo "str" | type_map_cpp
# Output: std::string
```

---

### 9. `cpp_signature_gen.adb` - Generate C++ Signatures

**Purpose**: Generate C++ function signatures  
**Size Target**: 80 lines  
**Responsibility**: Signature generation only

**Interface**:
```ada
procedure CPP_Signature_Gen is
   -- Reads function signature JSON from stdin
   -- Outputs C++ signature to stdout
   -- Format: return_type function_name(param_list);
   -- Options:
   --   --inline      : Add inline keyword
   --   --static      : Add static keyword
   --   --const       : Add const qualifier
end CPP_Signature_Gen;
```

**Exit Codes**:
- 0: Success
- 1: Invalid signature JSON

**Input Format**:
```json
{
  "name": "add",
  "return_type": "i32",
  "args": [
    {"name": "a", "type": "i32"},
    {"name": "b", "type": "i32"}
  ]
}
```

**Output**:
```cpp
int32_t add(int32_t a, int32_t b);
```

---

### 10. `cpp_header_gen.adb` - Generate C++ Header Structure

**Purpose**: Generate C++ header guards and includes  
**Size Target**: 60 lines  
**Responsibility**: Header structure only

**Interface**:
```ada
procedure CPP_Header_Gen is
   -- Reads module metadata from stdin
   -- Outputs header structure to stdout
   -- Options:
   --   --guard NAME  : Custom header guard
   --   --pragma-once : Use #pragma once instead
end CPP_Header_Gen;
```

**Output Format**:
```cpp
#ifndef MODULE_NAME_H
#define MODULE_NAME_H

#include <cstdint>
#include <string>

// Content goes here

#endif // MODULE_NAME_H
```

---

### 11. `cpp_namespace_wrap.adb` - Wrap Code in Namespace

**Purpose**: Wrap C++ code in namespace declaration  
**Size Target**: 40 lines  
**Responsibility**: Namespace wrapping only

**Interface**:
```ada
procedure CPP_Namespace_Wrap is
   -- Reads code from stdin
   -- Outputs code wrapped in namespace to stdout
   -- Options:
   --   --namespace NAME : Namespace name (required)
   --   --nested         : Support nested namespaces (a::b::c)
end CPP_Namespace_Wrap;
```

**Usage Examples**:
```bash
echo "void foo();" | cpp_namespace_wrap --namespace mylib
# Output:
# namespace mylib {
#   void foo();
# }
```

---

## Validation Utilities

### 12. `schema_check_required.adb` - Check Required Fields

**Purpose**: Verify required fields exist in JSON  
**Size Target**: 60 lines  
**Responsibility**: Field existence checking only

**Interface**:
```ada
procedure Schema_Check_Required is
   -- Reads JSON from stdin
   -- Reads required fields list from --fields argument
   -- Outputs validation result to stdout
   -- Options:
   --   --fields ARRAY : JSON array of required field paths
end Schema_Check_Required;
```

**Exit Codes**:
- 0: All fields present
- 1: Missing required field(s)

**Usage Examples**:
```bash
echo '{"a":1,"b":2}' | schema_check_required --fields '["a","b"]'
# Output: VALID

echo '{"a":1}' | schema_check_required --fields '["a","b"]'
# Output: INVALID: Missing field 'b'
```

---

### 13. `schema_check_types.adb` - Validate Field Types

**Purpose**: Verify field types match schema  
**Size Target**: 80 lines  
**Responsibility**: Type validation only

**Interface**:
```ada
procedure Schema_Check_Types is
   -- Reads JSON from stdin
   -- Reads type schema from --schema argument
   -- Outputs validation result to stdout
   -- Options:
   --   --schema FILE : JSON file with type definitions
end Schema_Check_Types;
```

**Exit Codes**:
- 0: All types valid
- 1: Type mismatch

**Schema Format**:
```json
{
  "name": "string",
  "version": "string",
  "functions": "array"
}
```

---

### 14. `schema_check_format.adb` - Validate Format Rules

**Purpose**: Verify format constraints (regex, ranges, etc.)  
**Size Target**: 70 lines  
**Responsibility**: Format validation only

**Interface**:
```ada
procedure Schema_Check_Format is
   -- Reads JSON from stdin
   -- Reads format rules from --rules argument
   -- Outputs validation result to stdout
   -- Options:
   --   --rules FILE : JSON file with format rules
end Schema_Check_Format;
```

**Format Rules**:
```json
{
  "version": {"pattern": "^\\d+\\.\\d+\\.\\d+$"},
  "name": {"minLength": 1, "maxLength": 64}
}
```

---

### 15. `validation_reporter.adb` - Format Validation Reports

**Purpose**: Generate consistent validation reports  
**Size Target**: 50 lines  
**Responsibility**: Report formatting only

**Interface**:
```ada
procedure Validation_Reporter is
   -- Reads validation results from stdin (JSON array)
   -- Outputs formatted report to stdout
   -- Options:
   --   --format {text,json} : Report format
   --   --verbose            : Include details
end Validation_Reporter;
```

**Input Format**:
```json
[
  {"field": "schema", "status": "ok"},
  {"field": "module.name", "status": "error", "message": "required but missing"}
]
```

**Output (text)**:
```
INVALID: 1 error(s)
  ✗ module.name: required but missing
```

---

### 16-18. IR Validation Utilities

#### 16. `ir_check_required.adb` - Check Required IR Fields

Similar to schema_check_required but for IR-specific fields.

#### 17. `ir_check_functions.adb` - Validate IR Functions

**Purpose**: Validate IR function structures  
**Size Target**: 90 lines  
**Responsibility**: Function validation only

**Checks**:
- Function has name
- Function has return_type
- Function has args array
- Each arg has name and type
- Steps array is valid

#### 18. `ir_check_types.adb` - Validate IR Types

**Purpose**: Validate IR type definitions  
**Size Target**: 70 lines  
**Responsibility**: Type validation only

**Checks**:
- Type definitions are well-formed
- No circular dependencies
- All referenced types exist

---

## Type Utilities

### 19. `type_lookup.adb` - Look Up Type Definitions

**Purpose**: Find type definitions in type registry  
**Size Target**: 80 lines  
**Responsibility**: Type lookup only

**Interface**:
```ada
procedure Type_Lookup is
   -- Reads type name from stdin or argument
   -- Reads type registry from --registry file
   -- Outputs type definition to stdout
   -- Options:
   --   --registry FILE : JSON file with type definitions
end Type_Lookup;
```

---

### 20. `type_expand.adb` - Expand Complex Types

**Purpose**: Expand type aliases and composite types  
**Size Target**: 100 lines  
**Responsibility**: Type expansion only

**Interface**:
```ada
procedure Type_Expand is
   -- Reads type reference from stdin
   -- Reads type registry from --registry file
   -- Outputs expanded type to stdout
   -- Options:
   --   --registry FILE : JSON file with type definitions
   --   --recursive     : Recursively expand nested types
end Type_Expand;
```

**Example**:
```
Input: "MyList"
Registry: {"MyList": {"type": "array", "element": "i32"}}
Output: {"type": "array", "element": "i32"}
```

---

### 21. `type_dependency.adb` - Resolve Type Dependencies

**Purpose**: Determine type dependency order  
**Size Target**: 70 lines  
**Responsibility**: Dependency resolution only

**Interface**:
```ada
procedure Type_Dependency is
   -- Reads type definitions from stdin (JSON array)
   -- Outputs dependency-ordered types to stdout
   -- Options:
   --   --check-cycles : Error on circular dependencies
end Type_Dependency;
```

**Output**: Array of type names in dependency order (leaf types first)

---

## File Utilities

### 22. `file_writer.adb` - Write Content to File

**Purpose**: Write content to file with error handling  
**Size Target**: 50 lines  
**Responsibility**: File writing only

**Interface**:
```ada
procedure File_Writer is
   -- Reads content from stdin
   -- Writes to file specified in argument
   -- Options:
   --   --create-dirs : Create parent directories
   --   --append      : Append instead of overwrite
   --   --backup      : Create backup if file exists
end File_Writer;
```

**Exit Codes**:
- 0: Success
- 2: Write error
- 3: Permission denied

---

## Utility Generation Prompts

### Template for Local Model Generation

```
Generate an Ada SPARK 2014 utility named `<utility_name>` that <purpose>.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: <exit codes>
- Target size: <target lines> lines
- Single responsibility: <responsibility>

Behavior:
<behavior description>

Interface:
<interface specification>

Exit Codes:
<exit code details>

Usage Examples:
<examples>

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- <other dependencies>

Follow Unix philosophy: do one thing well, stdin/stdout, composable
Keep code minimal and focused on single task
```

---

## Integration Examples

### Example 1: json_read using utilities

```bash
# Old json_read (216 lines)
cat file.json | json_read

# New json_read (120 lines) using utilities
json_read() {
  local file="$1"
  cli_parser --help --version "$@" || return $?
  file_reader "$file" | json_validate
}
```

### Example 2: json_extract using utilities

```bash
# Old json_extract (410 lines)
cat spec.json | json_extract --path "functions.0.name"

# New json_extract (140 lines) using utilities
json_extract() {
  local path="$1"
  json_path_parser "$path" | read components
  cat | json_path_eval --path "$components" | json_value_format
}
```

### Example 3: sig_gen_cpp using utilities

```bash
# Old sig_gen_cpp (445 lines)
cat spec.json | sig_gen_cpp --namespace mylib

# New sig_gen_cpp (150 lines) using utilities
sig_gen_cpp() {
  local namespace="$1"
  spec_extract_funcs | while read func; do
    echo "$func" | cpp_signature_gen | type_map_cpp
  done | cpp_header_gen | cpp_namespace_wrap --namespace "$namespace"
}
```

---

## Next Steps

1. ✅ Review utility specifications
2. ⬜ Generate utility implementations using prompts
3. ⬜ Test each utility independently
4. ⬜ Refactor oversized tools to use utilities
5. ⬜ Integration testing
6. ⬜ Update documentation

---

**End of Utility Specifications**
