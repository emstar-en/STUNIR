# STUNIR Powertools: Sequential Generation Guide for Local Models

**Version**: 0.1.0-alpha  
**Purpose**: Step-by-step prompts for local models to generate each powertool  
**Format**: Each section = one prompt → one complete Ada SPARK powertool  
**Usage**: Feed prompts sequentially to local model (Claude, GPT, etc.)

---

## How to Use This Guide

1. **Select a tool** from the phase you're working on
2. **Copy the entire prompt** for that tool
3. **Feed it to your local model** (e.g., Claude, GPT-4, local LLM)
4. **Save the generated code** to `tools/spark/src/powertools/<tool_name>.adb`
5. **Build and test** using `gprbuild -P powertools.gpr <tool_name>.adb`
6. **Move to next tool** once current tool works

---

## Phase 1: Core JSON & File Tools (6 tools)

### Tool 1: `json_read` - Read and Validate JSON

```
Generate an Ada SPARK 2014 powertool named `json_read` that reads JSON from a file or stdin, validates it, and outputs to stdout.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success, 1=validation error
- Implement --help, --version, --describe flags
- Input: JSON file path (optional, default stdin)
- Output: Validated JSON to stdout
- Validation: Check JSON is well-formed (braces match, basic structure)

Behavior:
- If file path given as argument, read from that file
- If no arguments, read from stdin
- Validate JSON structure (starts with { or [, braces balanced)
- Output validated JSON to stdout unchanged
- Exit 0 if valid, exit 1 if invalid

--describe output:
{
  "tool": "json_read",
  "version": "0.1.0-alpha",
  "description": "Read and validate JSON from file or stdin",
  "inputs": [{"name": "json_file", "type": "file", "source": "argument", "required": false}],
  "outputs": [{"name": "json_content", "type": "json", "source": "stdout"}],
  "options": ["--help", "--version", "--describe"],
  "complexity": "O(n)",
  "pipeline_stage": "core"
}

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded

Target size: ~70 lines
Follow template from POWERTOOLS_SPEC_FOR_AI.md section "Common Ada SPARK Patterns"
Use Unix philosophy: do one thing well, stdin/stdout, composable
```

---

### Tool 2: `json_write` - Write JSON to File

```
Generate an Ada SPARK 2014 powertool named `json_write` that reads JSON from stdin and writes it to a file with optional pretty-printing.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success, 2=write failed
- Implement --help, --version, --describe flags
- Input: JSON from stdin
- Output: Written file at specified path
- Options: --pretty (indent with 2 spaces)

Behavior:
- Read JSON from stdin
- Accept output file path as first positional argument
- Write JSON to file
- If --pretty flag given, indent JSON with 2 spaces
- Exit 0 if written successfully, exit 2 on write failure

--describe output:
{
  "tool": "json_write",
  "version": "0.1.0-alpha",
  "description": "Write JSON from stdin to file",
  "inputs": [{"name": "json_content", "type": "json", "source": "stdin"}],
  "outputs": [{"name": "output_file", "type": "file", "source": "argument"}],
  "options": ["--help", "--version", "--describe", "--pretty"],
  "complexity": "O(n)",
  "pipeline_stage": "core"
}

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded

Target size: ~50 lines
Handle file creation errors gracefully
Use Ada.Text_IO.Create for file writing
```

---

### Tool 3: `file_find` - Find Files by Pattern

```
Generate an Ada SPARK 2014 powertool named `file_find` that recursively finds files matching a pattern in a directory.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success, 2=directory not found
- Implement --help, --version, --describe flags
- Input: Directory path and pattern (e.g., "*.json")
- Output: List of matching file paths, one per line to stdout

Behavior:
- Accept directory path as first argument
- Accept pattern as second argument (default: "*")
- Recursively search directory for matching files
- Output full path of each matching file to stdout (one per line)
- Exit 0 if search completes, exit 2 if directory doesn't exist

--describe output:
{
  "tool": "file_find",
  "version": "0.1.0-alpha",
  "description": "Find files matching pattern recursively",
  "inputs": [
    {"name": "directory", "type": "argument", "required": true},
    {"name": "pattern", "type": "argument", "required": false}
  ],
  "outputs": [{"name": "file_paths", "type": "text", "source": "stdout"}],
  "options": ["--help", "--version", "--describe"],
  "complexity": "O(n) where n is number of files",
  "pipeline_stage": "core"
}

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Directories

Target size: ~80 lines
Use Ada.Directories.Search for recursive directory traversal
Handle permission errors gracefully
```

---

### Tool 4: `file_hash` - Compute SHA-256 Hash

```
Generate an Ada SPARK 2014 powertool named `file_hash` that computes SHA-256 hash of a file.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success, 2=file not found
- Implement --help, --version, --describe flags
- Input: File path
- Output: SHA-256 hash in hexadecimal to stdout

Behavior:
- Accept file path as first argument
- Compute SHA-256 hash of file contents
- Output hash as lowercase hexadecimal string to stdout
- Exit 0 if successful, exit 2 if file not found

--describe output:
{
  "tool": "file_hash",
  "version": "0.1.0-alpha",
  "description": "Compute SHA-256 hash of file",
  "inputs": [{"name": "file_path", "type": "argument", "required": true}],
  "outputs": [{"name": "sha256_hash", "type": "text", "source": "stdout"}],
  "options": ["--help", "--version", "--describe"],
  "complexity": "O(n) where n is file size",
  "pipeline_stage": "core"
}

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Streams
- Ada.Streams.Stream_IO
- GNAT.SHA256

Target size: ~60 lines
Use streaming to handle large files
Read file in 8KB chunks
Output only the hash (no extra text)
```

---

### Tool 5: `toolchain_verify` - Verify Toolchain Lockfile

```
Generate an Ada SPARK 2014 powertool named `toolchain_verify` that verifies a toolchain lockfile exists and is valid.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=valid, 1=invalid/not found
- Implement --help, --version, --describe flags
- Input: Lockfile path (default: "toolchain.lock.json")
- Output: Validation status message to stdout

Behavior:
- Accept lockfile path as first argument (optional)
- Check if lockfile exists
- Check if lockfile is valid JSON
- Check if lockfile contains required fields: "version", "tools"
- Output "VALID" or "INVALID: <reason>" to stdout
- Exit 0 if valid, exit 1 if invalid

--describe output:
{
  "tool": "toolchain_verify",
  "version": "0.1.0-alpha",
  "description": "Verify toolchain lockfile exists and is valid",
  "inputs": [{"name": "lockfile_path", "type": "argument", "required": false}],
  "outputs": [{"name": "status", "type": "text", "source": "stdout"}],
  "options": ["--help", "--version", "--describe"],
  "complexity": "O(n) where n is lockfile size",
  "pipeline_stage": "core"
}

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Directories

Target size: ~50 lines
Minimal JSON parsing (just check for required fields)
Clear error messages
```

---

### Tool 6: `manifest_generate` - Generate File Manifest

```
Generate an Ada SPARK 2014 powertool named `manifest_generate` that generates a JSON manifest of files with their hashes.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success, 2=processing error
- Implement --help, --version, --describe flags
- Input: List of file paths from stdin (one per line)
- Output: JSON manifest to stdout

Behavior:
- Read file paths from stdin (one per line)
- For each file, compute SHA-256 hash
- Generate JSON array with entries: {"path": "<path>", "hash": "<sha256>", "size": <bytes>}
- Output JSON array to stdout
- Exit 0 if successful, exit 2 on errors

--describe output:
{
  "tool": "manifest_generate",
  "version": "0.1.0-alpha",
  "description": "Generate JSON manifest with file hashes",
  "inputs": [{"name": "file_paths", "type": "text", "source": "stdin"}],
  "outputs": [{"name": "manifest_json", "type": "json", "source": "stdout"}],
  "options": ["--help", "--version", "--describe"],
  "complexity": "O(n*m) where n is files, m is avg file size",
  "pipeline_stage": "core"
}

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Directories
- GNAT.SHA256

Target size: ~80 lines
Output valid JSON array format
Handle missing files gracefully (skip with warning to stderr)
```

---

## Phase 2: Spec Processing Tools (5 tools)

### Tool 7: `spec_extract_funcs` - Extract Functions from Spec

```
Generate an Ada SPARK 2014 powertool named `spec_extract_funcs` that extracts the functions array from a STUNIR spec JSON.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success, 1=invalid spec
- Implement --help, --version, --describe flags
- Input: Spec JSON from stdin
- Output: Functions JSON array to stdout

Behavior:
- Read spec JSON from stdin
- Extract the "module.functions" array from spec
- Output functions array as JSON to stdout
- If no functions found, output empty array []
- Exit 0 if successful, exit 1 if spec is invalid JSON

Spec format example:
{
  "module": {
    "name": "example",
    "functions": [
      {"name": "add", "params": [...], "returns": "i32"}
    ]
  }
}

--describe output:
{
  "tool": "spec_extract_funcs",
  "version": "0.1.0-alpha",
  "description": "Extract functions array from STUNIR spec JSON",
  "inputs": [{"name": "spec_json", "type": "json", "source": "stdin"}],
  "outputs": [{"name": "functions_array", "type": "json", "source": "stdout"}],
  "options": ["--help", "--version", "--describe"],
  "complexity": "O(n) where n is spec size",
  "dependencies": ["json_read"],
  "pipeline_stage": "spec"
}

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded

Target size: ~100 lines
Use basic string searching to find "functions": [ ... ] in JSON
Output only the array contents (not the key)
```

---

### Tool 8: `spec_extract_types` - Extract Types from Spec

```
Generate an Ada SPARK 2014 powertool named `spec_extract_types` that extracts type definitions from a STUNIR spec JSON.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success, 1=invalid spec
- Implement --help, --version, --describe flags
- Input: Spec JSON from stdin
- Output: Types JSON array to stdout

Behavior:
- Read spec JSON from stdin
- Extract the "module.types" array from spec (if present)
- Output types array as JSON to stdout
- If no types found, output empty array []
- Exit 0 if successful, exit 1 if spec is invalid JSON

--describe output:
{
  "tool": "spec_extract_types",
  "version": "0.1.0-alpha",
  "description": "Extract types array from STUNIR spec JSON",
  "inputs": [{"name": "spec_json", "type": "json", "source": "stdin"}],
  "outputs": [{"name": "types_array", "type": "json", "source": "stdout"}],
  "options": ["--help", "--version", "--describe"],
  "complexity": "O(n) where n is spec size",
  "dependencies": ["json_read"],
  "pipeline_stage": "spec"
}

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded

Target size: ~80 lines
Similar logic to spec_extract_funcs but for "types" field
```

---

### Tool 9: `spec_extract_module` - Extract Module Metadata

```
Generate an Ada SPARK 2014 powertool named `spec_extract_module` that extracts module metadata from a STUNIR spec JSON.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success, 1=invalid spec
- Implement --help, --version, --describe flags
- Input: Spec JSON from stdin
- Output: Module metadata JSON to stdout

Behavior:
- Read spec JSON from stdin
- Extract module metadata: name, version, description
- Output as JSON object to stdout
- Exit 0 if successful, exit 1 if spec is invalid JSON

Output format:
{
  "name": "module_name",
  "version": "1.0.0",
  "description": "Module description"
}

--describe output:
{
  "tool": "spec_extract_module",
  "version": "0.1.0-alpha",
  "description": "Extract module metadata from STUNIR spec JSON",
  "inputs": [{"name": "spec_json", "type": "json", "source": "stdin"}],
  "outputs": [{"name": "module_metadata", "type": "json", "source": "stdout"}],
  "options": ["--help", "--version", "--describe"],
  "complexity": "O(n) where n is spec size",
  "dependencies": ["json_read"],
  "pipeline_stage": "spec"
}

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded

Target size: ~60 lines
Extract string values for "module.name", "module.version", "module.description"
```

---

### Tool 10: `spec_validate_schema` - Validate Spec Schema

```
Generate an Ada SPARK 2014 powertool named `spec_validate_schema` that validates a STUNIR spec JSON against basic schema requirements.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=valid, 1=invalid
- Implement --help, --version, --describe flags
- Input: Spec JSON from stdin
- Output: Validation report to stdout

Behavior:
- Read spec JSON from stdin
- Check required fields exist: "schema", "module", "module.name"
- Check schema field = "stunir.spec.v1"
- Check module has "functions" array
- Output "VALID" or "INVALID: <reason>" to stdout
- Exit 0 if valid, exit 1 if invalid

--describe output:
{
  "tool": "spec_validate_schema",
  "version": "0.1.0-alpha",
  "description": "Validate STUNIR spec JSON schema",
  "inputs": [{"name": "spec_json", "type": "json", "source": "stdin"}],
  "outputs": [{"name": "validation_report", "type": "text", "source": "stdout"}],
  "options": ["--help", "--version", "--describe"],
  "complexity": "O(n) where n is spec size",
  "dependencies": ["json_read"],
  "pipeline_stage": "spec"
}

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded

Target size: ~100 lines
Check for presence of required JSON fields
Clear validation error messages
```

---

### Tool 11: `type_normalize` - Normalize Type Names

```
Generate an Ada SPARK 2014 powertool named `type_normalize` that normalizes type names to canonical form.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success
- Implement --help, --version, --describe flags
- Input: Type name from stdin or argument
- Output: Normalized type name to stdout

Behavior:
- Accept type name from first argument or stdin
- Normalize type name using these rules:
  - "int" → "i32"
  - "uint" → "u32"
  - "long" → "i64"
  - "short" → "i16"
  - "byte" → "i8"
  - "char" → "u8"
  - "float" → "f32"
  - "double" → "f64"
  - "bool" → "bool"
  - "string" → "str"
  - Everything else → unchanged
- Output normalized type name to stdout
- Exit 0

--describe output:
{
  "tool": "type_normalize",
  "version": "0.1.0-alpha",
  "description": "Normalize type names to canonical form",
  "inputs": [{"name": "type_name", "type": "text", "source": "stdin"}],
  "outputs": [{"name": "normalized_type", "type": "text", "source": "stdout"}],
  "options": ["--help", "--version", "--describe"],
  "complexity": "O(1)",
  "pipeline_stage": "spec"
}

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded
- Ada.Characters.Handling (for To_Lower)

Target size: ~120 lines
Case-insensitive matching
Output only the normalized type (no newline if piping)
```

---

## Phase 3: IR Generation Tools (4 tools)

### Tool 12: `func_to_ir` - Convert Function Spec to IR

```
Generate an Ada SPARK 2014 powertool named `func_to_ir` that converts a function spec JSON to IR format.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success, 1=invalid function
- Implement --help, --version, --describe flags
- Input: Function spec JSON from stdin
- Output: IR function JSON to stdout

Behavior:
- Read function spec JSON from stdin
- Convert to IR format with fields:
  - "name": function name
  - "return_type": normalized return type
  - "args": array of {name, type} objects
  - "steps": empty array (body to be filled later)
  - "is_public": true
- Output IR function JSON to stdout
- Exit 0 if successful, exit 1 if invalid

Input format (spec):
{
  "name": "add",
  "params": [{"name": "a", "type": "int"}, {"name": "b", "type": "int"}],
  "returns": "int"
}

Output format (IR):
{
  "name": "add",
  "return_type": "i32",
  "args": [{"name": "a", "type": "i32"}, {"name": "b", "type": "i32"}],
  "steps": [],
  "is_public": true
}

--describe output:
{
  "tool": "func_to_ir",
  "version": "0.1.0-alpha",
  "description": "Convert function spec to IR format",
  "inputs": [{"name": "function_spec", "type": "json", "source": "stdin"}],
  "outputs": [{"name": "ir_function", "type": "json", "source": "stdout"}],
  "options": ["--help", "--version", "--describe"],
  "complexity": "O(n) where n is number of params",
  "dependencies": ["type_normalize"],
  "pipeline_stage": "ir_gen"
}

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded

Target size: ~130 lines
Normalize all type names using same rules as type_normalize
Handle missing fields gracefully (use defaults)
```

---

### Tool 13: `module_to_ir` - Convert Module Spec to IR

```
Generate an Ada SPARK 2014 powertool named `module_to_ir` that converts module metadata to IR format.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success, 1=invalid module
- Implement --help, --version, --describe flags
- Input: Module metadata JSON from stdin
- Output: IR module header JSON to stdout

Behavior:
- Read module metadata JSON from stdin
- Convert to IR format with fields:
  - "schema": "stunir_flat_ir_v1"
  - "ir_version": "0.1.0"
  - "module_name": from input
  - "description": from input (or "")
- Output IR module header JSON to stdout
- Exit 0 if successful, exit 1 if invalid

--describe output:
{
  "tool": "module_to_ir",
  "version": "0.1.0-alpha",
  "description": "Convert module metadata to IR format",
  "inputs": [{"name": "module_metadata", "type": "json", "source": "stdin"}],
  "outputs": [{"name": "ir_module_header", "type": "json", "source": "stdout"}],
  "options": ["--help", "--version", "--describe"],
  "complexity": "O(1)",
  "pipeline_stage": "ir_gen"
}

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded

Target size: ~90 lines
Simple JSON transformation
```

---

### Tool 14: `ir_merge_funcs` - Merge IR Function Arrays

```
Generate an Ada SPARK 2014 powertool named `ir_merge_funcs` that merges multiple IR function JSON objects into a single array.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success
- Implement --help, --version, --describe flags
- Input: Multiple IR function JSON objects from stdin (one per line or concatenated)
- Output: JSON array of all functions to stdout

Behavior:
- Read IR function JSON objects from stdin
- Each line or separate JSON object is one function
- Collect all functions into an array
- Output as JSON array to stdout
- Exit 0

Input (multiple JSON objects):
{"name":"add",...}
{"name":"sub",...}

Output (JSON array):
[
  {"name":"add",...},
  {"name":"sub",...}
]

--describe output:
{
  "tool": "ir_merge_funcs",
  "version": "0.1.0-alpha",
  "description": "Merge multiple IR functions into array",
  "inputs": [{"name": "ir_functions", "type": "json", "source": "stdin"}],
  "outputs": [{"name": "functions_array", "type": "json", "source": "stdout"}],
  "options": ["--help", "--version", "--describe"],
  "complexity": "O(n) where n is number of functions",
  "pipeline_stage": "ir_gen"
}

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded

Target size: ~110 lines
Handle both line-delimited JSON and concatenated JSON objects
Output valid JSON array
```

---

### Tool 15: `ir_add_metadata` - Add IR Metadata

```
Generate an Ada SPARK 2014 powertool named `ir_add_metadata` that adds metadata fields to IR JSON.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success
- Implement --help, --version, --describe flags
- Input: Partial IR JSON from stdin
- Output: Complete IR JSON with metadata to stdout
- Options: --schema (default: stunir_flat_ir_v1), --version (default: 0.1.0)

Behavior:
- Read partial IR JSON from stdin
- Add metadata fields:
  - "schema": from --schema flag or default
  - "ir_version": from --version flag or default
  - "generated_at": current timestamp (ISO 8601)
- Merge with existing JSON
- Output complete IR JSON to stdout
- Exit 0

--describe output:
{
  "tool": "ir_add_metadata",
  "version": "0.1.0-alpha",
  "description": "Add metadata to IR JSON",
  "inputs": [{"name": "partial_ir", "type": "json", "source": "stdin"}],
  "outputs": [{"name": "complete_ir", "type": "json", "source": "stdout"}],
  "options": ["--help", "--version", "--describe", "--schema", "--ir-version"],
  "complexity": "O(n) where n is IR size",
  "pipeline_stage": "ir_gen"
}

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded
- Ada.Calendar
- Ada.Calendar.Formatting

Target size: ~70 lines
Add timestamp in ISO 8601 format
Preserve existing JSON structure
```

---

## Phase 4: IR Processing Tools (5 tools)

### Tool 16: `ir_validate_schema` - Validate IR Schema

```
Generate an Ada SPARK 2014 powertool named `ir_validate_schema` that validates IR JSON against schema requirements.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=valid, 1=invalid
- Implement --help, --version, --describe flags
- Input: IR JSON from stdin
- Output: Validation report to stdout

Behavior:
- Read IR JSON from stdin
- Check required fields: "schema", "ir_version", "module_name", "functions"
- Check schema = "stunir_flat_ir_v1"
- Check functions is an array
- Output "VALID" or "INVALID: <reason>" to stdout
- Exit 0 if valid, exit 1 if invalid

--describe output:
{
  "tool": "ir_validate_schema",
  "version": "0.1.0-alpha",
  "description": "Validate IR JSON schema",
  "inputs": [{"name": "ir_json", "type": "json", "source": "stdin"}],
  "outputs": [{"name": "validation_report", "type": "text", "source": "stdout"}],
  "options": ["--help", "--version", "--describe"],
  "complexity": "O(n) where n is IR size",
  "pipeline_stage": "ir_process"
}

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded

Target size: ~90 lines
Check for required fields
Clear error messages
```

---

### Tool 17: `ir_extract_module` - Extract Module from IR

```
Generate an Ada SPARK 2014 powertool named `ir_extract_module` that extracts module metadata from IR JSON.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success, 1=invalid IR
- Implement --help, --version, --describe flags
- Input: IR JSON from stdin
- Output: Module metadata JSON to stdout

Behavior:
- Read IR JSON from stdin
- Extract module metadata: module_name, description (if present)
- Output as JSON object to stdout
- Exit 0 if successful, exit 1 if IR is invalid

Output format:
{
  "module_name": "example",
  "description": "Example module"
}

--describe output:
{
  "tool": "ir_extract_module",
  "version": "0.1.0-alpha",
  "description": "Extract module metadata from IR JSON",
  "inputs": [{"name": "ir_json", "type": "json", "source": "stdin"}],
  "outputs": [{"name": "module_metadata", "type": "json", "source": "stdout"}],
  "options": ["--help", "--version", "--describe"],
  "complexity": "O(n) where n is IR size",
  "pipeline_stage": "ir_process"
}

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded

Target size: ~70 lines
Similar to spec_extract_module but for IR format
```

---

### Tool 18: `ir_extract_funcs` - Extract Functions from IR

```
Generate an Ada SPARK 2014 powertool named `ir_extract_funcs` that extracts the functions array from IR JSON.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success, 1=invalid IR
- Implement --help, --version, --describe flags
- Input: IR JSON from stdin
- Output: Functions array JSON to stdout

Behavior:
- Read IR JSON from stdin
- Extract "functions" array
- Output functions array to stdout
- Exit 0 if successful, exit 1 if IR is invalid

--describe output:
{
  "tool": "ir_extract_funcs",
  "version": "0.1.0-alpha",
  "description": "Extract functions array from IR JSON",
  "inputs": [{"name": "ir_json", "type": "json", "source": "stdin"}],
  "outputs": [{"name": "functions_array", "type": "json", "source": "stdout"}],
  "options": ["--help", "--version", "--describe"],
  "complexity": "O(n) where n is IR size",
  "pipeline_stage": "ir_process"
}

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded

Target size: ~80 lines
Find "functions": [...] in IR JSON
Output array only
```

---

### Tool 19: `func_parse_sig` - Parse Function Signature

```
Generate an Ada SPARK 2014 powertool named `func_parse_sig` that parses function signature from IR function JSON.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success, 1=invalid function
- Implement --help, --version, --describe flags
- Input: IR function JSON from stdin
- Output: Function signature JSON to stdout

Behavior:
- Read IR function JSON from stdin
- Extract signature components: name, return_type, args (not steps)
- Output simplified signature JSON to stdout
- Exit 0 if successful, exit 1 if invalid

Output format:
{
  "name": "add",
  "return_type": "i32",
  "args": [{"name": "a", "type": "i32"}, {"name": "b", "type": "i32"}]
}

--describe output:
{
  "tool": "func_parse_sig",
  "version": "0.1.0-alpha",
  "description": "Parse function signature from IR function",
  "inputs": [{"name": "ir_function", "type": "json", "source": "stdin"}],
  "outputs": [{"name": "function_signature", "type": "json", "source": "stdout"}],
  "options": ["--help", "--version", "--describe"],
  "complexity": "O(n) where n is number of args",
  "pipeline_stage": "ir_process"
}

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded

Target size: ~90 lines
Extract name, return_type, args fields
Ignore steps/body
```

---

### Tool 20: `func_parse_body` - Parse Function Body

```
Generate an Ada SPARK 2014 powertool named `func_parse_body` that parses function body (steps) from IR function JSON.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success, 1=invalid function
- Implement --help, --version, --describe flags
- Input: IR function JSON from stdin
- Output: Steps array JSON to stdout

Behavior:
- Read IR function JSON from stdin
- Extract "steps" array
- Output steps array to stdout
- If no steps, output empty array []
- Exit 0 if successful, exit 1 if invalid

--describe output:
{
  "tool": "func_parse_body",
  "version": "0.1.0-alpha",
  "description": "Parse function body steps from IR function",
  "inputs": [{"name": "ir_function", "type": "json", "source": "stdin"}],
  "outputs": [{"name": "steps_array", "type": "json", "source": "stdout"}],
  "options": ["--help", "--version", "--describe"],
  "complexity": "O(n) where n is number of steps",
  "pipeline_stage": "ir_process"
}

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded

Target size: ~100 lines
Extract "steps" array from function
Output array or empty [] if missing
```

---

## Phase 5: Code Generation Tools (7 tools)

### Tool 21: `type_map_target` - Map Type to Target Language

```
Generate an Ada SPARK 2014 powertool named `type_map_target` that maps STUNIR type names to target language types.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success
- Implement --help, --version, --describe flags
- Input: Type name from stdin
- Options: --target LANG (required: rust|c|python|go)
- Output: Target language type to stdout

Behavior:
- Accept target language via --target flag
- Read type name from stdin
- Map to target language type using these rules:

C mappings:
- i8→int8_t, i16→int16_t, i32→int32_t, i64→int64_t
- u8→uint8_t, u16→uint16_t, u32→uint32_t, u64→uint64_t
- f32→float, f64→double
- bool→bool, str→const char*, void→void

Rust mappings:
- i8→i8, i16→i16, i32→i32, i64→i64
- u8→u8, u16→u16, u32→u32, u64→u64
- f32→f32, f64→f64
- bool→bool, str→&str, void→()

Python mappings:
- All integers→int, floats→float, bool→bool, str→str, void→None

- Output target type to stdout
- Exit 0

--describe output:
{
  "tool": "type_map_target",
  "version": "0.1.0-alpha",
  "description": "Map STUNIR type to target language type",
  "inputs": [{"name": "type_name", "type": "text", "source": "stdin"}],
  "outputs": [{"name": "target_type", "type": "text", "source": "stdout"}],
  "options": ["--help", "--version", "--describe", "--target"],
  "complexity": "O(1)",
  "pipeline_stage": "code_gen"
}

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded
- Ada.Characters.Handling

Target size: ~120 lines
Support rust, c, python, go
Clear error if unsupported target
```

---

### Tool 22: `code_gen_preamble` - Generate Code Preamble

```
Generate an Ada SPARK 2014 powertool named `code_gen_preamble` that generates language-specific code preamble (imports, headers).

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success
- Implement --help, --version, --describe flags
- Input: Module metadata JSON from stdin
- Options: --target LANG (required)
- Output: Code preamble to stdout

Behavior:
- Accept target language via --target flag
- Read module metadata from stdin
- Generate appropriate preamble for target:

Rust preamble:
```
// Module: <module_name>
// Generated by STUNIR

Python preamble:
```
#!/usr/bin/env python3
# Module: <module_name>
# Generated by STUNIR

C preamble:
```
#include <stdint.h>
#include <stdbool.h>

/* Module: <module_name> */
/* Generated by STUNIR */

- Output preamble to stdout
- Exit 0

--describe output:
{
  "tool": "code_gen_preamble",
  "version": "0.1.0-alpha",
  "description": "Generate code preamble for target language",
  "inputs": [{"name": "module_metadata", "type": "json", "source": "stdin"}],
  "outputs": [{"name": "code_preamble", "type": "text", "source": "stdout"}],
  "options": ["--help", "--version", "--describe", "--target"],
  "complexity": "O(1)",
  "pipeline_stage": "code_gen"
}

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded

Target size: ~110 lines
Template-based generation
Support rust, c, python
```

---

### Tool 23: `code_gen_func_sig` - Generate Function Signature

```
Generate an Ada SPARK 2014 powertool named `code_gen_func_sig` that generates function signature code for target language.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success
- Implement --help, --version, --describe flags
- Input: Function signature JSON from stdin
- Options: --target LANG (required)
- Output: Function signature code to stdout

Behavior:
- Accept target language via --target flag
- Read function signature JSON from stdin
- Generate function signature for target:

Rust example:
pub fn add(a: i32, b: i32) -> i32

C example:
int32_t add(int32_t a, int32_t b);

Python example:
def add(a: int, b: int) -> int:

- Use type_map_target logic for type mapping
- Output signature code to stdout
- Exit 0

--describe output:
{
  "tool": "code_gen_func_sig",
  "version": "0.1.0-alpha",
  "description": "Generate function signature for target language",
  "inputs": [{"name": "function_signature", "type": "json", "source": "stdin"}],
  "outputs": [{"name": "signature_code", "type": "text", "source": "stdout"}],
  "options": ["--help", "--version", "--describe", "--target"],
  "complexity": "O(n) where n is number of params",
  "dependencies": ["type_map_target"],
  "pipeline_stage": "code_gen"
}

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded

Target size: ~130 lines
Template-based generation
Support rust, c, python
Map all parameter types
```

---

### Tool 24: `code_gen_func_body` - Generate Function Body

```
Generate an Ada SPARK 2014 powertool named `code_gen_func_body` that generates function body code from IR steps.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success
- Implement --help, --version, --describe flags
- Input: Steps array JSON from stdin
- Options: --target LANG (required)
- Output: Function body code to stdout

Behavior:
- Accept target language via --target flag
- Read steps array JSON from stdin
- Generate code for each step based on op type:
  - "assign": variable = value
  - "return": return value
  - "call": function_name(args)
  - "if": if condition { ... }
  - "nop": // no-op comment
- Output function body code to stdout
- Exit 0

For empty steps, generate stub:
Rust: unimplemented!()
C: return 0; (or appropriate default)
Python: pass

--describe output:
{
  "tool": "code_gen_func_body",
  "version": "0.1.0-alpha",
  "description": "Generate function body from IR steps",
  "inputs": [{"name": "steps_array", "type": "json", "source": "stdin"}],
  "outputs": [{"name": "body_code", "type": "text", "source": "stdout"}],
  "options": ["--help", "--version", "--describe", "--target"],
  "complexity": "O(n) where n is number of steps",
  "pipeline_stage": "code_gen"
}

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded

Target size: ~200 lines (largest tool due to IR op handling)
Support basic ops: assign, return, call, if, nop
Generate stub if no steps
Proper indentation
```

---

### Tool 25: `code_add_comments` - Add Comments to Code

```
Generate an Ada SPARK 2014 powertool named `code_add_comments` that adds metadata comments to generated code.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success
- Implement --help, --version, --describe flags
- Input: Code from stdin
- Options: --metadata FILE (IR JSON file for metadata)
- Output: Code with comments to stdout

Behavior:
- Read code from stdin
- Read metadata from --metadata file if provided
- Add header comment with:
  - Generated timestamp
  - STUNIR version
  - Source IR version
- Prepend comment to code
- Output commented code to stdout
- Exit 0

Header comment format:
// Generated by STUNIR v0.1.0-alpha
// Timestamp: 2026-02-17T12:00:00Z
// IR Version: 0.1.0

--describe output:
{
  "tool": "code_add_comments",
  "version": "0.1.0-alpha",
  "description": "Add metadata comments to generated code",
  "inputs": [{"name": "code", "type": "text", "source": "stdin"}],
  "outputs": [{"name": "commented_code", "type": "text", "source": "stdout"}],
  "options": ["--help", "--version", "--describe", "--metadata"],
  "complexity": "O(n) where n is code size",
  "pipeline_stage": "code_gen"
}

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded
- Ada.Calendar
- Ada.Calendar.Formatting

Target size: ~60 lines
Read entire code from stdin
Prepend header comment
```

---

### Tool 26: `code_format_target` - Format Code

```
Generate an Ada SPARK 2014 powertool named `code_format_target` that formats code using language-specific formatters.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success, 2=formatter failed
- Implement --help, --version, --describe flags
- Input: Code from stdin
- Options: --target LANG (required)
- Output: Formatted code to stdout

Behavior:
- Accept target language via --target flag
- Read code from stdin
- Invoke appropriate formatter:
  - Rust: rustfmt (if available, otherwise pass through)
  - C: clang-format (if available, otherwise pass through)
  - Python: black or autopep8 (if available, otherwise pass through)
- Output formatted code to stdout
- If formatter not available, output code unchanged with warning to stderr
- Exit 0 if successful or pass-through, exit 2 if formatter error

--describe output:
{
  "tool": "code_format_target",
  "version": "0.1.0-alpha",
  "description": "Format code using language formatter",
  "inputs": [{"name": "code", "type": "text", "source": "stdin"}],
  "outputs": [{"name": "formatted_code", "type": "text", "source": "stdout"}],
  "options": ["--help", "--version", "--describe", "--target"],
  "complexity": "O(n) where n is code size",
  "pipeline_stage": "code_gen"
}

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- GNAT.OS_Lib (for spawning formatters)

Target size: ~80 lines
Check if formatter exists before invoking
Graceful fallback if formatter not available
Pipe code through formatter
```

---

### Tool 27: `code_write` - Write Code to File

```
Generate an Ada SPARK 2014 powertool named `code_write` that writes code from stdin to a file.

Requirements:
- Version: 0.1.0-alpha
- Exit codes: 0=success, 2=write failed
- Implement --help, --version, --describe flags
- Input: Code from stdin
- Output: Written file at specified path

Behavior:
- Accept file path as first positional argument
- Read code from stdin
- Write code to file
- Create parent directories if they don't exist
- Exit 0 if written successfully, exit 2 on write failure

--describe output:
{
  "tool": "code_write",
  "version": "0.1.0-alpha",
  "description": "Write code from stdin to file",
  "inputs": [{"name": "code", "type": "text", "source": "stdin"}],
  "outputs": [{"name": "output_file", "type": "file", "source": "argument"}],
  "options": ["--help", "--version", "--describe"],
  "complexity": "O(n) where n is code size",
  "pipeline_stage": "code_gen"
}

Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Directories

Target size: ~50 lines
Create directories if needed
Handle write errors gracefully
```

---

## Building and Testing Each Tool

After generating each tool, follow these steps:

### 1. Save the Generated Code
```bash
# Save to powertools directory
vim tools/spark/src/powertools/<tool_name>.adb
# Paste generated code
```

### 2. Build the Tool
```bash
cd tools/spark
gprbuild -P stunir_tools.gpr <tool_name>.adb
```

### 3. Test the Tool
```bash
# Test --help
./bin/<tool_name> --help

# Test --describe
./bin/<tool_name> --describe

# Test functionality (example for json_read)
echo '{"test": "value"}' | ./bin/json_read
```

### 4. Verify in Pipeline
```bash
# Test with real data
cat spec/examples/example.json | ./bin/json_read | ./bin/spec_extract_funcs
```

---

## Progress Tracking

Use this checklist to track progress:

**Phase 1: Core JSON & File Tools**
- [ ] json_read
- [ ] json_write
- [ ] file_find
- [ ] file_hash
- [ ] toolchain_verify
- [ ] manifest_generate

**Phase 2: Spec Processing Tools**
- [ ] spec_extract_funcs
- [ ] spec_extract_types
- [ ] spec_extract_module
- [ ] spec_validate_schema
- [ ] type_normalize

**Phase 3: IR Generation Tools**
- [ ] func_to_ir
- [ ] module_to_ir
- [ ] ir_merge_funcs
- [ ] ir_add_metadata

**Phase 4: IR Processing Tools**
- [ ] ir_validate_schema
- [ ] ir_extract_module
- [ ] ir_extract_funcs
- [ ] func_parse_sig
- [ ] func_parse_body

**Phase 5: Code Generation Tools**
- [ ] type_map_target
- [ ] code_gen_preamble
- [ ] code_gen_func_sig
- [ ] code_gen_func_body
- [ ] code_add_comments
- [ ] code_format_target
- [ ] code_write

---

## Tips for Local Model Generation

1. **One tool at a time** - Don't try to generate multiple tools in one prompt
2. **Review generated code** - Check for Ada syntax correctness
3. **Test immediately** - Compile and test before moving to next tool
4. **Fix compilation errors** - If gprbuild fails, iterate with the model
5. **Keep it simple** - If generated code is too complex, ask for simpler version
6. **Follow templates** - Refer back to POWERTOOLS_SPEC_FOR_AI.md for patterns
7. **Document issues** - Note any recurring problems for future refinement

---

## Example Workflow

```bash
# Step 1: Generate json_read
# Copy prompt #1 above, feed to model, save output

# Step 2: Save generated code
vim tools/spark/src/powertools/json_read.adb
# Paste model output

# Step 3: Build
cd tools/spark
gprbuild -P stunir_tools.gpr json_read.adb

# Step 4: Test
echo '{"test": "value"}' | ./bin/json_read
# Should output: {"test": "value"}

# Step 5: Move to next tool
# Repeat with prompt #2 (json_write)
```

---

**End of Sequential Generation Guide**
