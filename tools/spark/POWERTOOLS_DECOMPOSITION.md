# STUNIR Pipeline Decomposition: Monolithic → Unix Philosophy Powertools

**Version**: 0.1.0-alpha  
**Purpose**: Break down monolithic `spec_to_ir` and `ir_to_code` tools into small, composable powertools  
**Philosophy**: Many simple, robust tools that do one thing well > Few complex tools  
**Benefit**: Easier for AI to generate, test, and orchestrate

---

## Current Monolithic Pipeline

```
spec_to_ir (475 lines)          ir_to_code (1971 lines)
    ↓                                    ↓
[EVERYTHING]                       [EVERYTHING]
```

**Problems**:
- Hard to test individual components
- Difficult for local models to understand/modify (too much context)
- Failures in one step break entire pipeline
- No reusability across different workflows
- Heavy, complex codebases

---

## Decomposed Unix-Philosophy Pipeline

```
Spec JSON → [small tools] → IR JSON → [small tools] → Code
            ↓ ↓ ↓ ↓ ↓              ↓ ↓ ↓ ↓ ↓
         15 focused tools       12 focused tools
```

**Benefits**:
- Each tool <150 lines, easy to understand
- Local models can generate one tool at a time
- Test each tool independently
- AI can discover and compose pipelines dynamically
- Reusable across different workflows
- Robust error handling per-tool

---

## Decomposition: `spec_to_ir` → 15 Powertools

### Current `spec_to_ir` Does:
1. Verify toolchain lockfile
2. Find spec files recursively
3. Compute SHA-256 hashes
4. Parse JSON spec files
5. Extract functions from spec
6. Extract types from spec
7. Extract modules from spec
8. Normalize type names
9. Validate spec schema
10. Generate semantic IR
11. Merge multiple specs
12. Create manifest entries
13. Write IR JSON to file

### Decomposed Powertools:

| # | Powertool | Purpose | Input | Output | Lines |
|---|-----------|---------|-------|--------|-------|
| 1 | `toolchain_verify` | Verify toolchain lockfile exists and is valid | `toolchain.lock` | Exit code 0/1 | ~50 |
| 2 | `file_find` | Find files matching pattern recursively | Dir path, pattern (*.json) | List of file paths (stdout) | ~80 |
| 3 | `file_hash` | Compute SHA-256 hash of single file | File path | Hash string (stdout) | ~60 |
| 4 | `json_read` | Read and validate JSON file | JSON file | Validated JSON (stdout) | ~70 |
| 5 | `spec_extract_funcs` | Extract functions array from spec JSON | Spec JSON (stdin) | Functions JSON array (stdout) | ~100 |
| 6 | `spec_extract_types` | Extract types from spec JSON | Spec JSON (stdin) | Types JSON array (stdout) | ~80 |
| 7 | `spec_extract_module` | Extract module metadata | Spec JSON (stdin) | Module metadata JSON (stdout) | ~60 |
| 8 | `type_normalize` | Normalize type names (i32→int, etc.) | Type name (stdin) | Normalized type (stdout) | ~120 |
| 9 | `spec_validate_schema` | Validate spec against schema | Spec JSON (stdin) | Exit code 0/1 + report | ~100 |
| 10 | `func_to_ir` | Convert function spec to IR format | Function JSON (stdin) | IR function (stdout) | ~130 |
| 11 | `module_to_ir` | Convert module spec to IR format | Module JSON (stdin) | IR module (stdout) | ~90 |
| 12 | `ir_merge_funcs` | Merge multiple IR function arrays | IR functions (stdin, multiple) | Merged IR (stdout) | ~110 |
| 13 | `ir_add_metadata` | Add metadata to IR (version, schema, timestamp) | IR JSON (stdin) | IR with metadata (stdout) | ~70 |
| 14 | `manifest_generate` | Generate file manifest with hashes | File list + hashes (stdin) | Manifest JSON (stdout) | ~80 |
| 15 | `json_write` | Pretty-print and write JSON to file | JSON (stdin), output path | Written file | ~50 |

**Total**: ~1,330 lines across 15 tools vs. 475 lines in monolithic tool  
**Benefit**: Each tool is simple, focused, and independently testable

---

## Decomposition: `ir_to_code` → 12 Powertools

### Current `ir_to_code` Does:
1. Parse IR JSON
2. Validate IR schema
3. Extract module metadata
4. Extract functions array
5. Parse function signatures
6. Parse function parameters
7. Parse function steps (operations)
8. Map types to target language
9. Generate code preamble (imports, headers)
10. Generate function signatures
11. Generate function bodies
12. Emit comments and metadata
13. Write code to file

### Decomposed Powertools:

| # | Powertool | Purpose | Input | Output | Lines |
|---|-----------|---------|-------|--------|-------|
| 1 | `ir_validate_schema` | Validate IR JSON schema | IR JSON (stdin) | Exit code 0/1 + report | ~90 |
| 2 | `ir_extract_module` | Extract module metadata from IR | IR JSON (stdin) | Module metadata JSON (stdout) | ~70 |
| 3 | `ir_extract_funcs` | Extract functions array from IR | IR JSON (stdin) | Functions JSON array (stdout) | ~80 |
| 4 | `func_parse_sig` | Parse function signature (name, return, args) | Function JSON (stdin) | Signature JSON (stdout) | ~90 |
| 5 | `func_parse_body` | Parse function body (steps/operations) | Function JSON (stdin) | Steps JSON array (stdout) | ~100 |
| 6 | `type_map_target` | Map type to target language | Type name + target lang | Target type (stdout) | ~120 |
| 7 | `code_gen_preamble` | Generate language preamble (imports, headers) | Module meta + target | Preamble code (stdout) | ~110 (emitters) |
| 8 | `code_gen_func_sig` | Generate function signature for target | Func sig JSON + target | Function signature code (stdout) | ~130 (emitters) |
| 9 | `code_gen_func_body` | Generate function body for target | Steps JSON + target | Function body code (stdout) | ~200 (emitters) |
| 10 | `code_add_comments` | Add comments to generated code | Code (stdin) + metadata | Commented code (stdout) | ~60 |
| 11 | `code_format_target` | Format code using language formatter | Code (stdin) + target | Formatted code (stdout) | ~80 |
| 12 | `code_write` | Write code to file with proper encoding | Code (stdin) + path | Written file | ~50 |

**Total**: ~1,180 lines across 12 tools vs. 1,971 lines in monolithic tool  
**Benefit**: Modular code generation, easy to add new target languages

---

## Example: Decomposed Pipeline in Action

### Old Way (Monolithic):
```bash
# One big tool does everything
./stunir_spec_to_ir_main --spec-root specs/ --out output.ir.json
./code_emitter -i output.ir.json -o output -t rust
```

### New Way (Unix Philosophy):
```bash
# Step-by-step pipeline with small tools
file_find specs/ "*.json" | \
  while read spec; do
    json_read "$spec" | \
    spec_validate_schema && \
    spec_extract_funcs | \
    func_to_ir
  done | \
  ir_merge_funcs | \
  ir_add_metadata --schema stunir_flat_ir_v1 | \
  json_write output.ir.json

# IR to Rust code
json_read output.ir.json | \
  ir_extract_module | \
  code_gen_preamble --target rust > output.rs

json_read output.ir.json | \
  ir_extract_funcs | \
  while read func; do
    echo "$func" | func_parse_sig | code_gen_func_sig --target rust
    echo "$func" | func_parse_body | code_gen_func_body --target rust
  done | \
  code_add_comments --metadata output.ir.json | \
  code_format_target --target rust >> output.rs
```

**Benefits**:
- Each step is visible and debuggable
- Can swap out individual tools (e.g., different formatters)
- Easy to insert validation, logging, or transformation steps
- AI can discover optimal pipeline by trying different compositions

---

## AI-Friendly Pipeline Composition

With `--describe` introspection, AI can auto-compose pipelines:

```json
{
  "source": "spec_json",
  "target": "rust_code",
  "available_tools": [
    {"name": "json_read", "inputs": ["file"], "outputs": ["json"]},
    {"name": "spec_extract_funcs", "inputs": ["json"], "outputs": ["functions_array"]},
    {"name": "func_to_ir", "inputs": ["functions_array"], "outputs": ["ir_functions"]},
    {"name": "code_gen_func_sig", "inputs": ["ir_functions", "target"], "outputs": ["code"]}
  ],
  "optimal_pipeline": [
    "json_read specs/example.json",
    "spec_extract_funcs",
    "func_to_ir",
    "code_gen_func_sig --target rust"
  ]
}
```

AI discovers this pipeline by:
1. Querying all tools with `--describe`
2. Building dependency graph (output type → input type)
3. Finding shortest path from source to target
4. Generating executable shell script

---

## Implementation Priority

### Phase 1: Core JSON & File Tools (6 tools)
1. `json_read` - Read and validate JSON
2. `json_write` - Write JSON to file
3. `file_find` - Find files by pattern
4. `file_hash` - Compute SHA-256
5. `toolchain_verify` - Verify lockfile
6. `manifest_generate` - Generate manifest

**Why first**: These are foundational, used by all other tools

### Phase 2: Spec Processing Tools (5 tools)
7. `spec_extract_funcs` - Extract functions from spec
8. `spec_extract_types` - Extract types from spec
9. `spec_extract_module` - Extract module metadata
10. `spec_validate_schema` - Validate spec schema
11. `type_normalize` - Normalize type names

**Why second**: Enable spec→IR conversion

### Phase 3: IR Generation Tools (4 tools)
12. `func_to_ir` - Convert function to IR
13. `module_to_ir` - Convert module to IR
14. `ir_merge_funcs` - Merge IR functions
15. `ir_add_metadata` - Add IR metadata

**Why third**: Complete spec→IR pipeline

### Phase 4: IR Processing Tools (5 tools)
16. `ir_validate_schema` - Validate IR schema
17. `ir_extract_module` - Extract module from IR
18. `ir_extract_funcs` - Extract functions from IR
19. `func_parse_sig` - Parse function signature
20. `func_parse_body` - Parse function body

**Why fourth**: Enable IR→code conversion

### Phase 5: Code Generation Tools (7 tools)
21. `type_map_target` - Map types to target
22. `code_gen_preamble` - Generate preamble
23. `code_gen_func_sig` - Generate signatures
24. `code_gen_func_body` - Generate bodies
25. `code_add_comments` - Add comments
26. `code_format_target` - Format code
27. `code_write` - Write code to file

**Why last**: Complete IR→code pipeline

---

## Specification Template for Each Tool

Each powertool should follow this structure:

```ada
--  <tool_name> - <One-line purpose>
--  Unix philosophy: Do one thing well

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;

procedure <Tool_Name> is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   Exit_Success : constant := 0;
   Exit_Error   : constant := 1;
   
   Version : constant String := "0.1.0-alpha";
   
   Describe_Output : constant String := 
     "{" &
     "  \"tool\": \"<tool_name>\"," &
     "  \"version\": \"0.1.0-alpha\"," &
     "  \"description\": \"<purpose>\"," &
     "  \"inputs\": [{\"type\": \"<type>\", \"source\": \"<stdin|file>\"}]," &
     "  \"outputs\": [{\"type\": \"<type>\", \"source\": \"<stdout|file>\"}]," &
     "  \"options\": [\"--help\", \"--version\", \"--describe\"]" &
     "}";

   procedure Do_Work;  -- Single focused function
   
begin
   --  Parse args
   --  Do one thing well
   --  Output to stdout or file
   --  Exit with appropriate code
end <Tool_Name>;
```

**Key principles**:
- <150 lines per tool
- Single input, single output (usually stdin→stdout)
- Composable via pipes
- Clear error messages
- `--describe` for AI introspection

---

## Example: `json_read` Powertool

```ada
--  json_read - Read and validate JSON file or stdin
--  Outputs validated JSON to stdout

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;

procedure JSON_Read is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   Exit_Success : constant := 0;
   Exit_Error   : constant := 1;
   
   Input_File : Unbounded_String := Null_Unbounded_String;
   
   function Read_JSON return String is
      Result : Unbounded_String := Null_Unbounded_String;
      Line   : String (1 .. 4096);
      Last   : Natural;
   begin
      if Length (Input_File) > 0 then
         --  Read from file
         declare
            File : File_Type;
         begin
            Open (File, In_File, To_String (Input_File));
            while not End_Of_File (File) loop
               Get_Line (File, Line, Last);
               Append (Result, Line (1 .. Last) & ASCII.LF);
            end loop;
            Close (File);
         end;
      else
         --  Read from stdin
         while not End_Of_File loop
            Get_Line (Line, Last);
            Append (Result, Line (1 .. Last) & ASCII.LF);
         end loop;
      end if;
      return To_String (Result);
   end Read_JSON;
   
   function Validate_JSON (Content : String) return Boolean is
   begin
      --  Basic JSON validation (check braces match, etc.)
      if Content'Length = 0 then
         return False;
      end if;
      if Content (Content'First) /= '{' and Content (Content'First) /= '[' then
         return False;
      end if;
      --  TODO: More validation
      return True;
   end Validate_JSON;

begin
   --  Parse args
   for I in 1 .. Argument_Count loop
      declare
         Arg : constant String := Argument (I);
      begin
         if Arg = "--help" then
            Put_Line ("json_read - Read and validate JSON");
            Put_Line ("Usage: json_read [FILE]");
            Put_Line ("  Reads JSON from FILE or stdin, validates, outputs to stdout");
            Set_Exit_Status (Exit_Success);
            return;
         elsif Arg = "--describe" then
            Put_Line ("{""tool"":""json_read"",""version"":""0.1.0-alpha""}");
            Set_Exit_Status (Exit_Success);
            return;
         else
            Input_File := To_Unbounded_String (Arg);
         end if;
      end;
   end loop;
   
   --  Read and validate JSON
   declare
      JSON_Content : constant String := Read_JSON;
   begin
      if not Validate_JSON (JSON_Content) then
         Put_Line (Standard_Error, "ERROR: Invalid JSON");
         Set_Exit_Status (Exit_Error);
         return;
      end if;
      
      --  Output validated JSON
      Put (JSON_Content);
      Set_Exit_Status (Exit_Success);
   end;
   
exception
   when others =>
      Put_Line (Standard_Error, "ERROR: Failed to read JSON");
      Set_Exit_Status (Exit_Error);
end JSON_Read;
```

**Size**: ~90 lines  
**Complexity**: O(n) where n is JSON size  
**Testable**: Easy to test with various JSON inputs  
**Composable**: Pipes into other tools

---

## Testing Strategy for Decomposed Tools

### Unit Tests (Per Tool)
```bash
# Test json_read with valid input
echo '{"key": "value"}' | ./json_read && echo "PASS" || echo "FAIL"

# Test json_read with invalid input
echo 'not json' | ./json_read && echo "FAIL" || echo "PASS"

# Test file_find
./file_find test_dir/ "*.json" | wc -l  # Should find N files
```

### Integration Tests (Pipeline)
```bash
# Test full spec→IR pipeline
./file_find specs/ "*.json" | \
  xargs -I {} ./json_read {} | \
  ./spec_extract_funcs | \
  ./func_to_ir | \
  ./ir_merge_funcs | \
  ./json_write output.ir.json

# Verify output
./json_read output.ir.json && echo "PASS"
```

### AI-Generated Test Suite
AI can generate test cases for each tool using `--describe`:
```python
def generate_tests(tool_name):
    description = subprocess.check_output([tool_name, "--describe"])
    inputs = description["inputs"]
    outputs = description["outputs"]
    # Generate test cases based on input/output types
    return test_cases
```

---

## Migration Plan: Monolithic → Decomposed

### Step 1: Keep Existing Monolithic Tools
- `stunir_spec_to_ir_main` and `code_emitter` remain working
- No breaking changes

### Step 2: Build Decomposed Tools in Parallel
- Create new powertools in `tools/spark/src/powertools/`
- Each tool is independent
- Use existing SPARK libraries (STUNIR_JSON_Utils, etc.)

### Step 3: Test Decomposed Pipeline
- Run both monolithic and decomposed pipelines
- Compare outputs (should be identical)
- Measure performance

### Step 4: Gradual Adoption
- Use decomposed tools for new workflows
- Keep monolithic tools for backward compatibility
- Document migration path

### Step 5: Deprecate Monolithic Tools (Optional)
- After 6 months, mark monolithic tools as deprecated
- Eventually remove monolithic implementations

---

## Summary: Why Decompose?

| Aspect | Monolithic | Decomposed Unix Philosophy |
|--------|-----------|---------------------------|
| **Lines per tool** | 475-1971 | 50-200 |
| **Testability** | Hard (must test everything) | Easy (test one thing) |
| **AI Generation** | Difficult (too much context) | Easy (small, focused) |
| **Debugging** | Hard (where did it fail?) | Easy (which tool failed?) |
| **Reusability** | Low (all-or-nothing) | High (mix and match) |
| **Composability** | None | Full (pipe-able) |
| **Error Handling** | Monolithic try-catch | Per-tool exit codes |
| **AI Orchestration** | Manual | Auto-discovery via `--describe` |
| **Learning Curve** | Steep (must understand all) | Shallow (understand one) |
| **Maintenance** | Hard (change affects all) | Easy (change one tool) |

**Verdict**: Decomposed powertools are **significantly better** for AI-driven workflows, local model code generation, and robust pipeline composition.

---

**Next Step**: Generate detailed specifications for each of the 27 powertools, following the template in `POWERTOOLS_SPEC_FOR_AI.md`.
