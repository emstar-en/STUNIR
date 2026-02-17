# Ada SPARK Powertools Specification for AI Code Generation

**Version**: 0.1.0-alpha  
**Purpose**: Specifications for AI/local models to generate Ada SPARK powertools for STUNIR spec-to-code-to-env pipeline  
**Target Language**: Ada SPARK 2014 with GNAT extensions  
**Audience**: AI code generation models (Claude, GPT, local LLMs)

---

## Overview

This document provides **precise specifications** for generating Ada SPARK powertools. Each specification includes:
- **Purpose & Behavior** - What the tool does
- **Input/Output Contract** - Structured interfaces
- **Exit Codes** - Error handling semantics
- **AI Introspection** - `--describe` JSON schema
- **Implementation Requirements** - Code structure, dependencies, patterns

---

## Common Ada SPARK Patterns for All Powertools

### File Structure Template

```ada
--  <tool_name> - <One-line description>
--  <Additional context if needed>
--  Phase N Powertool for STUNIR

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;

procedure <Tool_Name> is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   --  Exit codes per powertools spec
   Exit_Success          : constant := 0;
   Exit_Validation_Error : constant := 1;
   Exit_Processing_Error : constant := 2;
   Exit_Resource_Error   : constant := 3;

   --  Configuration
   Input_File    : Unbounded_String := Null_Unbounded_String;
   Output_File   : Unbounded_String := Null_Unbounded_String;
   Verbose_Mode  : Boolean := False;
   Show_Version  : Boolean := False;
   Show_Help     : Boolean := False;
   Show_Describe : Boolean := False;

   Version : constant String := "0.1.0-alpha";

   --  Description output for --describe
   Describe_Output : constant String :=
     "{" & ASCII.LF &
     "  ""tool"": ""<tool_name>"","  & ASCII.LF &
     "  ""version"": ""0.1.0-alpha""," & ASCII.LF &
     "  ""description"": ""<Tool description>""," & ASCII.LF &
     "  ""inputs"": [...]," & ASCII.LF &
     "  ""outputs"": [...]," & ASCII.LF &
     "  ""options"": [...]" & ASCII.LF &
     "}";

   procedure Print_Usage;
   procedure Print_Error (Msg : String);
   procedure Print_Info (Msg : String);

   --  Tool-specific procedures here

begin
   --  Parse command line
   --  Execute main logic
   --  Handle errors with appropriate exit codes
end <Tool_Name>;
```

### Exit Code Contract

All powertools MUST use these exit codes consistently:

| Exit Code | Constant Name | Meaning | When to Use |
|-----------|---------------|---------|-------------|
| 0 | `Exit_Success` | Success | Operation completed successfully |
| 1 | `Exit_Validation_Error` | Validation failed | Input validation failed (bad JSON, missing fields, schema violation) |
| 2 | `Exit_Processing_Error` | Processing error | Runtime error during processing (file I/O, parsing, transformation) |
| 3 | `Exit_Resource_Error` | Resource error | System resource issue (out of memory, disk full, permission denied) |

### AI Introspection (`--describe`) Contract

All powertools MUST implement `--describe` flag that outputs JSON:

```json
{
  "tool": "<tool_name>",
  "version": "0.1.0-alpha",
  "description": "<Human-readable description>",
  "inputs": [
    {
      "name": "<input_name>",
      "type": "<json|file|stdin|argument>",
      "description": "<What this input is>",
      "required": true|false
    }
  ],
  "outputs": [
    {
      "name": "<output_name>",
      "type": "<json|file|stdout|exit_code>",
      "description": "<What this output is>"
    }
  ],
  "options": ["--help", "--version", "--describe", "<custom-options>"],
  "complexity": "O(n)" or similar,
  "dependencies": ["<other_tool_name>"],
  "pipeline_stage": "<spec|ir|code|env|verify>"
}
```

---

## Priority 1: Environment & Execution Powertools

### 1. `env_create` - Create Isolated Execution Environment

**Purpose**: Create an isolated execution environment (directory structure, config files, env vars) for running generated code.

**Behavior**:
- Creates a new directory structure for execution environment
- Generates platform-specific configuration files
- Sets up environment variables file
- Creates build/test directories
- Generates manifest JSON describing the environment

**Input**:
- `--spec-file FILE` - Spec JSON (to extract target language, dependencies)
- `--env-root DIR` - Root directory for environment (default: `./env_<timestamp>`)
- `--platform PLATFORM` - Target platform (linux|windows|macos|docker)
- `--targets LANGS` - Comma-separated target languages (rust,c,python)

**Output**:
- Directory structure created
- `env_manifest.json` - JSON describing environment structure
- Exit code indicating success/failure

**Exit Codes**:
- 0: Environment created successfully
- 2: Failed to create directories or write files
- 3: Insufficient permissions or disk space

**--describe Output**:
```json
{
  "tool": "env_create",
  "version": "0.1.0-alpha",
  "description": "Create isolated execution environment for generated code",
  "inputs": [
    {"name": "spec_file", "type": "file", "required": true},
    {"name": "env_root", "type": "argument", "required": false},
    {"name": "platform", "type": "argument", "required": false},
    {"name": "targets", "type": "argument", "required": false}
  ],
  "outputs": [
    {"name": "env_manifest", "type": "json", "description": "Environment manifest"},
    {"name": "exit_code", "type": "exit_code"}
  ],
  "options": ["--help", "--version", "--describe", "--spec-file", "--env-root", "--platform", "--targets"],
  "complexity": "O(1)",
  "pipeline_stage": "env"
}
```

**Implementation Requirements**:
- Create directories: `src/`, `build/`, `bin/`, `tests/`, `config/`
- Generate `env_manifest.json`:
  ```json
  {
    "env_id": "env_<timestamp>",
    "platform": "linux|windows|macos|docker",
    "targets": ["rust", "c", "python"],
    "root_path": "/path/to/env",
    "created_at": "2026-02-17T12:00:00Z",
    "directories": {
      "src": "src/",
      "build": "build/",
      "bin": "bin/",
      "tests": "tests/"
    }
  }
  ```
- Use `Ada.Directories` for directory creation
- Handle permission errors gracefully

**Ada Dependencies**:
```ada
with Ada.Directories;
with Ada.Calendar;
with Ada.Calendar.Formatting;
with GNAT.OS_Lib;
```

---

### 2. `env_destroy` - Clean Up Execution Environment

**Purpose**: Safely remove an execution environment created by `env_create`.

**Behavior**:
- Validates environment manifest
- Recursively removes directory structure
- Logs cleanup actions
- Handles file locks and permission issues

**Input**:
- `--env-root DIR` - Root directory of environment to destroy
- `--force` - Force removal even if files are locked
- `--dry-run` - Show what would be deleted without deleting

**Output**:
- Cleanup log printed to stdout
- Exit code indicating success/failure

**Exit Codes**:
- 0: Environment destroyed successfully
- 2: Failed to remove some files/directories
- 3: Permission denied or env not found

**--describe Output**:
```json
{
  "tool": "env_destroy",
  "version": "0.1.0-alpha",
  "description": "Clean up and remove execution environment",
  "inputs": [
    {"name": "env_root", "type": "argument", "required": true},
    {"name": "force", "type": "flag", "required": false},
    {"name": "dry_run", "type": "flag", "required": false}
  ],
  "outputs": [
    {"name": "cleanup_log", "type": "stdout"},
    {"name": "exit_code", "type": "exit_code"}
  ],
  "options": ["--help", "--version", "--describe", "--env-root", "--force", "--dry-run"],
  "complexity": "O(n) where n is number of files",
  "pipeline_stage": "env"
}
```

**Implementation Requirements**:
- Read and validate `env_manifest.json`
- Use `Ada.Directories.Delete_Tree` for recursive removal
- Handle exceptions for locked files
- Provide verbose logging of removed items

---

### 3. `package_install` - Install Language Dependencies

**Purpose**: Install language-specific dependencies required for generated code.

**Behavior**:
- Reads spec JSON to extract dependency requirements
- Invokes platform package managers (cargo, pip, npm, apt, etc.)
- Validates installation success
- Generates dependency lockfile

**Input**:
- `--spec-file FILE` - Spec JSON with dependency declarations
- `--target LANG` - Target language (rust|python|c|javascript)
- `--env-root DIR` - Environment root directory
- `--offline` - Use local cache only (no network)

**Output**:
- `dependencies.lock` - Lockfile with installed versions
- Installation log to stdout
- Exit code

**Exit Codes**:
- 0: All dependencies installed successfully
- 1: Dependency specification invalid
- 2: Installation failed (package not found, network error)
- 3: Package manager not available

**--describe Output**:
```json
{
  "tool": "package_install",
  "version": "0.1.0-alpha",
  "description": "Install language dependencies for generated code",
  "inputs": [
    {"name": "spec_file", "type": "file", "required": true},
    {"name": "target", "type": "argument", "required": true},
    {"name": "env_root", "type": "argument", "required": false}
  ],
  "outputs": [
    {"name": "dependencies_lock", "type": "json"},
    {"name": "install_log", "type": "stdout"},
    {"name": "exit_code", "type": "exit_code"}
  ],
  "options": ["--help", "--version", "--describe", "--spec-file", "--target", "--env-root", "--offline"],
  "complexity": "O(n) where n is number of dependencies",
  "dependencies": ["spec_validate"],
  "pipeline_stage": "env"
}
```

**Implementation Requirements**:
- Parse spec JSON for dependency sections
- Map language to package manager:
  - Rust → `cargo add` or `Cargo.toml`
  - Python → `pip install`
  - C → `apt-get` or `yum` (for system libraries)
  - JavaScript → `npm install`
- Use `GNAT.OS_Lib.Spawn` to invoke package managers
- Capture stdout/stderr
- Generate `dependencies.lock`:
  ```json
  {
    "target": "rust",
    "packages": [
      {"name": "serde", "version": "1.0.196", "installed_at": "2026-02-17T12:00:00Z"}
    ]
  }
  ```

**Ada Dependencies**:
```ada
with GNAT.OS_Lib;
with GNAT.Expect;
with Ada.Containers.Vectors;
```

---

### 4. `build_orchestrator` - Compile/Build Generated Code

**Purpose**: Orchestrate compilation and building of generated code for target languages.

**Behavior**:
- Detects build system for target language
- Invokes appropriate build commands
- Captures build output and errors
- Validates build artifacts
- Generates build report

**Input**:
- `--source-dir DIR` - Directory containing generated source code
- `--target LANG` - Target language (rust|c|python|go)
- `--output-dir DIR` - Where to place build artifacts
- `--build-type TYPE` - Build type (debug|release|test)

**Output**:
- Build artifacts (executables, libraries)
- `build_report.json` - Build results
- Build log to stdout

**Exit Codes**:
- 0: Build successful
- 2: Build failed (compilation errors)
- 3: Build tools not found or permission denied

**--describe Output**:
```json
{
  "tool": "build_orchestrator",
  "version": "0.1.0-alpha",
  "description": "Orchestrate compilation and building of generated code",
  "inputs": [
    {"name": "source_dir", "type": "argument", "required": true},
    {"name": "target", "type": "argument", "required": true},
    {"name": "output_dir", "type": "argument", "required": false},
    {"name": "build_type", "type": "argument", "required": false}
  ],
  "outputs": [
    {"name": "build_artifacts", "type": "file"},
    {"name": "build_report", "type": "json"},
    {"name": "exit_code", "type": "exit_code"}
  ],
  "options": ["--help", "--version", "--describe", "--source-dir", "--target", "--output-dir", "--build-type"],
  "complexity": "O(n) where n is number of source files",
  "pipeline_stage": "env"
}
```

**Implementation Requirements**:
- Map language to build command:
  - Rust → `cargo build --release`
  - C → `gcc -o output source.c` or `make`
  - Python → `python -m py_compile`
  - Go → `go build`
- Use `GNAT.OS_Lib.Spawn` to invoke build tools
- Parse build output for errors
- Generate `build_report.json`:
  ```json
  {
    "target": "rust",
    "build_type": "release",
    "success": true,
    "artifacts": ["bin/program.exe"],
    "errors": [],
    "warnings": 3,
    "build_time_ms": 4523
  }
  ```

---

### 5. `test_runner` - Execute Tests on Generated Code

**Purpose**: Run tests against generated code to validate correctness.

**Behavior**:
- Discovers test files in environment
- Executes tests using language-specific test frameworks
- Captures test results
- Generates test report

**Input**:
- `--test-dir DIR` - Directory containing tests
- `--target LANG` - Target language
- `--timeout SECONDS` - Test timeout (default: 60)

**Output**:
- `test_report.json` - Test results
- Test output to stdout

**Exit Codes**:
- 0: All tests passed
- 1: Some tests failed
- 2: Tests couldn't run (missing test framework)
- 3: Timeout or resource error

**--describe Output**:
```json
{
  "tool": "test_runner",
  "version": "0.1.0-alpha",
  "description": "Execute tests on generated code",
  "inputs": [
    {"name": "test_dir", "type": "argument", "required": true},
    {"name": "target", "type": "argument", "required": true},
    {"name": "timeout", "type": "argument", "required": false}
  ],
  "outputs": [
    {"name": "test_report", "type": "json"},
    {"name": "test_output", "type": "stdout"},
    {"name": "exit_code", "type": "exit_code"}
  ],
  "options": ["--help", "--version", "--describe", "--test-dir", "--target", "--timeout"],
  "complexity": "O(n) where n is number of tests",
  "dependencies": ["build_orchestrator"],
  "pipeline_stage": "env"
}
```

**Implementation Requirements**:
- Map language to test command:
  - Rust → `cargo test`
  - Python → `pytest` or `python -m unittest`
  - C → Custom test runner or `make test`
- Capture test framework output
- Parse results for pass/fail counts
- Generate `test_report.json`:
  ```json
  {
    "target": "rust",
    "total_tests": 42,
    "passed": 40,
    "failed": 2,
    "skipped": 0,
    "duration_ms": 1234,
    "failures": [
      {"test": "test_addition", "error": "assertion failed"}
    ]
  }
  ```

---

## Priority 2: Code Generation Enhancement Powertools

### 6. `sig_gen_c` - Generate C Function Signatures

**Purpose**: Generate C99 function signatures from STUNIR spec/IR JSON.

**Behavior**:
- Parse spec/IR JSON to extract function definitions
- Map STUNIR types to C99 types (i32→int32_t, str→char*, etc.)
- Generate forward declarations
- Generate header guards
- Output valid C header file

**Input**:
- Spec/IR JSON from stdin or `--input FILE`
- `--output FILE` - Output C header file (default: stdout)
- `--header-guard NAME` - Custom header guard macro

**Output**:
- C99 header file with function signatures

**Exit Codes**:
- 0: Success
- 1: Invalid JSON or missing function definitions
- 2: Type mapping failed

**--describe Output**:
```json
{
  "tool": "sig_gen_c",
  "version": "0.1.0-alpha",
  "description": "Generate C99 function signatures from STUNIR spec/IR",
  "inputs": [
    {"name": "spec_json", "type": "json", "source": ["stdin", "file"], "required": true}
  ],
  "outputs": [
    {"name": "c_header", "type": "stdout", "description": "C99 header file"}
  ],
  "options": ["--help", "--version", "--describe", "--input", "--output", "--header-guard"],
  "complexity": "O(n) where n is number of functions",
  "dependencies": ["type_map", "json_validate"],
  "pipeline_stage": "code"
}
```

**Implementation Requirements**:
- Type mapping table:
  - `i8` → `int8_t`
  - `i16` → `int16_t`
  - `i32` → `int32_t`
  - `i64` → `int64_t`
  - `u8` → `uint8_t`
  - `u16` → `uint16_t`
  - `u32` → `uint32_t`
  - `u64` → `uint64_t`
  - `f32` → `float`
  - `f64` → `double`
  - `bool` → `bool` (with `#include <stdbool.h>`)
  - `str` → `const char*`
  - `void` → `void`
- Generate header guards: `#ifndef MODULE_NAME_H` / `#define MODULE_NAME_H`
- Include stdint.h and stdbool.h
- Generate extern "C" wrapper for C++ compatibility

---

### 7. `code_format` - Auto-Format Generated Code

**Purpose**: Format generated code according to language-specific style guidelines.

**Behavior**:
- Detects target language
- Invokes appropriate formatter (rustfmt, clang-format, black, gofmt)
- Validates formatting success
- Overwrites source file or outputs to stdout

**Input**:
- `--input FILE` - Source file to format
- `--target LANG` - Target language (auto-detect if omitted)
- `--style STYLE` - Style preset (google, llvm, mozilla, etc.)
- `--in-place` - Overwrite input file (default: output to stdout)

**Output**:
- Formatted source code

**Exit Codes**:
- 0: Formatted successfully
- 2: Formatter failed or not found
- 3: File permission denied

**--describe Output**:
```json
{
  "tool": "code_format",
  "version": "0.1.0-alpha",
  "description": "Auto-format generated code using language-specific formatters",
  "inputs": [
    {"name": "source_file", "type": "file", "required": true},
    {"name": "target", "type": "argument", "required": false},
    {"name": "style", "type": "argument", "required": false}
  ],
  "outputs": [
    {"name": "formatted_code", "type": "stdout"}
  ],
  "options": ["--help", "--version", "--describe", "--input", "--target", "--style", "--in-place"],
  "complexity": "O(n) where n is file size",
  "pipeline_stage": "code"
}
```

**Implementation Requirements**:
- Map language to formatter:
  - Rust → `rustfmt`
  - C/C++ → `clang-format`
  - Python → `black` or `autopep8`
  - Go → `gofmt`
- Use `GNAT.OS_Lib.Spawn` to invoke formatter
- Pipe file content through formatter
- Capture formatted output

---

## Priority 3: Verification & Attestation Powertools

### 8. `signature_verify` - Verify Cryptographic Signatures

**Purpose**: Verify cryptographic signatures on generated code and artifacts.

**Behavior**:
- Load public key from file or environment
- Verify signature on file content
- Validate signature algorithm (RSA, Ed25519, ECDSA)
- Output verification result as JSON

**Input**:
- `--file FILE` - File to verify
- `--signature FILE` - Signature file (detached signature)
- `--pubkey FILE` - Public key file (PEM format)
- `--algorithm ALGO` - Signature algorithm (rsa2048|ed25519|ecdsa-p256)

**Output**:
- Verification result JSON to stdout

**Exit Codes**:
- 0: Signature valid
- 1: Signature invalid
- 2: Verification failed (missing key, bad format)

**--describe Output**:
```json
{
  "tool": "signature_verify",
  "version": "0.1.0-alpha",
  "description": "Verify cryptographic signatures on files",
  "inputs": [
    {"name": "file", "type": "file", "required": true},
    {"name": "signature", "type": "file", "required": true},
    {"name": "pubkey", "type": "file", "required": true}
  ],
  "outputs": [
    {"name": "verification_result", "type": "json"}
  ],
  "options": ["--help", "--version", "--describe", "--file", "--signature", "--pubkey", "--algorithm"],
  "complexity": "O(n) where n is file size",
  "dependencies": ["hash_compute"],
  "pipeline_stage": "verify"
}
```

**Implementation Requirements**:
- Use GNAT crypto libraries or call external tools (gpg, openssl)
- Output JSON:
  ```json
  {
    "valid": true,
    "algorithm": "ed25519",
    "signer": "STUNIR Build System",
    "verified_at": "2026-02-17T12:00:00Z"
  }
  ```

---

### 9. `doc_generate` - Generate Documentation from Specs

**Purpose**: Generate human-readable documentation from STUNIR spec JSON.

**Behavior**:
- Parse spec JSON
- Extract functions, types, modules
- Generate Markdown documentation
- Include function signatures, descriptions, parameter docs

**Input**:
- `--spec-file FILE` - Spec JSON
- `--output FILE` - Output Markdown file (default: stdout)
- `--format FORMAT` - Output format (markdown|html|man)

**Output**:
- Documentation file

**Exit Codes**:
- 0: Documentation generated
- 1: Invalid spec JSON
- 2: Failed to write output

**--describe Output**:
```json
{
  "tool": "doc_generate",
  "version": "0.1.0-alpha",
  "description": "Generate documentation from STUNIR specs",
  "inputs": [
    {"name": "spec_file", "type": "file", "required": true}
  ],
  "outputs": [
    {"name": "documentation", "type": "file"}
  ],
  "options": ["--help", "--version", "--describe", "--spec-file", "--output", "--format"],
  "complexity": "O(n) where n is number of functions",
  "dependencies": ["spec_validate"],
  "pipeline_stage": "code"
}
```

**Implementation Requirements**:
- Template-based generation
- Support Markdown tables for function signatures
- Generate table of contents
- Include spec metadata (version, profile, targets)

---

## Priority 4: Pipeline Orchestration Powertools

### 10. `pipeline_compose` - Auto-Compose Tool Pipelines

**Purpose**: Automatically compose powertool pipelines based on input/output type matching.

**Behavior**:
- Scan available powertools using `--describe`
- Build dependency graph based on input/output types
- Find optimal path from source to target
- Generate shell script or JSON workflow

**Input**:
- `--source TYPE` - Source data type (spec_json|ir_json|code)
- `--target TYPE` - Target output type (code|env|executable|documentation)
- `--tools-dir DIR` - Directory containing powertools
- `--output-format FORMAT` - Output format (shell|json|makefile)

**Output**:
- Pipeline script or JSON workflow

**Exit Codes**:
- 0: Pipeline generated
- 1: No valid pipeline found
- 2: Tool discovery failed

**--describe Output**:
```json
{
  "tool": "pipeline_compose",
  "version": "0.1.0-alpha",
  "description": "Auto-compose powertool pipelines",
  "inputs": [
    {"name": "source", "type": "argument", "required": true},
    {"name": "target", "type": "argument", "required": true},
    {"name": "tools_dir", "type": "argument", "required": false}
  ],
  "outputs": [
    {"name": "pipeline_script", "type": "stdout"}
  ],
  "options": ["--help", "--version", "--describe", "--source", "--target", "--tools-dir", "--output-format"],
  "complexity": "O(n²) where n is number of tools",
  "pipeline_stage": "orchestration"
}
```

**Implementation Requirements**:
- Invoke all tools with `--describe`
- Parse JSON output
- Build directed graph (tools as nodes, type compatibility as edges)
- Use Dijkstra's or BFS to find shortest path
- Generate executable pipeline script

---

## Ada SPARK Implementation Guidelines

### Naming Conventions
- **Procedures**: `Snake_Case` (e.g., `Print_Usage`)
- **Functions**: `Snake_Case` (e.g., `Read_File`)
- **Variables**: `Snake_Case` (e.g., `Input_File`)
- **Constants**: `Snake_Case` with clear naming (e.g., `Exit_Success`)
- **Types**: `Pascal_Case` (e.g., `File_Type`)

### Error Handling Pattern
```ada
begin
   --  Main logic
   Do_Work;
   Set_Exit_Status (Exit_Success);
exception
   when Validation_Error =>
      Print_Error ("Validation failed");
      Set_Exit_Status (Exit_Validation_Error);
   when IO_Error =>
      Print_Error ("I/O error");
      Set_Exit_Status (Exit_Processing_Error);
   when others =>
      Print_Error ("Unexpected error");
      Set_Exit_Status (Exit_Processing_Error);
end;
```

### JSON Output Pattern
```ada
procedure Print_JSON_Result (Success : Boolean; Message : String) is
begin
   Put_Line ("{");
   Put_Line ("  ""success"": " & (if Success then "true" else "false") & ",");
   Put_Line ("  ""message"": """ & Message & """");
   Put_Line ("}");
end Print_JSON_Result;
```

### Command Line Parsing Pattern
```ada
for I in 1 .. Argument_Count loop
   declare
      Arg : constant String := Argument (I);
   begin
      if Arg = "--help" or Arg = "-h" then
         Show_Help := True;
      elsif Arg = "--version" or Arg = "-v" then
         Show_Version := True;
      elsif Arg = "--describe" then
         Show_Describe := True;
      elsif Arg'Length > 8 and then Arg (Arg'First .. Arg'First + 7) = "--input=" then
         Input_File := To_Unbounded_String (Arg (Arg'First + 8 .. Arg'Last));
      end if;
   end;
end loop;
```

---

## Building and Testing Powertools

### GPR Project File Template

Create `powertools.gpr`:

```ada
project Powertools is
   for Source_Dirs use ("src/powertools");
   for Object_Dir use "obj";
   for Exec_Dir use "bin";
   
   for Main use (
      "env_create.adb",
      "env_destroy.adb",
      "package_install.adb",
      "build_orchestrator.adb",
      "test_runner.adb",
      "sig_gen_c.adb",
      "code_format.adb",
      "signature_verify.adb",
      "doc_generate.adb",
      "pipeline_compose.adb"
   );

   package Compiler is
      for Default_Switches ("Ada") use
        ("-gnat2022", "-gnatwa", "-gnatVa", "-gnatf", "-gnato", "-fstack-check", "-g");
   end Compiler;

   package Binder is
      for Default_Switches ("Ada") use ("-E", "-d32m");
   end Binder;

   package Builder is
      for Default_Switches ("Ada") use ("-s", "-j0");
   end Builder;

end Powertools;
```

### Build Commands

```powershell
# Build all powertools
cd tools/spark
gprbuild -P powertools.gpr

# Build individual powertool
gprbuild -P powertools.gpr env_create.adb

# Clean build artifacts
gprclean -P powertools.gpr
```

---

## Summary: Powertools for Spec→Code→Env Pipeline

| Powertool | Priority | Purpose | Inputs | Outputs |
|-----------|----------|---------|--------|---------|
| `env_create` | High | Create execution environment | Spec JSON | Environment manifest |
| `env_destroy` | High | Clean up environment | Environment root | Cleanup log |
| `package_install` | High | Install dependencies | Spec JSON, target | Dependencies lockfile |
| `build_orchestrator` | High | Build generated code | Source dir, target | Build artifacts, report |
| `test_runner` | High | Run tests | Test dir, target | Test report |
| `sig_gen_c` | Medium | Generate C signatures | Spec/IR JSON | C header file |
| `code_format` | Medium | Format generated code | Source file | Formatted code |
| `signature_verify` | Medium | Verify signatures | File, signature, pubkey | Verification result |
| `doc_generate` | Low | Generate docs | Spec JSON | Markdown docs |
| `pipeline_compose` | Low | Compose pipelines | Source type, target type | Pipeline script |

---

## AI Code Generation Prompt Template

When generating a powertool, use this prompt structure:

```
Generate an Ada SPARK 2014 powertool named `<tool_name>` that <purpose>.

Requirements:
- Follow the file structure template in POWERTOOLS_SPEC_FOR_AI.md
- Version: 0.1.0-alpha
- Exit codes: 0=success, 1=validation error, 2=processing error, 3=resource error
- Implement --help, --version, --describe flags
- <tool-specific requirements from spec>

Input: <input specification>
Output: <output specification>

Use these Ada dependencies:
- Ada.Command_Line
- Ada.Text_IO
- Ada.Strings.Unbounded
- <additional dependencies>

Generate complete, working Ada SPARK code.
```

---

**End of Specification**
