# STUNIR Ada SPARK Tool Contracts

**Purpose**: Machine-readable contracts for AI models to understand and orchestrate STUNIR pipeline tools.

**For AI Models**: Read these contracts to understand what each tool does, then invoke the actual Ada SPARK binaries. Do NOT reimplement these tools - orchestrate them.

**IMPORTANT - Development Status**:
- **Ada SPARK**: PRIMARY implementation, PARTIALLY FUNCTIONAL, being developed NOW
- **Python**: INCOMPLETE, development planned AFTER SPARK completion
- Both pipelines are under active development
- SPARK is the focus TODAY, Python will be focus LATER

---

## Tool: stunir_spec_to_ir

**Binary Location**:
- Precompiled: `precompiled/linux-x86_64/spark/bin/stunir_spec_to_ir_main`
- Source build: `tools/spark/bin/stunir_spec_to_ir_main`

**Purpose**: Converts specification JSON files to canonical Intermediate Reference (IR) format.

**Implementation**: `tools/spark/src/stunir_spec_to_ir.adb` (lines 1-475)

**Development Status**:
- 🔨 PARTIALLY FUNCTIONAL - basic pipeline works
- 🔨 Active development in progress
- ✅ Basic spec→IR conversion functional
- 🔨 Complete transformation logic in development

### Command Line Interface

```bash
stunir_spec_to_ir_main --spec-root <directory> --out <file> [--lockfile <file>]
```

**Arguments**:
- `--spec-root <directory>` (required): Root directory containing .json spec files
- `--out <file>` (required): Output path for generated IR JSON file
- `--lockfile <file>` (optional): Toolchain lock file path (default: `local_toolchain.lock.json`)

### Input Format

**Spec files** (in `--spec-root` directory):
- File pattern: `*.json` (searched recursively)
- Format: STUNIR spec JSON (various schemas supported)
- Processing order: Sorted alphabetically for determinism

**Toolchain lock file**:
- Format: JSON
- Purpose: Verify deterministic toolchain configuration
- Must exist for conversion to proceed

### Output Format

**IR file** (`--out` path):
- Schema: `stunir_semantic_ir_v1`
- Format: Canonical JSON with sorted keys
- Contains: Semantic IR AST (root module, declarations)

### Processing Steps

The tool executes these operations in order (see stunir_spec_to_ir.adb):

1. **Verify Toolchain** (lines 256-261)
   - Check lockfile exists
   - Output: `[INFO] Toolchain verified from: <path>`
   - Fails if lockfile missing

2. **Collect Spec Files** (lines 263-271)
   - Recursively search `--spec-root` for `*.json`
   - Sort files alphabetically
   - Output: `[INFO] Found N spec file(s)`
   - Fails if no files found

3. **Parse JSON Specs** (lines 282-342)
   - Read each JSON file
   - Parse as spec
   - Skip invalid files with warning
   - Output: `[WARN] Failed to parse <file>, skipping` (if invalid)
   - Fails if no valid specs found

4. **Merge Multiple Specs** (lines 305-346)
   - If multiple specs: merge functions from all files
   - Preserves order
   - Deduplicates types (first occurrence wins)
   - Output: Progress messages for each file

5. **Generate Semantic IR** (lines 352-377)
   - Convert merged spec to IR format
   - Apply semantic transformations
   - Generate canonical JSON
   - Output: `[INFO] Generating semantic IR with N function(s)...`

6. **Write Output** (lines 363-377)
   - Create output directory if needed
   - Write canonical IR JSON
   - Output: `[INFO] Wrote semantic IR to <path>`
  - Final: `[SUCCESS] Generated semantic IR with schema: stunir_semantic_ir_v1`

### Exit Codes

- **0 (Success)**: IR generated successfully
- **1 (Failure)**: Conversion failed (see stderr for details)

### Error States

Defined in `stunir_spec_to_ir.ads`:

- `Success`: Operation completed
- `Error_Toolchain_Verification_Failed`: Lockfile missing or invalid
- `Error_Spec_Not_Found`: No spec files found in directory
- `Error_Invalid_Spec`: No valid spec files (all parse failures)
- `Error_Output_Write_Failed`: Cannot write output file

### Determinism Guarantees

- **Spec file processing**: Alphabetical order (sorted)
- **JSON output**: Canonical format (sorted keys, stable separators)
- **Hash computation**: SHA256 of file contents
- **No timestamps**: Output doesn't include generation time
- **Bounded data**: Maximum functions/types per module enforced

### Example Usage

```bash
# Basic usage
./tools/spark/bin/stunir_spec_to_ir_main \
  --spec-root ./spec/examples \
  --out ./ir/example.json

# With custom lockfile
./tools/spark/bin/stunir_spec_to_ir_main \
  --spec-root ./spec/examples \
  --out ./ir/example.json \
  --lockfile ./custom_toolchain.lock.json
```

### Verification

**Check Success**:
```bash
if [ $? -eq 0 ]; then
  echo "IR generated successfully"
  # Verify output exists
  test -f ./ir/example.json || echo "ERROR: Output file missing"
fi
```

**Inspect Output**:
```bash
# Validate JSON
python3 -m json.tool ./ir/example.json > /dev/null

# Check schema
jq -r '.schema' ./ir/example.json
# Expected: "stunir_semantic_ir_v1"
```

---

## Tool: code_emitter

**Binary Location**:
- Precompiled: `precompiled/linux-x86_64/spark/bin/code_emitter`
- Source build: `tools/spark/bin/code_emitter`

**Purpose**: Generates target language code from STUNIR IR.

**Implementation**: `tools/spark/src/core/code_emitter.adb`

### Command Line Interface

```bash
code_emitter -i <file> -o <output_dir> -t <target>
```

**Arguments**:
- `-i <file>` (required): Input IR JSON file
- `-o <output_dir>` (required): Output directory for generated code
- `-t <target>` (required): Target language (see below)

### Supported Target Languages

**General-Purpose Languages**:
- `c` → `.c` (C99/C11)
- `cpp` → `.cpp` (C++)
- `rust` → `.rs` (Rust)
- `python` → `.py` (Python 3)
- `go` → `.go` (Go)
- `javascript` / `js` → `.js` (JavaScript ES6)
- `typescript` / `ts` → `.ts` (TypeScript)
- `java` → `.java` (Java)

**Low-Level Targets**:
- `x86` / `asm` → `.asm` (x86 Assembly)
- `arm` → `.s` (ARM Assembly)

**Other Targets**:
- `clojure` → `.clj`
- `cljs` / `clojurescript` → `.cljs`
- `prolog` → `.pl`
- `futhark` → `.fut`
- `lean4` → `.lean`

### Input Format

**IR file** (`-i` path):
- Schema: `stunir_semantic_ir_v1` (AST)
- Must contain: `root` module with `declarations`

### Output Format

**Code output** (`-o` path):
- Format: Target language source code
- Contents: Function declarations/definitions
- Style: Idiomatic for target language

**Current Limitation**: Generates function stubs (signatures + empty bodies)

### Processing Steps

1. **Parse IR JSON**
  - Read and validate IR file
  - Parse Semantic IR AST (root module)

2. **Select Code Generator** (lines 44-94)
   - Map target string to language enum
   - Get file extension
   - Initialize generator

3. **Generate Code** (lines 152-300+)
   - Generate file preamble (imports, headers)
   - For each function:
     - Generate type-mapped signature
     - Generate stub body
   - Format according to target conventions

4. **Write Output**
   - Create output directory if needed
   - Write generated code
   - Output: Success/failure message

### Exit Codes

- **0 (Success)**: Code generated successfully
- **1 (Failure)**: Generation failed

### Example Usage

```bash
# Generate C code
./tools/spark/bin/code_emitter \
  -i ./ir/example.json \
  -o ./generated \
  -t c

# Generate Rust code
./tools/spark/bin/code_emitter \
  -i ./ir/example.json \
  -o ./generated \
  -t rust

# Generate Python code
./tools/spark/bin/code_emitter \
  -i ./ir/example.json \
  -o ./generated \
  -t python
```

---

## Pipeline Orchestration

### End-to-End Workflow

```bash
# Step 1: Spec → IR
./tools/spark/bin/stunir_spec_to_ir_main \
  --spec-root ./spec \
  --out ./build/ir.json

# Step 2: IR → Code (multiple targets)
for target in c rust python; do
  ./tools/spark/bin/code_emitter \
    -i ./build/ir.json \
    -o ./generated \
    -t $target
done
```

### Receipt Generation

(Not yet implemented in Ada SPARK tools - planned for Phase 3)

Future: Each tool will support `--emit-receipt` flag to generate:
```json
{
  "schema": "stunir.receipt.v1",
  "tool": "stunir_spec_to_ir",
  "status": "success",
  "inputs": [{"path": "...", "sha256": "..."}],
  "outputs": [{"path": "...", "sha256": "..."}],
  "receipt_core_id_sha256": "..."
}
```

---

## Error Handling for AI Models

### Strategy for AI Orchestration

1. **Check exit code**: `$? == 0` means success
2. **Read stderr**: Error messages prefixed with `[ERROR]`
3. **Validate outputs**: Check file existence and format
4. **Chain verification**: Each tool's output is next tool's input

### Common Failure Modes

**spec_to_ir**:
- Lockfile missing → Provide `local_toolchain.lock.json`
- No spec files → Check `--spec-root` path is correct
- Invalid JSON → Fix spec file syntax
- Parse errors → Check spec schema version

**code_emitter**:
- IR file missing → Run spec_to_ir first
- Invalid schema → IR must be `stunir_semantic_ir_v1`
- Unknown target → Use supported language names
- Empty functions → Expected for current version (stubs only)

### Debugging Tips

```bash
# Verbose logging (future)
STUNIR_LOG_LEVEL=debug ./tools/spark/bin/stunir_spec_to_ir_main ...

# Dry run (future)
./tools/spark/bin/stunir_spec_to_ir_main --dry-run ...

# Schema validation (future)
./tools/spark/bin/stunir_spec_to_ir_main --validate-only ...
```

---

## Tool Discovery (Future)

### Planned: --describe Flag

Each tool will support:
```bash
./tools/spark/bin/stunir_spec_to_ir_main --describe
```

Output JSON:
```json
{
  "tool": "stunir_spec_to_ir",
  "version": "0.1.0-alpha",
  "description": "Converts specification JSON to canonical IR",
  "inputs": {
    "spec_root": {"type": "directory", "required": true},
    "lockfile": {"type": "file", "default": "local_toolchain.lock.json"}
  },
  "outputs": {"ir_file": {"type": "file", "format": "stunir_semantic_ir_v1"}},
  "operations": ["Verify toolchain", "Collect specs", "Parse JSON", "Generate IR"],
  "determinism": "full",
  "exit_codes": {"0": "success", "1": "failure"}
}
```

---

## Related Files

- **Ada Implementation**: `tools/spark/src/stunir_spec_to_ir.adb`, `tools/spark/src/core/code_emitter.adb`
- **Python Implementation** (under development): `tools/spec_to_ir.py`, `tools/ir_to_code.py`
- **Build Config**: `tools/spark/stunir_tools.gpr`
- **Plan**: `.abacusai/plans/stunir_ai_orchestration_plan.md`

---

## For AI Models: Next Steps

1. **Read this contract** to understand tool behavior
2. **Invoke binaries** (don't reimplement)
3. **Check exit codes** and validate outputs
4. **Chain tools**: spec → IR → code
5. **Report errors** with context from stderr

**Remember**: These Ada SPARK tools are the PRIMARY IMPLEMENTATION (partially functional):
- 🔨 UNDER ACTIVE DEVELOPMENT - being completed NOW
- ✅ Basic pipeline functional (spec→IR→code stubs)
- ✅ Deterministic execution architecture in place
- 🔨 Complete IR transformation logic in development
- 🔨 Full code generation in development
- 🔨 Receipt generation planned (Phase 3)
- 🔨 Formal verification with SPARK proofs in progress

**Current Capabilities**:
- Can convert specs to IR (basic functionality)
- Can generate code stubs in multiple languages
- Command-line tools functional
- Deterministic processing enforced

**In Development**:
- Complete transformation pipeline
- Full code body generation
- Receipt and verification system
- SPARK formal proofs

**Python implementations** in `tools/*.py` are **UNDER DEVELOPMENT** (incomplete):
- Status: Partial functionality, not aligned with current SPARK development
- Purpose: Will become full production alternative when complete
- Timeline: Development begins AFTER SPARK completion
- Current use: Can read for algorithm understanding only
