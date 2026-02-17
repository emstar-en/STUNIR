# STUNIR Powertools Status

**Last Updated**: February 17, 2026  
**Powertools Version**: `0.1.0-alpha` (NOT built or tested)

---

## What Are STUNIR Powertools?

**Powertools** are small, composable, single-purpose command-line utilities designed for **AI-driven workflows**. Instead of monolithic tools, powertools follow the Unix philosophy:

- ✅ **Do one thing well** - Each tool has a focused purpose
- ✅ **Compose via pipes** - Tools can be chained together
- ✅ **AI-introspectable** - All tools support `--describe` for JSON schema output
- ✅ **Exit codes** - Clear success/failure signaling (0=success, 1=validation error, 2=processing error, 3=resource error)
- ✅ **JSON I/O** - Structured input/output for machine consumption

**Key Insight**: AI models can discover, chain, and orchestrate these tools dynamically based on their `--describe` metadata.

---

## Current Status: ⚠️ NOT BUILT

**Critical Issue**: Powertools source code exists but **no executables are built**.

| Status | Description |
|--------|-------------|
| ❌ **Not Compiled** | No binaries in `tools/spark/bin/` (only 2 main tools exist) |
| ❌ **Version Mismatch** | All claim `1.0.0` but are clearly alpha prototypes |
| ❌ **No Build System** | No `powertools.gpr` or makefile for batch compilation |
| ❌ **Untested** | No test suite or integration tests |
| ⚠️ **AI-Ready Design** | `--describe` introspection implemented but untested |

**Reality Check**: These are `0.1.0-alpha` prototypes, not production tools.

---

## Powertools Inventory (19 Tools)

### Phase 1: Foundation Tools (6 tools)

| Tool | Purpose | Version | Status | AI-Introspectable |
|------|---------|---------|--------|-------------------|
| **json_validate** | Validate JSON syntax and structure | 1.0.0 → `0.1.0-alpha` | ❌ Not built | ✅ Yes |
| **json_extract** | Extract values from JSON by path | 1.0.0 → `0.1.0-alpha` | ❌ Not built | ✅ Yes |
| **json_merge** | Merge multiple JSON documents | ❓ No version | ❌ Not built | ❓ Unknown |
| **type_normalize** | Normalize type names across languages | 1.0.0 → `0.1.0-alpha` | ❌ Not built | ❓ Unknown |
| **type_map** | Map types between languages | 1.0.0 → `0.1.0-alpha` | ❌ Not built | ❓ Unknown |
| **func_dedup** | Deduplicate function signatures | 1.0.0 → `0.1.0-alpha` | ❌ Not built | ❓ Unknown |

**Purpose**: Low-level JSON manipulation and type system utilities.

### Phase 2: Code Analysis Tools (3 tools)

| Tool | Purpose | Version | Status | AI-Introspectable |
|------|---------|---------|--------|-------------------|
| **format_detect** | Detect source code format/language | 1.0.0 → `0.1.0-alpha` | ❌ Not built | ✅ Yes |
| **lang_detect** | Language detection for source files | ❓ No version | ❌ Not built | ❓ Unknown |
| **extraction_to_spec** | Extract spec from existing code | 1.0.0 → `0.1.0-alpha` | ❌ Not built | ✅ Yes |

**Purpose**: Code analysis and reverse engineering (Code→Spec pipeline).

### Phase 3: Code Generation Tools (4 tools)

| Tool | Purpose | Version | Status | AI-Introspectable |
|------|---------|---------|--------|-------------------|
| **sig_gen_cpp** | Generate C++ function signatures | 1.0.0 → `0.1.0-alpha` | ❌ Not built | ✅ Yes |
| **sig_gen_rust** | Generate Rust function signatures | 1.0.0 → `0.1.0-alpha` | ❌ Not built | ✅ Yes |
| **sig_gen_python** | Generate Python function signatures | ❓ No version | ❌ Not built | ❓ Unknown |
| **type_resolve** | Resolve type dependencies | ❓ No version | ❌ Not built | ❓ Unknown |

**Purpose**: Language-specific code emission.

### Verification & Attestation Tools (6 tools)

| Tool | Purpose | Version | Status | AI-Introspectable |
|------|---------|---------|--------|-------------------|
| **hash_compute** | Compute SHA-256 hashes | ❓ No version | ❌ Not built | ✅ Yes |
| **receipt_generate** | Generate verification receipts | ❓ No version | ❌ Not built | ✅ Yes |
| **toolchain_verify** | Verify toolchain.lock integrity | ❓ No version | ❌ Not built | ✅ Yes |
| **spec_validate** | Validate spec JSON against schema | ❓ No version | ❌ Not built | ❓ Unknown |
| **ir_validate** | Validate IR JSON against schema | ❓ No version | ❌ Not built | ❓ Unknown |
| **file_indexer** | Index files for verification | ❓ No version | ❌ Not built | ❓ Unknown |

**Purpose**: Cryptographic attestation, receipts, and verification for AI safety.

---

## AI-Introspection Design

All powertools implement `--describe` flag that outputs JSON schema:

```json
{
  "tool": "json_validate",
  "version": "0.1.0-alpha",
  "description": "Validates JSON structure and syntax",
  "inputs": [{
    "name": "json_input",
    "type": "json",
    "source": ["stdin", "file"],
    "required": true
  }],
  "outputs": [{
    "name": "validation_result",
    "type": "exit_code",
    "description": "0=valid, 1=invalid, 2=error"
  }],
  "complexity": "O(n)",
  "options": ["--help", "--version", "--describe", "--strict", "--verbose"]
}
```

**AI models can**:
1. Query `tool --describe` to understand capabilities
2. Chain tools based on input/output types
3. Handle errors via exit codes
4. Compose complex pipelines dynamically

**Example AI Workflow**:
```bash
# AI discovers these tools via --describe, then chains them:
cat spec.json | json_validate --strict && \
  json_extract --path ".functions[0]" | \
  sig_gen_rust --output func.rs && \
  hash_compute func.rs | \
  receipt_generate --spec spec.json --manifest manifest.json
```

---

## Critical Gaps for Complete Spec→Code→Env Pipeline

### ❌ Missing Powertools

1. **env_setup** - Setup execution environment (Docker, containers, VMs)
2. **dependency_resolve** - Resolve and install language dependencies
3. **build_orchestrator** - Compile/build generated code
4. **test_runner** - Execute generated code with test inputs
5. **spec_merge** - Merge multiple spec files
6. **ir_transform** - Apply IR optimization passes
7. **code_format** - Auto-format generated code
8. **doc_generate** - Generate documentation from specs
9. **contract_validate** - Validate pre/post conditions
10. **error_annotate** - Annotate error messages with context

### ❌ Missing Build Infrastructure

1. **No GPR project files** - Can't build powertools individually
2. **No makefile** - No batch compilation script
3. **No install script** - No way to deploy to `tools/spark/bin/`
4. **No test suite** - No validation that tools work
5. **No CI pipeline** - No automated builds

### ❌ Missing Documentation

1. **No usage examples** - How to chain tools together
2. **No AI workflow guides** - How models should use these
3. **No API contracts** - Unclear input/output formats
4. **No error handling guide** - Exit code semantics unclear

---

## Complete Ada SPARK Powertool Chain Requirements

For a **production-ready** Ada SPARK powertool ecosystem, we need:

### 1. Core Pipeline Tools ✅ (Partially Complete)
- ✅ `spec_to_ir` - Convert specs to IR (exists as monolithic tool)
- ✅ `ir_to_code` - Convert IR to code (exists as monolithic tool)
- ❌ `code_to_spec` - Reverse: extract specs from code (missing)
- ❌ `ir_to_ir` - IR transformation/optimization (missing)

### 2. Validation & Verification Tools ⚠️ (Incomplete)
- ✅ `json_validate`, `spec_validate`, `ir_validate` (source exists, not built)
- ✅ `hash_compute`, `receipt_generate`, `toolchain_verify` (source exists, not built)
- ❌ `signature_verify` - Verify cryptographic signatures (missing)
- ❌ `proof_check` - SPARK proof obligation checker integration (missing)

### 3. Code Generation Tools ⚠️ (Incomplete)
- ✅ `sig_gen_cpp`, `sig_gen_rust`, `sig_gen_python` (source exists, not built)
- ❌ `sig_gen_c`, `sig_gen_go`, `sig_gen_java`, etc. (missing)
- ❌ `body_gen_*` - Generate function bodies, not just signatures (missing)
- ❌ `import_gen_*` - Generate language imports/uses (missing)

### 4. Type System Tools ⚠️ (Incomplete)
- ✅ `type_map`, `type_normalize`, `type_resolve` (source exists, not built)
- ❌ `type_infer` - Infer types from usage (missing)
- ❌ `type_check` - Validate type consistency (missing)

### 5. Environment & Execution Tools ❌ (Missing)
- ❌ `env_create` - Create isolated execution environment
- ❌ `env_destroy` - Clean up environments
- ❌ `package_install` - Install language dependencies
- ❌ `build_run` - Compile and build code
- ❌ `test_execute` - Run tests in environment

### 6. AI Orchestration Tools ⚠️ (Partial)
- ✅ `--describe` introspection (implemented in some tools)
- ❌ `pipeline_compose` - Auto-compose tool pipelines (missing)
- ❌ `tool_discover` - Scan for available powertools (missing)
- ❌ `workflow_validate` - Validate tool chains (missing)

---

## Recommended Actions

### Immediate (v0.1.0-alpha → v0.2.0-alpha)

1. **Update all powertools versions** from `1.0.0` → `0.1.0-alpha`
2. **Create `powertools.gpr`** - GNAT project to build all 19 tools
3. **Build and test 5 core tools**:
   - `json_validate` - Critical for validation
   - `hash_compute` - Critical for attestation
   - `receipt_generate` - Critical for verification
   - `toolchain_verify` - Critical for determinism
   - `sig_gen_rust` - Test code generation path

4. **Document AI workflows** - Show how to chain tools

### Short-term (v0.2.0-alpha → v0.3.0-alpha)

1. **Build remaining 14 powertools**
2. **Add missing tools**: `env_setup`, `dependency_resolve`, `build_orchestrator`
3. **Create integration tests** - Validate full pipelines
4. **Add `--json` output** to all tools for machine consumption

### Long-term (v0.3.0-alpha → v1.0.0)

1. **Complete missing tools** (10+ identified above)
2. **AI orchestration layer** - Auto-discovery and composition
3. **SPARK formal verification** - Prove tool correctness
4. **Production deployment** - CI/CD, packaging, distribution

---

## Versioning Status

| Component | Current | Should Be | Notes |
|-----------|---------|-----------|-------|
| **Powertools (source)** | `1.0.0` | `0.1.0-alpha` | Inflated, not production |
| **Powertools (binaries)** | N/A | `0.1.0-alpha` | Not built |
| **Build system** | None | `0.1.0-alpha` | Needs creation |
| **Documentation** | Minimal | `0.1.0-alpha` | This file is a start |

---

## Summary

**STUNIR Powertools are a brilliant design** for AI-driven code generation:
- Small, composable, introspectable tools
- JSON-based machine interfaces
- Cryptographic attestation built-in
- Exit code-based error handling

**BUT**: They're currently **vaporware** - source code exists but nothing is built, tested, or deployable.

**To reach v1.0.0**, we need:
- ✅ Build all 19 existing powertools
- ✅ Add ~10 missing critical tools
- ✅ Create build/test infrastructure
- ✅ Document AI workflows
- ✅ Integrate with formal verification

**Current realistic state: 0.1.0-alpha** - Interesting prototypes, not production tools.
