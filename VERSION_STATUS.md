# STUNIR Version Status

**Last Updated**: February 17, 2026  
**Overall Project Status**: `0.1.0-alpha` (ALPHA Prototype)

## Overall Project Version

**Version**: `0.1.0-alpha`  
**Status**: ALPHA Prototype - Experimental/Testing Phase  
**Location**: `pyproject.toml`

## What `0.1.0-alpha` Means

- **0.x.x** = Pre-production development
- **0.1.x** = Early experimental stage with basic functionality
- **-alpha** = Alpha testing phase - functional but incomplete

**NOT production ready** - This is an early prototype for testing and development.

---

## Component Versions

### Ada SPARK Pipeline (Primary Implementation)

**Status**: ✅ Functional - Basic Spec→IR→Code pipeline working

| Component | Version | File | Status | Capabilities |
|-----------|---------|------|--------|--------------|
| **IR to Code Emitter** | `0.1.0-alpha` | `tools/spark/src/stunir_ir_to_code.ads` | ✅ Functional | IR→Code for C/Rust/Python (function stubs) |
| **Spec to IR Converter** | No version constant | `tools/spark/src/stunir_spec_to_ir.ads` | ✅ Functional | Spec→IR conversion, manifest generation |
| **Toolchain** | `0.1.0-alpha` | `local_toolchain.lock.json` | ✅ Locked | Deterministic tool versioning |

**Tested Capabilities**:
- ✅ Spec JSON → IR JSON conversion
- ✅ IR JSON → Code generation (C, Rust, Python)
- ✅ Function stub generation
- ❌ Full function implementation (not implemented)
- ❌ Code → Spec reverse pipeline (not implemented)

### Python Pipeline (Alternative Implementation)

**Status**: ⚠️ Untested - Alternative to Ada SPARK

| Component | Version | Status | Notes |
|-----------|---------|--------|-------|
| Python bridge scripts | TBD | ⚠️ Untested | Alternative when SPARK unavailable |
| Python emitters | Various | ⚠️ Mixed | Many emitter versions at 1.0.0 or 2.0.0 |

### Utility Libraries

| Component | Version | File | Status |
|-----------|---------|------|--------|
| **String Builder** | `0.7.0` | `tools/spark/src/stunir_string_builder.ads` | ✅ Stable |
| **Optimizer** | `0.8.9` | `tools/spark/src/stunir_optimizer.ads` | ⚠️ Needs version review |
| **Semantic IR Schema** | `1.0.0` | `tools/spark/src/semantic_ir/semantic_ir.ads` | ✅ Schema stable |

### Ada SPARK Powertools (Composable AI-Driven Tools)

**Status**: ❌ Not Built - Source code exists but no executables compiled

**See**: [`POWERTOOLS_STATUS.md`](./POWERTOOLS_STATUS.md) for complete details

| Category | Tools | Version | Status |
|----------|-------|---------|--------|
| **Foundation Tools** | json_validate, json_extract, json_merge, type_normalize, type_map, func_dedup | `1.0.0` → `0.1.0-alpha` | ❌ Source only, not built |
| **Code Analysis** | format_detect, lang_detect, extraction_to_spec | `1.0.0` → `0.1.0-alpha` | ❌ Source only, not built |
| **Code Generation** | sig_gen_cpp, sig_gen_rust, sig_gen_python, type_resolve | `1.0.0` → `0.1.0-alpha` | ❌ Source only, not built |
| **Verification** | hash_compute, receipt_generate, toolchain_verify, spec_validate, ir_validate, file_indexer | Various → `0.1.0-alpha` | ❌ Source only, not built |

**Total**: 19 powertools defined, 0 built

**Key Design Features**:
- ✅ AI-introspectable via `--describe` flag
- ✅ Composable via Unix pipes
- ✅ JSON I/O for machine consumption
- ✅ Attestation/receipt generation for AI safety
- ❌ No build system (`.gpr` files needed)
- ❌ No test suite

**Critical Gaps**: See `POWERTOOLS_STATUS.md` for ~10 missing tools needed for complete Spec→Code→Env pipeline.

### Language Emitters (Planned/Incomplete)

Most emitters in `src/ada/targets/` report version `1.0.0` but are **not tested** in the current pipeline.

---

## What Works in 0.1.0-alpha

### ✅ Functional

1. **Ada SPARK Spec→IR→Code Pipeline**
   - Input: Spec JSON files
   - Output: IR JSON + function stubs (C/Rust/Python)
   - Tools: `stunir_spec_to_ir_main.exe` + `stunir_ir_to_code_main.exe`

2. **Supported Languages (stub generation only)**
   - C (C99)
   - Rust
   - Python

3. **Example Workflow**
   ```powershell
   # Tested and working
   .\tools\spark\bin\stunir_spec_to_ir_main.exe --spec-root spec\examples --out test_ir.json
   .\tools\spark\bin\stunir_ir_to_code_main.exe --input test_ir.json --output output.c --target c
   ```

### ❌ Not Implemented

1. **Full Code Implementation**
   - Current: Function stubs with TODO comments
   - Missing: Complete function body generation

2. **Code→Spec Reverse Pipeline**
   - Extract specs from existing code

3. **Additional Language Support**
   - JavaScript, TypeScript, Go, Java, C#, etc. (listed but untested)
   - WASM, Assembly (not implemented)

4. **Advanced Features**
   - Control flow (if/while/for)
   - Error handling (try/catch)
   - Module system
   - Generics/templates

---

## Version Roadmap to 1.0.0

According to `docs/reports/general/VERSIONING_STRATEGY.md`, **v1.0.0 requires**:

- ✅ All 4 pipelines (Python, Rust, SPARK, **Haskell**) at 100%
- ✅ ZERO known critical or high-severity issues
- ✅ Comprehensive testing
- ✅ Full documentation
- ✅ Production deployments validated
- ✅ Security audit completed

**Current Status**: Only SPARK pipeline partially functional → **0.1.0-alpha is accurate**

---

## Next Version Milestones

### 0.1.x → 0.2.0 (Next Minor)
- Complete function body generation (not just stubs)
- Test and verify Python pipeline
- Add control flow support

### 0.2.x → 0.3.0
- Implement Code→Spec reverse pipeline
- Add error handling support

### 0.3.x → 1.0.0 (Production)
- Complete all 4 pipelines (Python, Rust, SPARK, Haskell)
- Full language support for all listed targets
- Production testing and security audit

---

## Version History

| Version | Date | Status | Notes |
|---------|------|--------|-------|
| `0.8.9` | Jan 2026 | Deprecated | Inflated version - did not reflect actual state |
| `0.7.1` | Jan 2026 | Deprecated | Tool-specific version before overall project versioning |
| `0.1.0-alpha` | Feb 17, 2026 | **Current** | Accurate ALPHA prototype versioning |

---

## How to Check Versions

**Overall Project**:
```bash
grep "version" pyproject.toml
```

**Ada SPARK Tools**:
```bash
.\tools\spark\bin\stunir_ir_to_code_main.exe --version
```

**Toolchain Lock**:
```bash
cat local_toolchain.lock.json
```

---

## Versioning Policy

STUNIR follows **Semantic Versioning 2.0.0** adapted for pre-1.0 development:

- **MAJOR.MINOR.PATCH**
- **0.x.x** = Pre-production (current)
- **1.0.0** = First production-ready release
- **-alpha/-beta** = Pre-release designations

See `docs/reports/general/VERSIONING_STRATEGY.md` for full policy.
