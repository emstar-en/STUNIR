# STUNIR Component Architecture

> Part of `docs/architecture/1139`

## Component Overview

### 1. IR Emitter (`tools/ir_emitter/`)
**Purpose:** Convert spec files to deterministic IR

**Key Functions:**
- `spec_to_ir()` - Transform spec to IR format
- `canonical_json()` - RFC 8785 compliant output
- `compute_sha256()` - Hash computation for manifests

**Files:**
- `emit_ir.py` - Main IR emission logic

### 2. Canonicalizers (`tools/canonicalizers/`)
**Purpose:** Ensure deterministic output formats

**Capabilities:**
- JSON canonicalization (sorted keys, no whitespace variance)
- dCBOR encoding for binary artifacts
- Hash verification

### 3. Target Emitters (`targets/`)
**Purpose:** Generate platform-specific code

**Structure:**
```
targets/
├── asm/ir/         # ASM/IR format
├── polyglot/       # C89, C99, Rust
├── assembly/       # x86, ARM
├── wasm/           # WebAssembly
├── bytecode/       # Bytecode targets
├── gpu/            # GPU compute
├── fpga/           # FPGA synthesis
├── embedded/       # Embedded systems
└── mobile/         # Mobile platforms
```

### 4. Manifest System (`manifests/`)
**Purpose:** Track and verify all artifacts

**Manifest Types:**
- IR Manifest - `receipts/ir_manifest.json`
- Receipts Manifest - `receipts/receipts_manifest.json`
- Contracts Manifest - `receipts/contracts_manifest.json`
- Targets Manifest - `receipts/targets_manifest.json`
- Pipeline Manifest - `receipts/pipeline_manifest.json`

**Schema:** `stunir.manifest.<type>.v1`

### 5. Native Tools

#### Haskell Native (`tools/native/haskell/`)
- `Stunir.Manifest` - Deterministic manifest generation
- `Stunir.Provenance` - Build provenance tracking
- `Main.hs` - CLI entry point

#### Rust Native (`tools/native/rust/`)
- High-performance parsing
- Memory-safe operations

### 6. Verification (`scripts/`)
**Key Scripts:**
- `build.sh` - Polyglot build entrypoint
- `verify.sh` - Standard verification
- `verify_strict.sh` - Strict manifest verification

## Data Flow

```
┌─────────┐     ┌─────────┐     ┌─────────┐
│  Spec   │ ──▶ │   IR    │ ──▶ │ Targets │
└─────────┘     └─────────┘     └─────────┘
                     │               │
                     ▼               ▼
               ┌─────────┐     ┌─────────┐
               │Manifests│ ◀── │Receipts │
               └─────────┘     └─────────┘
                     │
                     ▼
               ┌─────────┐
               │ Verify  │
               └─────────┘
```

## Related
- [Architecture Overview](README.md)
- [Internals - IR Format](../internals/ir_format.md)

---
*STUNIR Components v1.0*
