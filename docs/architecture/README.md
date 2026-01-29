# STUNIR Architecture Documentation

> Issue: `docs/architecture/1139` - Complete docs → architecture pipeline stage

## Overview

STUNIR (Structured Universal Native Intermediate Representation) is a deterministic build pipeline for generating reproducible, verifiable code artifacts across multiple target platforms.

## Core Principles

### 1. Determinism
- All outputs are reproducible given the same inputs
- SHA-256 hashes verify artifact integrity
- Canonical JSON encoding (RFC 8785/JCS subset)
- Epoch-based timestamps for reproducibility

### 2. Multi-Target Support
- Native: Haskell, Rust
- Polyglot: C89, C99, Python
- Assembly: x86, ARM
- Specialized: WASM, FPGA, GPU, Mobile

### 3. Pipeline Architecture
```
Spec → IR → Targets → Receipts → Verification
```

## System Components

### Pipeline Stages
| Stage | Description | Tools |
|-------|-------------|-------|
| Spec Parsing | Parse input specifications | `tools/parsers/` |
| IR Emission | Generate intermediate representation | `tools/ir_emitter/` |
| Target Emission | Generate platform-specific code | `tools/emitters/`, `targets/` |
| Manifest Generation | Create deterministic manifests | `manifests/` |
| Receipt Generation | Produce verification receipts | `tools/receipt_emitter/` |
| Verification | Verify build integrity | `scripts/verify.sh` |

### Key Directories
- `tools/` - Core pipeline tools
- `targets/` - Target-specific emitters
- `manifests/` - Manifest generators/verifiers
- `contracts/` - Build contracts and profiles
- `receipts/` - Generated verification receipts
- `asm/ir/` - Intermediate representation artifacts

## Build Profiles

| Profile | Runtime | Use Case |
|---------|---------|----------|
| Profile 1 | Haskell Native | Full determinism, optimal |
| Profile 2 | Python | Rapid development |
| Profile 3 | Shell | Minimal dependencies |
| Profile 4 | Rust Native | High performance |

## Related Documentation
- [API Reference](../api/README.md)
- [Internals](../internals/README.md)
- [Design Documents](../design/README.md)
- [Components Detail](components.md)

---
*STUNIR Architecture Documentation v1.0*
