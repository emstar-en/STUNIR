# STUNIR Pipeline Design

> Issue: `docs/design/pipeline/1037` - Complete docs → design → pipeline pipeline stage

## Overview

The STUNIR build pipeline transforms specifications into deterministic, verifiable code artifacts.

## Pipeline Stages

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  PARSE   │───▶│   IR     │───▶│  TARGET  │───▶│ MANIFEST │───▶│  VERIFY  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
    │               │               │               │               │
    ▼               ▼               ▼               ▼               ▼
spec.json      ir.json      targets/*    manifests/*   PASS/FAIL
               ir.dcbor     receipts/*
```

## Stage Details

### Stage 1: Parse

**Purpose:** Parse and validate input specifications.

**Input:**
- `spec.json` - Module specification

**Output:**
- Validated spec AST
- Parse receipts

**Tools:**
- `tools/parsers/spec_parser.py`
- `tools/validators/spec_validator.py`

### Stage 2: IR Emission

**Purpose:** Generate deterministic Intermediate Representation.

**Input:**
- Validated spec AST

**Output:**
- `asm/ir/<module>.json` - Canonical JSON IR
- `asm/ir/<module>.dcbor` - dCBOR binary IR

**Tools:**
- `tools/ir_emitter/emit_ir.py`
- `scripts/lib/emit_dcbor.sh`

### Stage 3: Target Emission

**Purpose:** Generate platform-specific code from IR.

**Input:**
- IR artifacts from Stage 2

**Output:**
- Platform code in `targets/<platform>/`
- Build scripts (Makefiles, Cargo.toml, etc.)

**Tools:**
- `tools/emitters/emit_code.py`
- Target-specific emitters in `targets/*/emitter.py`

### Stage 4: Manifest Generation

**Purpose:** Create deterministic manifests for all artifacts.

**Input:**
- All generated artifacts

**Output:**
- `receipts/ir_manifest.json`
- `receipts/targets_manifest.json`
- `receipts/receipts_manifest.json`
- `receipts/pipeline_manifest.json`

**Tools:**
- `manifests/*/gen_*_manifest.py`
- `stunir-native gen-ir-manifest`

### Stage 5: Verification

**Purpose:** Verify build integrity and determinism.

**Input:**
- Manifests from Stage 4
- All artifacts

**Output:**
- PASS/FAIL status
- Verification report

**Tools:**
- `scripts/verify.sh`
- `scripts/verify_strict.sh`
- `manifests/*/verify_*_manifest.py`

## Build Profiles

| Profile | Stages Available | Notes |
|---------|------------------|-------|
| Native (Haskell) | All | Full determinism |
| Python | All | Development mode |
| Shell | Stages 1-3 | Minimal deps |
| Rust | All | High performance |

## Execution

```bash
# Full pipeline
./scripts/build.sh

# Profile-specific
./scripts/build.sh --profile=native
./scripts/build.sh --profile=python

# Individual stages
python -m tools.ir_emitter.emit_ir spec.json
python -m manifests.ir.gen_ir_manifest
./scripts/verify.sh
```

## Related
- [Design Overview](README.md)
- [Receipts Design](receipts.md)
- [Architecture](../architecture/README.md)

---
*STUNIR Pipeline Design v1.0*
