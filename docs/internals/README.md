# STUNIR Internals Documentation

> Issue: `docs/internals/1138` - Complete docs → internals pipeline stage

## Overview

This documentation covers the internal workings of STUNIR, including the IR format, determinism guarantees, and implementation details.

## Contents

| Topic | Description |
|-------|-------------|
| [IR Format](ir_format.md) | Intermediate Representation specification |
| [Determinism](determinism.md) | Determinism guarantees and implementation |

## Key Concepts

### Intermediate Representation (IR)

STUNIR uses a JSON-based IR format that:
- Captures function definitions, types, and module structure
- Is fully deterministic (canonical JSON encoding)
- Can be serialized to dCBOR for compact binary storage
- Includes cryptographic hashes for verification

### Pipeline Stages

1. **Parsing**: Spec files → AST
2. **IR Emission**: AST → Canonical IR
3. **Target Emission**: IR → Platform code
4. **Receipt Generation**: Artifacts → Verification receipts
5. **Manifest Generation**: Receipts → Deterministic manifests

### Hash Chain

Every artifact is connected via SHA-256 hashes:

```
spec_hash → ir_hash → artifact_hash → receipt_hash → manifest_hash
```

This chain enables full provenance tracking and verification.

## File Formats

### IR Files (`.json`, `.dcbor`)
- Location: `asm/ir/`
- Schema: `stunir.ir.v1`

### Receipt Files (`.json`)
- Location: `receipts/`
- Schema: `stunir.receipt.v1`

### Manifest Files (`.json`)
- Location: `receipts/`
- Schema: `stunir.manifest.<type>.v1`

## Related
- [Architecture](../architecture/README.md)
- [IR Format](ir_format.md)
- [Determinism](determinism.md)

---
*STUNIR Internals v1.0*
