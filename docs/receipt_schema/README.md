# STUNIR Receipt Schema Documentation

> Issue: `docs/receipt_schema/1012` - Complete docs → receipt_schema pipeline stage

## Overview

This documentation covers the STUNIR receipt and manifest schemas used for build verification.

## Contents

| Document | Description |
|----------|-------------|
| [Schema v1](schema_v1.md) | Receipt bundle schema v1 specification |

## Schema Categories

### 1. Receipt Schema
**Purpose:** Track individual artifact provenance.

**Version:** `stunir.receipt.v1`

**Fields:**
- `schema` - Schema identifier
- `epoch` - Build timestamp
- `receipt_type` - Type of receipt
- `artifact` - Artifact metadata
- `inputs` - Input dependencies
- `receipt_hash` - Self-referential hash

### 2. Manifest Schema
**Purpose:** Aggregate multiple artifacts.

**Versions:**
- `stunir.manifest.ir.v1` - IR artifacts
- `stunir.manifest.receipts.v1` - Receipt files
- `stunir.manifest.contracts.v1` - Build contracts
- `stunir.manifest.targets.v1` - Generated targets
- `stunir.manifest.pipeline.v1` - Pipeline stages
- `stunir.manifest.runtime.v1` - Runtime artifacts
- `stunir.manifest.security.v1` - Security artifacts
- `stunir.manifest.performance.v1` - Performance metrics

## Quick Reference

### Creating Receipts
```python
from tools.receipt_emitter import emit_receipt

receipt = emit_receipt.generate(
    artifact_path='output.json',
    receipt_type='build'
)
```

### Creating Manifests
```bash
# IR Manifest
python -m manifests.ir.gen_ir_manifest

# Receipts Manifest
python -m manifests.receipts.gen_receipts_manifest
```

### Verification
```bash
# Verify manifest
python -m manifests.ir.verify_ir_manifest receipts/ir_manifest.json

# Strict verification
./scripts/verify_strict.sh --strict
```

## Schema Evolution

| Version | Status | Changes |
|---------|--------|--------|
| v1 | Active | Initial release |

## Related
- [Schema v1 Specification](schema_v1.md)
- [Receipts Design](../design/receipts.md)
- [API Reference](../api/manifests.md)

---
*STUNIR Receipt Schema Documentation v1.0*
