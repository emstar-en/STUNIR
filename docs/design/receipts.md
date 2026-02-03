# STUNIR Receipts Design

> Issue: `docs/design/receipts/1038` - Complete docs → design → receipts pipeline stage

## Overview

The receipts system provides verifiable proof of build artifacts and their provenance.

## Receipt Purpose

Receipts serve three key functions:
1. **Integrity**: Prove artifact content via SHA-256 hashes
2. **Provenance**: Track artifact origin and dependencies
3. **Reproducibility**: Enable deterministic rebuild verification

## Receipt Schema

### Version: `stunir.receipt.v1`

```json
{
  "schema": "stunir.receipt.v1",
  "epoch": 1735500000,
  "receipt_type": "build|ir|target|manifest",
  "artifact": {
    "name": "<artifact_name>",
    "path": "<relative_path>",
    "hash": "sha256:<hex_digest>",
    "size": 1024
  },
  "inputs": [
    {
      "name": "<input_name>",
      "hash": "sha256:<hex_digest>"
    }
  ],
  "receipt_hash": "sha256:<hex_digest>"
}
```

## Receipt Types

### Build Receipt
Generated for complete build runs.

```json
{
  "receipt_type": "build",
  "artifact": {
    "name": "build_output",
    "path": "output/"
  },
  "build_profile": "native",
  "stages_completed": ["parse", "ir", "target", "manifest"]
}
```

### IR Receipt
Generated for IR emission.

```json
{
  "receipt_type": "ir",
  "artifact": {
    "name": "module.dcbor",
    "path": "asm/ir/module.dcbor"
  },
  "inputs": [
    {"name": "spec.json", "hash": "sha256:..."}
  ]
}
```

### Target Receipt
Generated for target code emission.

```json
{
  "receipt_type": "target",
  "artifact": {
    "name": "module.rs",
    "path": "targets/polyglot/rust/src/module.rs"
  },
  "target_type": "rust",
  "inputs": [
    {"name": "module.json", "hash": "sha256:..."}
  ]
}
```

### Manifest Receipt
Generated for manifest creation.

```json
{
  "receipt_type": "manifest",
  "artifact": {
    "name": "ir_manifest.json",
    "path": "receipts/ir_manifest.json"
  },
  "manifest_type": "ir",
  "entry_count": 5
}
```

## Storage

Receipts are stored in `receipts/` directory:

```
receipts/
├── build_receipt.json
├── ir_manifest.json
├── receipts_manifest.json
├── targets_manifest.json
├── contracts_manifest.json
└── pipeline_manifest.json
```

## Verification

### Receipt Verification
```python
from manifests.receipts.verify_receipts_manifest import ReceiptsManifestVerifier

verifier = ReceiptsManifestVerifier()
valid, errors, warnings = verifier.verify('receipts/receipts_manifest.json')
```

### CLI Verification
```bash
# Verify all receipts
python -m manifests.receipts.verify_receipts_manifest receipts/receipts_manifest.json

# Strict verification
./scripts/verify_strict.sh --strict
```

## Hash Chain

Receipts form a hash chain enabling full provenance:

```
spec_hash ────▶ ir_receipt.hash
                   │
                   ▼
           target_receipt.hash
                   │
                   ▼
           manifest_receipt.hash
                   │
                   ▼
           receipts_manifest.hash
```

## Related
- [Design Overview](README.md)
- [Pipeline Design](pipeline.md)
- [Receipt Schema Documentation](../receipt_schema/README.md)

---
*STUNIR Receipts Design v1.0*
