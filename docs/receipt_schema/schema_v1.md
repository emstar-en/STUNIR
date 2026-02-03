# Receipt Bundle Schema v1

> Issue: `docs/receipt_schema/1007` - Receipt bundle schema v1

## Overview

This document specifies the Receipt Bundle Schema v1 (`stunir.receipt.v1`) used for artifact verification in STUNIR.

## Schema Definition

### Receipt Schema v1

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "stunir.receipt.v1",
  "title": "STUNIR Receipt Schema v1",
  "type": "object",
  "required": ["schema", "epoch", "receipt_type", "artifact", "receipt_hash"],
  "properties": {
    "schema": {
      "type": "string",
      "const": "stunir.receipt.v1"
    },
    "epoch": {
      "type": "integer",
      "description": "Unix timestamp of receipt generation"
    },
    "receipt_type": {
      "type": "string",
      "enum": ["build", "ir", "target", "manifest", "verification"]
    },
    "artifact": {
      "$ref": "#/$defs/artifact"
    },
    "inputs": {
      "type": "array",
      "items": {"$ref": "#/$defs/input"}
    },
    "receipt_hash": {
      "type": "string",
      "pattern": "^sha256:[a-f0-9]{64}$"
    }
  },
  "$defs": {
    "artifact": {
      "type": "object",
      "required": ["name", "hash"],
      "properties": {
        "name": {"type": "string"},
        "path": {"type": "string"},
        "hash": {"type": "string", "pattern": "^sha256:[a-f0-9]{64}$"},
        "size": {"type": "integer"}
      }
    },
    "input": {
      "type": "object",
      "required": ["name", "hash"],
      "properties": {
        "name": {"type": "string"},
        "hash": {"type": "string", "pattern": "^sha256:[a-f0-9]{64}$"}
      }
    }
  }
}
```

## Field Specifications

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `schema` | string | Must be `"stunir.receipt.v1"` |
| `epoch` | integer | Unix timestamp (seconds since 1970-01-01) |
| `receipt_type` | string | One of: `build`, `ir`, `target`, `manifest`, `verification` |
| `artifact` | object | Primary artifact being receipted |
| `receipt_hash` | string | SHA-256 hash of receipt content (excluding this field) |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `inputs` | array | Input artifacts that produced this artifact |

### Artifact Object

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `name` | Yes | string | Artifact identifier |
| `hash` | Yes | string | SHA-256 content hash |
| `path` | No | string | Relative file path |
| `size` | No | integer | File size in bytes |

### Input Object

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `name` | Yes | string | Input identifier |
| `hash` | Yes | string | SHA-256 content hash |

## Receipt Types

### Build Receipt
```json
{
  "schema": "stunir.receipt.v1",
  "epoch": 1735500000,
  "receipt_type": "build",
  "artifact": {
    "name": "full_build",
    "path": "output/",
    "hash": "sha256:abc123..."
  },
  "inputs": [
    {"name": "spec.json", "hash": "sha256:def456..."}
  ],
  "receipt_hash": "sha256:789ghi..."
}
```

### IR Receipt
```json
{
  "schema": "stunir.receipt.v1",
  "epoch": 1735500000,
  "receipt_type": "ir",
  "artifact": {
    "name": "module.dcbor",
    "path": "asm/ir/module.dcbor",
    "hash": "sha256:abc123...",
    "size": 2048
  },
  "inputs": [
    {"name": "module_spec.json", "hash": "sha256:def456..."}
  ],
  "receipt_hash": "sha256:789ghi..."
}
```

## Hash Computation

The `receipt_hash` is computed by:
1. Serialize receipt without `receipt_hash` field
2. Convert to canonical JSON (sorted keys, no whitespace)
3. Compute SHA-256 of UTF-8 bytes
4. Prefix with `sha256:`

```python
import json
import hashlib

def compute_receipt_hash(receipt):
    # Remove hash field for computation
    receipt_copy = {k: v for k, v in receipt.items() if k != 'receipt_hash'}
    
    # Canonical JSON
    canonical = json.dumps(receipt_copy, sort_keys=True, separators=(',', ':'))
    
    # SHA-256
    h = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    return f"sha256:{h}"
```

## Validation

```python
def validate_receipt(receipt):
    """Validate receipt against schema v1."""
    assert receipt['schema'] == 'stunir.receipt.v1'
    assert isinstance(receipt['epoch'], int)
    assert receipt['receipt_type'] in ['build', 'ir', 'target', 'manifest', 'verification']
    assert 'name' in receipt['artifact']
    assert 'hash' in receipt['artifact']
    
    # Verify hash
    expected_hash = compute_receipt_hash(receipt)
    assert receipt['receipt_hash'] == expected_hash
```

## Related
- [Receipt Schema Overview](README.md)
- [Receipts Design](../design/receipts.md)

---
*STUNIR Receipt Bundle Schema v1.0*
