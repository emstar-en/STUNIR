# STUNIR Tools API

> Part of `docs/api/1068`

## IR Emitter (`tools/ir_emitter/`)

### `emit_ir.py`

**Functions:**

#### `spec_to_ir(spec_data: dict) -> dict`
Converts a spec dictionary to IR format.

**Parameters:**
- `spec_data`: Parsed JSON spec

**Returns:**
- IR dictionary with `module`, `functions`, `types`, `imports`, `exports`

**Example:**
```python
ir = spec_to_ir({
    'module': 'example',
    'functions': [...]
})
```

#### `canonical_json(data: Any) -> str`
Generates RFC 8785/JCS compliant canonical JSON.

**Parameters:**
- `data`: Any JSON-serializable data

**Returns:**
- Canonical JSON string (sorted keys, no extra whitespace)

#### `compute_sha256(data: Union[bytes, str]) -> str`
Computes SHA-256 hash.

**Returns:**
- Hash string prefixed with `sha256:`

---

## Parsers (`tools/parsers/`)

### Spec Parser

**Usage:**
```python
from tools.parsers import spec_parser

spec = spec_parser.parse_spec('spec.json')
```

---

## Canonicalizers (`tools/canonicalizers/`)

### JSON Canonicalizer

**Functions:**
- `canonicalize_json(data)` - Sort keys, normalize output
- `to_dcbor(data)` - Convert to dCBOR format

---

## Validators (`tools/validators/`)

### Spec Validator

**Usage:**
```python
from tools.validators import spec_validator

valid = spec_validator.validate(spec_data)
errors = spec_validator.get_errors()
```

---

## Receipt Emitter (`tools/receipt_emitter/`)

### Receipt Generation

**Usage:**
```python
from tools.receipt_emitter import emit_receipt

receipt = emit_receipt.generate(
    artifact_path='output.json',
    receipt_type='build'
)
```

**Receipt Schema:**
```json
{
  "schema": "stunir.receipt.v1",
  "epoch": 1735500000,
  "artifact_hash": "sha256:...",
  "receipt_hash": "sha256:..."
}
```

---

## Related
- [API Overview](README.md)
- [Manifests API](manifests.md)

---
*STUNIR Tools API v1.0*
