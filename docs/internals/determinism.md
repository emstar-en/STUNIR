# STUNIR Determinism Guarantees

> Part of `docs/internals/1138`

## Overview

STUNIR is designed around the principle of **deterministic builds**: given the same inputs, STUNIR always produces byte-for-byte identical outputs.

## Determinism Pillars

### 1. Canonical JSON Encoding

All JSON output follows RFC 8785 (JSON Canonicalization Scheme):

```python
import json

def canonical_json(data):
    """Generate canonical JSON with sorted keys."""
    return json.dumps(
        data,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False
    )
```

**Rules:**
- Keys sorted alphabetically
- No extra whitespace
- Minimal separators
- UTF-8 encoding

**IR Normal Form:** The IR normal form rules are codified in `tools/spark/schema/stunir_ir_v1.dcbor.json` → `normal_form` section. Phase 2b normalization enforces these rules before code emission.

### 1.1 Output Confluence

**Output confluence** extends determinism to generated outputs via receipts:

- Target source outputs may differ by platform/toolchain, but MUST be provably bound to the same `cir_sha256` (Canonical IR hash) via receipts.
- Receipts/manifests MUST include explicit output artifact hashes anchored to `cir_sha256`.
- Root attestation MUST bind receipt bundles and output artifacts to the same `cir_sha256`.

**Key principle:**
```
same cir_sha256 ⟹ semantically equivalent outputs (proven via receipts)
```

This enables cross-environment reproducibility and attested provenance.

### 2. SHA-256 Hash Verification

Every artifact includes its SHA-256 hash:

```python
import hashlib

def compute_sha256(data):
    """Compute SHA-256 with prefix."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    h = hashlib.sha256(data).hexdigest()
    return f"sha256:{h}"
```

**Hash Chain:**
```
spec_hash → ir_hash → target_hash → receipt_hash → manifest_hash
```

### 3. Epoch Timestamps

Instead of wall-clock time, STUNIR uses **epochs**:

```python
EPOCH = 1735500000  # Fixed build epoch
```

- Epochs are reproducible across builds
- Can be overridden via `STUNIR_EPOCH` environment variable
- Used for all timestamp fields in manifests

### 4. Sorted File Ordering

When processing directories, files are always sorted:

```python
def scan_directory(path, pattern):
    """Scan directory with deterministic ordering."""
    files = list(pathlib.Path(path).glob(pattern))
    return sorted(files, key=lambda f: f.name)
```

### 5. dCBOR Encoding

Binary artifacts use Deterministic CBOR:
- Same canonical ordering as JSON
- Minimal encoding (shortest form)
- No floating-point special values

**IR dCBOR Profile:** See `tools/spark/schema/stunir_ir_v1.dcbor.json` for the full dCBOR profile specification, including:
- Float policy: `forbid_floats`
- String normalization: NFC UTF-8
- Map key order: canonical (lexicographic)
- Integer encoding: shortest

## Verification

### Manual Verification
```bash
# Run build twice
./scripts/build.sh
cp -r receipts/ receipts1/

./scripts/build.sh
cp -r receipts/ receipts2/

# Compare hashes
diff <(sha256sum receipts1/*) <(sha256sum receipts2/*)
```

### Automated Verification
```bash
./scripts/verify_strict.sh --strict
```

## Common Pitfalls

| Issue | Solution |
|-------|----------|
| Non-sorted dict keys | Use `sort_keys=True` |
| Wall-clock time | Use epoch timestamps |
| Random ordering | Sort all collections |
| Floating point | Use integer/fixed-point |
| OS-specific paths | Normalize path separators |

## Best Practices

1. **Always use canonical_json()** for JSON output
2. **Never use `datetime.now()`** - use epochs
3. **Sort all file lists** before processing
4. **Verify with strict mode** after changes
5. **Run determinism checks** in CI

## Related
- [Internals Overview](README.md)
- [IR Format](ir_format.md)

---
*STUNIR Determinism v1.0*
