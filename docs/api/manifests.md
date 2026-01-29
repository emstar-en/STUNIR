# STUNIR Manifests API

> Part of `docs/api/1068`

## Base Classes (`manifests/base.py`)

### `BaseManifestGenerator`

Abstract base class for manifest generators.

**Class Attributes:**
- `MANIFEST_TYPE: str` - Type identifier (e.g., 'ir', 'receipts')
- `SCHEMA_VERSION: str` - Version string (e.g., 'v1')

**Methods:**

#### `generate(output_path: str = None) -> dict`
Generate manifest and optionally write to file.

#### `_collect_entries() -> List[dict]`
Override to collect manifest entries.

**Example:**
```python
class IRManifestGenerator(BaseManifestGenerator):
    MANIFEST_TYPE = 'ir'
    SCHEMA_VERSION = 'v1'
    
    def _collect_entries(self):
        entries = []
        for f in scan_directory('asm/ir', '*.dcbor'):
            entries.append({
                'name': f.name,
                'path': f.path,
                'hash': compute_file_hash(f.path),
                'size': f.size
            })
        return entries
```

### `BaseManifestVerifier`

Abstract base class for manifest verifiers.

**Methods:**

#### `verify(manifest_path: str) -> Tuple[bool, List[str], List[str]]`
Verify manifest against actual files.

**Returns:**
- `(valid, errors, warnings)` tuple

#### `_verify_entries(entries, base_dir) -> Tuple[List[str], List[str]]`
Override to verify specific entry types.

---

## Manifest Types

### IR Manifest (`manifests/ir/`)

**Generator:** `gen_ir_manifest.py`
```bash
python -m manifests.ir.gen_ir_manifest --ir-dir asm/ir --output receipts/ir_manifest.json
```

**Verifier:** `verify_ir_manifest.py`
```bash
python -m manifests.ir.verify_ir_manifest receipts/ir_manifest.json
```

### Receipts Manifest (`manifests/receipts/`)

**Generator:** `gen_receipts_manifest.py`
```bash
python -m manifests.receipts.gen_receipts_manifest --output receipts/receipts_manifest.json
```

### Contracts Manifest (`manifests/contracts/`)

**Generator:** `gen_contracts_manifest.py`
```bash
python -m manifests.contracts.gen_contracts_manifest --output receipts/contracts_manifest.json
```

### Targets Manifest (`manifests/targets/`)

**Generator:** `gen_targets_manifest.py`
```bash
python -m manifests.targets.gen_targets_manifest --output receipts/targets_manifest.json
```

### Pipeline Manifest (`manifests/pipeline/`)

**Generator:** `gen_pipeline_manifest.py`
```bash
python -m manifests.pipeline.gen_pipeline_manifest --output receipts/pipeline_manifest.json
```

---

## Manifest Schema

```json
{
  "schema": "stunir.manifest.<type>.v1",
  "epoch": 1735500000,
  "entries": [
    {
      "name": "example.dcbor",
      "path": "asm/ir/example.dcbor",
      "hash": "sha256:abc123...",
      "size": 1024
    }
  ],
  "manifest_hash": "sha256:def456..."
}
```

---

## Related
- [API Overview](README.md)
- [Targets API](targets.md)

---
*STUNIR Manifests API v1.0*
