# STUNIR API Reference

> Issue: `docs/api/1068` - Complete docs → api pipeline stage

## Overview

This document provides API reference for STUNIR tools, manifests, and target emitters.

## Quick Reference

| Category | Documentation |
|----------|---------------|
| Tools API | [tools.md](tools.md) |
| Manifests API | [manifests.md](manifests.md) |
| Targets API | [targets.md](targets.md) |

## CLI Commands

### Build Pipeline
```bash
# Full build
./scripts/build.sh

# Profile-specific build
./scripts/build.sh --profile=native
./scripts/build.sh --profile=python
./scripts/build.sh --profile=shell
```

### Native Tools
```bash
# Generate IR manifest
stunir-native gen-ir-manifest --ir-dir asm/ir --out receipts/ir_manifest.json

# Generate provenance
stunir-native gen-provenance --spec spec.json --out provenance.h
```

### Verification
```bash
# Standard verification
./scripts/verify.sh

# Strict verification (manifest matching)
./scripts/verify_strict.sh --strict
```

## Python API

### IR Emission
```python
from tools.ir_emitter import emit_ir

# Convert spec to IR
ir_data = emit_ir.spec_to_ir(spec_data)

# Generate canonical JSON
json_output = emit_ir.canonical_json(ir_data)
```

### Manifest Generation
```python
from manifests.base import BaseManifestGenerator

class CustomManifestGen(BaseManifestGenerator):
    MANIFEST_TYPE = 'custom'
    SCHEMA_VERSION = 'v1'
    
    def _collect_entries(self):
        # Return list of entry dicts
        return [...]
```

### Manifest Verification
```python
from manifests.base import BaseManifestVerifier

verifier = CustomManifestVerifier()
valid, errors, warnings = verifier.verify(manifest_path)
```

## Schema Reference

### Manifest Schema
```json
{
  "schema": "stunir.manifest.<type>.v1",
  "epoch": 1735500000,
  "entries": [...],
  "manifest_hash": "sha256:..."
}
```

### IR Schema
```json
{
  "module": "<name>",
  "ir_epoch": 1735500000,
  "ir_spec_hash": "sha256:...",
  "functions": [...],
  "types": [...]
}
```

## Related
- [Architecture](../architecture/README.md)
- [Internals](../internals/README.md)

---
*STUNIR API Reference v1.0*
