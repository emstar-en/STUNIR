# STUNIR Output Emitter

**Pipeline Stage:** spec → outputs  
**Issue:** #1049

## Overview

This module emits structured output specifications for STUNIR pipeline artifacts.

## Files

| File | Description |
|------|-------------|
| `output_schema.json` | JSON Schema for output specification |
| `emit_output.py` | Output emitter implementation |

## Usage

### CLI

```bash
# Emit output spec with artifacts
python emit_output.py \
  --input-id my-spec-001 \
  --input-hash abc123... \
  --type ir \
  --target rust \
  --artifact "module" "asm/ir/module.dcbor" \
  -o receipts/output_spec.json
```

### Python API

```python
from emit_output import OutputEmitter
from pathlib import Path

# Create emitter with source reference
emitter = OutputEmitter(
    input_id='my-spec-001',
    input_hash='abc123...'
)

# Add artifacts
emitter.add_artifact('module', Path('asm/ir/module.dcbor'))
emitter.add_artifact('manifest', Path('receipts/ir_manifest.json'))

# Emit output spec
output = emitter.emit(output_type='ir', target='rust')

# Write to file
hash_value = emitter.write(Path('receipts/output.json'), 'ir', 'rust')
```

## Output Structure

```json
{
  "schema": "stunir.output.v1",
  "input_id": "spec-id",
  "input_hash": "sha256...",
  "output_type": "ir",
  "target": "rust",
  "artifacts": [
    {
      "name": "artifact-name",
      "path": "relative/path",
      "hash": "sha256...",
      "size": 1234,
      "format": "dcbor"
    }
  ],
  "metadata": {
    "epoch": 1735500000,
    "tool_version": "1.0.0",
    "duration_ms": 150
  }
}
```
