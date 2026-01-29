# STUNIR IR Emitter

Part of the `tools → ir_emitter` pipeline stage.

## Overview

The IR Emitter converts STUNIR spec files to deterministic Intermediate Representation (IR) format.

## Usage

```bash
python emit_ir.py <spec.json> [output.json]
```

## Output Format

The output is canonical JSON (RFC 8785 / JCS subset):
- Keys sorted alphabetically
- No unnecessary whitespace
- UTF-8 encoded

## Schema

```json
{
  "schema": "stunir.ir.v1",
  "ir_module": "module_name",
  "ir_epoch": 1735500000,
  "ir_spec_hash": "sha256...",
  "ir_functions": [...],
  "ir_types": [...],
  "ir_imports": [...],
  "ir_exports": [...]
}
```

## Determinism

Running the emitter twice on the same input will produce identical output bytes.
