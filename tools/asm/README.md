# STUNIR ASM Tools

Part of the `tools → asm` pipeline stage.

## Overview

ASM tools handle assembly-level artifacts in the STUNIR pipeline.

## Tools

### asm_emit.py

Emits ASM artifacts from IR.

```bash
python asm_emit.py <ir.json> [--output-dir=<dir>]
```

### asm_verify.py

Verifies ASM artifacts against manifest.

```bash
python asm_verify.py <asm.json> [--manifest=<manifest.json>]
```

## Output Schema

```json
{
  "schema": "stunir.asm.v1",
  "asm_module": "module_name",
  "asm_ir_hash": "sha256...",
  "asm_instructions": [...],
  "asm_labels": [...],
  "asm_data": [...]
}
```

## Default Output Directory

ASM artifacts are emitted to `asm/ir/` by default.
