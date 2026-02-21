# STUNIR Receipt Emitter

Part of the `tools → receipt_emitter` pipeline stage.

## Overview

The Receipt Emitter generates STUNIR build receipts that track tool invocations and their results.

## Usage

```bash
python emit_receipt.py <target> <status> <epoch> <tool_name> <tool_path> <tool_hash> <tool_ver> [args...]
```

## Arguments

| Argument | Description |
|----------|-------------|
| target | Receipt target identifier |
| status | Build status (success/failure) |
| epoch | Build timestamp |
| tool_name | Name of the tool |
| tool_path | Path to the tool |
| tool_hash | SHA256 hash of the tool |
| tool_ver | Tool version |
| args... | Additional arguments passed to tool |

## Output Schema

```json
{
  "schema": "stunir.receipt.build.v1",
  "receipt_target": "target_name",
  "receipt_status": "success",
  "receipt_build_epoch": 1735500000,
  "receipt_epoch_json": "build/epoch.json",
  "receipt_inputs": {},
  "receipt_tool": {
    "name": "tool_name",
    "path": "/path/to/tool",
    "hash": "sha256...",
    "version": "1.0.0"
  },
  "receipt_argv": []
}
```

## Determinism

Output is canonical JSON (RFC 8785 subset) - identical inputs produce identical bytes.
