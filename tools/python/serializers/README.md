# STUNIR Serializers

Part of the `tools → serializers` pipeline stage.

## Overview

Serializers convert STUNIR data structures to canonical byte formats.

## Tools

### serialize_ir.py

Serializes IR to JSON or dCBOR format.

```bash
python serialize_ir.py <ir.json> [--format=json|dcbor] [--output=<file>]
```

Options:
- `--format=json`: Canonical JSON output (default)
- `--format=dcbor`: Deterministic CBOR output
- `--output=<file>`: Write to file instead of stdout

### serialize_receipt.py

Serializes receipts to canonical JSON.

```bash
python serialize_receipt.py <receipt.json> [--output=<file>]
```

## Determinism

All serializers produce deterministic output:
- Keys sorted alphabetically
- No unnecessary whitespace
- Consistent encoding
