# STUNIR Parsers

Part of the `tools → parsers` pipeline stage.

## Overview

Parsers read and validate STUNIR input files.

## Tools

### parse_ir.py

Parses and validates IR JSON files.

```bash
python parse_ir.py <ir.json> [--strict]
```

### parse_spec.py

Parses and validates spec JSON files.

```bash
python parse_spec.py <spec.json> [--strict]
```

## Validation

- **Normal mode**: Checks required fields only
- **Strict mode**: Also checks optional fields and emits warnings

## Exit Codes

- `0`: Valid file
- `1`: Invalid file or error
