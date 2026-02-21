# STUNIR Validators

Part of the `tools → validators` pipeline stage.

## Overview

Validators check STUNIR files against their schemas.

## Tools

### validate_ir.py

Validates IR files against the stunir.ir.v1 schema.

```bash
python validate_ir.py <ir.json> [--strict] [--hash]
```

Options:
- `--strict`: Also check optional fields
- `--hash`: Display content hash

### validate_receipt.py

Validates receipt files against schema.

```bash
python validate_receipt.py <receipt.json> [--strict]
```

## Exit Codes

- `0`: Validation passed
- `1`: Validation failed or error
