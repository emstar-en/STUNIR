# STUNIR C99 Target

This target emits ISO C99 code from STUNIR IR.

## Purpose

The C99 target provides:
- Modern C features (inline, flexible declarations)
- Standard integer types via `<stdint.h>`
- Boolean type via `<stdbool.h>`
- Line comments (`//`)

## C99 Features Used

| Feature | C89 | C99 |
|---------|-----|-----|
| `inline` keyword | ❌ | ✓ |
| `//` comments | ❌ | ✓ |
| Variable declarations | Block start only | Anywhere ✓ |
| `<stdbool.h>` | ❌ | ✓ |
| `<stdint.h>` | ❌ | ✓ |

## Usage

```bash
python3 emitter.py <input.ir.json> --output=<output_dir>
```

## Output Structure

```
output_dir/
├── module.h            # Header file
├── module.c            # Implementation
├── Makefile            # Build configuration
├── manifest.json       # STUNIR manifest
└── README.md           # Documentation
```

## Build

```bash
make
```

Compiles with `-std=c99` flag for C99 compliance.

## Files

- `emitter.py` - C99 emitter implementation
- `../c_base.py` - Shared C emitter base

## Schema

`stunir.target.c99.v1`
