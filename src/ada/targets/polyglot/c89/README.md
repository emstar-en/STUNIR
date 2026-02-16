# STUNIR C89 Target

This target emits ANSI C (C89) code from STUNIR IR.

## Purpose

The C89 target provides:
- Maximum portability (ANSI C standard)
- Compatibility with older compilers
- Strict standard compliance

## C89 Restrictions

C89 (ANSI C) has specific restrictions compared to C99:

| Feature | C89 | C99 |
|---------|-----|-----|
| `inline` keyword | ❌ | ✓ |
| `//` comments | ❌ | ✓ |
| Variable declarations | Block start only | Anywhere |
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

Compiles with `-ansi -pedantic` flags to ensure strict C89 compliance.

## Files

- `emitter.py` - C89 emitter implementation
- `../c_base.py` - Shared C emitter base

## Schema

`stunir.target.c89.v1`
