# STUNIR Polyglot Test Vectors

Test vectors for validating the STUNIR polyglot target generation.

## Issue

**test_vectors/polyglot/1035**: Complete test_vectors → polyglot pipeline stage

## Overview

This module provides deterministic test vectors for:
- Rust target code generation
- C89 target code generation (ANSI C)
- C99 target code generation
- Cross-language type mapping
- Polyglot determinism verification

## Usage

### Generate Test Vectors

```bash
python gen_vectors.py [--output <dir>]
```

### Validate Test Vectors

```bash
python validate.py [--dir <dir>]
```

## Test Cases

| ID | Name | Description |
|----|------|-------------|
| tv_polyglot_001 | Rust Target Generation | Verify IR-to-Rust generation |
| tv_polyglot_002 | C89 Target Generation | Verify IR-to-C89 generation |
| tv_polyglot_003 | C99 Target Generation | Verify IR-to-C99 generation |
| tv_polyglot_004 | Cross-Language Type Mapping | Verify type mappings |
| tv_polyglot_005 | Polyglot Determinism | Verify deterministic output |

## Schema

Test vectors follow the `stunir.test_vector.polyglot.v1` schema.
