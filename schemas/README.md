# STUNIR Schemas

## Overview

This directory contains JSON Schema definitions for STUNIR data structures.

## Schemas

### stunir_ir_v1.schema.json
Base IR schema defining the intermediate representation format.

### symbolic_ir.json
Extensions for symbolic programming constructs (Phase 5A):
- Symbols
- Quoted expressions
- Quasiquoted templates
- Lambda expressions
- Macro definitions
- List and cons constructs

## Usage

```python
import json
from jsonschema import validate

# Load schema
with open('schemas/symbolic_ir.json') as f:
    schema = json.load(f)

# Validate IR
validate(ir_data, schema)
```

## Symbolic IR Extensions

The `symbolic_ir.json` schema adds support for:

| Kind | Description |
|------|-------------|
| symbol | Named symbol reference |
| quote | Quoted datum (not evaluated) |
| quasiquote | Template with selective evaluation |
| unquote | Force evaluation within quasiquote |
| unquote_splicing | Splice list into template |
| lambda | Anonymous function |
| list | List constructor |
| cons | Cons cell (pair) |
| defmacro | Macro definition |
| multiple_value | Multiple return values |

## Part of Phase 5A

These schemas support the Core Lisp emitters:
- Common Lisp
- Scheme (R7RS)
- Clojure
- Racket
