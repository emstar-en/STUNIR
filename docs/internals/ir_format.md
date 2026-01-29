# STUNIR IR Format Specification

> Part of `docs/internals/1138`

## Overview

The STUNIR Intermediate Representation (IR) is a JSON-based format that captures the structure and semantics of compiled modules.

## IR Schema

### Version: `stunir.ir.v1`

```json
{
  "module": "<module_name>",
  "ir_epoch": 1735500000,
  "ir_spec_hash": "sha256:...",
  "functions": [
    {
      "name": "<func_name>",
      "params": [
        {"name": "<param>", "type": "<type>"}
      ],
      "return_type": "<type>",
      "body": [...]
    }
  ],
  "types": [
    {
      "name": "<type_name>",
      "kind": "struct|enum|alias",
      "fields": [...]
    }
  ],
  "imports": ["<module>", ...],
  "exports": ["<symbol>", ...]
}
```

## Field Definitions

### Module Metadata

| Field | Type | Description |
|-------|------|-------------|
| `module` | string | Module name (required) |
| `ir_epoch` | integer | Unix timestamp of IR generation |
| `ir_spec_hash` | string | SHA-256 hash of source spec |

### Functions

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Function identifier |
| `params` | array | Parameter list |
| `return_type` | string | Return type identifier |
| `body` | array | IR statements |

### Types

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Type identifier |
| `kind` | string | `struct`, `enum`, or `alias` |
| `fields` | array | Field definitions (for structs) |

## IR Statements

### Variable Declaration
```json
{"op": "var_decl", "name": "x", "type": "i32", "value": {...}}
```

### Assignment
```json
{"op": "assign", "target": "x", "value": {...}}
```

### Return
```json
{"op": "return", "value": {...}}
```

### Function Call
```json
{"op": "call", "func": "add", "args": [...]}
```

### Binary Operation
```json
{"op": "binop", "operator": "+", "left": {...}, "right": {...}}
```

## Type System

### Primitive Types
- `i8`, `i16`, `i32`, `i64` - Signed integers
- `u8`, `u16`, `u32`, `u64` - Unsigned integers
- `f32`, `f64` - Floating point
- `bool` - Boolean
- `void` - No return value
- `string` - UTF-8 string

### Composite Types
- `array<T, N>` - Fixed-size array
- `slice<T>` - Dynamic slice
- `ptr<T>` - Pointer

## Canonical Encoding

IR must be encoded in canonical JSON (RFC 8785/JCS subset):
1. Keys sorted alphabetically
2. No extra whitespace
3. UTF-8 encoding
4. No duplicate keys

## dCBOR Encoding

For binary storage, IR can be encoded as dCBOR:
- Location: `asm/ir/*.dcbor`
- Deterministic CBOR encoding
- Same canonical ordering as JSON

## Related
- [Internals Overview](README.md)
- [Determinism](determinism.md)

---
*STUNIR IR Format v1.0*
