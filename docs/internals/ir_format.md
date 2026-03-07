# STUNIR IR Format Specification

> Part of `docs/internals/1138`

## Overview

The STUNIR Intermediate Representation (IR) is a JSON-based format that captures the structure and semantics of compiled modules.

## IR Normal Form (SSoT)

**Models MUST NOT invent their own IR formats.** The canonical normal form rules are codified in:

```
tools/spark/schema/stunir_ir_v1.dcbor.json → normal_form section
```

Key rules enforced by Phase 2b normalization:
- **Field ordering**: All object keys sorted lexicographically (UTF-8 byte order)
- **Array ordering**: Types/functions sorted alphabetically by name; args preserve source order
- **Alpha renaming**: Bound variables use `_t0`, `_t1`, ...; top-level names preserved
- **Literal normalization**: Shortest CBOR encoding; NFC-normalized UTF-8 strings
- **Floats**: Forbidden in IR payloads (hard reject)
- **Confluence**: Two semantically equivalent programs produce identical canonical IR bytes

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
      "body_hint": "<hint_kind>",
      "hint_detail": "<hint_description>",
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
| `body_hint` | string | Token-saving hint for body pattern |
| `hint_detail` | string | Additional hint description |
| `emission_mode` | string | `stub_hints` or `best_effort` (default: `stub_hints`) |
| `body` | array | IR statements |

### Emission Mode

The `emission_mode` field controls how emitters handle unsupported constructs:

| Mode | Behavior |
|------|----------|
| `stub_hints` | Emit structured stub hints with JSONPath pointers for unsupported constructs (default) |
| `best_effort` | Attempt real code generation even if imperfect |

When `emission_mode` is `stub_hints` (default), emitters will:
- Generate real code for all supported constructs
- Emit structured stub comments with JSONPath pointers for unsupported constructs
- Include key fields in stub hints for manual completion

When `emission_mode` is `best_effort`, emitters will:
- Attempt to generate real code for all constructs
- May produce syntactically correct but semantically incomplete code
- Useful for rapid prototyping or when manual completion is not needed

### Body Hints

Body hints provide token-saving guidance for manual completion of function bodies:

| Hint | Description |
|------|-------------|
| `none` | No hint available |
| `simple_return` | Function just returns a value |
| `getter` | Simple getter/accessor |
| `setter` | Simple setter/mutator |
| `loop_accum` | Loop with accumulation pattern |
| `conditional` | Conditional logic (if/else) |
| `switch` | Switch/match pattern |
| `try_catch` | Exception handling |
| `recursive` | Recursive function |
| `callback` | Callback/async pattern |
| `complex` | Complex body, no simplification |

### Types

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Type identifier |
| `kind` | string | `struct`, `enum`, or `alias` |
| `fields` | array | Field definitions (for structs) |

## Stub Hint Convention

When emitters cannot fully generate code for an IR construct, they emit stub comments with spec-mapped hints to aid manual completion.

### JSONPath Pointer Format

Stub comments include a JSONPath pointer to the IR location and key fields:

```
/* STUB: $.functions[2].steps[4] op=switch expr=status */
/* TODO: Implement switch on 'status' */
```

### Required Hint Fields by Step Type

| Step Type | Required Fields |
|-----------|-----------------|
| `assign` | `target`, `value` |
| `return` | `value` |
| `call` | `value` (func name), `args` |
| `if` | `condition` |
| `while` | `condition` |
| `for` | `init`, `condition`, `increment` |
| `switch` | `expr` |
| `try` | `value` |
| `throw` | `exception_type`, `exception_message` |
| `error` | `error_msg` |

### Example Stub Comments

**C/C++:**
```c
/* STUB: $.functions[0].steps[3] op=call func=process_data args=data,len */
/* TODO: Implement call to 'process_data' */
```

**SPARK/Ada:**
```ada
--  STUB: $.functions[1].steps[2] op=for init=i=0 condition=i<n increment=i++
--  TODO: Implement for loop
```

**Lisp:**
```lisp
;; STUB: $.functions[0].steps[1] op=if condition=x>0
;; TODO: Implement conditional
```

**Futhark:**
```futhark
-- STUB: $.functions[0].steps[2] op=loop condition=i<n
-- TODO: Implement loop
```

**Lean4:**
```lean
-- STUB: $.functions[0].steps[1] op=if condition=x>0
-- TODO: Implement conditional
```

**Prolog:**
```prolog
% STUB: $.functions[0].steps[1] op=if condition=X>0
% TODO: Implement conditional
```

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
