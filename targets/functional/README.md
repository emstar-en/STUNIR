# STUNIR Functional Language Emitters

This module provides code emitters for functional programming languages, specifically Haskell and OCaml.

## Overview

### Haskell Emitter

Generates idiomatic Haskell code including:
- Data type declarations (`data`, `newtype`, `type`)
- Function definitions with pattern matching
- Type signatures and type classes
- Monadic do notation
- List comprehensions
- Guards and where clauses

### OCaml Emitter

Generates idiomatic OCaml code including:
- Variant type declarations
- Record type declarations
- Function definitions with pattern matching
- Module definitions and signatures
- Functors
- Imperative features (`ref`, mutable fields)

## Module Structure

```
targets/functional/
├── __init__.py          # Package exports
├── base.py              # Shared base class and utilities
├── haskell_emitter.py   # Haskell code generator
├── ocaml_emitter.py     # OCaml code generator
└── README.md            # This file
```

## Usage

### Haskell Emitter

```python
from targets.functional import HaskellEmitter
from ir.functional import (
    Module, DataType, DataConstructor, TypeParameter, TypeVar,
    FunctionDef, FunctionClause, VarPattern, VarExpr
)

# Create a module
module = Module(
    name='Example',
    exports=['identity', 'Maybe'],
    type_definitions=[
        DataType(
            name='Maybe',
            type_params=[TypeParameter(name='a')],
            constructors=[
                DataConstructor(name='Nothing'),
                DataConstructor(name='Just', fields=[TypeVar(name='a')])
            ],
            deriving=['Eq', 'Show']
        )
    ],
    functions=[
        FunctionDef(
            name='identity',
            clauses=[FunctionClause(
                patterns=[VarPattern(name='x')],
                body=VarExpr(name='x')
            )]
        )
    ]
)

# Generate Haskell code
emitter = HaskellEmitter()
code = emitter.emit_module(module)
print(code)
```

**Output:**
```haskell
module Example (identity, Maybe) where

data Maybe a
  = Nothing
  | Just a
  deriving (Eq, Show)

identity x = x
```

### OCaml Emitter

```python
from targets.functional import OCamlEmitter
from ir.functional import (
    Module, DataType, DataConstructor, TypeParameter, TypeVar,
    RecordType, RecordField, TypeCon,
    FunctionDef, FunctionClause, VarPattern, VarExpr
)

# Create a module with variant and record types
module = Module(
    name='Example',
    type_definitions=[
        DataType(
            name='option',
            type_params=[TypeParameter(name='a')],
            constructors=[
                DataConstructor(name='None'),
                DataConstructor(name='Some', fields=[TypeVar(name='a')])
            ]
        ),
        RecordType(
            name='person',
            fields=[
                RecordField(name='name', field_type=TypeCon(name='String')),
                RecordField(name='age', field_type=TypeCon(name='Int'), mutable=True)
            ]
        )
    ],
    functions=[
        FunctionDef(
            name='identity',
            clauses=[FunctionClause(
                patterns=[VarPattern(name='x')],
                body=VarExpr(name='x')
            )]
        )
    ]
)

# Generate OCaml code
emitter = OCamlEmitter()
code = emitter.emit_module(module)
print(code)
```

**Output:**
```ocaml
type ('a) option =
  | None
  | Some of 'a

type person = {
  name: string;
  mutable age: int;
}

let identity x = x
```

## Feature Comparison

| Feature | Haskell | OCaml |
|---------|---------|-------|
| Data types | `data`, `newtype`, `type` | `type` (variants, records) |
| Pattern matching | ✓ | ✓ |
| Type classes | ✓ | (via modules) |
| Monads/do notation | ✓ | (manual) |
| Modules | (basic) | ✓ (signatures, functors) |
| Lazy evaluation | ✓ | ✗ (strict by default) |
| Imperative features | ✗ | ✓ (ref, mutable) |
| List comprehensions | ✓ | ✗ |
| Guards | ✓ | `when` clause |

## Haskell-Specific Features

### Type Classes

```python
from ir.functional import TypeClass, MethodSignature, FunctionType, TypeVar

functor = TypeClass(
    name='Functor',
    type_params=[TypeParameter(name='f')],
    methods=[
        MethodSignature(
            name='fmap',
            type_signature=FunctionType(
                param_type=FunctionType(
                    param_type=TypeVar(name='a'),
                    return_type=TypeVar(name='b')
                ),
                return_type=FunctionType(
                    param_type=TypeCon(name='f', args=[TypeVar(name='a')]),
                    return_type=TypeCon(name='f', args=[TypeVar(name='b')])
                )
            )
        )
    ]
)
```

### Do Notation

```python
from ir.functional import DoExpr, DoBindStatement, DoLetStatement, DoExprStatement

do_expr = DoExpr(statements=[
    DoBindStatement(
        pattern=VarPattern(name='x'),
        action=AppExpr(func=VarExpr(name='getLine'), arg=None)
    ),
    DoLetStatement(name='y', value=AppExpr(
        func=VarExpr(name='read'),
        arg=VarExpr(name='x')
    )),
    DoExprStatement(expr=AppExpr(
        func=VarExpr(name='return'),
        arg=VarExpr(name='y')
    ))
])
```

**Output:**
```haskell
do
  x <- getLine
  let y = (read x)
  (return y)
```

## OCaml-Specific Features

### Functors

```python
body = Module(
    name='SetImpl',
    functions=[FunctionDef(
        name='empty',
        clauses=[FunctionClause(
            patterns=[],
            body=ListExpr(elements=[])
        )]
    )]
)

result = emitter.emit_functor('MakeSet', 'Ord', 'OrderedType', body)
```

**Output:**
```ocaml
module MakeSet (Ord : OrderedType) = struct
  let empty  = []
end
```

### Imperative Features

```python
# Reference creation
ref_code = emitter.emit_ref_operations('counter', LiteralExpr(value=0, literal_type='int'))
# Output: let counter = ref 0

# Assignment
assign_code = emitter.emit_assignment('counter', LiteralExpr(value=1, literal_type='int'))
# Output: counter := 1

# Dereference
deref_code = emitter.emit_deref('counter')
# Output: !counter
```

## Type Mappings

### Haskell Types

| IR Type | Haskell Type |
|---------|-------------|
| `int` | `Int` |
| `i32` | `Int32` |
| `i64` | `Int64` |
| `float` | `Float` |
| `f64` | `Double` |
| `bool` | `Bool` |
| `string` | `String` |
| `char` | `Char` |
| `unit` | `()` |

### OCaml Types

| IR Type | OCaml Type |
|---------|------------|
| `int` | `int` |
| `i32` | `int32` |
| `i64` | `int64` |
| `float` | `float` |
| `f64` | `float` |
| `bool` | `bool` |
| `string` | `string` |
| `char` | `char` |
| `unit` | `unit` |

## Operator Mappings

### Haskell Operators

| IR Op | Haskell Op | Description |
|-------|------------|-------------|
| `+` | `+` | Addition |
| `-` | `-` | Subtraction |
| `*` | `*` | Multiplication |
| `/` | `div` | Integer division |
| `%` | `mod` | Modulo |
| `==` | `==` | Equality |
| `/=` | `/=` | Inequality |
| `++` | `++` | List concatenation |
| `:` | `:` | Cons |
| `.` | `.` | Function composition |
| `$` | `$` | Application |

### OCaml Operators

| IR Op | OCaml Op | Description |
|-------|----------|-------------|
| `+` | `+` | Integer addition |
| `-` | `-` | Integer subtraction |
| `*` | `*` | Integer multiplication |
| `/` | `/` | Integer division |
| `==` | `=` | Equality |
| `/=` | `<>` | Inequality |
| `++` | `@` | List concatenation |
| `:` | `::` | Cons |
| `+.` | `+.` | Float addition |
| `^` | `^` | String concatenation |

## See Also

- [Functional IR](../../ir/functional/README.md) - IR constructs for functional programming
- [Phase 8A HLI](../../../stunir_implementation_framework/phase8/HLI_FUNCTIONAL_LANGUAGES.md) - Design document
