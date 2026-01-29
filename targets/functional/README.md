# STUNIR Functional Language Emitters

This module provides code emitters for functional programming languages, including Haskell, OCaml, and F#.

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

### F# Emitter

Generates idiomatic F# code including:
- Discriminated unions (sum types)
- Record types
- Function definitions with pattern matching
- Computation expressions (`async`, `seq`, `query`)
- Units of measure
- Active patterns
- .NET interoperability (classes, interfaces)
- Module definitions

## Module Structure

```
targets/functional/
├── __init__.py          # Package exports
├── base.py              # Shared base class and utilities
├── haskell_emitter.py   # Haskell code generator
├── ocaml_emitter.py     # OCaml code generator
├── fsharp_emitter.py    # F# code generator
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

### F# Emitter

```python
from targets.functional import FSharpEmitter
from ir.functional import (
    Module, DataType, DataConstructor, TypeParameter, TypeVar, TypeCon,
    RecordType, RecordField, FunctionDef, FunctionClause, VarPattern,
    VarExpr, BinaryOpExpr, ComputationExpr, ComputationLet, ComputationReturn,
    MeasureDeclaration
)

# Create a module with F# features
module = Module(
    name='Example',
    imports=[Import(module='System')],
    type_definitions=[
        # Discriminated union
        DataType(
            name='Result',
            type_params=[TypeParameter(name='a'), TypeParameter(name='e')],
            constructors=[
                DataConstructor(name='Ok', fields=[TypeVar(name='a')]),
                DataConstructor(name='Error', fields=[TypeVar(name='e')])
            ]
        ),
        # Record type
        RecordType(
            name='Person',
            fields=[
                RecordField(name='Name', field_type=TypeCon(name='string')),
                RecordField(name='Age', field_type=TypeCon(name='int'))
            ]
        ),
        # Unit of measure
        MeasureDeclaration(name='m')
    ],
    functions=[
        FunctionDef(
            name='add',
            clauses=[FunctionClause(
                patterns=[VarPattern(name='x'), VarPattern(name='y')],
                body=BinaryOpExpr(op='+', left=VarExpr(name='x'), right=VarExpr(name='y'))
            )]
        )
    ]
)

# Generate F# code
emitter = FSharpEmitter()
code = emitter.emit_module(module)
print(code)
```

**Output:**
```fsharp
module Example

open System

type Result<'a, 'e> =
    | Ok of 'a
    | Error of 'e

type Person =
    {
        Name: string
        Age: int
    }

[<Measure>] type m

let add x y = (x + y)
```

## Feature Comparison

| Feature | Haskell | OCaml | F# |
|---------|---------|-------|-----|
| Data types | `data`, `newtype`, `type` | `type` (variants, records) | `type` (DU, records) |
| Pattern matching | ✓ | ✓ | ✓ |
| Type classes | ✓ | (via modules) | (via interfaces) |
| Monads/do notation | ✓ | (manual) | Computation expressions |
| Modules | (basic) | ✓ (signatures, functors) | ✓ (with namespaces) |
| Lazy evaluation | ✓ | ✗ (strict by default) | `lazy` keyword |
| Imperative features | ✗ | ✓ (ref, mutable) | ✓ (full .NET interop) |
| List comprehensions | ✓ | ✗ | Sequence expressions |
| Guards | ✓ | `when` clause | `when` clause |
| Units of measure | ✗ | ✗ | ✓ |
| Active patterns | ✗ | ✗ | ✓ |
| .NET interop | ✗ | ✗ | ✓ |

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

## F#-Specific Features

### Computation Expressions

```python
from ir.functional import (
    ComputationExpr, ComputationLet, ComputationReturn,
    ComputationYield, ComputationFor, VarPattern, VarExpr, AppExpr
)

# Async computation expression
async_expr = ComputationExpr(
    builder='async',
    body=[
        ComputationLet(
            pattern=VarPattern(name='data'),
            value=AppExpr(func=VarExpr(name='fetchAsync'), arg=None),
            is_bang=True  # let!
        ),
        ComputationReturn(
            value=VarExpr(name='data'),
            is_bang=False
        )
    ]
)
```

**Output:**
```fsharp
async {
    let! data = (fetchAsync)
    return data
}
```

### Units of Measure

```python
from ir.functional import (
    MeasureDeclaration, MeasureType, MeasureUnit, MeasureDiv, TypeCon
)

# Declare units
emitter._emit_measure_declaration(MeasureDeclaration(name='m'))  # [<Measure>] type m
emitter._emit_measure_declaration(MeasureDeclaration(name='s'))  # [<Measure>] type s

# Velocity type: float<m/s>
velocity_type = MeasureType(
    base_type=TypeCon(name='float'),
    measure=MeasureDiv(
        numerator=MeasureUnit(name='m'),
        denominator=MeasureUnit(name='s')
    )
)
```

**Output:**
```fsharp
[<Measure>] type m
[<Measure>] type s
float<m / s>
```

### Active Patterns

```python
from ir.functional import ActivePattern, IfExpr, BinaryOpExpr, VarExpr, LiteralExpr

# Total active pattern
even_odd = ActivePattern(
    cases=['Even', 'Odd'],
    is_partial=False,
    params=['n'],
    body=IfExpr(
        condition=BinaryOpExpr(
            op='=',
            left=BinaryOpExpr(op='%', left=VarExpr(name='n'), right=LiteralExpr(value=2, literal_type='int')),
            right=LiteralExpr(value=0, literal_type='int')
        ),
        then_branch=VarExpr(name='Even'),
        else_branch=VarExpr(name='Odd')
    )
)
```

**Output:**
```fsharp
let (|Even|Odd|) n = if ((n % 2) = 0) then Even else Odd
```

### .NET Interoperability

```python
from ir.functional import ClassDef, ClassField, ClassMethod, InterfaceDef, InterfaceMember, Attribute

# Class definition
counter_class = ClassDef(
    name='Counter',
    attributes=[Attribute(name='Serializable')],
    members=[
        ClassField(
            name='count',
            field_type=TypeCon(name='int'),
            is_mutable=True,
            default_value=LiteralExpr(value=0, literal_type='int')
        ),
        ClassMethod(
            name='Increment',
            params=[],
            body=BinaryOpExpr(op='+', left=VarExpr(name='count'), right=LiteralExpr(value=1, literal_type='int'))
        )
    ]
)
```

**Output:**
```fsharp
[<Serializable>]
type Counter() =
    let mutable count = 0
    member this.Increment() = (count + 1)
```

### F# Types

| IR Type | F# Type |
|---------|---------|
| `int` | `int` |
| `i32` | `int` |
| `i64` | `int64` |
| `float` | `float` |
| `f64` | `float` |
| `bool` | `bool` |
| `string` | `string` |
| `char` | `char` |
| `unit` | `unit` |
| `list` | `list` (postfix) |
| `option` | `option` (postfix) |
| `seq` | `seq` (postfix) |

### F# Operators

| IR Op | F# Op | Description |
|-------|-------|-------------|
| `+` | `+` | Addition |
| `-` | `-` | Subtraction |
| `*` | `*` | Multiplication |
| `/` | `/` | Division |
| `%` | `%` | Modulo |
| `==` | `=` | Equality |
| `!=` | `<>` | Inequality |
| `@` | `@` | List concatenation |
| `::` | `::` | Cons |
| `\|>` | `\|>` | Forward pipe |
| `<\|` | `<\|` | Backward pipe |
| `>>` | `>>` | Forward composition |
| `<<` | `<<` | Backward composition |

## See Also

- [Functional IR](../../ir/functional/README.md) - IR constructs for functional programming
- [Phase 8A HLI](../../../stunir_implementation_framework/phase8/HLI_FUNCTIONAL_LANGUAGES.md) - Haskell/OCaml design document
- [Phase 8B HLI](../../../stunir_implementation_framework/phase8/HLI_FSHARP.md) - F# design document
