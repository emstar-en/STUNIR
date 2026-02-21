# STUNIR Functional IR

This module provides IR (Intermediate Representation) constructs for functional programming languages, including Haskell and OCaml.

## Overview

The Functional IR supports:

- **Type System**: Type variables, type constructors, function types, tuple/list types
- **Expressions**: Literals, variables, application, lambda, let, if, case
- **Pattern Matching**: Wildcards, variables, literals, constructors, tuples, lists
- **Algebraic Data Types**: Sum types, product types, newtypes, records
- **Type Inference**: Basic Hindley-Milner style unification
- **Type Classes**: Haskell-style type classes and instances
- **Do Notation**: Monadic do notation for Haskell

## Module Structure

```
ir/functional/
├── __init__.py          # Package exports
├── functional_ir.py     # Core expressions and types
├── pattern.py           # Pattern matching definitions
├── adt.py               # Algebraic data types
├── type_system.py       # Type inference
└── README.md            # This file
```

## Usage

### Basic Expressions

```python
from ir.functional import (
    LiteralExpr, VarExpr, AppExpr, LambdaExpr, LetExpr,
    BinaryOpExpr, TypeCon, FunctionType
)

# Integer literal
lit = LiteralExpr(value=42, literal_type='int')

# Variable reference
var = VarExpr(name='x')

# Function application: f x
app = AppExpr(func=VarExpr(name='f'), arg=VarExpr(name='x'))

# Lambda: \x -> x + 1
lam = LambdaExpr(
    param='x',
    body=BinaryOpExpr(
        op='+',
        left=VarExpr(name='x'),
        right=LiteralExpr(value=1, literal_type='int')
    )
)

# Let binding: let x = 5 in x * 2
let_expr = LetExpr(
    name='x',
    value=LiteralExpr(value=5, literal_type='int'),
    body=BinaryOpExpr(
        op='*',
        left=VarExpr(name='x'),
        right=LiteralExpr(value=2, literal_type='int')
    )
)
```

### Pattern Matching

```python
from ir.functional import (
    WildcardPattern, VarPattern, ConstructorPattern,
    ListPattern, CaseExpr, CaseBranch
)

# Wildcard: _
wildcard = WildcardPattern()

# Variable: x
var_pat = VarPattern(name='x')

# Constructor: Just x
just_pat = ConstructorPattern(
    constructor='Just',
    args=[VarPattern(name='x')]
)

# List cons: h:t
cons_pat = ListPattern(
    elements=[VarPattern(name='h')],
    rest=VarPattern(name='t')
)

# Case expression
case = CaseExpr(
    scrutinee=VarExpr(name='maybe'),
    branches=[
        CaseBranch(
            pattern=ConstructorPattern(constructor='Nothing'),
            body=LiteralExpr(value=0, literal_type='int')
        ),
        CaseBranch(
            pattern=ConstructorPattern(
                constructor='Just',
                args=[VarPattern(name='x')]
            ),
            body=VarExpr(name='x')
        )
    ]
)
```

### Algebraic Data Types

```python
from ir.functional import (
    DataType, DataConstructor, TypeParameter, TypeVar, TypeCon,
    RecordType, RecordField
)

# Maybe type
maybe = DataType(
    name='Maybe',
    type_params=[TypeParameter(name='a')],
    constructors=[
        DataConstructor(name='Nothing'),
        DataConstructor(name='Just', fields=[TypeVar(name='a')])
    ],
    deriving=['Eq', 'Show']
)

# Tree type
tree = DataType(
    name='Tree',
    type_params=[TypeParameter(name='a')],
    constructors=[
        DataConstructor(name='Leaf', fields=[TypeVar(name='a')]),
        DataConstructor(
            name='Node',
            fields=[
                TypeCon(name='Tree', args=[TypeVar(name='a')]),
                TypeVar(name='a'),
                TypeCon(name='Tree', args=[TypeVar(name='a')])
            ]
        )
    ]
)

# Record type
person = RecordType(
    name='Person',
    fields=[
        RecordField(name='name', field_type=TypeCon(name='String')),
        RecordField(name='age', field_type=TypeCon(name='Int'), mutable=True)
    ]
)
```

### Type Inference

```python
from ir.functional import TypeInference, TypeEnvironment, TypeCon, FunctionType

# Create inference engine and environment
inference = TypeInference()
env = TypeEnvironment()

# Add known bindings
env = env.extend('succ', FunctionType(
    param_type=TypeCon(name='Int'),
    return_type=TypeCon(name='Int')
))

# Infer type of expression
expr = AppExpr(
    func=VarExpr(name='succ'),
    arg=LiteralExpr(value=5, literal_type='int')
)
result_type = inference.infer(expr, env)
# result_type is TypeCon(name='Int')
```

### Module Definitions

```python
from ir.functional import (
    Module, FunctionDef, FunctionClause, Import
)

# Define a module
module = Module(
    name='Example',
    exports=['identity', 'Maybe'],
    imports=[
        Import(module='Prelude'),
        Import(module='Data.List', qualified=True, alias='L')
    ],
    type_definitions=[maybe, tree],
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
```

## Type Expressions

| Type | Description | Example |
|------|-------------|----------|
| `TypeVar` | Type variable | `a`, `b` |
| `TypeCon` | Type constructor | `Int`, `List a` |
| `FunctionType` | Function type | `a -> b` |
| `TupleType` | Tuple type | `(a, b, c)` |
| `ListType` | List type | `[a]` |

## Supported Patterns

| Pattern | Description | Example |
|---------|-------------|----------|
| `WildcardPattern` | Match anything | `_` |
| `VarPattern` | Bind to variable | `x` |
| `LiteralPattern` | Match literal | `42`, `True` |
| `ConstructorPattern` | Match constructor | `Just x` |
| `TuplePattern` | Match tuple | `(a, b)` |
| `ListPattern` | Match list/cons | `[a, b]`, `h:t` |
| `AsPattern` | Bind and match | `xs@(h:t)` |

## Utilities

```python
from ir.functional import (
    make_app, make_lambda, make_let_chain,
    get_pattern_variables, is_exhaustive,
    free_type_vars, type_to_string
)

# Create curried application: f x y z
app = make_app(VarExpr(name='f'), 
               VarExpr(name='x'), 
               VarExpr(name='y'), 
               VarExpr(name='z'))

# Create curried lambda: \x y z -> body
lam = make_lambda(['x', 'y', 'z'], body_expr)

# Extract variables from pattern
vars = get_pattern_variables(TuplePattern(
    elements=[VarPattern(name='a'), VarPattern(name='b')]
))  # ['a', 'b']

# Pretty-print type
print(type_to_string(FunctionType(
    param_type=TypeCon(name='Int'),
    return_type=TypeCon(name='Bool')
)))  # '(Int -> Bool)'
```

## See Also

- [Haskell Emitter](../../targets/functional/README.md) - Generate Haskell code
- [OCaml Emitter](../../targets/functional/README.md) - Generate OCaml code
