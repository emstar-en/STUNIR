# STUNIR OOP Emitters

**Version:** 1.0  
**Phase:** 10B (OOP & Historical Languages)

## Overview

This module provides code emitters for object-oriented and historical
programming languages:

- **Smalltalk** - Pure OOP with message passing
- **ALGOL** - Block-structured with call-by-name

## Architecture

```
targets/oop/
├── __init__.py           # Package exports
├── smalltalk_emitter.py  # Smalltalk code generation
├── algol_emitter.py      # ALGOL code generation
└── README.md             # This file
```

## Smalltalk Emitter

### Usage

```python
from targets.oop import SmalltalkEmitter

emitter = SmalltalkEmitter()
result = emitter.emit(ir)

print(result.code)      # Generated Smalltalk code
print(result.manifest)  # Build manifest
print(result.warnings)  # Any warnings
```

### Supported Features

#### Class Definitions

```smalltalk
Object subclass: #Point
    instanceVariableNames: 'x y'
    classVariableNames: 'Origin'
    poolDictionaries: ''
    category: 'Graphics-Primitives'
```

#### Method Definitions

```smalltalk
"Unary method"
x
    ^x

"Keyword method"
at: index put: value
    array at: index put: value.
    ^self
```

#### Message Passing

```smalltalk
"Unary: receiver selector"
collection size

"Binary: receiver operator argument"
3 + 4

"Keyword: receiver keyword: arg keyword: arg"
dict at: #key put: 'value'
```

#### Blocks

```smalltalk
"Simple block"
[ 42 ]

"Block with parameters"
[ :x | x squared ]

"Block with temporaries"
[ :x | | temp | temp := x. temp * 2 ]
```

#### Cascading

```smalltalk
stream
    nextPutAll: 'Hello';
    cr;
    flush
```

#### Control Structures

```smalltalk
"Conditional"
flag ifTrue: [ 'yes' ] ifFalse: [ 'no' ]

"While loop"
[ i < 10 ] whileTrue: [ i := i + 1 ]

"Times repeat"
5 timesRepeat: [ self doSomething ]

"Collection iteration"
collection do: [ :each | each print ]
```

#### Literals

```smalltalk
"Numbers"
42
3.14

"Strings"
'Hello, World!'

"Symbols"
#mySymbol

"Characters"
$a

"Arrays"
#(1 2 3)
{expr1. expr2. expr3}
```

## ALGOL Emitter

### Usage

```python
from targets.oop import ALGOLEmitter

emitter = ALGOLEmitter(config={'use_ascii': True})
result = emitter.emit(ir)

print(result.code)      # Generated ALGOL code
print(result.manifest)  # Build manifest
```

### Supported Features

#### Block Structure

```algol
begin
    integer x;
    real y;
    
    x := 10;
    y := 3.14
end
```

#### Procedures and Functions

```algol
procedure swap(a, b);
    integer a, b;
begin
    integer temp;
    temp := a;
    a := b;
    b := temp
end

real procedure sum(i, lo, hi, term);
    value lo, hi;
    integer i, lo, hi;
    real term;
begin
    real temp;
    temp := 0;
    for i := lo step 1 until hi do
        temp := temp + term;
    sum := temp
end
```

#### Parameter Passing

```algol
"Call by value - evaluated once"
value x;

"Call by name - re-evaluated (thunk)"
"Parameters without 'value' are call-by-name"
integer i;  "call-by-name"
```

#### For Loops

```algol
"Step-until form"
for i := 1 step 1 until n do
    sum := sum + A[i];

"Step with increment"
for i := 1 step 2 until 20 do
    process(i);

"While form"
for i := 1 while i < 100 do
    i := i * 2
```

#### Arrays with Dynamic Bounds

```algol
array A[1:n, 1:m];
integer array B[lo:hi];
```

#### Switch Statements

```algol
switch S := L1, L2, L3, L4;
...
goto S[i];  "Computed goto"
```

#### Own Variables

```algol
procedure counter;
begin
    own integer count := 0;
    count := count + 1;
    counter := count
end
```

#### Conditional Statements

```algol
if x < 0 then x := 0

if a > b then max := a else max := b
```

## Type Mapping

### Smalltalk (dynamically typed)
Smalltalk is dynamically typed, so IR types are not directly mapped.

### ALGOL

| IR Type | ALGOL Type |
|---------|------------|
| i8, i16, i32, i64 | integer |
| f32, f64 | real |
| bool | Boolean |
| string | string |

## Example: Jensen's Device

The classic demonstration of call-by-name:

```python
ir = {
    'kind': 'algol_procedure',
    'name': 'sum',
    'result_type': 'real',
    'parameters': [
        {'name': 'i', 'param_type': 'integer', 'mode': 'name'},
        {'name': 'lo', 'param_type': 'integer', 'mode': 'value'},
        {'name': 'hi', 'param_type': 'integer', 'mode': 'value'},
        {'name': 'term', 'param_type': 'real', 'mode': 'name'}
    ],
    'body': {...}
}

result = emitter.emit(ir)
```

Generates:

```algol
real procedure sum(i, lo, hi, term);
    value lo, hi;
    integer i, lo, hi;
    real term;
begin
    real temp;
    temp := 0;
    for i := lo step 1 until hi do
        temp := temp + term;
    sum := temp
end
```

## Manifest Format

Both emitters generate deterministic manifests:

```json
{
    "schema": "stunir.manifest.targets.v1",
    "generator": "stunir.smalltalk.emitter",
    "dialect": "smalltalk",
    "ir_hash": "sha256...",
    "output": {
        "hash": "sha256...",
        "size": 1234,
        "format": "smalltalk",
        "extension": ".st"
    }
}
```

## Related Modules

- `ir/oop/` - OOP IR definitions
- `tests/codegen/test_oop_emitters.py` - Emitter tests
- STUNIR HLI Phase 10B

## References

- Smalltalk-80: The Language
- Blue Book (Smalltalk implementation)
- Revised Report on ALGOL 60
- Jensen's Device and ALGOL semantics
