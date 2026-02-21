# STUNIR Scientific IR

Scientific computing IR layer for Fortran and Pascal code generation.

## Overview

This module provides intermediate representation (IR) nodes for scientific computing
and legacy programming languages. It supports:

- **Module systems** - Fortran modules/submodules, Pascal units
- **Array operations** - Multi-dimensional arrays, slicing, whole-array operations
- **Numerical computing** - Mathematical intrinsics, complex numbers
- **Parallel constructs** - Fortran DO CONCURRENT, coarrays
- **Structured programming** - Records, objects, sets

## Architecture

```
ir/scientific/
├── __init__.py         # Package exports
├── scientific_ir.py    # Core IR classes
├── arrays.py           # Array operations
├── numerical.py        # Numerical primitives
└── README.md          # This file
```

## Core Components

### scientific_ir.py

**Modules and Programs:**
- `Module` - Fortran module or Pascal unit
- `Program` - Main program definition
- `Import` - USE/uses statement
- `Subprogram` - Subroutine or function

**Type Declarations:**
- `RecordType` - Fortran derived type or Pascal record
- `VariantRecord` - Pascal variant record
- `EnumType` - Enumeration type
- `SetType` - Pascal set type
- `ClassType` - Object Pascal class

**Statements:**
- `Assignment`, `IfStatement`, `ForLoop`, `WhileLoop`
- `CaseStatement`, `CallStatement`, `ReturnStatement`
- `WithStatement`, `TryStatement` (Pascal)

### arrays.py

**Array Types:**
- `ArrayType` - Multi-dimensional array with bounds
- `ArrayDimension` - Single dimension specification
- `PascalArrayType` - Pascal array with index ranges
- `DynamicArray` - Pascal dynamic array

**Array Operations:**
- `ArraySlice` - Array sectioning (A(1:5,:))
- `ArrayConstructor` - Array constructor [1,2,3]
- `ArrayIntrinsic` - Intrinsic functions (SUM, MAXVAL)
- `WhereStatement` - Masked array assignment
- `ForallStatement` - FORALL construct

### numerical.py

**Numeric Types:**
- `NumericType` - Integer/real with KIND
- `ComplexType` - Complex number type
- `ComplexLiteral` - Complex literal (1.0, 2.0)

**Intrinsic Functions:**
- `MathIntrinsic` - Mathematical functions (SIN, COS, etc.)
- `INTRINSIC_MAP` - Mapping to Fortran/Pascal names

**Parallel Constructs:**
- `DoConcurrent` - DO CONCURRENT loop
- `LocalitySpec` - LOCAL, SHARED, REDUCE
- `Coarray` - Coarray declaration
- `SyncAll`, `SyncImages` - Synchronization

## Usage

### Creating a Module

```python
from ir.scientific import Module, Subprogram, Parameter, TypeRef

mod = Module(
    name='math_utils',
    exports=['add', 'multiply'],
    subprograms=[
        Subprogram(
            name='add',
            is_function=True,
            parameters=[
                Parameter(name='a', type_ref=TypeRef(name='f64')),
                Parameter(name='b', type_ref=TypeRef(name='f64'))
            ],
            return_type=TypeRef(name='f64'),
            is_pure=True
        )
    ]
)

# Serialize to dict
ir_dict = mod.to_dict()
```

### Array Operations

```python
from ir.scientific import (
    ArrayType, ArrayDimension, ArraySlice, SliceSpec,
    Literal, VarRef
)

# 2D array with explicit bounds
matrix = ArrayType(
    element_type=TypeRef(name='f64'),
    dimensions=[
        ArrayDimension(lower=Literal(value=1), upper=Literal(value=100)),
        ArrayDimension(lower=Literal(value=1), upper=Literal(value=100))
    ],
    allocatable=True
)

# Array slice: A(1:50, :)
slice_op = ArraySlice(
    array=VarRef(name='A'),
    slices=[
        SliceSpec(start=Literal(value=1), stop=Literal(value=50)),
        SliceSpec()  # Full dimension
    ]
)
```

### Parallel Constructs

```python
from ir.scientific import (
    DoConcurrent, LoopIndex, LocalitySpec, ReduceSpec,
    Assignment, ArrayAccess, BinaryOp, VarRef, Literal
)

# DO CONCURRENT with reduction
do_conc = DoConcurrent(
    indices=[
        LoopIndex(variable='i', start=Literal(value=1), end=VarRef(name='n'))
    ],
    locality=LocalitySpec(
        local_vars=['temp'],
        reduce_ops=[ReduceSpec(op='+', variable='sum')]
    ),
    body=[
        Assignment(
            target=VarRef(name='sum'),
            value=BinaryOp(op='+', left=VarRef(name='sum'), right=VarRef(name='temp'))
        )
    ]
)
```

## Enumerations

- `Visibility` - PUBLIC, PRIVATE
- `Intent` - IN, OUT, INOUT (Fortran)
- `ParameterMode` - VALUE, VAR, CONST, OUT (Pascal)
- `ArrayOrder` - COLUMN_MAJOR (Fortran), ROW_MAJOR (Pascal)

## Type Reference

| IR Type | Fortran | Pascal |
|---------|---------|--------|
| i8 | INTEGER(KIND=1) | ShortInt |
| i16 | INTEGER(KIND=2) | SmallInt |
| i32 | INTEGER(KIND=4) | LongInt |
| i64 | INTEGER(KIND=8) | Int64 |
| f32 | REAL(KIND=4) | Single |
| f64 | REAL(KIND=8) | Double |
| bool | LOGICAL | Boolean |
| string | CHARACTER(*) | String |

## See Also

- `targets/scientific/fortran_emitter.py` - Fortran code generation
- `targets/scientific/pascal_emitter.py` - Pascal code generation
- `tests/ir/test_scientific_ir.py` - IR tests
