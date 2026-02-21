# STUNIR Systems IR

**Version:** 1.0  
**Phase:** 9A  

## Overview

The Systems IR provides intermediate representation nodes for systems programming languages including Ada (with SPARK formal verification support) and D. It supports strong typing, memory management, concurrency primitives, and formal verification constructs.

## Module Structure

```
ir/systems/
├── __init__.py        # Package exports
├── systems_ir.py      # Core IR nodes (Package, Subprogram, etc.)
├── memory.py          # Memory management (Access types, Allocator)
├── types.py           # Type system (Records, Arrays, Enums)
├── concurrency.py     # Concurrency (Tasks, Protected types)
├── verification.py    # SPARK verification annotations
└── README.md          # This file
```

## Core Components

### 1. Systems IR Core (`systems_ir.py`)

Core IR nodes for packages, subprograms, expressions, and statements.

```python
from ir.systems import Package, Subprogram, Parameter, Mode

pkg = Package(
    name='Math_Utils',
    spark_mode=True,
    subprograms=[...]
)

func = Subprogram(
    name='Add',
    parameters=[
        Parameter(name='A', type_ref=TypeRef(name='Integer'), mode=Mode.IN),
        Parameter(name='B', type_ref=TypeRef(name='Integer'), mode=Mode.IN),
    ],
    return_type=TypeRef(name='Integer'),
    body=[...]
)
```

### 2. Memory Management (`memory.py`)

Support for pointers, allocation, and memory operations.

```python
from ir.systems import AccessType, Allocator, Deallocate, AddressOf

# Access type (pointer)
int_ptr = AccessType(
    target_type=TypeRef(name='Integer'),
    not_null=True
)

# Allocation
alloc = Allocator(
    type_ref=TypeRef(name='Integer'),
    initializer=Literal(value=42)
)
```

### 3. Type System (`types.py`)

Strong type system support including subtypes, records, arrays, and enumerations.

```python
from ir.systems import SubtypeDecl, RecordType, ArrayType, EnumType

# Subtype with constraint
positive = SubtypeDecl(
    name='Positive',
    base_type=TypeRef(name='Integer'),
    constraint=RangeConstraint(lower=Literal(1), upper=None)
)

# Record type
point = RecordType(
    name='Point',
    components=[
        ComponentDecl(name='X', type_ref=TypeRef(name='Integer')),
        ComponentDecl(name='Y', type_ref=TypeRef(name='Integer')),
    ]
)
```

### 4. Concurrency (`concurrency.py`)

Ada-style tasking and protected objects.

```python
from ir.systems import TaskType, Entry, ProtectedType

# Task type with entries
worker = TaskType(
    name='Worker_Task',
    entries=[
        Entry(name='Start'),
        Entry(name='Stop'),
    ],
    body=[...]
)

# Protected type
counter = ProtectedType(
    name='Counter',
    procedures=[Subprogram(name='Increment')],
    functions=[Subprogram(name='Get', return_type=TypeRef(name='Integer'))],
    private_components=[
        ComponentDecl(name='Value', type_ref=TypeRef(name='Integer'))
    ]
)
```

### 5. Verification (`verification.py`)

SPARK formal verification annotations.

```python
from ir.systems import (
    Contract, ContractCase, GlobalSpec, DependsSpec,
    LoopInvariant, LoopVariant, GhostVariable
)

# Precondition
pre = Contract(
    condition=BinaryOp(op='>', left=VarExpr(name='X'), right=Literal(0)),
    message="X must be positive"
)

# Global specification
global_spec = GlobalSpec(
    inputs=['Input_Data'],
    outputs=['Output_Data'],
    in_outs=['Counter']
)

# Depends specification
depends = DependsSpec(
    dependencies={'Result': ['X', 'Y']}
)

# Loop invariant
invariant = LoopInvariant(
    condition=BinaryOp(op='<=', left=VarExpr('I'), right=VarExpr('N'))
)
```

## Supported Languages

### Ada (with SPARK)

- Strong typing with subtypes and derived types
- Contracts (Pre, Post, Type_Invariant)
- Tasking (concurrent programming)
- Protected objects
- Packages and child packages
- Generic units
- Exception handling
- SPARK annotations:
  - SPARK_Mode pragma
  - Global/Depends specifications
  - Loop invariants and variants
  - Ghost code
  - Contract_Cases

### D Language

- Templates and metaprogramming
- CTFE (Compile-Time Function Execution)
- Mixins
- Contracts (in, out, invariant)
- Memory safety (@safe, @trusted, @system)
- Concurrency (shared, synchronized)

## Usage Example

```python
from ir.systems import (
    Package, Subprogram, Parameter, Mode, TypeRef,
    Contract, GlobalSpec, BinaryOp, VarExpr, Literal,
    ReturnStatement
)

# Create a SPARK-verified function
pkg = Package(
    name='Math_Utils',
    spark_mode=True,
    subprograms=[
        Subprogram(
            name='Abs_Value',
            parameters=[
                Parameter(name='X', type_ref=TypeRef(name='Integer'))
            ],
            return_type=TypeRef(name='Natural'),
            spark_mode=True,
            preconditions=[
                Contract(condition=BinaryOp(op='/=', left=VarExpr(name='X'), 
                                           right=Literal(value=-2147483648)))
            ],
            global_spec=GlobalSpec(is_null=True),
            body=[...]
        )
    ]
)
```

## Serialization

All IR nodes support serialization to dictionary format:

```python
node = Package(name='Test', spark_mode=True)
d = node.to_dict()
# {'kind': 'package', 'name': 'Test', 'spark_mode': True, ...}
```

## Related Components

- **targets/systems/ada_emitter.py** - Ada code generator
- **targets/systems/d_emitter.py** - D code generator
- **tests/ir/test_systems_ir.py** - IR tests
- **tests/codegen/test_systems_emitters.py** - Emitter tests
