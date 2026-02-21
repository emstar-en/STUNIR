# STUNIR Scientific Language Emitters

Code emitters for scientific and legacy programming languages.

## Overview

This module provides code generation for:

- **Fortran** - Modern Fortran (2003/2008/2018) with parallel constructs
- **Pascal** - Standard Pascal, Object Pascal (Delphi/FPC)

## Architecture

```
targets/scientific/
├── __init__.py           # Package exports
├── fortran_emitter.py    # Fortran code generator
├── pascal_emitter.py     # Pascal code generator
└── README.md            # This file
```

## Usage

### Fortran Emitter

```python
from targets.scientific import FortranEmitter

emitter = FortranEmitter()

ir = {
    'kind': 'module',
    'name': 'math_utils',
    'exports': ['add'],
    'subprograms': [
        {
            'kind': 'subprogram',
            'name': 'add',
            'is_function': True,
            'is_pure': True,
            'parameters': [
                {'name': 'a', 'type_ref': {'name': 'f64'}, 'intent': 'in'},
                {'name': 'b', 'type_ref': {'name': 'f64'}, 'intent': 'in'}
            ],
            'return_type': {'name': 'f64'},
            'body': [
                {
                    'kind': 'return_statement',
                    'value': {
                        'kind': 'binary_op',
                        'op': '+',
                        'left': {'kind': 'var_ref', 'name': 'a'},
                        'right': {'kind': 'var_ref', 'name': 'b'}
                    }
                }
            ]
        }
    ]
}

result = emitter.emit(ir)
print(result.code)
```

**Output:**
```fortran
MODULE math_utils
  IMPLICIT NONE
  PUBLIC :: add

CONTAINS

  PURE REAL(KIND=8) FUNCTION add(a, b)
    REAL(KIND=8), INTENT(IN) :: a
    REAL(KIND=8), INTENT(IN) :: b

    RETURN (a + b)
  END FUNCTION add

END MODULE math_utils
```

### Pascal Emitter

```python
from targets.scientific import PascalEmitter

emitter = PascalEmitter()

ir = {
    'kind': 'module',
    'name': 'MathUtils',
    'exports': ['Add'],
    'subprograms': [
        {
            'kind': 'subprogram',
            'name': 'Add',
            'is_function': True,
            'parameters': [
                {'name': 'A', 'type_ref': {'name': 'f64'}, 'mode': 'value'},
                {'name': 'B', 'type_ref': {'name': 'f64'}, 'mode': 'value'}
            ],
            'return_type': {'name': 'f64'},
            'body': [
                {
                    'kind': 'return_statement',
                    'value': {
                        'kind': 'binary_op',
                        'op': '+',
                        'left': {'kind': 'var_ref', 'name': 'A'},
                        'right': {'kind': 'var_ref', 'name': 'B'}
                    }
                }
            ]
        }
    ]
}

result = emitter.emit(ir)
print(result.code)
```

**Output:**
```pascal
unit MathUtils;

interface

function Add(A: Double; B: Double): Double;

implementation

function Add(A: Double; B: Double): Double;
begin
  Result := (A + B);
  Exit;
end;

end.
```

## Fortran Features

### Supported Constructs

- **Programs** - PROGRAM ... END PROGRAM
- **Modules** - MODULE ... END MODULE with CONTAINS
- **Submodules** - SUBMODULE for separate compilation
- **Subprograms** - SUBROUTINE and FUNCTION with attributes
  - PURE, ELEMENTAL, RECURSIVE
  - INTENT(IN/OUT/INOUT)
- **Derived Types** - TYPE with EXTENDS, BIND(C)
- **Arrays** - DIMENSION, ALLOCATABLE, POINTER
- **Array Operations** - Slicing, whole-array operations
- **Intrinsics** - SIN, COS, MATMUL, SUM, etc.
- **DO CONCURRENT** - Parallel loops with locality
- **Coarrays** - SYNC ALL, coarray access
- **Interface Blocks** - Generic procedures

### Type Mappings

| IR Type | Fortran Type |
|---------|--------------|
| i8 | INTEGER(KIND=1) |
| i16 | INTEGER(KIND=2) |
| i32 | INTEGER(KIND=4) |
| i64 | INTEGER(KIND=8) |
| f32 | REAL(KIND=4) |
| f64 | REAL(KIND=8) |
| bool | LOGICAL |
| string | CHARACTER(LEN=*) |
| complex64 | COMPLEX(KIND=8) |

## Pascal Features

### Supported Constructs

- **Programs** - program ... end.
- **Units** - unit with interface/implementation sections
- **Procedures/Functions** - With VAR, CONST, OUT parameters
- **Records** - record ... end with fields
- **Variant Records** - case ... of with variants
- **Object Pascal Classes** - class with private/public sections
- **Properties** - read/write accessors
- **Sets** - set of type
- **Enums** - (value1, value2, ...)
- **Pointers** - ^Type
- **TRY-EXCEPT-FINALLY** - Exception handling
- **WITH** - Record field shorthand

### Type Mappings

| IR Type | Pascal Type |
|---------|-------------|
| i8 | ShortInt |
| i16 | SmallInt |
| i32 | LongInt |
| i64 | Int64 |
| u8 | Byte |
| u16 | Word |
| u32 | LongWord |
| f32 | Single |
| f64 | Double |
| bool | Boolean |
| string | String |

## EmitterResult

Both emitters return an `EmitterResult` with:

- `code` - Generated source code (str)
- `manifest` - Build manifest (dict) containing:
  - `schema` - Manifest schema version
  - `generator` - Emitter identifier
  - `epoch` - Generation timestamp
  - `ir_hash` - SHA-256 of input IR
  - `output.hash` - SHA-256 of generated code
  - `output.size` - Code size in bytes

## CLI Usage

```bash
# Fortran
python -m targets.scientific.fortran_emitter input.json output.f90

# Pascal
python -m targets.scientific.pascal_emitter input.json output.pas
```

## Configuration

### Fortran Emitter

```python
emitter = FortranEmitter(config={
    'free_form': True  # Use free-form formatting (default)
})
```

### Pascal Emitter

```python
emitter = PascalEmitter(config={
    'object_pascal': True,  # Enable Object Pascal features (default)
    'fpc_mode': True        # Free Pascal Compiler compatibility (default)
})
```

## See Also

- `ir/scientific/` - Scientific IR definitions
- `tests/codegen/test_scientific_emitters.py` - Emitter tests
