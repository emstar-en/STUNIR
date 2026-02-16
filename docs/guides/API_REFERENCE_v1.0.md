# STUNIR API Reference v1.0

**Version:** 1.0.0  
**Date:** January 31, 2026  
**Status:** Production Ready

## Table of Contents

1. [Core API](#core-api)
2. [Type System API](#type-system-api)
3. [Emitter API](#emitter-api)
4. [IR Generation API](#ir-generation-api)
5. [Code Generation API](#code-generation-api)

---

## Core API

### `stunir.spec_to_ir`

Convert specification files to STUNIR Intermediate Reference (IR).

**Function Signature:**
```python
def spec_to_ir(spec_path: str, output_path: str, lockfile_path: Optional[str] = None) -> Dict[str, Any]
```

**Parameters:**
- `spec_path` (str): Path to the specification file (JSON format)
- `output_path` (str): Where to write the generated IR
- `lockfile_path` (Optional[str]): Path to toolchain lockfile for verification

**Returns:**
- `Dict[str, Any]`: IR manifest with metadata

**Example:**
```python
from tools import spec_to_ir

result = spec_to_ir(
    spec_path="./spec/my_protocol.json",
    output_path="./output/ir.json"
)
print(f"Generated IR: {result['ir_hash']}")
```

---

### `stunir.ir_to_code`

Emit code from STUNIR IR to target language.

**Function Signature:**
```python
def ir_to_code(ir_path: str, target: str, output_dir: str, options: Optional[Dict] = None) -> EmissionResult
```

**Parameters:**
- `ir_path` (str): Path to IR file
- `target` (str): Target language/platform (e.g., "c", "rust", "python")
- `output_dir` (str): Output directory for generated code
- `options` (Optional[Dict]): Target-specific options

**Returns:**
- `EmissionResult`: Contains generated files, metadata, and status

**Example:**
```python
from tools import ir_to_code

result = ir_to_code(
    ir_path="./output/ir.json",
    target="rust",
    output_dir="./generated/rust"
)
print(f"Generated {len(result.files)} files")
```

---

## Type System API

### `STUNIR Type System`

The STUNIR type system provides comprehensive type representations for cross-language code generation.

**Module:** `tools.stunir_types.type_system`

#### Core Type Classes

##### `STUNIRType` (Abstract Base)

All types inherit from `STUNIRType`.

**Methods:**
- `kind() -> TypeKind`: Return the type kind
- `to_ir() -> Dict[str, Any]`: Convert to IR representation
- `is_primitive() -> bool`: Check if primitive type
- `is_pointer_like() -> bool`: Check if pointer-like
- `is_compound() -> bool`: Check if compound type

##### `IntType`

Represents integer types.

**Constructor:**
```python
IntType(bits: int = 32, signed: bool = True)
```

**Example:**
```python
from tools.stunir_types.type_system import IntType

i32 = IntType(bits=32, signed=True)  # i32
u64 = IntType(bits=64, signed=False)  # u64
```

##### `PointerType`

Represents pointer types.

**Constructor:**
```python
PointerType(pointee: STUNIRType, mutability: Mutability = Mutability.MUTABLE, nullable: bool = True)
```

**Example:**
```python
from tools.stunir_types.type_system import PointerType, IntType, Mutability

ptr_int = PointerType(pointee=IntType(), mutability=Mutability.CONST)
```

##### `StructType`

Represents struct/record types.

**Constructor:**
```python
StructType(name: str, fields: List[StructField] = [], generics: List[str] = [], packed: bool = False)
```

**Example:**
```python
from tools.stunir_types.type_system import StructType, StructField, IntType

point = StructType(
    name="Point",
    fields=[
        StructField(name="x", type=IntType()),
        StructField(name="y", type=IntType())
    ]
)
```

##### `FunctionType`

Represents function types.

**Constructor:**
```python
FunctionType(params: List[STUNIRType] = [], returns: STUNIRType = VoidType(), variadic: bool = False)
```

**Example:**
```python
from tools.stunir_types.type_system import FunctionType, IntType

add_fn = FunctionType(
    params=[IntType(), IntType()],
    returns=IntType()
)
```

---

## Emitter API

### Base Emitter Interface

All emitters implement a common interface for code generation.

**Module:** `targets.<category>.<language>_emitter`

#### Core Methods

##### `emit(ir: Dict) -> str`

Main emission method.

**Parameters:**
- `ir` (Dict): STUNIR IR data structure

**Returns:**
- `str`: Generated code

##### `emit_function(func: IRFunction) -> str`

Emit a single function.

##### `emit_type(type: IRType) -> str`

Emit type declaration/definition.

##### `map_type(ir_type: str) -> str`

Map STUNIR IR type to target language type.

---

### Emitter Categories (26 Total)

#### 1. Assembly Emitters
- **ARM** (`targets.assembly.arm_emitter`)
- **X86** (`targets.assembly.x86_emitter`)

#### 2. Answer Set Programming (ASP)
- **Clingo** (`targets.asp.clingo_emitter`)
- **DLV** (`targets.asp.dlv_emitter`)

#### 3. BEAM VM Languages
- **Elixir** (`targets.beam.elixir_emitter`)
- **Erlang** (`targets.beam.erlang_emitter`)

#### 4. Business Languages
- **BASIC** (`targets.business.basic_emitter`)
- **COBOL** (`targets.business.cobol_emitter`)

#### 5. Constraint Programming
- **MiniZinc** (`targets.constraints.minizinc_emitter`)
- **CHR** (`targets.constraints.chr_emitter`)

#### 6. Expert Systems
- **CLIPS** (`targets.expert_systems.clips_emitter`)
- **JESS** (`targets.expert_systems.jess_emitter`)

#### 7. Functional Languages
- **Haskell** (`targets.functional.haskell_emitter`)
- **F#** (`targets.functional.fsharp_emitter`)
- **OCaml** (`targets.functional.ocaml_emitter`)

#### 8. Grammar Specifications
- **ANTLR** (`targets.grammar.antlr_emitter`)
- **BNF** (`targets.grammar.bnf_emitter`)
- **EBNF** (`targets.grammar.ebnf_emitter`)
- **PEG** (`targets.grammar.peg_emitter`)
- **Yacc** (`targets.grammar.yacc_emitter`)

#### 9. OOP Languages
- **Smalltalk** (`targets.oop.smalltalk_emitter`)
- **ALGOL** (`targets.oop.algol_emitter`)

#### 10. Planning Languages
- **PDDL** (`targets.planning.pddl_emitter`)

#### 11. Scientific Languages
- **Fortran** (`targets.scientific.fortran_emitter`)
- **Pascal** (`targets.scientific.pascal_emitter`)

#### 12. Systems Languages
- **Ada** (`targets.systems.ada_emitter`)
- **D** (`targets.systems.d_emitter`)

---

## IR Generation API

### `semantic_ir.parse_spec`

Parse specification and generate semantic IR.

**Function:**
```python
def parse_spec(spec_data: Dict) -> IRModule
```

### `IRModule`

Represents a complete IR module.

**Attributes:**
- `schema`: IR schema version
- `module_name`: Module identifier
- `functions`: List of IR functions
- `types`: List of IR types
- `constants`: List of constants

---

## Code Generation API

### Codegen Modules

- `tools.codegen.c99_generator`
- `tools.codegen.cpp_generator`
- `tools.codegen.rust_generator`
- `tools.codegen.python_generator`
- `tools.codegen.go_generator`
- `tools.codegen.java_generator`
- `tools.codegen.javascript_generator`
- `tools.codegen.typescript_generator`

Each provides:
- `generate(ir: IRModule) -> str`: Generate code from IR
- `generate_function(func: IRFunction) -> str`: Generate function code
- `generate_type(type: IRType) -> str`: Generate type declaration

---

## Error Handling

### Exception Types

**Module:** `tools.errors`

- `STUNIRError`: Base exception
- `SpecParseError`: Specification parsing errors
- `IRGenerationError`: IR generation errors
- `EmissionError`: Code emission errors
- `ValidationError`: Validation errors

**Example:**
```python
from tools.errors import STUNIRError, EmissionError

try:
    result = ir_to_code(...)
except EmissionError as e:
    print(f"Code generation failed: {e}")
```

---

## Configuration

### Build Configuration

**File:** `local_toolchain.lock.json`

Defines toolchain configuration:
```json
{
  "version": "1.0.0",
  "toolchain": {
    "spec_to_ir": "tools/spark/bin/stunir_spec_to_ir_main",
    "ir_to_code": "tools/spark/bin/stunir_ir_to_code_main"
  },
  "hashes": {
    "spec_to_ir": "sha256:...",
    "ir_to_code": "sha256:..."
  }
}
```

---

## Version Information

**Current Version:** 1.0.0  
**API Stability:** Production  
**Breaking Changes:** None (initial release)

---

## See Also

- [Emitter Usage Guide](EMITTER_USAGE_GUIDE.md)
- [Type System Guide](TYPE_SYSTEM_GUIDE.md)
- [Migration Guide (Ada SPARK)](MIGRATION_SUMMARY_ADA_SPARK.md)
- [Release Notes](../RELEASE_NOTES_v1.0.md)

---

**Last Updated:** January 31, 2026  
**Maintained by:** STUNIR Team
