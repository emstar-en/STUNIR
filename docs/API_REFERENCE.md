# STUNIR API Reference

## Version 0.8.9

Complete API reference for STUNIR tools and libraries.

## Table of Contents

1. [Command-Line Tools](#command-line-tools)
2. [Python API](#python-api)
3. [Rust API](#rust-api)
4. [Ada SPARK API](#ada-spark-api)
5. [IR Format](#ir-format)
6. [Specification Format](#specification-format)

---

## Command-Line Tools

### spec_to_ir

Convert specification files to Intermediate Representation (IR).

**Ada SPARK (Primary)**
```bash
stunir_spec_to_ir_main --spec-root <dir> --out <file> [options]
```

**Python (Reference)**
```bash
python tools/spec_to_ir.py --spec-root <dir> --out <file> [options]
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--spec-root` | Directory containing spec JSON files | Required |
| `--out` | Output IR JSON file path | Required |
| `--emit-comments` | Include docstrings and comments | False |
| `--emit-receipt` | Generate verification receipt | False |

**Example:**
```bash
stunir_spec_to_ir_main \
  --spec-root specs/ \
  --out output.ir.json \
  --emit-comments \
  --emit-receipt
```

### ir_to_code

Convert IR to target language code.

**Ada SPARK (Primary)**
```bash
stunir_ir_to_code_main --ir <file> --target <lang> --out <dir> [options]
```

**Python (Reference)**
```bash
python tools/ir_to_code.py --ir <file> --target <lang> --out <dir> [options]
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--ir` | Input IR JSON file | Required |
| `--target` | Target language (rust, c, python, js, zig, go, ada) | Required |
| `--out` | Output directory | Required |
| `--package` | Package/module name | Derived from IR |

**Example:**
```bash
stunir_ir_to_code_main \
  --ir output.ir.json \
  --target rust \
  --out generated/ \
  --package my_module
```

### ir_optimize

Optimize IR using SPARK-verified passes.

**Ada SPARK (Primary)**
```bash
stunir_ir_optimize_main --ir <file> --out <file> [options]
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--ir` | Input IR JSON file | Required |
| `--out` | Output optimized IR file | Required |
| `--level` | Optimization level (0-3) | 2 |
| `--passes` | Comma-separated pass list | All |

**Optimization Passes:**
- `constant-folding`: Fold constant expressions
- `constant-propagation`: Propagate constants through code
- `dead-code-elimination`: Remove unreachable code
- `unreachable-code-elimination`: Remove unreachable branches

**Example:**
```bash
stunir_ir_optimize_main \
  --ir input.ir.json \
  --out optimized.ir.json \
  --level 3 \
  --passes constant-folding,dead-code-elimination
```

---

## Python API

### spec_to_ir Module

**Location:** `tools/spec_to_ir.py`

#### Functions

##### `convert_spec_to_ir(specs, emit_comments=True)`

Convert specification(s) to IR.

**Parameters:**
- `specs` (List[Dict]): List of specification dictionaries
- `emit_comments` (bool): Include documentation comments

**Returns:**
- `Dict[str, Any]`: IR module with keys:
  - `ir_functions`: List of function definitions
  - `ir_types`: List of type definitions
  - `ir_imports`: List of imports
  - `ir_exports`: List of exports

**Example:**
```python
from tools.spec_to_ir import convert_spec_to_ir, load_spec_dir

specs = load_spec_dir("specs/")
ir = convert_spec_to_ir(specs, emit_comments=True)
```

##### `load_spec_dir(spec_root)`

Load all JSON specs from directory.

**Parameters:**
- `spec_root` (str): Root directory containing spec files

**Returns:**
- `List[Dict[str, Any]]`: List of parsed specifications

**Raises:**
- `FileNotFoundError`: If directory doesn't exist

##### `convert_type(type_str)`

Map specification types to IR types.

**Parameters:**
- `type_str` (str): Type string from spec

**Returns:**
- `str`: IR type string

**Supported Types:**
- Primitives: `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64`, `f32`, `f64`, `bool`, `string`, `void`
- Arrays: `byte[]`
- Complex: Arrays, maps, sets, optionals, generics

---

## Rust API

### stunir Crate

**Location:** `src/lib.rs`

#### Core Types

##### `IRModule`

Intermediate representation module.

```rust
pub struct IRModule {
    pub ir_functions: Vec<IRFunction>,
    pub ir_types: Vec<IRType>,
    pub ir_imports: Vec<IRImport>,
    pub ir_exports: Vec<IRExport>,
}
```

##### `IRFunction`

Function definition in IR.

```rust
pub struct IRFunction {
    pub name: String,
    pub docstring: Option<String>,
    pub params: Vec<IRParam>,
    pub return_type: String,
    pub body: Vec<IRStatement>,
}
```

#### Functions

##### `parse_ir(json: &str) -> Result<IRModule, ParseError>`

Parse IR from JSON string.

**Example:**
```rust
use stunir::parse_ir;

let ir = parse_ir(json_content)?;
```

##### `emit_rust(ir: &IRModule, package: &str) -> String`

Generate Rust code from IR.

**Example:**
```rust
use stunir::emit_rust;

let code = emit_rust(&ir, "my_module");
```

---

## Ada SPARK API

### STUNIR.Semantic_IR Package

**Location:** `tools/spark/src/emitters/stunir-semantic_ir.ads`

#### Types

##### `IR_Function`

Function record with SPARK contracts.

```ada
type IR_Function is record
   Name        : IR_Name_String;
   Docstring   : IR_Doc_String;
   Args        : Arg_Array (1 .. Max_Args);
   Return_Type : IR_Type_String;
   Statements  : Statement_Array (1 .. Max_Statements);
   Arg_Cnt     : Natural range 0 .. Max_Args := 0;
   Stmt_Cnt    : Natural range 0 .. Max_Statements := 0;
end record
with Dynamic_Predicate =>
  Arg_Cnt <= Max_Args and Stmt_Cnt <= Max_Statements;
```

##### `IR_Statement`

Statement variant record.

```ada
type IR_Statement (Kind : Statement_Kind := Stmt_Nop) is record
   case Kind is
      when Stmt_Assign =>
         Target : IR_Name_String;
         Value  : Code_Buffer;
      when Stmt_Return =>
         Return_Value : Code_Buffer;
      when Stmt_If =>
         Condition    : Code_Buffer;
         Block_Start  : Positive;
         Block_End    : Positive;
      when Stmt_While =>
         While_Condition : Code_Buffer;
         While_Start     : Positive;
         While_End       : Positive;
      when others =>
         null;
   end case;
end record;
```

### STUNIR.Optimizer Package

**Location:** `tools/spark/src/optimizer/stunir-optimizer.ads`

#### Functions

##### `Is_Constant_Value`

Check if a value is a compile-time constant.

```ada
function Is_Constant_Value (Value : Code_Buffer) return Boolean
  with Global => null,
       Post => (if Is_Constant_Value'Result then
                  Is_Numeric (Value) or Is_Boolean (Value));
```

##### `Fold_Constant`

Fold constant expressions.

```ada
procedure Fold_Constant (
   Expr   : in out Code_Buffer;
   Folded : out Boolean
) with
  Pre  => Is_Constant_Value (Expr),
  Post => Folded = Is_Constant_Value (Expr);
```

##### `Eliminate_Dead_Code`

Remove dead code from function.

```ada
procedure Eliminate_Dead_Code (
   Func       : in out IR_Function;
   Eliminated : out Natural
) with
  Pre  => Func.Stmt_Cnt <= Max_Statements,
  Post => Func.Stmt_Cnt <= Max_Statements and
          Eliminated <= Func.Stmt_Cnt'Old;
```

---

## IR Format

### Version 2 Specification

#### Top-Level Structure

```json
{
  "ir_version": "2.0",
  "ir_functions": [...],
  "ir_types": [...],
  "ir_imports": [...],
  "ir_exports": [...]
}
```

#### IRFunction

```json
{
  "name": "function_name",
  "docstring": "Documentation comment",
  "params": [...],
  "return_type": "i32",
  "body": [...]
}
```

#### IRStatement

Assignment:
```json
{
  "kind": "assign",
  "target": "variable_name",
  "value": "expression"
}
```

Return:
```json
{
  "kind": "return",
  "value": "expression"
}
```

If:
```json
{
  "kind": "if",
  "condition": "boolean_expression",
  "block_start": 1,
  "block_end": 3
}
```

While:
```json
{
  "kind": "while",
  "condition": "boolean_expression",
  "block_start": 1,
  "block_end": 5
}
```

---

## Specification Format

### Version 2 Specification

#### Function Specification

```json
{
  "name": "add",
  "docstring": "Add two integers",
  "params": [
    {"name": "a", "type": "i32"},
    {"name": "b", "type": "i32"}
  ],
  "return_type": "i32",
  "body": "return a + b;"
}
```

#### Type Definition

```json
{
  "name": "Point",
  "kind": "struct",
  "fields": [
    {"name": "x", "type": "f64"},
    {"name": "y", "type": "f64"}
  ]
}
```

#### Complex Types

Array:
```json
{
  "kind": "array",
  "element_type": "i32",
  "size": 10
}
```

Map:
```json
{
  "kind": "map",
  "key_type": "string",
  "value_type": "i32"
}
```

Optional:
```json
{
  "kind": "optional",
  "inner": "string"
}
```

---

## Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| E001 | Spec file not found | Check path and file existence |
| E002 | Invalid JSON format | Validate JSON syntax |
| E003 | Missing required field | Add required field to spec |
| E004 | Type not supported | Use supported type or custom type |
| E005 | IR version mismatch | Update tools or regenerate IR |
| E006 | Target not supported | Use supported target language |

---

Last updated: 2026-02-03