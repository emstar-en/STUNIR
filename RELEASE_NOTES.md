# STUNIR Release Notes

## v0.8.9 - Generics & Optimization (2026-02-01)

### New Features

#### Generics Support
- **Type Parameters**: Generic types with constraints and default values
  - `type_params` field on types and functions
  - Constraint support (e.g., `Numeric`, `Comparable`)
  - Default type values
- **Generic Instantiation**: Template instantiation for concrete types
  - `generic_instantiations` for pre-defined instantiations
  - `generic_call` operation for calling generic functions
  - `type_cast` operation for explicit type conversion
- **C++ Template-like Syntax**: Support for `Container<T>`, `Pair<K, V>`, etc.

#### Optimization Framework
- **Dead Code Elimination**
  - Removes code after unconditional returns
  - Removes `nop` operations
  - Removes steps marked with `dead_code: true`
- **Constant Folding**
  - Evaluates arithmetic expressions: `1 + 2` → `3`
  - Evaluates boolean expressions: `true && false` → `false`
  - Evaluates comparisons: `1 > 0` → `true`
- **Unreachable Code Elimination**
  - Removes branches with constant `false` conditions
  - Simplifies branches with constant `true` conditions
  - Removes loops with constant `false` conditions
- **Optimization Levels**
  - O0: No optimization
  - O1: Basic (dead code, constant folding)
  - O2: Standard (O1 + unreachable code)
  - O3: Aggressive (all passes)

### Schema Updates (stunir_ir_v1)
- Added `type_param` definition with `name`, `constraint`, `default`
- Added `generic` kind for complex types with `base_type`, `type_args`
- Added `optimization_hint` definition with `pure`, `inline`, `const_eval`, `dead_code`
- Added `type_params` to types and functions
- Added `generic_instantiations` to module
- Added `optimization_level` to module
- Added `generic_call` and `type_cast` operations

### Implementation Updates
- **Python**: New `optimizer.py` module with all optimization passes
- **Rust**: New `optimizer.rs` module matching Python behavior
- **SPARK**: New `stunir_optimizer.ads/adb` with SPARK-verified passes

### Test Coverage
- New test specs in `test_specs/v0.8.9/`:
  - `generics.json` - Basic generic types and functions
  - `templates.json` - C++ style template patterns
  - `optimization_dead_code.json` - Dead code elimination tests
  - `optimization_constant_fold.json` - Constant folding tests
  - `combined_optimization.json` - Multi-pass optimization tests
- All pipelines passing (Python, Rust, SPARK)
- Coverage target: >60%

---

## v0.8.8 - Data Structures (2026-02-01)

### New Features
- Array operations: `array_new`, `array_get`, `array_set`, `array_push`, `array_pop`, `array_len`
- Map operations: `map_new`, `map_get`, `map_set`, `map_delete`, `map_has`, `map_keys`
- Set operations: `set_new`, `set_add`, `set_remove`, `set_has`, `set_union`, `set_intersect`
- Struct operations: `struct_new`, `struct_get`, `struct_set`

---

## v0.8.7 - Exception Handling (2026-02-01)

### New Features
- `try` operation with `try_block`, `catch_blocks`, `finally_block`
- `throw` operation with `exception_type`, `exception_message`
- Multiple catch block support

---

## v0.8.6 - Control Flow (2026-01-31)

### New Features
- `break` and `continue` operations
- `switch` statement with `cases` and `default`
- Fixed SPARK stack overflow issues

---

## Earlier Versions

See CHANGELOG.md for complete history.
