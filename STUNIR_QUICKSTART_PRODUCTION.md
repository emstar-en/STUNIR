# STUNIR ALPHA Prototype Quickstart Guide

## ⚠️ ALPHA PROTOTYPE - Experimental Pipeline

**Status**: ALPHA Prototype - SPARK-only pipeline functional on Windows
**Last Verified**: February 22, 2026
**Build Tools**: Ada SPARK via GNAT/gprbuild
**Stability**: Experimental - For testing and development only

---

## Pipeline Overview

```
IR JSON → Multi-Language Code (C, Rust, Clojure, Futhark, Lean4, etc.)
    ↓              ↓
  Schema      Function Bodies
 Validated    Control Flow
```

**Note**: The SPARK pipeline is the canonical implementation. All tools are Ada SPARK binaries.

---

## Prerequisites

✅ **GNAT & gprbuild** (Installed via Alire)
- Location: `C:\Users\MSTAR\AppData\Local\alire\cache\toolchains\`
- GNAT: v15.2.1
- gprbuild: v25.0.1

---

## Quick Start: Spec to Code in 2 Steps

### Step 1: Write Your Spec

Create `my_project_spec.json`:

```json
{
  "schema": "stunir.spec.v1",
  "id": "my-project",
  "name": "My Project",
  "version": "1.0.0",
  "profile": "profile3",
  "stages": ["STANDARDIZATION", "UNIQUE_NORMALS", "IR", "BINARY", "RECEIPT"],
  "targets": ["rust", "c99", "python"],
  "module": {
    "name": "my_module",
    "functions": [
      {
        "name": "calculate",
        "params": [
          {"name": "x", "type": "i32"},
          {"name": "y", "type": "i32"}
        ],
        "returns": "i32",
        "body": [
          {"op": "return", "value": {"op": "add", "left": "x", "right": "y"}}
        ]
      }
    ]
  }
}
```

### Step 2: Generate Code

**Option A: Generate from single spec file (put in a directory)**
```powershell
# Create spec directory
mkdir my_project_specs
copy my_project_spec.json my_project_specs\

# Generate IR
.\tools\spark\bin\stunir_spec_to_ir_main.exe --spec-root my_project_specs --out my_project.ir.json

# Generate code for multiple targets
.\tools\spark\bin\stunir_ir_to_code_main.exe --input my_project.ir.json --output output.c --target c
.\tools\spark\bin\stunir_ir_to_code_main.exe --input my_project.ir.json --output output.rs --target rust
.\tools\spark\bin\stunir_ir_to_code_main.exe --input my_project.ir.json --output output.py --target python
```

**Option B: Use example specs**
```powershell
# Generate IR from existing examples
.\tools\spark\bin\stunir_spec_to_ir_main.exe --spec-root spec\examples --out test_ir.json

# Generate code
.\tools\spark\bin\stunir_ir_to_code_main.exe --input test_ir.json --output generated.c --target c
```

---

## Supported Target Languages

### Primary Targets (Tested & Working)
- **C** (`c`) - C99 with stdint.h
- **Rust** (`rust`) - Modern Rust with proper types
- **Python** (`python`) - Python 3 with type hints

### Additional Targets (Available)
- **C++** (`cpp`)
- **Go** (`go`)
- **JavaScript** (`javascript`)
- **TypeScript** (`typescript`)
- **Java** (`java`)
- **C#** (`csharp`)
- **WebAssembly** (`wasm`)
- **Assembly** (`x86`, `arm`)

---

## ⚠️ Determinism & Schema Rules

STUNIR enforces strict determinism for reproducible builds. **IR JSON must comply with:**

| Rule | Requirement | Why |
|------|-------------|-----|
| **No Floats** | Use integers only | Avoid cross-encoder divergence |
| **Sorted Keys** | JSON object keys must be lexicographically sorted | Canonical encoding |
| **NFC Strings** | All strings must be Unicode NFC normalized | Byte-exact hashing |
| **No Duplicates** | Duplicate keys forbidden | Deterministic parsing |

**Reference**: `tools/spark/schema/stunir_ir_v1.dcbor.json` and `contracts/stunir_profile3_contract.json`

**Example - Valid IR:**
```json
{
  "functions": [{"name": "add", "return_type": "i32"}],
  "module_name": "my_module",
  "types": []
}
```

**Example - INVALID (unsorted keys):**
```json
{
  "module_name": "my_module",   // ❌ keys not sorted
  "functions": [],
  "types": []
}
```

---

## Example Output

**Input Spec**:
```json
{
  "name": "add",
  "params": [
    {"name": "a", "type": "i32"},
    {"name": "b", "type": "i32"}
  ],
  "returns": "i32"
}
```

**Generated C**:
```c
int32_t add(int32_t a, int32_t b) {
    /* TODO: Implement */
    return 0;
}
```

**Generated Rust**:
```rust
pub fn add(a: i32, b: i32) -> i32 {
    todo!()  // TODO: Implement
}
```

**Generated Python**:
```python
def add(a: i32, b: i32) -> i32:
    pass  # TODO: Implement
```

---

## Spec Format Reference

### Minimal Spec
```json
{
  "schema": "stunir.spec.v1",
  "id": "example",
  "name": "Example",
  "version": "1.0.0",
  "profile": "profile3",
  "stages": ["STANDARDIZATION", "UNIQUE_NORMALS", "IR", "BINARY", "RECEIPT"],
  "targets": ["rust"]
}
```

### Full Function Spec
```json
{
  "module": {
    "name": "math_utils",
    "functions": [
      {
        "name": "multiply",
        "params": [
          {"name": "x", "type": "i32"},
          {"name": "y", "type": "i32"}
        ],
        "returns": "i32",
        "body": [
          {"op": "return", "value": {"op": "mul", "left": "x", "right": "y"}}
        ]
      }
    ]
  }
}
```

### Supported Types
- **Integers**: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`
- **Floats**: `f32`, `f64`
- **Boolean**: `bool`
- **Custom**: Any string (maps to target language types)

---

## Alternative: Python Bridge Scripts

If you prefer Python over Ada SPARK tools:

```powershell
# Spec to IR
python tools\scripts\bridge_spec_to_ir.py -i my_spec.json -o my_ir.json

# IR to Code
python tools\scripts\bridge_ir_to_code.py -i my_ir.json -o output_dir --target cpp
```

---

## Troubleshooting

### Build Issues

**Problem**: `gprbuild` fails with path errors  
**Solution**: Paths were fixed in `tools/spark/stunir_tools.gpr` - rebuild:
```powershell
cd tools\spark
gprbuild -P stunir_tools.gpr stunir_spec_to_ir_main.adb stunir_ir_to_code_main.adb
```

**Problem**: Binaries not found  
**Solution**: Check `tools\spark\bin\` for:
- `stunir_spec_to_ir_main.exe` (3.3 MB)
- `stunir_ir_to_code_main.exe` (3.5 MB)

### Spec Issues

**Problem**: "Invalid spec" error  
**Solution**: Ensure your spec has:
- `schema: "stunir.spec.v1"`
- Valid `stages` array
- Proper function structure with `name`, `params`, `returns`

**Problem**: Empty IR generated  
**Solution**: Check that spec files are in a directory (not single file as argument)

---

## For Your Production Project

### Recommended Workflow

1. **Create spec directory**: `mkdir project_specs`
2. **Write modular specs**: One file per module/component
3. **Generate IR**: `stunir_spec_to_ir_main.exe --spec-root project_specs --out project.ir.json`
4. **Generate code**: Target all needed languages in one pass
5. **Implement functions**: Fill in `/* TODO: Implement */` stubs
6. **Version control**: Commit specs + generated code

### Best Practices

- **Keep specs in `spec/` directory** - organized by feature/module
- **Generate IR to `ir/` directory** - intermediate artifacts
- **Generate code to `src/generated/`** - clear separation
- **Document custom types** - in spec or separate schema
- **Use CI/CD** - automate spec→IR→code pipeline

---

## Next Steps for Your Project

1. ✅ **Specs ready**: STUNIR pipeline is ALPHA Prototype
2. ✅ **Tools built**: Ada SPARK binaries compiled and working
3. ✅ **Tested**: Verified with 6 functions across 3 languages
4. **Start coding**: Write your project spec JSON
5. **Generate**: Run the pipeline to get multi-language output
6. **Implement**: Fill in the generated function stubs

---

## Support & References

- **Example Specs**: `spec/examples/` (minimal.json, with_functions.json, etc.)
- **Pipeline Docs**: `docs/guides/SPARK_PIPELINE.md`
- **Ada SPARK Source**: `tools/spark/src/`
- **Python Bridge**: `tools/scripts/bridge_*.py`

**This ALPHA prototype pipeline is ready for testing. Your handmade spec can now generate multi-language prototypes automatically.** 🚀
