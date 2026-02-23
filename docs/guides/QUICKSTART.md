# STUNIR Quick Start Guide

> **⚠️ PRE-ALPHA (v0.1.0-alpha)** — SPARK-only pipeline functional. See [VERSION_STATUS.md](../../VERSION_STATUS.md) for limitations.

Get started with STUNIR in 5 minutes.

## Prerequisites

- **Ada SPARK Tools** (required): `gprbuild`, `gnat` via Alire or GNAT Community
- **Python 3.8+** (not recommended): Experimental only, not canonical

## Installation

### Option 1: Precompiled Binaries (Recommended)

Download precompiled Ada SPARK binaries:

```bash
# Linux x86_64
export PATH="$PWD/precompiled/linux-x86_64/spark/bin:$PATH"

# Verify
stunir_spec_to_ir_main --help
```

### Option 2: Build from Source

```bash
# Clone repository
git clone <repository-url>
cd STUNIR-main

# Build Ada SPARK tools
cd tools/spark
gprbuild -P stunir_tools.gpr
cd ../..

# Add to PATH
export PATH="$PWD/tools/spark/bin:$PATH"
```

### Option 3: Python Reference

```bash
# Install Python dependencies
pip install -r requirements.txt

# Use Python tools directly
python tools/spec_to_ir.py --help
```

## Your First STUNIR Project

### 1. Create a Specification

Create `specs/hello.json`:

```json
{
  "name": "hello",
  "docstring": "A simple greeting function",
  "params": [
    {"name": "name", "type": "string"}
  ],
  "return_type": "string",
  "body": "return 'Hello, ' + name + '!';"
}
```

### 2. Convert to IR

```bash
# Using Ada SPARK (recommended)
stunir_spec_to_ir_main \
  --spec-root specs/ \
  --out hello.ir.json \
  --emit-comments

# Or using Python
python tools/spec_to_ir.py \
  --spec-root specs/ \
  --out hello.ir.json \
  --emit-comments
```

### 3. Generate Code

```bash
# Generate Rust code
stunir_ir_to_code_main \
  --ir hello.ir.json \
  --target rust \
  --out generated/ \
  --package hello

# Or generate C
stunir_ir_to_code_main \
  --ir hello.ir.json \
  --target c \
  --out generated/ \
  --package hello
```

### 4. View Generated Code

```bash
cat generated/hello.rs
```

Output:
```rust
/// A simple greeting function
pub fn hello(name: &str) -> String {
    return format!("Hello, {}!", name);
}
```

## Next Steps

### ⚠️ Determinism & Schema Rules

STUNIR enforces strict determinism for reproducible builds. **IR JSON must comply with:**

| Rule | Requirement | Reference |
|------|-------------|-----------|
| **No Floats** | Use integers only | `tools/spark/schema/stunir_ir_v1.dcbor.json` |
| **Sorted Keys** | JSON keys lexicographically sorted | `contracts/stunir_profile3_contract.json` |
| **NFC Strings** | Unicode NFC normalized | dCBOR spec |
| **No Duplicates** | Duplicate keys forbidden | Canonical JSON |

### Optimize Your IR

```bash
stunir_ir_optimize_main \
  --ir hello.ir.json \
  --out hello_optimized.ir.json \
  --level 3
```

### Run Tests

```bash
# Run all tests
cd tests/spark/optimizer
gprbuild -P optimizer_test.gpr
./bin/test_optimizer
```

### Explore Examples

Check the `examples/` directory for more complex specifications:

- `examples/calculator.json` - Arithmetic operations
- `examples/data_structures.json` - Structs and arrays
- `examples/control_flow.json` - If/else and loops

## Common Workflows

### Development Workflow

```bash
# 1. Edit spec
vim specs/my_module.json

# 2. Convert to IR
stunir_spec_to_ir_main --spec-root specs/ --out my_module.ir.json

# 3. Optimize
stunir_ir_optimize_main --ir my_module.ir.json --out optimized.ir.json

# 4. Generate code
stunir_ir_to_code_main --ir optimized.ir.json --target rust --out src/

# 5. Test generated code
cd src && cargo test
```

### CI/CD Workflow

```yaml
# .github/workflows/stunir.yml
name: STUNIR Build

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup STUNIR
        run: |
          export PATH="$PWD/precompiled/linux-x86_64/spark/bin:$PATH"
          
      - name: Generate IR
        run: |
          stunir_spec_to_ir_main --spec-root specs/ --out output.ir.json
          
      - name: Generate Rust
        run: |
          stunir_ir_to_code_main --ir output.ir.json --target rust --out generated/
          
      - name: Test
        run: |
          cd generated && cargo test
```

## Documentation

- **[API Reference](API_REFERENCE.md)** - Complete API documentation
- **[Migration Guide](MIGRATION_GUIDE.md)** - Version migration instructions
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues and solutions
- **[Architecture](ARCHITECTURE.md)** - System design and components

## Getting Help

- **Issues**: Check [Troubleshooting](TROUBLESHOOTING.md)
- **Questions**: Review [API Reference](API_REFERENCE.md)
- **Bugs**: File an issue with system information

## Version Info

- **Current Version**: 0.8.9
- **IR Version**: 2.0
- **Last Updated**: 2026-02-03

---

Happy coding with STUNIR!