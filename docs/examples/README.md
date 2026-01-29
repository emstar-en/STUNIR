# STUNIR Examples Documentation

> Issue: `docs/examples/complete/1039` - Complete docs → examples → complete pipeline stage

## Overview

This section provides practical examples of using STUNIR.

## Examples Index

| Example | Description |
|---------|-------------|
| [Complete Workflow](complete_workflow.md) | End-to-end build example |

## Quick Start

### 1. Create a Spec

```json
// spec.json
{
  "module": "example",
  "functions": [
    {
      "name": "add",
      "params": [
        {"name": "a", "type": "i32"},
        {"name": "b", "type": "i32"}
      ],
      "return_type": "i32",
      "body": {
        "op": "binop",
        "operator": "+",
        "left": {"op": "var", "name": "a"},
        "right": {"op": "var", "name": "b"}
      }
    }
  ]
}
```

### 2. Run the Build

```bash
./scripts/build.sh
```

### 3. Verify Output

```bash
./scripts/verify.sh
```

### 4. Check Artifacts

```bash
ls -la asm/ir/       # IR artifacts
ls -la receipts/     # Verification receipts
ls -la targets/      # Generated code
```

## Example Categories

### Basic Examples
- Simple function compilation
- Multi-function modules
- Type definitions

### Advanced Examples
- Multi-target generation
- Custom manifest generation
- CI/CD integration

### Integration Examples
- Rust crate generation
- C library generation
- WASM module generation

## Running Examples

Examples are located in `docs/examples/` with corresponding spec files.

```bash
# Run specific example
cd docs/examples/
python -m tools.ir_emitter.emit_ir basic_function.json
```

## Related
- [Complete Workflow](complete_workflow.md)
- [API Reference](../api/README.md)
- [Architecture](../architecture/README.md)

---
*STUNIR Examples v1.0*
