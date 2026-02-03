# STUNIR Ada SPARK Tools

## PRIMARY IMPLEMENTATION

**Ada SPARK is the DEFAULT and PRIMARY implementation language for STUNIR tools.**

The Python versions of these tools (`tools/spec_to_ir.py`, `tools/ir_to_code.py`) exist only as reference implementations for readability. For all production use, verification, and deterministic builds, use the Ada SPARK tools in this directory.

## Overview

This directory contains the Ada SPARK implementations of STUNIR's core tools:

- **`spec_to_ir`** - Converts specification files to STUNIR Intermediate Reference (IR)
- **`ir_to_code`** - Generates deterministic source code from IR

## Why Ada SPARK?

1. **Formal Verification**: SPARK proofs guarantee absence of runtime errors
2. **Determinism**: Ada's predictable execution model ensures reproducible builds
3. **Safety**: Strong typing and contracts prevent entire classes of bugs
4. **DO-178C Compliance**: Ada SPARK is the standard for safety-critical systems
5. **Performance**: Native compilation with no interpreter overhead

## Building

### Prerequisites

- GNAT (Ada compiler) >= 12.0
- GNATprove (for SPARK proofs)

### Build Commands

```bash
# Build all tools
cd tools/spark
gprbuild -P stunir_tools.gpr

# Build specific tool
gprbuild -P stunir_tools.gpr stunir_spec_to_ir_main.adb
gprbuild -P stunir_tools.gpr stunir_ir_to_code_main.adb

# Run SPARK proofs
gnatprove -P stunir_tools.gpr --level=2
```

### Output

Compiled binaries are placed in `bin/`:
- `bin/stunir_spec_to_ir_main` - Spec to IR converter
- `bin/stunir_ir_to_code_main` - IR to code emitter

## Usage

### spec_to_ir

```bash
./bin/stunir_spec_to_ir_main --spec-root spec/ --out asm/spec_ir.json
```

Options:
- `--spec-root <dir>` - Directory containing spec JSON files
- `--out <file>` - Output IR manifest file
- `--lockfile <file>` - Toolchain lockfile (default: local_toolchain.lock.json)

### ir_to_code

```bash
./bin/stunir_ir_to_code_main --input asm/spec_ir.json --output asm/output.py --target python
```

Options:
- `--input <file>` - Input IR JSON file
- `--output <file>` - Output source code file
- `--target <lang>` - Target language (python, rust, c, cpp, go, javascript, typescript, java, csharp, wasm, x86, arm)
- `--templates <dir>` - Custom templates directory

## Supported Targets

| Target | Extension | Notes |
|--------|-----------|-------|
| python | .py | Reference implementation |
| rust | .rs | Memory-safe systems |
| c | .c | Embedded/legacy |
| cpp | .cpp | C++ with OOP |
| go | .go | Concurrent systems |
| javascript | .js | Browser/Node.js |
| typescript | .ts | Type-safe JavaScript |
| java | .java | JVM platform |
| csharp | .cs | .NET platform |
| wasm | .wasm | WebAssembly |
| x86 | .asm | x86-64 assembly |
| arm | .s | ARM assembly |

## Integration with STUNIR Build System

The main build script (`scripts/build.sh`) automatically detects and uses Ada SPARK tools when available:

```bash
# Ada SPARK is the default (STUNIR_PROFILE=spark)
./scripts/build.sh

# Force Ada SPARK even if detection fails
STUNIR_PROFILE=spark ./scripts/build.sh

# Fall back to Python reference implementation (not recommended)
STUNIR_PROFILE=python ./scripts/build.sh
```

## SPARK Verification

All core packages are written with SPARK_Mode(On) and include:

- **Preconditions**: Input validation contracts
- **Postconditions**: Output guarantees
- **Data Flow Contracts**: Information flow verification
- **Absence of Runtime Errors**: Proven via GNATprove

To verify:

```bash
# Full SPARK proof
gnatprove -P stunir_tools.gpr --level=2 --report=all

# Quick check (level 1)
gnatprove -P stunir_tools.gpr --level=1
```

## Directory Structure

```
tools/spark/
├── README.md           # This file
├── stunir_tools.gpr    # GNAT project file
├── src/
│   ├── stunir_spec_to_ir.ads       # Spec to IR specification
│   ├── stunir_spec_to_ir.adb       # Spec to IR implementation
│   ├── stunir_spec_to_ir_main.adb  # Spec to IR entry point
│   ├── stunir_ir_to_code.ads       # IR to Code specification
│   ├── stunir_ir_to_code.adb       # IR to Code implementation
│   └── stunir_ir_to_code_main.adb  # IR to Code entry point
├── obj/                # Build objects
└── bin/                # Compiled binaries
```

## Relationship to Python Tools

The Python implementations in `tools/` are **reference implementations only**:

| Python (Reference) | Ada SPARK (Primary) |
|--------------------|--------------------|
| `tools/spec_to_ir.py` | `tools/spark/bin/stunir_spec_to_ir_main` |
| `tools/ir_to_code.py` | `tools/spark/bin/stunir_ir_to_code_main` |

Python files include headers marking them as reference implementations. They should NOT be used for:
- Production builds
- Verification workflows
- DO-178C compliance
- Any safety-critical applications

## License

MIT License - Copyright (c) 2026 STUNIR Project
