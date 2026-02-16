# STUNIR Bytecode Target

Stack-based bytecode target emitter for STUNIR.

## Overview

This emitter converts STUNIR IR to a stack-based bytecode format suitable for
interpreted execution or JIT compilation.

## Usage

```bash
python emitter.py <ir.json> --output=<output_dir>
```

## Output Files

- `<module>.bc` - Binary bytecode
- `<module>.bc.asm` - Human-readable assembly
- `<module>.bc.json` - JSON representation
- `manifest.json` - Deterministic file manifest
- `README.md` - Generated documentation

## Features

- Stack-based instruction set
- Constant pool management
- Compact binary encoding
- Debug symbols in assembly output

## Schema

`stunir.bytecode.v1`
