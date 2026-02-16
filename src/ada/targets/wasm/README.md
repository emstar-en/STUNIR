# STUNIR WASM Target

WebAssembly (WASM) target emitter for STUNIR.

## Overview

This emitter converts STUNIR IR to WebAssembly Text Format (WAT), which can be
compiled to WASM binary for execution in browsers and other WASM runtimes.

## Usage

```bash
python emitter.py <ir.json> --output=<output_dir>
```

## Output Files

- `<module>.wat` - WebAssembly Text Format source
- `build.sh` - Compilation script (requires WABT toolkit)
- `manifest.json` - Deterministic file manifest
- `README.md` - Generated documentation

## Features

- WAT (S-expression) format output
- Linear memory management
- Function exports
- Type-safe IR mapping

## Dependencies

- WABT toolkit for compilation (`wat2wasm`)

## Schema

`stunir.wasm.v1`
