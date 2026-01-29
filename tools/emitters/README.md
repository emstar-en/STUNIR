# STUNIR Emitters

Part of the `tools → emitters` pipeline stage.

## Overview

Emitters generate output code and data from STUNIR IR.

## Tools

### emit_code.py

Dispatcher for code generation to various targets.

```bash
python emit_code.py <ir.json> --target=<lang> [--output=<file>]
```

Supported targets:
- python, rust, c, c89, c99, cpp
- go, haskell, java, node
- wasm, ruby, php, csharp, dotnet
- erlang, prolog, lisp, smt2, asm

### emit_receipt_json.py

Emits receipts in canonical JSON format.

```bash
python emit_receipt_json.py --target=<name> --status=<status> [options]
```

Options:
- `--target=<name>`: Receipt target name
- `--status=<status>`: Status (success/failure)
- `--epoch=<timestamp>`: Build epoch
- `--tool-name=<name>`: Tool name
- `--output=<file>`: Output file

## Determinism

All emitters produce deterministic output when given the same input.
