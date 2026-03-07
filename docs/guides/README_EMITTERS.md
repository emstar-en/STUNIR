# STUNIR Native Emitters

This directory contains the Rust implementation of the STUNIR Native Core, capable of emitting code for multiple targets from a canonical Intermediate Reference (IR).

## IR Normal Form

**Emitters expect normalized IR input.** The IR normal form rules are enforced by Phase 2b normalization:

```
tools/spark/schema/stunir_ir_v1.dcbor.json → normal_form section
```

Key rules:
- Field ordering: lexicographic (UTF-8 byte order)
- Array ordering: types/functions alphabetically by name
- Floats: forbidden in IR payloads

## Supported Targets

| Target       | Flag          | Description                                      |
|--------------|---------------|--------------------------------------------------|
| **Python**   | `--target python` | Generates Python 3 code (Hosted Reference).      |
| **WASM**     | `--target wat`    | Generates WebAssembly Text (Compiled Reference). |
| **Node.js**  | `--target js`     | Generates JavaScript for Node.js.                |
| **Bash**     | `--target bash`   | Generates strict Bash scripts (Linux/macOS).     |
| **PowerShell**| `--target powershell` | Generates PowerShell scripts (Windows).      |

## Usage

1. **Generate IR:**
   ```bash
   ./stunir_native spec-to-ir --in-json spec.json --out-ir ir.json
   ```

2. **Emit Code:**
   ```bash
   ./stunir_native emit --in-ir ir.json --target <TARGET> --out-file <OUTPUT_FILE>
   ```

## Building

```bash
cd tools/native/rust/stunir-native
cargo build --release
```
