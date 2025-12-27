# STUNIR Native Emitters

This directory contains the Rust implementation of the STUNIR Native Core, capable of emitting code for multiple targets from a canonical Intermediate Reference (IR).

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
