# STUNIR Rust Target

This target emits idiomatic Rust code from STUNIR IR.

## Purpose

The Rust target provides:
- Memory-safe code generation
- Cargo-based project structure
- Type-safe IR translation

## Usage

```bash
python3 emitter.py <input.ir.json> --output=<output_dir>
```

## Output Structure

```
output_dir/
├── Cargo.toml          # Package manifest
├── src/
│   ├── lib.rs          # Library module
│   └── main.rs         # Binary entry point
├── manifest.json       # STUNIR manifest
└── README.md           # Documentation
```

## Type Mapping

| IR Type | Rust Type |
|---------|----------|
| `i32` | `i32` |
| `i64` | `i64` |
| `f32` | `f32` |
| `f64` | `f64` |
| `bool` | `bool` |
| `string` | `String` |
| `void` | `()` |

## Files

- `emitter.py` - Rust emitter implementation
- `manifest.json` - Output manifest (after emission)

## Schema

`stunir.target.rust.v1`
