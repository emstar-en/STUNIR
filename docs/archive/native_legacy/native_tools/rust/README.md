# STUNIR Rust Native Tools

Part of the `tools → native → rust` pipeline stage.

## Overview

Rust native tools provide high-performance implementations of STUNIR core functionality.

## Modules

### canonical.rs

RFC 8785 / JCS subset canonicalization for deterministic JSON output.

```rust
use stunir_native::canonical::{canonicalize, canonical_hash};

let value = json!({"z": 1, "a": 2});
let canonical = canonicalize(&value);
assert_eq!(canonical, r#"{"a":2,"z":1}"#);
```

### errors.rs

Error types for STUNIR operations.

```rust
use stunir_native::errors::{StunirError, StunirResult};

fn validate_ir(data: &str) -> StunirResult<()> {
    // ...
}
```

## Building

```bash
cd tools/native/rust
cargo build --release
```

## CLI Commands

```bash
# Canonicalize JSON
stunir-native canon --input file.json --output canonical.json

# Compute hash
stunir-native hash --path file.json

# Verify receipt
stunir-native verify --receipt receipt.json

# Convert spec to IR
stunir-native spec-to-ir --input spec.json --output ir.json

# Emit code
stunir-native emit --input ir.json --target python --output output.py
```

## Determinism

All outputs are deterministic - identical inputs produce identical outputs.
