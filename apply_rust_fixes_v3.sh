#!/bin/bash
set -e

echo "Applying Rust Fixes V3 (Dependencies & Errors)..."

BASE_DIR="tools/native/rust/stunir-native"
SRC_DIR="$BASE_DIR/src"

if [ ! -d "$SRC_DIR" ]; then
    echo "Error: Cannot find $SRC_DIR"
    exit 1
fi

# 1. Update Cargo.toml (Add thiserror)
cat > "$BASE_DIR/Cargo.toml" << 'TOML_EOF'
[package]
name = "stunir-native"
version = "0.5.0"
edition = "2021"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
sha2 = "0.10"
hex = "0.4"
walkdir = "2.3"
chrono = "0.4"
clap = { version = "4.4", features = ["derive"] }
anyhow = "1.0"
thiserror = "1.0"
TOML_EOF

# 2. Fix errors.rs (Proper formatting)
cat > "$SRC_DIR/errors.rs" << 'RUST_EOF'
use thiserror::Error;

#[derive(Error, Debug)]
pub enum StunirError {
    #[error("IO: {0}")]
    Io(String),
    #[error("JSON: {0}")]
    Json(String),
    #[error("Validation: {0}")]
    Validation(String),
    #[error("Verify Failed: {0}")]
    VerifyFailed(String),
}
RUST_EOF

echo "Fixes applied. Rebuilding..."
cd "$BASE_DIR"
cargo build --release
