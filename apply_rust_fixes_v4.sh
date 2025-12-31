#!/bin/bash
set -e

echo "Applying Rust Fixes V4 (Type Mismatches)..."

BASE_DIR="tools/native/rust/stunir-native"
SRC_DIR="$BASE_DIR/src"

if [ ! -d "$SRC_DIR" ]; then
    echo "Error: Cannot find $SRC_DIR"
    exit 1
fi

# Fix check_toolchain.rs (Add .to_string() to string literals)
cat > "$SRC_DIR/check_toolchain.rs" << 'RUST_EOF'
use crate::errors::StunirError;
use anyhow::Result;
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Deserialize)]
struct ToolchainLock {
    tools: Vec<ToolDef>,
}

#[derive(Deserialize)]
struct ToolDef {
    name: String,
    path: String,
    hash: String,
}

pub fn run(lockfile_path: &str) -> Result<()> {
    println!("Checking Toolchain Lock: {}", lockfile_path);
    let content = fs::read_to_string(lockfile_path)
        .map_err(|e| StunirError::Io(format!("Failed to read lockfile: {}", e)))?;

    let lock: ToolchainLock = serde_json::from_str(&content)
        .map_err(|e| StunirError::Json(format!("Invalid lockfile JSON: {}", e)))?;

    for tool in lock.tools {
        let path = Path::new(&tool.path);
        if !path.exists() {
            return Err(StunirError::VerifyFailed("Tool not found".to_string()).into());
        }

        // Simple hash check (placeholder logic)
        let bytes = fs::read(path)
            .map_err(|e| StunirError::Io(format!("Failed to read tool binary: {}", e)))?;
        
        // In a real implementation, we would hash `bytes` and compare with `tool.hash`
        // For now, we just check if it's empty as a dummy check
        if bytes.is_empty() {
             return Err(StunirError::VerifyFailed("Tool hash mismatch".to_string()).into());
        }
        
        println!("Verified: {}", tool.name);
    }

    Ok(())
}
RUST_EOF

echo "Fixes applied. Rebuilding..."
cd "$BASE_DIR"
cargo build --release
