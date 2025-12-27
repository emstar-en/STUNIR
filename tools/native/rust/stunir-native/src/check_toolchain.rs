use crate::errors::StunirError;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

#[derive(Serialize, Deserialize, Debug)]
pub struct ToolEntry {
    pub name: String,
    pub path: String,
    #[serde(default)]
    pub hash: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ToolchainLock {
    pub kind: String,
    #[serde(default)]
    pub tools: Vec<ToolEntry>,
}

pub fn run(lockfile_path: &str) -> Result<()> {
    println!("Checking Toolchain Lock: {}", lockfile_path);
    let content = fs::read_to_string(lockfile_path)
        .map_err(|e| StunirError::Io(format!("Failed to read lockfile: {}", e)))?;
    let lock: ToolchainLock = serde_json::from_str(&content)
        .map_err(|e| StunirError::Json(format!("Invalid lockfile JSON: {}", e)))?;

    if lock.kind != "toolchain_lock" {
        return Err(StunirError::Validation("kind must be 'toolchain_lock'".into()).into());
    }

    for tool in lock.tools {
        let path = Path::new(&tool.path);
        if !path.exists() {
            return Err(StunirError::VerifyFailed("Tool not found").into());
        }
        if let Some(expected_hash) = tool.hash {
            if path.is_file() {
                let bytes = fs::read(path)
                    .map_err(|e| StunirError::Io(format!("Failed to read tool binary: {}", e)))?;
                let actual_hash = hex::encode(Sha256::digest(&bytes));
                if actual_hash != expected_hash {
                    eprintln!("MISMATCH: Tool '{}'", tool.name);
                    return Err(StunirError::VerifyFailed("Tool hash mismatch").into());
                }
            }
        }
        println!("  [OK] {} ({})", tool.name, tool.path);
    }
    println!("Toolchain Verified.");
    Ok(())
}
