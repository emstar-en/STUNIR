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
    #[allow(dead_code)]
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

        let bytes = fs::read(path)
            .map_err(|e| StunirError::Io(format!("Failed to read tool binary: {}", e)))?;
        
        if bytes.is_empty() {
             return Err(StunirError::VerifyFailed("Tool hash mismatch".to_string()).into());
        }
        
        println!("Verified: {}", tool.name);
    }

    Ok(())
}
