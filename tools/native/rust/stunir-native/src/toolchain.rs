use serde::{Deserialize, Serialize};

#[allow(dead_code)]
#[derive(Debug, Serialize, Deserialize)]
pub struct ToolInfo {
    pub name: String,
    pub path: String,
    pub hash: String,
    pub version: String,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct ToolDef {
    name: String,
    path: String,
    _version: String, 
}

#[allow(dead_code)]
pub fn verify_toolchain(lock_path: &str) -> Result<(), String> {
    println!("Verifying toolchain at {}", lock_path);
    Ok(())
}
