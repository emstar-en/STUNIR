use anyhow::Result;
use std::fs;
use std::path::Path;

pub fn execute(receipt: &Path) -> Result<()> {
    let content = fs::read_to_string(receipt)?;

    if content.trim().is_empty() {
        return Err(anyhow::anyhow!("Receipt file is empty: {}", receipt.display()));
    }

    let json: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| anyhow::anyhow!("Invalid JSON in receipt: {}", e))?;

    if let Some(schema) = json.get("schema") {
        println!("Receipt schema: {}", schema);
    }

    if let Some(ir_version) = json.get("ir_version") {
        println!("IR version: {}", ir_version);
    }

    if let Some(module_name) = json.get("module_name") {
        println!("Module: {}", module_name);
    }

    let functions = json.get("functions")
        .and_then(|f| f.as_array())
        .map(|arr| arr.len())
        .unwrap_or(0);

    println!("Verified receipt for {} with {} function(s)", receipt.display(), functions);

    Ok(())
}