use anyhow::Result;
use serde_json::Value;

pub fn normalize(json_str: &str) -> Result<String> {
    // Parse as generic Value
    let v: Value = serde_json::from_str(json_str)?;

    // Serialize with pretty printing turned OFF, keys sorted (preserve_order feature in Cargo.toml handles this if map is used, 
    // but standard serde_json maps are BTreeMaps which are sorted by key).
    // Note: True JCS requires more strict handling of floats/unicode, but this is the "v0" implementation.
    let normalized = serde_json::to_string(&v)?;

    Ok(normalized)
}
