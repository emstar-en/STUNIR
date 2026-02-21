use crate::errors::StunirError;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

#[derive(Serialize, Deserialize, Debug)]
pub struct ProvenanceV1 {
    pub kind: String,
    pub generator: String,
    pub epoch: String,
    pub input_ir_hash: String,
}

pub fn run(ir_path: &str, epoch_path: &str, out_path: &str) -> Result<()> {
    let ir_bytes = fs::read(ir_path)
        .map_err(|e| StunirError::Io(format!("Failed to read IR: {}", e)))?;
    let ir_hash = hex::encode(Sha256::digest(&ir_bytes));

    let epoch_content = fs::read_to_string(epoch_path)
        .map_err(|e| StunirError::Io(format!("Failed to read Epoch: {}", e)))?;

    let epoch_val = if let Ok(json) = serde_json::from_str::<serde_json::Value>(&epoch_content) {
        json["epoch"].as_str().unwrap_or("unknown").to_string()
    } else {
        epoch_content.trim().to_string()
    };

    let prov = ProvenanceV1 {
        kind: "provenance".to_string(),
        generator: "stunir-native-rust".to_string(),
        epoch: epoch_val,
        input_ir_hash: ir_hash,
    };

    let prov_json = serde_json::to_string_pretty(&prov)
        .map_err(|e| StunirError::Json(e.to_string()))?;

    if let Some(parent) = Path::new(out_path).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out_path, prov_json)?;
    println!("Generated Provenance at {}", out_path);
    Ok(())
}
