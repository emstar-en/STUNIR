use crate::errors::StunirError;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Serialize, Deserialize, Debug)]
pub struct SpecModule {
    pub name: String,
    pub code: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Spec {
    pub kind: String,
    pub modules: Vec<SpecModule>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct IrMetadata {
    pub kind: String,
    pub modules: Vec<SpecModule>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct IrV1 {
    pub kind: String,
    pub generator: String,
    pub ir_version: String,
    pub module_name: String,
    pub functions: Vec<String>,
    pub modules: Vec<String>,
    pub metadata: IrMetadata,
}

pub fn run(in_json: &str, out_ir: &str) -> Result<()> {
    let content = fs::read_to_string(in_json)
        .map_err(|e| StunirError::Io(format!("Failed to read spec: {}", e)))?;
    let spec: Spec = serde_json::from_str(&content)
        .map_err(|e| StunirError::Json(format!("Invalid spec JSON: {}", e)))?;

    let ir = IrV1 {
        kind: "ir".to_string(),
        generator: "stunir-native-rust".to_string(),
        ir_version: "v1".to_string(),
        module_name: "main".to_string(),
        functions: vec![],
        modules: vec![],
        metadata: IrMetadata {
            kind: spec.kind,
            modules: spec.modules,
        },
    };

    let ir_json = serde_json::to_string_pretty(&ir)
        .map_err(|e| StunirError::Json(e.to_string()))?;

    if let Some(parent) = Path::new(out_ir).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out_ir, ir_json)?;
    println!("Generated IR at {}", out_ir);
    Ok(())
}
