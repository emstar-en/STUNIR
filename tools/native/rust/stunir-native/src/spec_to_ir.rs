use crate::errors::StunirError;
use crate::ir_v1::{IrV1, IrMetadata, IrFunction, IrInstruction};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SpecModule {
    pub name: String,
    pub code: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Spec {
    pub kind: String,
    pub modules: Vec<SpecModule>,
}

pub fn run(in_json: &str, out_ir: &str) -> Result<()> {
    let content = fs::read_to_string(in_json)
        .map_err(|e| StunirError::Io(format!("Failed to read spec: {}", e)))?;
    let spec: Spec = serde_json::from_str(&content)
        .map_err(|e| StunirError::Json(format!("Invalid spec JSON: {}", e)))?;

    // DEMO LOGIC: Inject a "main" function that prints "Hello from STUNIR"
    // In a real implementation, this would parse spec.modules[].code
    let demo_func = IrFunction {
        name: "main".to_string(),
        body: vec![
            IrInstruction {
                op: "print".to_string(),
                args: vec!["Hello from STUNIR Python Emitter!".to_string()],
            }
        ],
    };

    let ir = IrV1 {
        kind: "ir".to_string(),
        generator: "stunir-native-rust".to_string(),
        ir_version: "v1".to_string(),
        module_name: "main".to_string(),
        functions: vec![demo_func],
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
