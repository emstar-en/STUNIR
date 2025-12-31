use crate::errors::StunirError;
use crate::ir_v1::{IrV1, IrMetadata, IrFunction, IrInstruction, Spec, SpecModule};
use anyhow::Result;
use std::fs;
use std::path::Path;

pub fn run(in_json: &str, out_ir: &str) -> Result<()> {
    // 1. Read Input Spec
    let content = fs::read_to_string(in_json)
        .map_err(|e| StunirError::Io(format!("Failed to read spec: {}", e)))?;

    let spec: Spec = serde_json::from_str(&content)
        .map_err(|e| StunirError::Json(format!("Invalid spec JSON: {}", e)))?;

    // 2. Transform Spec -> IR
    let mut functions = Vec::new();

    for module in &spec.modules {
        let func = IrFunction {
            name: module.name.clone(),
            body: vec![
                IrInstruction {
                    op: "comment".to_string(),
                    args: vec![format!("Source Language: {}", module.lang)],
                },
                IrInstruction {
                    op: "print".to_string(),
                    args: vec![format!("Executing module: {}", module.name)],
                },
                // Pass-through: Emit source as a print for now
                IrInstruction {
                    op: "print".to_string(),
                    args: vec![module.source.clone()],
                },
            ],
        };
        functions.push(func);
    }

    // If no modules, add a default main
    if functions.is_empty() {
        functions.push(IrFunction {
            name: "main".to_string(),
            body: vec![
                IrInstruction {
                    op: "print".to_string(),
                    args: vec!["STUNIR: No modules defined in spec.".to_string()],
                }
            ],
        });
    }

    // 3. Construct Final IR
    let ir = IrV1 {
        kind: "ir".to_string(),
        generator: "stunir-native-rust".to_string(),
        ir_version: "v1".to_string(),
        module_name: "main".to_string(),
        functions,
        modules: vec![],
        metadata: IrMetadata {
            original_spec_kind: spec.kind,
            source_modules: spec.modules,
        },
    };

    // 4. Write Output
    let ir_json = serde_json::to_string_pretty(&ir)
        .map_err(|e| StunirError::Json(e.to_string()))?;

    if let Some(parent) = Path::new(out_ir).parent() {
        fs::create_dir_all(parent)?;
    }

    fs::write(out_ir, ir_json)?;
    println!("Generated IR at {}", out_ir);

    Ok(())
}
