use crate::errors::StunirError;
use crate::ir_v1::{IrV1, IrMetadata, IrFunction, IrInstruction, Spec, SpecModule};
use anyhow::Result;
use std::fs;
use std::path::Path;

pub fn run(input_path: &str, out_ir: &str) -> Result<()> {
    let path = Path::new(input_path);
    let mut combined_spec = Spec {
        kind: "spec".to_string(),
        modules: Vec::new(),
        metadata: std::collections::HashMap::new(),
    };

    // 1. Load Spec(s)
    if path.is_dir() {
        println!("Scanning directory: {}", input_path);
        for entry in fs::read_dir(path).map_err(|e| StunirError::Io(e.to_string()))? {
            let entry = entry.map_err(|e| StunirError::Io(e.to_string()))?;
            let path = entry.path();
            if path.is_file() && path.extension().map_or(false, |ext| ext == "json") {
                // Robust Loading: Check 'kind' before parsing schema
                let content = fs::read_to_string(&path)
                    .map_err(|e| StunirError::Io(format!("Failed to read {:?}: {}", path, e)))?;

                let v: serde_json::Value = match serde_json::from_str(&content) {
                    Ok(v) => v,
                    Err(e) => {
                        println!("WARNING: Skipping invalid JSON {:?}: {}", path, e);
                        continue;
                    }
                };

                if let Some(kind) = v.get("kind").and_then(|k| k.as_str()) {
                    if kind == "spec" {
                        println!("Loading spec: {:?}", path);
                        let spec: Spec = serde_json::from_value(v)
                            .map_err(|e| StunirError::Json(format!("Invalid Spec schema in {:?}: {}", path, e)))?;
                        combined_spec.modules.extend(spec.modules);
                        combined_spec.metadata.extend(spec.metadata);
                    } else {
                        println!("Skipping non-spec file (kind='{}'): {:?}", kind, path);
                    }
                } else {
                    println!("Skipping unknown JSON (no 'kind' field): {:?}", path);
                }
            }
        }
    } else {
        // Single file mode - Enforce strictness
        let content = fs::read_to_string(input_path)
            .map_err(|e| StunirError::Io(format!("Failed to read spec: {}", e)))?;
        let spec: Spec = serde_json::from_str(&content)
            .map_err(|e| StunirError::Json(format!("Invalid spec JSON: {}", e)))?;
        combined_spec = spec;
    }

    // 2. Transform Spec -> IR
    let mut functions = Vec::new();
    let mut module_names = Vec::new();

    for module in &combined_spec.modules {
        let func = IrFunction {
            name: module.name.clone(),
            body: vec![
                IrInstruction {
                    op: "comment".to_string(),
                    args: vec![format!("Source Language: {}", module.lang)],
                },
                IrInstruction {
                    op: "raw".to_string(),
                    args: vec![module.source.clone()],
                },
            ],
        };
        functions.push(func);
        module_names.push(module.name.clone());
    }

    // 3. Generate Main Orchestrator
    let mut main_body = Vec::new();

    if module_names.is_empty() {
        main_body.push(IrInstruction {
            op: "print".to_string(),
            args: vec!["STUNIR: No modules defined in spec.".to_string()],
        });
    } else {
        main_body.push(IrInstruction {
            op: "print".to_string(),
            args: vec!["STUNIR: Orchestrating modules...".to_string()],
        });
        for name in module_names {
            main_body.push(IrInstruction {
                op: "call".to_string(),
                args: vec![name],
            });
        }
    }

    functions.push(IrFunction {
        name: "main".to_string(),
        body: main_body,
    });

    // 4. Construct Final IR
    let ir = IrV1 {
        kind: "ir".to_string(),
        generator: "stunir-native-rust".to_string(),
        ir_version: "v1".to_string(),
        module_name: "main".to_string(),
        functions,
        modules: vec![],
        metadata: IrMetadata {
            original_spec_kind: combined_spec.kind,
            source_modules: combined_spec.modules,
        },
    };

    // 5. Write Output
    let ir_json = serde_json::to_string_pretty(&ir)
        .map_err(|e| StunirError::Json(e.to_string()))?;

    if let Some(parent) = Path::new(out_ir).parent() {
        fs::create_dir_all(parent).map_err(|e| StunirError::Io(e.to_string()))?;
    }

    fs::write(out_ir, ir_json).map_err(|e| StunirError::Io(e.to_string()))?;
    println!("Generated IR at {}", out_ir);

    Ok(())
}
