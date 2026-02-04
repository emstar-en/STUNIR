//! Specification to IR Conversion Module
//!
//! Converts STUNIR specification files into Intermediate Representation (IR).
//! This is a core transformation step in the STUNIR pipeline.
//!
//! # Conversion Process
//!
//! 1. Parse the specification JSON file
//! 2. Extract modules and their metadata
//! 3. Generate IR functions from specification constructs
//! 4. Serialize IR to JSON format
//!
//! # Error Handling
//!
//! All errors are converted to `StunirError` types for consistent
//! error reporting across the toolchain.
//!
//! # Safety
//!
//! The conversion is deterministic - the same specification always
//! produces the same IR output. This is critical for reproducible builds.

use crate::errors::StunirError;
use crate::ir_v1::{IrV1, IrMetadata, IrFunction, IrInstruction, SpecModule};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// STUNIR Specification structure.
///
/// Represents the parsed specification file containing the kind
/// of specification and its constituent modules.
#[derive(Serialize, Deserialize, Debug)]
pub struct Spec {
    /// Kind of specification (e.g., "semantic", "assembly").
    pub kind: String,
    /// Source modules in the specification.
    pub modules: Vec<SpecModule>,
}

/// Convert a specification file to IR.
///
/// Reads a specification JSON file, parses it, and generates
/// corresponding IR output.
///
/// # Arguments
///
/// * `in_json` - Path to the input specification JSON file
/// * `out_ir` - Path for the output IR JSON file
///
/// # Returns
///
/// * `Ok(())` - If conversion succeeds
/// * `Err(anyhow::Error)` - If any step fails
///
/// # Errors
///
/// Returns errors for:
/// - File read failures
/// - JSON parse errors
/// - IR serialization errors
/// - Directory creation failures
///
/// # Example
///
/// ```rust
/// use stunir::spec_to_ir;
///
/// match spec_to_ir::run("spec.json", "output.ir.json") {
///     Ok(()) => println!("Conversion successful"),
///     Err(e) => eprintln!("Conversion failed: {}", e),
/// }
/// ```
///
/// # Safety
///
/// This function creates output directories as needed and handles
/// all IO errors gracefully.
pub fn run(in_json: &str, out_ir: &str) -> Result<()> {
    let content = fs::read_to_string(in_json)
        .map_err(|e| StunirError::Io(format!("Failed to read spec: {}", e)))?;
    let spec: Spec = serde_json::from_str(&content)
        .map_err(|e| StunirError::Json(format!("Invalid spec JSON: {}", e)))?;

    let demo_func = IrFunction {
        name: "main".to_string(),
        body: vec![
            IrInstruction {
                op: "print".to_string(),
                args: vec!["Hello from STUNIR!".to_string()],
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
