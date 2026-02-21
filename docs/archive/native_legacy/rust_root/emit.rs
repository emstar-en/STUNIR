//! Code Emission Module
//!
//! Provides code generation for multiple target languages from STUNIR IR.
//! Supports Python, WebAssembly Text (WAT), JavaScript, Bash, and PowerShell.
//!
//! # Supported Targets
//!
//! - `python`: Python 3 code generation
//! - `wat`: WebAssembly Text format
//! - `js`: JavaScript/Node.js code
//! - `bash`: Bash shell scripts
//! - `powershell`: PowerShell scripts
//!
//! # Architecture
//!
//! Each target language has its own submodule implementing the `emit` function.
//! The main `emit::run` function dispatches to the appropriate target handler.
//!
//! # Safety
//!
//! Code emission is deterministic - the same IR always produces the same
//! output code. This is essential for reproducible builds in critical systems.

use crate::errors::StunirError;
use crate::ir_v1::IrV1;
use anyhow::Result;
use std::fs;

mod python;
mod wat;
mod js;
mod bash;
mod powershell;

/// Emit code for a specific target language.
///
/// Reads IR from a JSON file and generates code in the specified
/// target language, writing it to the output file.
///
/// # Arguments
///
/// * `in_ir` - Path to the input IR JSON file
/// * `target` - Target language identifier ("python", "wat", "js", "bash", "powershell")
/// * `out_file` - Path for the generated output file
///
/// # Returns
///
/// * `Ok(())` - If emission succeeds
/// * `Err(anyhow::Error)` - If emission fails
///
/// # Errors
///
/// Returns errors for:
/// - File read failures
/// - JSON parse errors
/// - Unsupported target languages
/// - Code generation failures
///
/// # Example
///
/// ```rust
/// use stunir::emit;
///
/// match emit::run("input.ir.json", "python", "output.py") {
///     Ok(()) => println!("Code generation successful"),
///     Err(e) => eprintln!("Code generation failed: {}", e),
/// }
/// ```
///
/// # Safety
///
/// Target-specific emitters handle their own error conditions.
/// The dispatch is exhaustive for supported targets.
pub fn run(in_ir: &str, target: &str, out_file: &str) -> Result<()> {
    println!("Emitting code for target: {}", target);

    let content = fs::read_to_string(in_ir)
        .map_err(|e| StunirError::Io(format!("Failed to read IR: {}", e)))?;
    let ir: IrV1 = serde_json::from_str(&content)
        .map_err(|e| StunirError::Json(format!("Invalid IR JSON: {}", e)))?;

    match target {
        "python" => python::emit(&ir, out_file),
        "wat" => wat::emit(&ir, out_file),
        "js" => js::emit(&ir, out_file),
        "bash" => bash::emit(&ir, out_file),
        "powershell" => powershell::emit(&ir, out_file),
        _ => Err(StunirError::Usage(format!("Unsupported emission target: {}", target)).into()),
    }
}
