use crate::errors::StunirError;
use crate::ir_v1::IrV1;
use anyhow::Result;
use std::fs;

mod python;
mod wat;
mod js;
mod bash;
mod powershell;

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
