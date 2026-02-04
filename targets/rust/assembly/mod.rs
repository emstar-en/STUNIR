//! Assembly code emitters
//!
//! Supports: ARM, ARM64, x86, x86_64, RISC-V, MIPS, AVR

use crate::types::*;
use std::fmt;

pub mod arm;
pub mod x86;

/// Error type for unsupported architectures
#[derive(Debug)]
pub struct UnsupportedArchitectureError {
    pub arch: Architecture,
}

impl fmt::Display for UnsupportedArchitectureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Unsupported architecture: {:?}", self.arch)
    }
}

impl std::error::Error for UnsupportedArchitectureError {}

/// Assembly emitter trait
pub trait AssemblyEmitter {
    fn emit_prologue(&self, function_name: &str, stack_size: usize) -> String;
    fn emit_epilogue(&self, function_name: &str) -> String;
    fn emit_load(&self, reg: &str, offset: i32) -> String;
    fn emit_store(&self, reg: &str, offset: i32) -> String;
}

/// Get emitter for architecture
///
/// # Errors
/// Returns `UnsupportedArchitectureError` if the architecture is not supported
pub fn get_emitter(arch: Architecture) -> Result<Box<dyn AssemblyEmitter>, UnsupportedArchitectureError> {
    match arch {
        Architecture::ARM | Architecture::ARM64 => Ok(Box::new(arm::ARMEmitter::new(arch))),
        Architecture::X86 | Architecture::X86_64 => Ok(Box::new(x86::X86Emitter::new(arch))),
        _ => Err(UnsupportedArchitectureError { arch }),
    }
}