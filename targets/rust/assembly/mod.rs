//! Assembly code emitters
//!
//! Supports: ARM, ARM64, x86, x86_64, RISC-V, MIPS, AVR

use crate::types::*;

pub mod arm;
pub mod x86;

/// Assembly emitter trait
pub trait AssemblyEmitter {
    fn emit_prologue(&self, function_name: &str, stack_size: usize) -> String;
    fn emit_epilogue(&self, function_name: &str) -> String;
    fn emit_load(&self, reg: &str, offset: i32) -> String;
    fn emit_store(&self, reg: &str, offset: i32) -> String;
}

/// Get emitter for architecture
pub fn get_emitter(arch: Architecture) -> Box<dyn AssemblyEmitter> {
    match arch {
        Architecture::ARM | Architecture::ARM64 => Box::new(arm::ARMEmitter::new(arch)),
        Architecture::X86 | Architecture::X86_64 => Box::new(x86::X86Emitter::new(arch)),
        _ => panic!("Unsupported architecture: {}", arch),
    }
}
