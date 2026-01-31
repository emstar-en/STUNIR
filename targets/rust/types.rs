//! Common emitter types for STUNIR

use std::fmt;

/// Architecture types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    ARM,
    ARM64,
    X86,
    X86_64,
    RISCV,
    MIPS,
    AVR,
}

impl fmt::Display for Architecture {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Architecture::ARM => write!(f, "ARM"),
            Architecture::ARM64 => write!(f, "ARM64"),
            Architecture::X86 => write!(f, "x86"),
            Architecture::X86_64 => write!(f, "x86_64"),
            Architecture::RISCV => write!(f, "RISC-V"),
            Architecture::MIPS => write!(f, "MIPS"),
            Architecture::AVR => write!(f, "AVR"),
        }
    }
}

/// Emitter result
pub type EmitterResult<T> = Result<T, EmitterError>;

/// Emitter errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmitterError {
    UnsupportedTarget(String),
    InvalidConfiguration(String),
    GenerationFailed(String),
}

impl fmt::Display for EmitterError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EmitterError::UnsupportedTarget(s) => write!(f, "Unsupported target: {}", s),
            EmitterError::InvalidConfiguration(s) => write!(f, "Invalid configuration: {}", s),
            EmitterError::GenerationFailed(s) => write!(f, "Generation failed: {}", s),
        }
    }
}

impl std::error::Error for EmitterError {}
