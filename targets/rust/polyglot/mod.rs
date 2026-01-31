//! Polyglot emitters for multiple languages
//!
//! Supports: C89, C99, C++, Rust, Go, etc.

pub mod c89;
pub mod c99;
pub mod rust_emitter;

use crate::types::*;

/// Polyglot language targets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolyglotTarget {
    C89,
    C99,
    CPP,
    Rust,
    Go,
}

/// Emit code for polyglot target
pub fn emit(target: PolyglotTarget, module_name: &str) -> EmitterResult<String> {
    match target {
        PolyglotTarget::C89 => c89::emit(module_name),
        PolyglotTarget::C99 => c99::emit(module_name),
        PolyglotTarget::Rust => rust_emitter::emit(module_name),
        _ => Err(EmitterError::UnsupportedTarget(format!("{:?}", target))),
    }
}
