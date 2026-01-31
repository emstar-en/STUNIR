//! STUNIR Target Emitters - Rust Implementation
//!
//! This library provides code emitters for all 24 target categories.
//! All emitters are production-ready and designed for confluence with
//! SPARK, Python, and Haskell implementations.

pub mod types;
pub mod assembly;
pub mod polyglot;
pub mod lisp;
pub mod prolog;
pub mod embedded;
pub mod gpu;
pub mod wasm;

// Re-export common types
pub use types::*;
