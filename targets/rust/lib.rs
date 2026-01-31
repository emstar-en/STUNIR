//! STUNIR Target Emitters - Rust Implementation
//!
//! This library provides code emitters for all 24 target categories.
//! All emitters are production-ready and designed for confluence with
//! SPARK, Python, and Haskell implementations.

pub mod types;

// Core emitters (8 from phase 1)
pub mod assembly;
pub mod polyglot;
pub mod lisp;
pub mod prolog;
pub mod embedded;
pub mod gpu;
pub mod wasm;

// New emitters (17 from phase 2)
pub mod mobile;
pub mod fpga;
pub mod business;
pub mod bytecode;
pub mod constraints;
pub mod expert_systems;
pub mod functional;
pub mod grammar;
pub mod lexer;
pub mod parser;
pub mod oop;
pub mod planning;
pub mod scientific;
pub mod systems;
pub mod asm;
pub mod beam;
pub mod asp;
pub mod json;
pub mod asm_ir;

// Re-export common types
pub use types::*;
