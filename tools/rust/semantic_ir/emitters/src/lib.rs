//! STUNIR Semantic IR Emitters - Rust Implementation
//!
//! This crate provides Rust implementations for all 24 STUNIR semantic IR
//! emitters, based on the DO-178C Level A compliant Ada SPARK implementations.
//!
//! All emitters produce identical outputs to their SPARK counterparts (confluence).

#![deny(missing_docs)]
#![deny(unsafe_code)]

pub mod types;
pub mod base;
pub mod visitor;
pub mod codegen;
pub mod core;
pub mod language_families;
pub mod specialized;

pub use base::{BaseEmitter, EmitterConfig, EmitterResult, EmitterStatus};
pub use types::*;

/// Emitter version
pub const VERSION: &str = "1.0.0";
