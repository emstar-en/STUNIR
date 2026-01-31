//! STUNIR Core Emitters - Rust Implementation
//!
//! Core emitters for fundamental code generation capabilities.

pub mod assembly;
pub mod embedded;
pub mod gpu;
pub mod polyglot;
pub mod wasm;

// Re-export emitter types
pub use assembly::{AssemblyConfig, AssemblyEmitter, AssemblySyntax};
pub use embedded::{EmbeddedConfig, EmbeddedEmitter};
pub use gpu::{GPUConfig, GPUEmitter, GPUPlatform};
pub use polyglot::{PolyglotConfig, PolyglotEmitter, PolyglotLanguage, RustEdition};
pub use wasm::{WasmConfig, WasmEmitter, WasmTarget};
