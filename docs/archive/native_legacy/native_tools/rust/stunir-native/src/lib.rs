//! # STUNIR Native Rust Toolchain
//!
//! This crate provides the native Rust implementation of the STUNIR (Structured Toolchain
//! for Unified Native IR) toolchain. It offers high-performance cryptographic operations,
//! JSON canonicalization, and IR processing capabilities.
//!
//! ## Overview
//!
//! STUNIR is a deterministic build and verification toolchain that ensures reproducible
//! outputs across different platforms and runtimes. This Rust implementation provides:
//!
//! - **Cryptographic Hashing**: SHA-256 hashing for files and directories using Merkle trees
//! - **JSON Canonicalization**: RFC 8785 (JCS) compliant JSON normalization
//! - **IR Processing**: Intermediate Representation parsing and transformation
//! - **Receipt Management**: Build receipt generation and verification
//! - **Code Emission**: Target code generation from IR
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use stunir_native::crypto;
//! use stunir_native::canonical;
//!
//! // Hash a file
//! let hash = crypto::hash_file(std::path::Path::new("input.json"))?;
//! println!("SHA-256: {}", hash);
//!
//! // Canonicalize JSON
//! let json = r#"{"b": 2, "a": 1}"#;
//! let canonical = canonical::normalize(json)?;
//! assert_eq!(canonical, r#"{"a":1,"b":2}"#);
//! ```
//!
//! ## Architecture
//!
//! The crate is organized into the following modules:
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`canonical`] | JSON canonicalization (JCS subset) |
//! | [`crypto`] | Cryptographic hashing utilities |
//! | [`errors`] | Error types and handling |
//! | [`ir_v1`] | Intermediate Representation v1 schema |
//! | [`receipt`] | Build receipt structures |
//! | [`spec_to_ir`] | Spec to IR transformation |
//! | [`emit`] | Code emission utilities |
//! | [`commands`] | CLI command implementations |
//!
//! ## Security Considerations
//!
//! This crate follows security best practices:
//!
//! - No symlink following (prevents traversal attacks)
//! - File size limits (prevents DoS)
//! - Directory depth limits (prevents stack overflow)
//! - Input validation on all public APIs
//!
//! ## Feature Flags
//!
//! - `derive` (serde): Enables derive macros for serialization
//! - `preserve_order` (serde_json): Maintains JSON key ordering
//!
//! ## Examples
//!
//! ### Hashing a Directory
//!
//! ```rust,ignore
//! use stunir_native::crypto::hash_directory;
//! use std::path::Path;
//!
//! let hash = hash_directory(Path::new("./src"), 0)?;
//! println!("Directory Merkle root: {}", hash);
//! ```
//!
//! ### Processing IR
//!
//! ```rust,ignore
//! use stunir_native::ir_v1::{Spec, IrV1};
//! use stunir_native::spec_to_ir::convert_spec_to_ir;
//!
//! let spec: Spec = serde_json::from_str(spec_json)?;
//! let ir: IrV1 = convert_spec_to_ir(&spec)?;
//! ```

pub mod canonical;
pub mod crypto;
pub mod errors;
pub mod ir_v1;
pub mod receipt;
pub mod spec_to_ir;
pub mod emit;
pub mod commands;

// Re-export commonly used items at crate root
pub use canonical::normalize;
pub use crypto::{hash_file, hash_directory, hash_path};
pub use errors::StunirError;
pub use ir_v1::{Spec, IrV1, IrFunction, IrInstruction};
pub use receipt::{Receipt, ToolInfo};
