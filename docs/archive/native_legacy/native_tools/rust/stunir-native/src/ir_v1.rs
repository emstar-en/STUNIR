//! # STUNIR Intermediate Representation (IR) v1
//!
//! This module defines the data structures for STUNIR's Intermediate Representation
//! version 1. The IR serves as the canonical intermediate format between input
//! specifications and target code generation.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────┐     ┌──────────┐     ┌──────────────┐
//! │   Spec   │ --> │   IR v1  │ --> │ Target Code  │
//! │  (JSON)  │     │ (JSON)   │     │ (Rust/C/etc) │
//! └──────────┘     └──────────┘     └──────────────┘
//! ```
//!
//! ## Schema Overview
//!
//! ### Input: Spec
//!
//! The input specification defines modules with source code:
//!
//! ```json
//! {
//!   "kind": "spec",
//!   "modules": [
//!     {"name": "main", "source": "print('hello')", "lang": "python"}
//!   ],
//!   "metadata": {"author": "stunir"}
//! }
//! ```
//!
//! ### Output: IR v1
//!
//! The IR normalizes specs into a target-agnostic format:
//!
//! ```json
//! {
//!   "kind": "ir",
//!   "generator": "stunir-native-rust",
//!   "ir_version": "v1",
//!   "module_name": "main",
//!   "functions": [
//!     {"name": "main", "body": [{"op": "call", "args": ["print", "hello"]}]}
//!   ],
//!   "modules": [],
//!   "metadata": {...}
//! }
//! ```
//!
//! ## Usage Examples
//!
//! ### Parsing a Spec
//!
//! ```rust
//! use stunir_native::ir_v1::Spec;
//!
//! let json = r#"{"kind": "spec", "modules": [], "metadata": {}}"#;
//! let spec: Spec = serde_json::from_str(json).unwrap();
//! assert_eq!(spec.kind, "spec");
//! ```
//!
//! ### Creating IR Programmatically
//!
//! ```rust
//! use stunir_native::ir_v1::{IrV1, IrFunction, IrInstruction, IrMetadata, IrModule};
//!
//! let ir = IrV1 {
//!     kind: "ir".to_string(),
//!     generator: "example".to_string(),
//!     ir_version: "v1".to_string(),
//!     module_name: "test".to_string(),
//!     functions: vec![
//!         IrFunction {
//!             name: "main".to_string(),
//!             body: vec![
//!                 IrInstruction {
//!                     op: "return".to_string(),
//!                     args: vec!["0".to_string()],
//!                 }
//!             ],
//!         }
//!     ],
//!     modules: vec![],
//!     metadata: IrMetadata {
//!         original_spec_kind: "spec".to_string(),
//!         source_modules: vec![],
//!     },
//! };
//! ```
//!
//! ## Instruction Set
//!
//! The IR instruction set includes:
//!
//! | Op | Arguments | Description |
//! |----|-----------|-------------|
//! | `call` | `[func, ...args]` | Call a function |
//! | `return` | `[value]` | Return from function |
//! | `assign` | `[var, value]` | Assign to variable |
//! | `load` | `[var]` | Load variable value |
//! | `store` | `[var, value]` | Store to variable |
//! | `branch` | `[cond, true_label, false_label]` | Conditional branch |
//! | `jump` | `[label]` | Unconditional jump |
//! | `label` | `[name]` | Label definition |
//!
//! ## Versioning
//!
//! The IR format is versioned to ensure compatibility:
//!
//! - **v1**: Current stable version (this module)
//! - IR version is embedded in the `ir_version` field
//! - Tools should check version before processing

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Input Spec Schema
// ============================================================================

/// Input specification format.
///
/// A Spec contains one or more modules with source code in various languages.
/// This is the primary input format for the STUNIR toolchain.
///
/// # Fields
///
/// - `kind`: Always "spec" for specification documents
/// - `modules`: List of source modules
/// - `metadata`: Optional key-value metadata
///
/// # Examples
///
/// ```rust
/// use stunir_native::ir_v1::Spec;
///
/// let spec = Spec {
///     kind: "spec".to_string(),
///     modules: vec![],
///     metadata: std::collections::HashMap::new(),
/// };
/// ```
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Spec {
    /// Document kind identifier, always "spec".
    pub kind: String,
    
    /// List of source modules in the specification.
    pub modules: Vec<SpecModule>,
    
    /// Optional metadata key-value pairs.
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// A source module within a specification.
///
/// Each module contains source code in a specific language that will be
/// transformed into IR.
///
/// # Fields
///
/// - `name`: Unique module identifier
/// - `source`: The actual source code or logic definition
/// - `lang`: Source language (e.g., "python", "bash", "javascript")
///
/// # Examples
///
/// ```rust
/// use stunir_native::ir_v1::SpecModule;
///
/// let module = SpecModule {
///     name: "main".to_string(),
///     source: "print('Hello, World!')".to_string(),
///     lang: "python".to_string(),
/// };
/// ```
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct SpecModule {
    /// Unique module name/identifier.
    pub name: String,
    
    /// The actual source code or logic definition.
    pub source: String,
    
    /// Source language identifier (e.g., "python", "bash", "javascript").
    pub lang: String,
}

// ============================================================================
// Output IR Schema (Intermediate Representation)
// ============================================================================

/// STUNIR Intermediate Representation version 1.
///
/// The IR is a normalized, target-agnostic representation of the input
/// specification. It contains structured function definitions that can
/// be emitted to various target languages.
///
/// # Invariants
///
/// - `kind` must be "ir"
/// - `ir_version` must be "v1" for this struct
/// - `module_name` must be a valid identifier
/// - Functions must have unique names within the IR
///
/// # Examples
///
/// ```rust
/// use stunir_native::ir_v1::{IrV1, IrMetadata};
///
/// let ir = IrV1 {
///     kind: "ir".to_string(),
///     generator: "stunir-native-rust".to_string(),
///     ir_version: "v1".to_string(),
///     module_name: "example".to_string(),
///     functions: vec![],
///     modules: vec![],
///     metadata: IrMetadata {
///         original_spec_kind: "spec".to_string(),
///         source_modules: vec![],
///     },
/// };
/// ```
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct IrV1 {
    /// Document kind identifier, always "ir".
    pub kind: String,
    
    /// Generator tool identifier (e.g., "stunir-native-rust").
    pub generator: String,
    
    /// IR schema version (currently "v1").
    pub ir_version: String,
    
    /// Primary module name for this IR.
    pub module_name: String,
    
    /// List of functions defined in this IR.
    pub functions: Vec<IrFunction>,
    
    /// External module dependencies.
    pub modules: Vec<IrModule>,
    
    /// Metadata preserving spec information.
    pub metadata: IrMetadata,
}

/// Metadata associated with an IR document.
///
/// Preserves information from the original specification for
/// traceability and debugging.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct IrMetadata {
    /// The kind of the original specification ("spec").
    pub original_spec_kind: String,
    
    /// Source modules that contributed to this IR.
    pub source_modules: Vec<SpecModule>,
}

/// A function definition in the IR.
///
/// Functions contain a sequence of instructions that define
/// the function's behavior.
///
/// # Examples
///
/// ```rust
/// use stunir_native::ir_v1::{IrFunction, IrInstruction};
///
/// let func = IrFunction {
///     name: "add".to_string(),
///     body: vec![
///         IrInstruction {
///             op: "return".to_string(),
///             args: vec!["a + b".to_string()],
///         }
///     ],
/// };
/// ```
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct IrFunction {
    /// Function name (must be a valid identifier).
    pub name: String,
    
    /// Sequence of IR instructions forming the function body.
    pub body: Vec<IrInstruction>,
}

/// A single IR instruction.
///
/// Instructions are the atomic operations in the IR. Each instruction
/// has an operation code and a list of arguments.
///
/// # Common Operations
///
/// - `call`: Call a function with arguments
/// - `return`: Return a value from a function
/// - `assign`: Assign a value to a variable
/// - `branch`: Conditional branch
///
/// # Examples
///
/// ```rust
/// use stunir_native::ir_v1::IrInstruction;
///
/// // Function call
/// let call = IrInstruction {
///     op: "call".to_string(),
///     args: vec!["print".to_string(), "hello".to_string()],
/// };
///
/// // Return statement
/// let ret = IrInstruction {
///     op: "return".to_string(),
///     args: vec!["0".to_string()],
/// };
/// ```
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct IrInstruction {
    /// Operation code (e.g., "call", "return", "assign").
    pub op: String,
    
    /// Operation arguments (semantics depend on op).
    pub args: Vec<String>,
}

/// An external module dependency.
///
/// Represents a module that this IR depends on but is defined elsewhere.
///
/// # Examples
///
/// ```rust
/// use stunir_native::ir_v1::IrModule;
///
/// let dep = IrModule {
///     name: "utils".to_string(),
///     path: "./lib/utils".to_string(),
/// };
/// ```
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct IrModule {
    /// Module name for imports.
    pub name: String,
    
    /// Path to the module (relative or absolute).
    pub path: String,
}

// ============================================================================
// Utility Functions
// ============================================================================

impl IrV1 {
    /// Create a new empty IR v1 document.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stunir_native::ir_v1::IrV1;
    ///
    /// let ir = IrV1::new("my_module");
    /// assert_eq!(ir.module_name, "my_module");
    /// ```
    pub fn new(module_name: &str) -> Self {
        IrV1 {
            kind: "ir".to_string(),
            generator: "stunir-native-rust".to_string(),
            ir_version: "v1".to_string(),
            module_name: module_name.to_string(),
            functions: vec![],
            modules: vec![],
            metadata: IrMetadata {
                original_spec_kind: "spec".to_string(),
                source_modules: vec![],
            },
        }
    }

    /// Add a function to this IR.
    pub fn add_function(&mut self, func: IrFunction) {
        self.functions.push(func);
    }

    /// Check if this IR is valid.
    pub fn validate(&self) -> Result<(), String> {
        if self.kind != "ir" {
            return Err("kind must be 'ir'".to_string());
        }
        if self.ir_version != "v1" {
            return Err("ir_version must be 'v1'".to_string());
        }
        if self.module_name.is_empty() {
            return Err("module_name cannot be empty".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spec_serialization() {
        let spec = Spec {
            kind: "spec".to_string(),
            modules: vec![SpecModule {
                name: "test".to_string(),
                source: "code".to_string(),
                lang: "python".to_string(),
            }],
            metadata: HashMap::new(),
        };
        
        let json = serde_json::to_string(&spec).unwrap();
        let parsed: Spec = serde_json::from_str(&json).unwrap();
        assert_eq!(spec, parsed);
    }

    #[test]
    fn test_ir_new() {
        let ir = IrV1::new("test");
        assert_eq!(ir.kind, "ir");
        assert_eq!(ir.ir_version, "v1");
        assert!(ir.validate().is_ok());
    }

    #[test]
    fn test_ir_validation() {
        let mut ir = IrV1::new("test");
        ir.kind = "wrong".to_string();
        assert!(ir.validate().is_err());
    }
}
