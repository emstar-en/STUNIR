//! STUNIR Intermediate Representation (IR) v1 Module
//!
//! Defines the core data structures for STUNIR IR v1.
//! These structures represent the typed, verifiable intermediate representation
//! used throughout the STUNIR pipeline.
//!
//! # Architecture
//!
//! The IR is structured hierarchically:
//! - `IrV1`: Top-level container with metadata
//! - `IrFunction`: Individual function definitions
//! - `IrInstruction`: Low-level operations
//! - `SpecModule`: Source module references
//! - `IrMetadata`: Compilation metadata
//!
//! # Serialization
//!
//! All types implement `Serialize` and `Deserialize` for JSON/CBOR interchange.
//! The `Debug` and `Clone` traits are provided for diagnostics and manipulation.
//!
//! # Safety
//!
//! These structures are designed for critical systems. All fields are public
//! for transparency, but immutability is encouraged through construction patterns.

use serde::{Deserialize, Serialize};

/// A source module from the specification.
///
/// Represents a single module in the original specification, containing
/// the module name and its source code.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SpecModule {
    /// Module name as declared in the specification.
    pub name: String,
    /// Source code content of the module.
    pub code: String,
}

/// A single IR instruction.
///
/// Represents a low-level operation in the intermediate representation.
/// Instructions are the building blocks of IR functions.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IrInstruction {
    /// Operation code (e.g., "add", "call", "ret").
    pub op: String,
    /// Instruction arguments as strings.
    pub args: Vec<String>,
}

/// An IR function definition.
///
/// Contains the function name and its body as a sequence of instructions.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IrFunction {
    /// Function name including namespace if applicable.
    pub name: String,
    /// Sequence of instructions forming the function body.
    pub body: Vec<IrInstruction>,
}

/// Metadata about the IR compilation.
///
/// Contains information about the compilation process, including
/// the kind of IR and referenced source modules.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IrMetadata {
    /// Kind of IR (e.g., "semantic", "assembly").
    pub kind: String,
    /// Source modules that contributed to this IR.
    pub modules: Vec<SpecModule>,
}

/// Top-level IR v1 structure.
///
/// The root container for STUNIR IR v1. Contains all functions,
/// metadata, and versioning information.
///
/// # Example
///
/// ```rust
/// use stunir::ir_v1::{IrV1, IrFunction, IrInstruction, IrMetadata, SpecModule};
///
/// let ir = IrV1 {
///     kind: "semantic".to_string(),
///     generator: "stunir-native".to_string(),
///     ir_version: "1.0.0".to_string(),
///     module_name: "main".to_string(),
///     functions: vec![],
///     modules: vec![],
///     metadata: IrMetadata {
///         kind: "semantic".to_string(),
///         modules: vec![],
///     },
/// };
/// ```
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IrV1 {
    /// Kind of IR (e.g., "semantic", "assembly", "bytecode").
    pub kind: String,
    /// Name and version of the generator tool.
    pub generator: String,
    /// Version of the IR schema.
    pub ir_version: String,
    /// Name of the primary module.
    pub module_name: String,
    /// All functions defined in this IR.
    pub functions: Vec<IrFunction>,
    /// Names of referenced modules.
    pub modules: Vec<String>,
    /// Compilation metadata.
    pub metadata: IrMetadata,
}
