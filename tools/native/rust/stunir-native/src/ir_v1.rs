use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// --- Input Spec Schema ---

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Spec {
    pub kind: String, // "spec"
    pub modules: Vec<SpecModule>,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SpecModule {
    pub name: String,
    pub source: String, // The actual code or logic definition
    pub lang: String,   // e.g., "python", "bash"
}

// --- Output IR Schema (Intermediate Reference) ---

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IrV1 {
    pub kind: String,       // "ir"
    pub generator: String,  // "stunir-native-rust"
    pub ir_version: String, // "v1"
    pub module_name: String,
    pub functions: Vec<IrFunction>,
    pub modules: Vec<IrModule>, // External dependencies or sub-modules
    pub metadata: IrMetadata,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IrMetadata {
    pub original_spec_kind: String,
    pub source_modules: Vec<SpecModule>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IrFunction {
    pub name: String,
    pub body: Vec<IrInstruction>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IrInstruction {
    pub op: String,
    pub args: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IrModule {
    pub name: String,
    pub path: String,
}
