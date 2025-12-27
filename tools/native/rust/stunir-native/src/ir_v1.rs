use serde::{Deserialize, Serialize};
use crate::spec_to_ir::SpecModule; // Import from spec_to_ir or move definitions here. 
// To avoid circular deps, let's redefine or move. 
// For simplicity in this patch, I'll redefine the Metadata struct here or assume it's passed correctly.
// Actually, let's keep it self-contained.

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IrInstruction {
    pub op: String,
    pub args: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IrFunction {
    pub name: String,
    pub body: Vec<IrInstruction>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IrMetadata {
    pub kind: String,
    pub modules: Vec<crate::spec_to_ir::SpecModule>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IrV1 {
    pub kind: String,
    pub generator: String,
    pub ir_version: String,
    pub module_name: String,
    pub functions: Vec<IrFunction>,
    pub modules: Vec<String>,
    pub metadata: IrMetadata,
}
