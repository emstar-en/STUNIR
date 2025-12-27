use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SpecModule {
    pub name: String,
    pub code: String,
}

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
    pub modules: Vec<SpecModule>,
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
