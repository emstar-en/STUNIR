//! STUNIR type definitions

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// IR data types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IRDataType {
    TypeI8,
    TypeI16,
    TypeI32,
    TypeI64,
    TypeU8,
    TypeU16,
    TypeU32,
    TypeU64,
    TypeF32,
    TypeF64,
    TypeBool,
    TypeString,
    TypeVoid,
}

impl IRDataType {
    /// Map to C type name
    pub fn to_c_type(&self) -> &'static str {
        match self {
            IRDataType::TypeI8 => "int8_t",
            IRDataType::TypeI16 => "int16_t",
            IRDataType::TypeI32 => "int32_t",
            IRDataType::TypeI64 => "int64_t",
            IRDataType::TypeU8 => "uint8_t",
            IRDataType::TypeU16 => "uint16_t",
            IRDataType::TypeU32 => "uint32_t",
            IRDataType::TypeU64 => "uint64_t",
            IRDataType::TypeF32 => "float",
            IRDataType::TypeF64 => "double",
            IRDataType::TypeBool => "bool",
            IRDataType::TypeString => "char*",
            IRDataType::TypeVoid => "void",
        }
    }

    /// Map to Rust type name
    pub fn to_rust_type(&self) -> &'static str {
        match self {
            IRDataType::TypeI8 => "i8",
            IRDataType::TypeI16 => "i16",
            IRDataType::TypeI32 => "i32",
            IRDataType::TypeI64 => "i64",
            IRDataType::TypeU8 => "u8",
            IRDataType::TypeU16 => "u16",
            IRDataType::TypeU32 => "u32",
            IRDataType::TypeU64 => "u64",
            IRDataType::TypeF32 => "f32",
            IRDataType::TypeF64 => "f64",
            IRDataType::TypeBool => "bool",
            IRDataType::TypeString => "String",
            IRDataType::TypeVoid => "()",
        }
    }
}

/// IR Function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRFunction {
    pub name: String,
    pub return_type: IRDataType,
    pub parameters: Vec<IRParameter>,
    pub body: Vec<IRStatement>,
}

/// IR Parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRParameter {
    pub name: String,
    pub param_type: IRDataType,
}

/// IR Statement
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum IRStatement {
    Return { value: Option<IRExpression> },
    Assignment { target: String, value: IRExpression },
    Call { function: String, arguments: Vec<IRExpression> },
}

/// IR Expression
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum IRExpression {
    Literal { value: serde_json::Value },
    Variable { name: String },
    BinaryOp { op: String, left: Box<IRExpression>, right: Box<IRExpression> },
}

/// IR Module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRModule {
    pub name: String,
    pub version: String,
    pub functions: Vec<IRFunction>,
}

/// IR Manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRManifest {
    pub schema: String,
    pub ir_hash: String,
    pub module: IRModule,
}
