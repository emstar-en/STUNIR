//! STUNIR type definitions - stunir_ir_v1 schema compliant

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// IR data types (kept for internal use and backward compatibility)
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

    /// Convert to schema-compatible string
    pub fn to_schema_string(&self) -> String {
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
            IRDataType::TypeString => "string",
            IRDataType::TypeVoid => "void",
        }.to_string()
    }
}

/// IR Field (for type definitions) - matches stunir_ir_v1 schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRField {
    pub name: String,
    #[serde(rename = "type")]
    pub field_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optional: Option<bool>,
}

/// IR Type definition - matches stunir_ir_v1 schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRType {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub docstring: Option<String>,
    pub fields: Vec<IRField>,
}

/// IR Argument - matches stunir_ir_v1 schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRArg {
    pub name: String,
    #[serde(rename = "type")]
    pub arg_type: String,
}

/// IR Case entry for switch statements (v0.9.0)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRCase {
    pub value: serde_json::Value,
    pub body: Vec<IRStep>,
}

/// IR Step (operation) - matches stunir_ir_v1 schema with control flow support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRStep {
    pub op: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<serde_json::Value>,
    
    // Control flow fields (v0.6.1+)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub then_block: Option<Vec<IRStep>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub else_block: Option<Vec<IRStep>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<Vec<IRStep>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub init: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub increment: Option<String>,
    
    // Switch/case fields (v0.9.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expr: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cases: Option<Vec<IRCase>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<Vec<IRStep>>,
}

/// IR Function definition - matches stunir_ir_v1 schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRFunction {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub docstring: Option<String>,
    pub args: Vec<IRArg>,
    pub return_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steps: Option<Vec<IRStep>>,
}

/// IR Module - matches stunir_ir_v1 schema (flat structure)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRModule {
    pub schema: String,
    pub ir_version: String,
    pub module_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub docstring: Option<String>,
    pub types: Vec<IRType>,
    pub functions: Vec<IRFunction>,
}

/// Legacy types for backward compatibility
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
