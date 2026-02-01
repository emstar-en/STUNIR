//! STUNIR type definitions - stunir_ir_v1 schema compliant (v0.8.8+)

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

/// Type reference - can be simple string or complex type (v0.8.8+)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TypeRef {
    Simple(String),
    Complex(ComplexType),
}

/// Complex type definition (v0.8.8+)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexType {
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub element_type: Option<Box<TypeRef>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key_type: Option<Box<TypeRef>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value_type: Option<Box<TypeRef>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inner: Option<Box<TypeRef>>,
}

impl TypeRef {
    /// Convert type reference to C type string
    pub fn to_c_type(&self) -> String {
        match self {
            TypeRef::Simple(s) => map_simple_type_to_c(s),
            TypeRef::Complex(c) => match c.kind.as_str() {
                "array" => {
                    let elem = c.element_type.as_ref()
                        .map(|t| t.to_c_type())
                        .unwrap_or_else(|| "int32_t".to_string());
                    if c.size.is_some() {
                        elem
                    } else {
                        format!("{}*", elem)
                    }
                }
                "map" | "set" => "void*".to_string(),
                "optional" => {
                    let inner = c.inner.as_ref()
                        .map(|t| t.to_c_type())
                        .unwrap_or_else(|| "void".to_string());
                    if inner == "void" {
                        "void*".to_string()
                    } else {
                        format!("{}*", inner)
                    }
                }
                _ => "void*".to_string(),
            },
        }
    }

    /// Convert type reference to Rust type string
    pub fn to_rust_type(&self) -> String {
        match self {
            TypeRef::Simple(s) => map_simple_type_to_rust(s),
            TypeRef::Complex(c) => match c.kind.as_str() {
                "array" => {
                    let elem = c.element_type.as_ref()
                        .map(|t| t.to_rust_type())
                        .unwrap_or_else(|| "i32".to_string());
                    if let Some(size) = c.size {
                        format!("[{}; {}]", elem, size)
                    } else {
                        format!("Vec<{}>", elem)
                    }
                }
                "map" => {
                    let key = c.key_type.as_ref()
                        .map(|t| t.to_rust_type())
                        .unwrap_or_else(|| "String".to_string());
                    let val = c.value_type.as_ref()
                        .map(|t| t.to_rust_type())
                        .unwrap_or_else(|| "i32".to_string());
                    format!("std::collections::BTreeMap<{}, {}>", key, val)
                }
                "set" => {
                    let elem = c.element_type.as_ref()
                        .map(|t| t.to_rust_type())
                        .unwrap_or_else(|| "i32".to_string());
                    format!("std::collections::BTreeSet<{}>", elem)
                }
                "optional" => {
                    let inner = c.inner.as_ref()
                        .map(|t| t.to_rust_type())
                        .unwrap_or_else(|| "()".to_string());
                    format!("Option<{}>", inner)
                }
                _ => "()".to_string(),
            },
        }
    }
}

fn map_simple_type_to_c(s: &str) -> String {
    match s {
        "i8" => "int8_t".to_string(),
        "i16" => "int16_t".to_string(),
        "i32" => "int32_t".to_string(),
        "i64" => "int64_t".to_string(),
        "u8" => "uint8_t".to_string(),
        "u16" => "uint16_t".to_string(),
        "u32" => "uint32_t".to_string(),
        "u64" => "uint64_t".to_string(),
        "f32" => "float".to_string(),
        "f64" => "double".to_string(),
        "bool" => "bool".to_string(),
        "string" => "const char*".to_string(),
        "void" => "void".to_string(),
        "byte[]" => "const uint8_t*".to_string(),
        _ => format!("struct {}", s),
    }
}

fn map_simple_type_to_rust(s: &str) -> String {
    match s {
        "i8" => "i8".to_string(),
        "i16" => "i16".to_string(),
        "i32" => "i32".to_string(),
        "i64" => "i64".to_string(),
        "u8" => "u8".to_string(),
        "u16" => "u16".to_string(),
        "u32" => "u32".to_string(),
        "u64" => "u64".to_string(),
        "f32" => "f32".to_string(),
        "f64" => "f64".to_string(),
        "bool" => "bool".to_string(),
        "string" => "String".to_string(),
        "void" => "()".to_string(),
        "byte[]" => "Vec<u8>".to_string(),
        _ => s.to_string(),
    }
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
    pub field_type: serde_json::Value,  // Can be string or complex type object (v0.8.8+)
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

/// IR Catch block entry for exception handling (v0.8.7)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRCatch {
    pub exception_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exception_var: Option<String>,
    pub body: Vec<IRStep>,
}

/// IR Step (operation) - matches stunir_ir_v1 schema with control flow support
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IRStep {
    #[serde(default)]
    pub op: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<serde_json::Value>,
    
    // Control flow fields (v0.6.1+)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub then_block: Option<Vec<IRStep>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub else_block: Option<Vec<IRStep>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<Vec<IRStep>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub increment: Option<String>,
    
    // Switch/case fields (v0.9.0)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expr: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cases: Option<Vec<IRCase>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<Vec<IRStep>>,
    
    // Exception handling fields (v0.8.7)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub try_block: Option<Vec<IRStep>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub catch_blocks: Option<Vec<IRCatch>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finally_block: Option<Vec<IRStep>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exception_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exception_message: Option<String>,
    
    // Data structure fields (v0.8.8+)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub field: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub element_type: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key_type: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value_type: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub struct_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source2: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fields: Option<serde_json::Value>,
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
