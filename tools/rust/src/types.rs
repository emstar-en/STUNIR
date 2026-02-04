//! STUNIR type definitions - stunir_ir_v1 schema compliant (v0.8.9+)

use serde::{Deserialize, Serialize};

/// IR data types (kept for internal use and backward compatibility)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IRDataType {
    /// Signed 8-bit integer
    TypeI8,
    /// Signed 16-bit integer
    TypeI16,
    /// Signed 32-bit integer
    TypeI32,
    /// Signed 64-bit integer
    TypeI64,
    /// Unsigned 8-bit integer
    TypeU8,
    /// Unsigned 16-bit integer
    TypeU16,
    /// Unsigned 32-bit integer
    TypeU32,
    /// Unsigned 64-bit integer
    TypeU64,
    /// 32-bit floating point
    TypeF32,
    /// 64-bit floating point
    TypeF64,
    /// Boolean type
    TypeBool,
    /// String type
    TypeString,
    /// Void/empty type
    TypeVoid,
}

/// Type reference - can be simple string or complex type (v0.8.8+)
/// Type reference - can be simple string or complex type (v0.8.8+)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TypeRef {
    /// Simple type name as a string
    Simple(String),
    /// Complex type with parameters
    Complex(ComplexType),
}

/// Complex type definition (v0.8.9+)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexType {
    /// Type kind (e.g., "array", "map", "optional", "result")
    pub kind: String,
    /// Element type for arrays and optionals
    #[serde(skip_serializing_if = "Option::is_none")]
    pub element_type: Option<Box<TypeRef>>,
    /// Key type for maps
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key_type: Option<Box<TypeRef>>,
    /// Value type for maps
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value_type: Option<Box<TypeRef>>,
    /// Size for fixed-size arrays
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<i64>,
    /// Inner type for result types
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inner: Option<Box<TypeRef>>,
    /// Base type name for generic types (v0.8.9+)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_type: Option<String>,
    /// Type arguments for generic types (v0.8.9+)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub type_args: Option<Vec<TypeRef>>,
}

/// Generic type parameter (v0.8.9+)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeParam {
    /// Parameter name
    pub name: String,
    /// Optional type constraint
    #[serde(skip_serializing_if = "Option::is_none")]
    pub constraint: Option<String>,
    /// Optional default type
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<TypeRef>,
}

/// Generic type instantiation (v0.8.9+)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenericInstantiation {
    /// Instantiation name
    pub name: String,
    /// Base type being instantiated
    pub base_type: String,
    /// Type arguments for instantiation
    pub type_args: Vec<TypeRef>,
}

/// Optimization hints (v0.8.9+)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OptimizationHint {
    /// Whether the function is pure (no side effects)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pure: Option<bool>,
    /// Whether the function should be inlined
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inline: Option<bool>,
    /// Whether the function can be evaluated at compile time
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub const_eval: Option<bool>,
    /// Whether this is dead code
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dead_code: Option<bool>,
    /// Constant value if known
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constant_value: Option<serde_json::Value>,
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
    /// Field name
    pub name: String,
    /// Field type (can be string or complex type object v0.8.8+)
    #[serde(rename = "type")]
    pub field_type: serde_json::Value,
    /// Whether the field is optional
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optional: Option<bool>,
}

/// IR Type definition - matches stunir_ir_v1 schema (v0.8.9+)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRType {
    /// Type name
    pub name: String,
    /// Optional documentation string
    #[serde(skip_serializing_if = "Option::is_none")]
    pub docstring: Option<String>,
    /// Generic type parameters (v0.8.9+)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub type_params: Option<Vec<TypeParam>>,
    /// Type fields
    pub fields: Vec<IRField>,
}

/// IR Argument - matches stunir_ir_v1 schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRArg {
    /// Argument name
    pub name: String,
    /// Argument type
    #[serde(rename = "type")]
    pub arg_type: String,
}

/// IR Case entry for switch statements (v0.9.0)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRCase {
    /// Case value to match against
    pub value: serde_json::Value,
    /// Body steps to execute when case matches
    pub body: Vec<IRStep>,
    /// Flattened IR: case block start index (1-based)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_start: Option<usize>,
    /// Flattened IR: case block length
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_count: Option<usize>,
}

/// IR Catch block entry for exception handling (v0.8.7)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRCatch {
    /// Exception type to catch
    pub exception_type: String,
    /// Optional variable name to bind the exception to
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exception_var: Option<String>,
    /// Body steps to execute in catch block
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
    // Flattened IR: block indices for if/then/else
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_start: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_count: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub else_start: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub else_count: Option<usize>,

    // Switch/case fields (v0.9.0)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expr: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cases: Option<Vec<IRCase>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<Vec<IRStep>>,
    // Flattened IR: block indices for default case
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_start: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_count: Option<usize>,

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
    // v0.9.0: Exception variable and throw value for tests
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exception_var: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub throw_value: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<serde_json::Value>,

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
    // v0.9.0: Args for tests
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub args: Option<Vec<serde_json::Value>>,
    
    // v0.8.9: Generic call and type cast fields
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub type_args: Option<Vec<TypeRef>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cast_type: Option<TypeRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optimization: Option<OptimizationHint>,
}

/// IR Function definition - matches stunir_ir_v1 schema (v0.8.9+)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRFunction {
    /// Function name
    pub name: String,
    /// Optional documentation string
    #[serde(skip_serializing_if = "Option::is_none")]
    pub docstring: Option<String>,
    /// Generic type parameters (v0.8.9+)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub type_params: Option<Vec<TypeParam>>,
    /// Optimization hints (v0.8.9+)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optimization: Option<OptimizationHint>,
    /// Generic instantiations for tests (v0.9.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generic_instantiations: Option<Vec<GenericInstantiation>>,
    /// Parameters as Option for test compatibility (v0.9.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Vec<IRArg>>,
    /// Function arguments
    pub args: Vec<IRArg>,
    /// Return type as Option for test compatibility (v0.9.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_type: Option<String>,
    /// Function body steps
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steps: Option<Vec<IRStep>>,
}

/// IR Module - matches stunir_ir_v1 schema (v0.8.9+)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRModule {
    /// Schema identifier
    pub schema: String,
    /// IR version
    pub ir_version: String,
    /// Module name
    pub module_name: String,
    /// Optional documentation string
    #[serde(skip_serializing_if = "Option::is_none")]
    pub docstring: Option<String>,
    /// Module-level type parameters (v0.8.9+)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub type_params: Option<Vec<TypeParam>>,
    /// Optimization level (v0.8.9+)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optimization_level: Option<i32>,
    /// Type definitions as Option for test compatibility (v0.9.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub types: Option<Vec<IRType>>,
    /// Generic types for tests (v0.9.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generic_types: Option<Vec<IRType>>,
    /// Imports for tests (v0.9.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub imports: Option<Vec<serde_json::Value>>,
    /// Generic instantiations (v0.8.9+)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generic_instantiations: Option<Vec<GenericInstantiation>>,
    /// Module functions
    pub functions: Vec<IRFunction>,
}

/// Parameter for IR functions (used in ir_to_code)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    pub name: String,
    pub param_type: TypeRef,
}

/// Field for struct definitions (used in ir_to_code)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Field {
    pub name: String,
    pub field_type: TypeRef,
}

/// IR Type with default for convenience
impl Default for IRType {
    fn default() -> Self {
        IRType {
            name: String::new(),
            docstring: None,
            type_params: None,
            fields: Vec::new(),
        }
    }
}

/// IR Parameter definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRParameter {
    /// Parameter name
    pub name: String,
    /// Parameter data type
    pub param_type: IRDataType,
}

/// IR Statement - represents a single statement in IR
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum IRStatement {
    /// Return statement with optional value
    Return {
        /// Optional return value expression
        value: Option<IRExpression>,
    },
    /// Assignment statement
    Assignment {
        /// Target variable name
        target: String,
        /// Value expression to assign
        value: IRExpression,
    },
    /// Function call statement
    Call {
        /// Function name to call
        function: String,
        /// Arguments to pass to the function
        arguments: Vec<IRExpression>,
    },
}

/// IR Expression - represents an expression in IR
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum IRExpression {
    /// Literal value expression
    Literal {
        /// The literal value as JSON
        value: serde_json::Value,
    },
    /// Variable reference expression
    Variable {
        /// Variable name
        name: String,
    },
    /// Binary operation expression
    BinaryOp {
        /// Binary operator (e.g., "+", "-", "*", "/")
        op: String,
        /// Left operand
        left: Box<IRExpression>,
        /// Right operand
        right: Box<IRExpression>,
    },
}
