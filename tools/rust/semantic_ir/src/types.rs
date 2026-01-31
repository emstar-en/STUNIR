//! STUNIR Semantic IR Core Types
//!
//! Type definitions and enumerations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// IR Name (bounded string)
pub type IRName = String;

/// IR Hash (SHA-256 digest)
pub type IRHash = String;

/// IR Path
pub type IRPath = String;

/// Node ID
pub type NodeID = String;

/// Primitive type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IRPrimitiveType {
    Void,
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    String,
    Char,
}

/// Node kind discriminator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IRNodeKind {
    // Module
    Module,
    // Declarations
    FunctionDecl,
    TypeDecl,
    ConstDecl,
    VarDecl,
    // Statements
    BlockStmt,
    ExprStmt,
    IfStmt,
    WhileStmt,
    ForStmt,
    ReturnStmt,
    BreakStmt,
    ContinueStmt,
    VarDeclStmt,
    AssignStmt,
    // Expressions
    IntegerLiteral,
    FloatLiteral,
    StringLiteral,
    BoolLiteral,
    VarRef,
    BinaryExpr,
    UnaryExpr,
    FunctionCall,
    MemberExpr,
    ArrayAccess,
    CastExpr,
    TernaryExpr,
    ArrayInit,
    StructInit,
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BinaryOperator {
    #[serde(rename = "+")]
    Add,
    #[serde(rename = "-")]
    Sub,
    #[serde(rename = "*")]
    Mul,
    #[serde(rename = "/")]
    Div,
    #[serde(rename = "%")]
    Mod,
    #[serde(rename = "==")]
    Eq,
    #[serde(rename = "!=")]
    Neq,
    #[serde(rename = "<")]
    Lt,
    #[serde(rename = "<=")]
    Leq,
    #[serde(rename = ">")]
    Gt,
    #[serde(rename = ">=")]
    Geq,
    #[serde(rename = "&&")]
    And,
    #[serde(rename = "||")]
    Or,
    #[serde(rename = "&")]
    BitAnd,
    #[serde(rename = "|")]
    BitOr,
    #[serde(rename = "^")]
    BitXor,
    #[serde(rename = "<<")]
    Shl,
    #[serde(rename = ">>")]
    Shr,
    #[serde(rename = "=")]
    Assign,
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnaryOperator {
    #[serde(rename = "-")]
    Neg,
    #[serde(rename = "!")]
    Not,
    #[serde(rename = "~")]
    BitNot,
    #[serde(rename = "++")]
    PreInc,
    #[serde(rename = "--")]
    PreDec,
    #[serde(rename = "++")]
    PostInc,
    #[serde(rename = "--")]
    PostDec,
    #[serde(rename = "*")]
    Deref,
    #[serde(rename = "&")]
    AddrOf,
}

/// Storage class
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StorageClass {
    Auto,
    Static,
    Extern,
    Register,
    Stack,
    Heap,
    Global,
}

/// Visibility kind
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VisibilityKind {
    Public,
    Private,
    Protected,
    Internal,
}

/// Mutability kind
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MutabilityKind {
    Mutable,
    Immutable,
    Const,
}

/// Inline hint
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InlineHint {
    Always,
    Never,
    Hint,
    None,
}

/// Target categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetCategory {
    Embedded,
    Realtime,
    SafetyCritical,
    Gpu,
    Wasm,
    Native,
    Jit,
    Interpreter,
    Functional,
    Logic,
}

/// Safety level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SafetyLevel {
    None,
    #[serde(rename = "DO-178C_Level_D")]
    DO178C_D,
    #[serde(rename = "DO-178C_Level_C")]
    DO178C_C,
    #[serde(rename = "DO-178C_Level_B")]
    DO178C_B,
    #[serde(rename = "DO-178C_Level_A")]
    DO178C_A,
}

/// Source location
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceLocation {
    pub file: IRPath,
    pub line: u32,
    pub column: u32,
    #[serde(default)]
    pub length: u32,
}

impl SourceLocation {
    /// Create a new source location
    pub fn new(file: IRPath, line: u32, column: u32) -> Self {
        Self {
            file,
            line,
            column,
            length: 0,
        }
    }
}
