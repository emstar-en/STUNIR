//! STUNIR Semantic IR Expression Nodes
//!
//! Expression node definitions.

use serde::{Deserialize, Serialize};

use crate::types::*;
use crate::nodes::*;

/// Expression node (contains type information)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpressionNode {
    #[serde(flatten)]
    pub base: IRNodeBase,
    #[serde(rename = "type")]
    pub expr_type: TypeReference,
}

/// Integer literal
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntegerLiteral {
    #[serde(flatten)]
    pub base: ExpressionNode,
    pub value: i64,
    #[serde(default = "default_radix")]
    pub radix: u8,
}

fn default_radix() -> u8 { 10 }

/// Float literal
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FloatLiteral {
    #[serde(flatten)]
    pub base: ExpressionNode,
    pub value: f64,
}

/// String literal
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StringLiteral {
    #[serde(flatten)]
    pub base: ExpressionNode,
    pub value: String,
}

/// Boolean literal
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoolLiteral {
    #[serde(flatten)]
    pub base: ExpressionNode,
    pub value: bool,
}

/// Variable reference
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VarRef {
    #[serde(flatten)]
    pub base: ExpressionNode,
    pub name: IRName,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub binding: Option<NodeID>,
}

/// Binary expression
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryExpr {
    #[serde(flatten)]
    pub base: ExpressionNode,
    pub op: BinaryOperator,
    pub left: NodeID,
    pub right: NodeID,
}

/// Unary expression
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnaryExpr {
    #[serde(flatten)]
    pub base: ExpressionNode,
    pub op: UnaryOperator,
    pub operand: NodeID,
}

/// Function call
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionCall {
    #[serde(flatten)]
    pub base: ExpressionNode,
    pub function: IRName,
    pub arguments: Vec<NodeID>,
}

/// Member access
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemberExpr {
    #[serde(flatten)]
    pub base: ExpressionNode,
    pub object: NodeID,
    pub member: IRName,
    #[serde(default)]
    pub is_arrow: bool,
}

/// Array access
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArrayAccess {
    #[serde(flatten)]
    pub base: ExpressionNode,
    pub array: NodeID,
    pub index: NodeID,
}
