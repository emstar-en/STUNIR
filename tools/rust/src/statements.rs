//! STUNIR Semantic IR Statement Nodes
//!
//! Statement node definitions.

use serde::{Deserialize, Serialize};

use crate::types::*;
use crate::nodes::*;

/// Statement node
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatementNode {
    #[serde(flatten)]
    pub base: IRNodeBase,
}

/// Block statement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlockStmt {
    #[serde(flatten)]
    pub base: StatementNode,
    pub statements: Vec<NodeID>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope_id: Option<IRName>,
}

/// Expression statement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExprStmt {
    #[serde(flatten)]
    pub base: StatementNode,
    pub expression: NodeID,
}

/// If statement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IfStmt {
    #[serde(flatten)]
    pub base: StatementNode,
    pub condition: NodeID,
    pub then_branch: NodeID,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub else_branch: Option<NodeID>,
}

/// While statement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WhileStmt {
    #[serde(flatten)]
    pub base: StatementNode,
    pub condition: NodeID,
    pub body: NodeID,
    #[serde(default)]
    pub loop_bound: u32,
    #[serde(default)]
    pub unroll: bool,
}

/// For statement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForStmt {
    #[serde(flatten)]
    pub base: StatementNode,
    pub init: NodeID,
    pub condition: NodeID,
    pub increment: NodeID,
    pub body: NodeID,
    #[serde(default)]
    pub loop_bound: u32,
    #[serde(default)]
    pub unroll: bool,
    #[serde(default)]
    pub vectorize: bool,
}

/// Return statement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReturnStmt {
    #[serde(flatten)]
    pub base: StatementNode,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<NodeID>,
}

/// Break statement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BreakStmt {
    #[serde(flatten)]
    pub base: StatementNode,
}

/// Continue statement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContinueStmt {
    #[serde(flatten)]
    pub base: StatementNode,
}

/// Variable declaration statement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VarDeclStmt {
    #[serde(flatten)]
    pub base: StatementNode,
    pub name: IRName,
    #[serde(skip_serializing_if = "Option::is_none", rename = "var_type")]
    pub var_type: Option<TypeReference>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initializer: Option<NodeID>,
    #[serde(default)]
    pub storage: StorageClass,
    #[serde(default)]
    pub mutability: MutabilityKind,
}

impl Default for StorageClass {
    fn default() -> Self { StorageClass::Auto }
}

impl Default for MutabilityKind {
    fn default() -> Self { MutabilityKind::Mutable }
}

/// Assignment statement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssignStmt {
    #[serde(flatten)]
    pub base: StatementNode,
    pub target: NodeID,
    pub value: NodeID,
}
