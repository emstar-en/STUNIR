//! STUNIR Semantic IR Declaration Nodes
//!
//! Declaration node definitions.

use serde::{Deserialize, Serialize};

use crate::types::*;
use crate::nodes::*;

/// Declaration node
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeclarationNode {
    #[serde(flatten)]
    pub base: IRNodeBase,
    pub name: IRName,
    #[serde(default)]
    pub visibility: VisibilityKind,
}

impl Default for VisibilityKind {
    fn default() -> Self { VisibilityKind::Public }
}

/// Function parameter
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Parameter {
    pub name: IRName,
    #[serde(rename = "type")]
    pub param_type: TypeReference,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<NodeID>,
}

/// Function declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionDecl {
    #[serde(flatten)]
    pub base: DeclarationNode,
    pub return_type: TypeReference,
    #[serde(default)]
    pub parameters: Vec<Parameter>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<NodeID>,
    #[serde(default)]
    pub inline: InlineHint,
    #[serde(default)]
    pub is_pure: bool,
    #[serde(default)]
    pub stack_usage: u32,
    #[serde(default)]
    pub priority: i32,
    #[serde(default)]
    pub interrupt_vector: u32,
    #[serde(default)]
    pub entry_point: bool,
}

impl Default for InlineHint {
    fn default() -> Self { InlineHint::None }
}

/// Type declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypeDecl {
    #[serde(flatten)]
    pub base: DeclarationNode,
    pub type_definition: TypeReference,
}

/// Constant declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstDecl {
    #[serde(flatten)]
    pub base: DeclarationNode,
    #[serde(skip_serializing_if = "Option::is_none", rename = "const_type")]
    pub const_type: Option<TypeReference>,
    pub value: NodeID,
    #[serde(default = "default_true")]
    pub compile_time: bool,
}

fn default_true() -> bool { true }

/// Variable declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VarDecl {
    #[serde(flatten)]
    pub base: DeclarationNode,
    #[serde(skip_serializing_if = "Option::is_none", rename = "var_type")]
    pub var_type: Option<TypeReference>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initializer: Option<NodeID>,
    #[serde(default)]
    pub storage: StorageClass,
    #[serde(default)]
    pub mutability: MutabilityKind,
    #[serde(default = "default_alignment")]
    pub alignment: u32,
    #[serde(default)]
    pub is_volatile: bool,
}

fn default_alignment() -> u32 { 1 }
