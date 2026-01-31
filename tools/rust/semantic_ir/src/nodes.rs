//! STUNIR Semantic IR Node Structures
//!
//! Base node types and type references.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::types::*;

/// Type kind discriminator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TypeKind {
    PrimitiveType,
    ArrayType,
    PointerType,
    StructType,
    FunctionType,
    TypeRef,
}

/// Type reference
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TypeReference {
    PrimitiveType {
        primitive: IRPrimitiveType,
    },
    TypeRef {
        name: IRName,
        #[serde(skip_serializing_if = "Option::is_none")]
        binding: Option<NodeID>,
    },
    // Simplified for now - can be extended
}

/// Base IR node structure
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IRNodeBase {
    pub node_id: NodeID,
    pub kind: IRNodeKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<SourceLocation>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "type")]
    pub type_info: Option<TypeReference>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub attributes: HashMap<String, serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hash: Option<IRHash>,
}

impl IRNodeBase {
    /// Create a new IR node base
    pub fn new(node_id: NodeID, kind: IRNodeKind) -> Self {
        Self {
            node_id,
            kind,
            location: None,
            type_info: None,
            attributes: HashMap::new(),
            hash: None,
        }
    }

    /// Check if node ID is valid
    pub fn is_valid_node_id(node_id: &str) -> bool {
        node_id.starts_with("n_") && node_id.len() > 2
    }

    /// Check if hash is valid
    pub fn is_valid_hash(hash: &str) -> bool {
        hash.starts_with("sha256:") && hash.len() == 71
    }
}
