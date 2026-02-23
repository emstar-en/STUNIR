//! STUNIR Semantic IR Module Structures
//!
//! Module and import definitions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::types::*;
use crate::nodes::*;

/// Import statement
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportStmt {
    #[serde(rename = "module")]
    pub module_name: IRName,
    #[serde(default)]
    pub symbols: ImportSymbols,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alias: Option<IRName>,
}

/// Import symbols (list or wildcard)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ImportSymbols {
    All(String), // "*"
    List(Vec<IRName>),
}

impl Default for ImportSymbols {
    fn default() -> Self { ImportSymbols::List(Vec::new()) }
}

/// Module metadata
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModuleMetadata {
    #[serde(default)]
    pub target_categories: Vec<TargetCategory>,
    #[serde(default)]
    pub safety_level: SafetyLevel,
    #[serde(default)]
    pub optimization_level: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_standard: Option<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub custom_attributes: HashMap<String, serde_json::Value>,
}

impl Default for ModuleMetadata {
    fn default() -> Self {
        Self {
            target_categories: Vec::new(),
            safety_level: SafetyLevel::None,
            optimization_level: 0,
            language_standard: None,
            custom_attributes: HashMap::new(),
        }
    }
}

impl Default for SafetyLevel {
    fn default() -> Self { SafetyLevel::None }
}

/// IR Module structure
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IRModule {
    #[serde(flatten)]
    pub base: IRNodeBase,
    pub name: IRName,
    #[serde(default)]
    pub imports: Vec<ImportStmt>,
    #[serde(default)]
    pub exports: Vec<IRName>,
    #[serde(default)]
    pub declarations: Vec<NodeID>,
    #[serde(default)]
    pub metadata: ModuleMetadata,
}

impl IRModule {
    /// Create a new module
    pub fn new(node_id: NodeID, name: IRName) -> Self {
        Self {
            base: IRNodeBase::new(node_id, IRNodeKind::Module),
            name,
            imports: Vec::new(),
            exports: Vec::new(),
            declarations: Vec::new(),
            metadata: ModuleMetadata::default(),
        }
    }

    /// Add an import
    pub fn add_import(&mut self, import: ImportStmt) {
        self.imports.push(import);
    }

    /// Add an export
    pub fn add_export(&mut self, name: IRName) {
        self.exports.push(name);
    }

    /// Add a declaration
    pub fn add_declaration(&mut self, decl_id: NodeID) {
        self.declarations.push(decl_id);
    }
}
