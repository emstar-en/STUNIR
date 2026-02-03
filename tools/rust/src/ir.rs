//! IR processing utilities

use crate::types::*;
use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::BTreeMap;

/// Parse IR from JSON
pub fn parse_ir(json: &Value) -> Result<IRModule> {
    serde_json::from_value(json.clone())
        .context("Failed to parse IR module")
}

/// Convert IR module to JSON
pub fn ir_to_json(module: &IRModule) -> Result<Value> {
    serde_json::to_value(module)
        .context("Failed to convert IR to JSON")
}
