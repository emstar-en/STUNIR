//! # STUNIR Receipt Management
//!
//! This module provides data structures and utilities for managing build receipts.
//! Receipts are cryptographic proofs that link build inputs to outputs, enabling
//! verification of build reproducibility.
//!
//! ## Overview
//!
//! A receipt contains:
//! - Unique identifier
//! - List of tools used in the build
//! - Input/output hashes
//! - Timestamps and metadata
//!
//! ## Receipt Structure
//!
//! ```json
//! {
//!   "id": "receipt-2026-01-28-abc123",
//!   "tools": [
//!     {"name": "stunir-native", "version": "0.1.0"},
//!     {"name": "cargo", "version": "1.75.0"}
//!   ],
//!   "inputs": {...},
//!   "outputs": {...},
//!   "timestamp": "2026-01-28T12:00:00Z"
//! }
//! ```
//!
//! ## Usage Examples
//!
//! ### Creating a Receipt
//!
//! ```rust
//! use stunir_native::receipt::{Receipt, ToolInfo};
//!
//! let receipt = Receipt::new("build-001")
//!     .add_tool("stunir-native", "0.1.0")
//!     .add_tool("rustc", "1.75.0");
//! ```
//!
//! ### Verifying a Receipt
//!
//! ```rust,ignore
//! use stunir_native::receipt::Receipt;
//!
//! let receipt: Receipt = serde_json::from_str(receipt_json)?;
//! if receipt.verify()? {
//!     println!("Receipt verified successfully");
//! }
//! ```
//!
//! ## Security Considerations
//!
//! - Receipts should be signed in production environments
//! - Receipt IDs should be collision-resistant
//! - Tool versions must be exact (no ranges)
//! - Timestamps should use UTC

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Information about a tool used in the build process.
///
/// Each tool is identified by its name and exact version, enabling
/// reproducible builds by specifying the exact toolchain.
///
/// # Fields
///
/// - `name`: Tool name (e.g., "stunir-native", "cargo", "rustc")
/// - `version`: Exact version string (e.g., "0.1.0", "1.75.0")
///
/// # Examples
///
/// ```rust
/// use stunir_native::receipt::ToolInfo;
///
/// let tool = ToolInfo {
///     name: "stunir-native".to_string(),
///     version: "0.1.0".to_string(),
/// };
/// ```
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ToolInfo {
    /// Tool name/identifier.
    pub name: String,
    
    /// Exact version string.
    pub version: String,
}

impl ToolInfo {
    /// Create a new ToolInfo.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stunir_native::receipt::ToolInfo;
    ///
    /// let tool = ToolInfo::new("cargo", "1.75.0");
    /// assert_eq!(tool.name, "cargo");
    /// ```
    pub fn new(name: &str, version: &str) -> Self {
        ToolInfo {
            name: name.to_string(),
            version: version.to_string(),
        }
    }
}

/// Build receipt linking inputs to outputs.
///
/// A receipt is a cryptographic proof of a build operation. It contains
/// all the information needed to verify that a build was performed
/// correctly and reproducibly.
///
/// # Fields
///
/// - `id`: Unique receipt identifier
/// - `tools`: List of tools used in the build
/// - `inputs`: Hash map of input file hashes
/// - `outputs`: Hash map of output file hashes
/// - `timestamp`: ISO 8601 timestamp of receipt creation
/// - `metadata`: Additional key-value metadata
///
/// # Examples
///
/// ```rust
/// use stunir_native::receipt::Receipt;
///
/// let receipt = Receipt::new("build-001")
///     .add_tool("stunir-native", "0.1.0")
///     .with_timestamp("2026-01-28T12:00:00Z");
/// ```
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Receipt {
    /// Unique receipt identifier.
    pub id: String,
    
    /// List of tools used in the build.
    pub tools: Vec<ToolInfo>,
    
    /// Input file hashes (path -> SHA256 hash).
    #[serde(default)]
    pub inputs: HashMap<String, String>,
    
    /// Output file hashes (path -> SHA256 hash).
    #[serde(default)]
    pub outputs: HashMap<String, String>,
    
    /// ISO 8601 timestamp of receipt creation.
    #[serde(default)]
    pub timestamp: Option<String>,
    
    /// Additional metadata.
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl Receipt {
    /// Create a new receipt with the given ID.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for this receipt
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stunir_native::receipt::Receipt;
    ///
    /// let receipt = Receipt::new("build-001");
    /// assert_eq!(receipt.id, "build-001");
    /// ```
    pub fn new(id: &str) -> Self {
        Receipt {
            id: id.to_string(),
            tools: vec![],
            inputs: HashMap::new(),
            outputs: HashMap::new(),
            timestamp: None,
            metadata: HashMap::new(),
        }
    }

    /// Add a tool to the receipt (builder pattern).
    ///
    /// # Arguments
    ///
    /// * `name` - Tool name
    /// * `version` - Tool version
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stunir_native::receipt::Receipt;
    ///
    /// let receipt = Receipt::new("build-001")
    ///     .add_tool("cargo", "1.75.0")
    ///     .add_tool("rustc", "1.75.0");
    /// assert_eq!(receipt.tools.len(), 2);
    /// ```
    pub fn add_tool(mut self, name: &str, version: &str) -> Self {
        self.tools.push(ToolInfo::new(name, version));
        self
    }

    /// Set the timestamp (builder pattern).
    ///
    /// # Arguments
    ///
    /// * `timestamp` - ISO 8601 formatted timestamp
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stunir_native::receipt::Receipt;
    ///
    /// let receipt = Receipt::new("build-001")
    ///     .with_timestamp("2026-01-28T12:00:00Z");
    /// ```
    pub fn with_timestamp(mut self, timestamp: &str) -> Self {
        self.timestamp = Some(timestamp.to_string());
        self
    }

    /// Add an input file hash.
    ///
    /// # Arguments
    ///
    /// * `path` - File path (relative to project root)
    /// * `hash` - SHA256 hash of the file
    pub fn add_input(&mut self, path: &str, hash: &str) {
        self.inputs.insert(path.to_string(), hash.to_string());
    }

    /// Add an output file hash.
    ///
    /// # Arguments
    ///
    /// * `path` - File path (relative to project root)
    /// * `hash` - SHA256 hash of the file
    pub fn add_output(&mut self, path: &str, hash: &str) {
        self.outputs.insert(path.to_string(), hash.to_string());
    }

    /// Verify the receipt against actual files.
    ///
    /// Checks that all input and output files exist and match
    /// their recorded hashes.
    ///
    /// # Returns
    ///
    /// * `Ok(true)` - All hashes match
    /// * `Ok(false)` - One or more hashes don't match
    /// * `Err` - Error reading files
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use stunir_native::receipt::Receipt;
    ///
    /// let receipt: Receipt = serde_json::from_str(json)?;
    /// if receipt.verify()? {
    ///     println!("Build verified!");
    /// }
    /// ```
    pub fn verify(&self) -> anyhow::Result<bool> {
        use crate::crypto::hash_file;
        use std::path::Path;

        // Check all inputs
        for (path, expected_hash) in &self.inputs {
            let actual_hash = hash_file(Path::new(path))?;
            if actual_hash != *expected_hash {
                return Ok(false);
            }
        }

        // Check all outputs
        for (path, expected_hash) in &self.outputs {
            let actual_hash = hash_file(Path::new(path))?;
            if actual_hash != *expected_hash {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Serialize the receipt to canonical JSON.
    ///
    /// Uses deterministic JSON serialization for consistent hashing.
    pub fn to_canonical_json(&self) -> anyhow::Result<String> {
        let json = serde_json::to_string(self)?;
        crate::canonical::normalize(&json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_info() {
        let tool = ToolInfo::new("cargo", "1.75.0");
        assert_eq!(tool.name, "cargo");
        assert_eq!(tool.version, "1.75.0");
    }

    #[test]
    fn test_receipt_builder() {
        let receipt = Receipt::new("test-001")
            .add_tool("tool1", "1.0")
            .add_tool("tool2", "2.0")
            .with_timestamp("2026-01-28T00:00:00Z");

        assert_eq!(receipt.id, "test-001");
        assert_eq!(receipt.tools.len(), 2);
        assert!(receipt.timestamp.is_some());
    }

    #[test]
    fn test_receipt_serialization() {
        let receipt = Receipt::new("test")
            .add_tool("stunir", "0.1.0");

        let json = serde_json::to_string(&receipt).unwrap();
        let parsed: Receipt = serde_json::from_str(&json).unwrap();
        assert_eq!(receipt.id, parsed.id);
    }

    #[test]
    fn test_add_hashes() {
        let mut receipt = Receipt::new("test");
        receipt.add_input("input.txt", "abc123");
        receipt.add_output("output.txt", "def456");

        assert_eq!(receipt.inputs.get("input.txt"), Some(&"abc123".to_string()));
        assert_eq!(receipt.outputs.get("output.txt"), Some(&"def456".to_string()));
    }
}
