//! STUNIR Base Emitter - Rust Implementation
//!
//! Base trait and common types for all semantic IR emitters.
//! Based on Ada SPARK emitter specifications.

use crate::types::{GeneratedFile, IRModule};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Emitter result status matching SPARK Emitter_Status enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmitterStatus {
    /// Success
    Success,
    /// Invalid IR error
    ErrorInvalidIR,
    /// Write failed error
    ErrorWriteFailed,
    /// Unsupported type error
    ErrorUnsupportedType,
    /// Buffer overflow error
    ErrorBufferOverflow,
    /// Invalid architecture error
    ErrorInvalidArchitecture,
}

/// Emitter error type
#[derive(Error, Debug)]
pub enum EmitterError {
    /// Invalid IR
    #[error("Invalid IR: {0}")]
    InvalidIR(String),
    /// Write failed
    #[error("Write failed: {0}")]
    WriteFailed(String),
    /// Unsupported type
    #[error("Unsupported type: {0}")]
    UnsupportedType(String),
    /// Buffer overflow
    #[error("Buffer overflow: {0}")]
    BufferOverflow(String),
    /// Invalid architecture
    #[error("Invalid architecture: {0}")]
    InvalidArchitecture(String),
    /// I/O error
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Emitter result matching SPARK Emitter_Result record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmitterResult {
    /// Result status
    pub status: EmitterStatus,
    /// Generated files
    pub files: Vec<GeneratedFile>,
    /// Total size in bytes
    pub total_size: usize,
    /// Error message (if any)
    pub error_message: Option<String>,
}

impl EmitterResult {
    /// Create success result
    pub fn success(files: Vec<GeneratedFile>, total_size: usize) -> Self {
        Self {
            status: EmitterStatus::Success,
            files,
            total_size,
            error_message: None,
        }
    }

    /// Create error result
    pub fn error(status: EmitterStatus, message: String) -> Self {
        Self {
            status,
            files: Vec::new(),
            total_size: 0,
            error_message: Some(message),
        }
    }

    /// Get number of generated files
    pub fn files_count(&self) -> usize {
        self.files.len()
    }
}

/// Base emitter configuration
#[derive(Debug, Clone)]
pub struct EmitterConfig {
    /// Output directory
    pub output_dir: PathBuf,
    /// Module name
    pub module_name: String,
    /// Add comments
    pub add_comments: bool,
    /// Add DO-178C headers
    pub add_do178c_headers: bool,
    /// Maximum line length
    pub max_line_length: usize,
    /// Indentation size
    pub indent_size: usize,
    /// Deterministic output
    pub deterministic: bool,
}

impl EmitterConfig {
    /// Create new emitter configuration
    pub fn new(output_dir: impl AsRef<Path>, module_name: impl Into<String>) -> Self {
        Self {
            output_dir: output_dir.as_ref().to_path_buf(),
            module_name: module_name.into(),
            add_comments: true,
            add_do178c_headers: true,
            max_line_length: 100,
            indent_size: 4,
            deterministic: true,
        }
    }
}

/// Base emitter trait
///
/// All emitters must implement this trait.
/// This ensures consistent behavior across all 24 emitter categories.
pub trait BaseEmitter {
    /// Emit code from IR module
    ///
    /// This method must be implemented by all concrete emitters.
    fn emit(&self, ir_module: &IRModule) -> Result<EmitterResult, EmitterError>;

    /// Validate IR module structure
    fn validate_ir(&self, ir_module: &IRModule) -> bool {
        !ir_module.ir_version.is_empty()
            && !ir_module.module_name.is_empty()
    }

    /// Compute SHA-256 hash of file content
    fn compute_file_hash(&self, content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// Write content to file and return generated file record
    fn write_file(
        &self,
        output_dir: &Path,
        relative_path: &str,
        content: &str,
    ) -> Result<GeneratedFile, EmitterError> {
        let file_path = output_dir.join(relative_path);
        
        // Create parent directories
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| EmitterError::WriteFailed(e.to_string()))?;
        }
        
        // Write file
        fs::write(&file_path, content)
            .map_err(|e| EmitterError::WriteFailed(e.to_string()))?;
        
        // Compute hash and size
        let hash = self.compute_file_hash(content);
        let size = content.as_bytes().len();
        
        Ok(GeneratedFile {
            path: relative_path.to_string(),
            hash,
            size,
        })
    }

    /// Generate DO-178C compliant header comment
    fn get_do178c_header(&self, config: &EmitterConfig, description: &str) -> String {
        if !config.add_do178c_headers {
            return String::new();
        }

        format!(
            "/*\n * STUNIR Generated Code\n * DO-178C Level A Compliant\n * {}\n * \n * This file was generated by STUNIR Semantic IR Emitter\n * Based on formally verified Ada SPARK implementation\n * \n * WARNING: Do not modify this file manually.\n * All changes must be made to the source IR.\n */\n\n",
            description
        )
    }

    /// Generate indentation string
    fn indent(&self, config: &EmitterConfig, level: usize) -> String {
        " ".repeat(config.indent_size * level)
    }
}
