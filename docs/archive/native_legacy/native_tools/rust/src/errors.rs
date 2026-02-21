//! STUNIR Error Types
//!
//! Defines error types used throughout the STUNIR Rust native tools.
//!
//! This module is part of the tools → native → rust pipeline stage.

use std::fmt;
use std::io;

/// Main error type for STUNIR operations.
#[derive(Debug)]
pub enum StunirError {
    /// IO error (file operations)
    Io(io::Error),
    /// JSON parsing error
    Json(serde_json::Error),
    /// Validation error
    Validation(String),
    /// Schema error
    Schema(String),
    /// Hash mismatch error
    HashMismatch {
        expected: String,
        actual: String,
    },
    /// Missing required field
    MissingField(String),
    /// Invalid input
    InvalidInput(String),
    /// General error
    General(String),
}

impl fmt::Display for StunirError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StunirError::Io(e) => write!(f, "IO error: {}", e),
            StunirError::Json(e) => write!(f, "JSON error: {}", e),
            StunirError::Validation(msg) => write!(f, "Validation error: {}", msg),
            StunirError::Schema(msg) => write!(f, "Schema error: {}", msg),
            StunirError::HashMismatch { expected, actual } => {
                write!(f, "Hash mismatch: expected {}, got {}", expected, actual)
            }
            StunirError::MissingField(field) => write!(f, "Missing required field: {}", field),
            StunirError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            StunirError::General(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl std::error::Error for StunirError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StunirError::Io(e) => Some(e),
            StunirError::Json(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for StunirError {
    fn from(err: io::Error) -> Self {
        StunirError::Io(err)
    }
}

impl From<serde_json::Error> for StunirError {
    fn from(err: serde_json::Error) -> Self {
        StunirError::Json(err)
    }
}

impl From<String> for StunirError {
    fn from(msg: String) -> Self {
        StunirError::General(msg)
    }
}

impl From<&str> for StunirError {
    fn from(msg: &str) -> Self {
        StunirError::General(msg.to_string())
    }
}

/// Result type for STUNIR operations.
pub type StunirResult<T> = Result<T, StunirError>;

/// Validation result with errors and warnings.
#[derive(Debug, Default)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl ValidationResult {
    pub fn new() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    pub fn add_error(&mut self, error: impl Into<String>) {
        self.errors.push(error.into());
        self.is_valid = false;
    }

    pub fn add_warning(&mut self, warning: impl Into<String>) {
        self.warnings.push(warning.into());
    }
}
