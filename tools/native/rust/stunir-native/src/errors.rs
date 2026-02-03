//! # STUNIR Error Types
//!
//! This module defines the error types used throughout the STUNIR toolchain.
//! All errors implement the standard `Error` trait and provide meaningful
//! error messages for debugging and user feedback.
//!
//! ## Error Categories
//!
//! | Error Type | Description | Example |
//! |------------|-------------|---------|
//! | [`StunirError::Io`] | File system operations | "File not found" |
//! | [`StunirError::Json`] | JSON parsing/serialization | "Invalid JSON syntax" |
//! | [`StunirError::Validation`] | Input validation failures | "Invalid spec format" |
//! | [`StunirError::VerifyFailed`] | Verification mismatches | "Hash mismatch" |
//! | [`StunirError::Usage`] | CLI usage errors | "Missing argument" |
//! | [`StunirError::Security`] | Security violations | "Path traversal detected" |
//! | [`StunirError::Config`] | Configuration errors | "Invalid config value" |
//!
//! ## Usage Examples
//!
//! ### Creating Errors
//!
//! ```rust
//! use stunir_native::errors::StunirError;
//!
//! fn validate_input(input: &str) -> Result<(), StunirError> {
//!     if input.is_empty() {
//!         return Err(StunirError::Validation(
//!             "Input cannot be empty".to_string()
//!         ));
//!     }
//!     Ok(())
//! }
//! ```
//!
//! ### Error Conversion
//!
//! ```rust,ignore
//! use stunir_native::errors::StunirError;
//!
//! // IO errors convert automatically
//! let content = std::fs::read_to_string("file.txt")
//!     .map_err(|e| StunirError::Io(e.to_string()))?;
//! ```
//!
//! ## Error Codes
//!
//! Each error type corresponds to an error code range:
//!
//! - `E1xxx`: IO errors
//! - `E2xxx`: JSON errors  
//! - `E3xxx`: Validation errors
//! - `E4xxx`: Verification errors
//! - `E5xxx`: Usage errors
//! - `E6xxx`: Security errors
//! - `E7xxx`: Configuration errors

use thiserror::Error;

/// Main error type for STUNIR operations.
///
/// This enum captures all possible error conditions that can occur
/// during STUNIR toolchain operations. Each variant includes a
/// descriptive message for debugging.
///
/// # Examples
///
/// ```rust
/// use stunir_native::errors::StunirError;
///
/// let err = StunirError::Validation("Invalid module name".to_string());
/// assert!(err.to_string().contains("Validation"));
/// ```
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum StunirError {
    /// I/O operation failed (file read/write, directory operations).
    ///
    /// **Error code range**: E1xxx
    ///
    /// # Common Causes
    /// - File not found
    /// - Permission denied
    /// - Disk full
    /// - Network file system errors
    #[error("[E1000] IO Error: {0}")]
    Io(String),

    /// JSON parsing or serialization failed.
    ///
    /// **Error code range**: E2xxx
    ///
    /// # Common Causes
    /// - Malformed JSON syntax
    /// - Unexpected data types
    /// - Missing required fields
    /// - Encoding issues (non-UTF8)
    #[error("[E2000] JSON Error: {0}")]
    Json(String),

    /// Input validation failed.
    ///
    /// **Error code range**: E3xxx
    ///
    /// # Common Causes
    /// - Invalid spec format
    /// - Missing required fields
    /// - Value out of range
    /// - Invalid identifier syntax
    #[error("[E3000] Validation Error: {0}")]
    Validation(String),

    /// Verification check failed.
    ///
    /// **Error code range**: E4xxx
    ///
    /// # Common Causes
    /// - Hash mismatch
    /// - Signature verification failure
    /// - Manifest entry not found
    /// - Receipt tampering detected
    #[error("[E4000] Verification Failed: {0}")]
    VerifyFailed(String),

    /// CLI usage error.
    ///
    /// **Error code range**: E5xxx
    ///
    /// # Common Causes
    /// - Missing required argument
    /// - Invalid argument value
    /// - Conflicting options
    /// - Unknown subcommand
    #[error("[E5000] Usage Error: {0}")]
    Usage(String),

    /// Security violation detected.
    ///
    /// **Error code range**: E6xxx
    ///
    /// # Common Causes
    /// - Path traversal attempt (../)
    /// - Symlink following blocked
    /// - File size limit exceeded
    /// - Unauthorized operation
    #[error("[E6000] Security Error: {0}")]
    Security(String),

    /// Configuration error.
    ///
    /// **Error code range**: E7xxx
    ///
    /// # Common Causes
    /// - Invalid config file
    /// - Missing config section
    /// - Unsupported config version
    /// - Environment variable not set
    #[error("[E7000] Configuration Error: {0}")]
    Config(String),
}

impl StunirError {
    /// Get the error code for this error.
    ///
    /// Returns a string like "E1000" for categorization.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stunir_native::errors::StunirError;
    ///
    /// let err = StunirError::Io("test".to_string());
    /// assert_eq!(err.code(), "E1000");
    /// ```
    pub fn code(&self) -> &'static str {
        match self {
            StunirError::Io(_) => "E1000",
            StunirError::Json(_) => "E2000",
            StunirError::Validation(_) => "E3000",
            StunirError::VerifyFailed(_) => "E4000",
            StunirError::Usage(_) => "E5000",
            StunirError::Security(_) => "E6000",
            StunirError::Config(_) => "E7000",
        }
    }

    /// Check if this is a user-recoverable error.
    ///
    /// Returns `true` if the user can fix the error by
    /// correcting their input or environment.
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            StunirError::Usage(_) | StunirError::Validation(_) | StunirError::Config(_)
        )
    }

    /// Get a suggestion for how to fix this error.
    ///
    /// Returns a human-readable hint for resolving the error.
    pub fn suggestion(&self) -> &'static str {
        match self {
            StunirError::Io(_) => "Check file path and permissions",
            StunirError::Json(_) => "Validate JSON syntax with a linter",
            StunirError::Validation(_) => "Review input format requirements",
            StunirError::VerifyFailed(_) => "Re-run build and check for modifications",
            StunirError::Usage(_) => "Run with --help for usage information",
            StunirError::Security(_) => "Check for symlinks or path traversal",
            StunirError::Config(_) => "Review configuration file format",
        }
    }
}

/// Create an IO error with context.
///
/// # Examples
///
/// ```rust
/// use stunir_native::errors::io_error;
///
/// let err = io_error("config.json", "file not found");
/// assert!(err.to_string().contains("config.json"));
/// ```
pub fn io_error(path: &str, detail: &str) -> StunirError {
    StunirError::Io(format!("{}: {}", path, detail))
}

/// Create a validation error with field context.
///
/// # Examples
///
/// ```rust
/// use stunir_native::errors::validation_error;
///
/// let err = validation_error("module.name", "must be non-empty");
/// assert!(err.to_string().contains("module.name"));
/// ```
pub fn validation_error(field: &str, detail: &str) -> StunirError {
    StunirError::Validation(format!("'{}': {}", field, detail))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes() {
        assert_eq!(StunirError::Io("test".into()).code(), "E1000");
        assert_eq!(StunirError::Json("test".into()).code(), "E2000");
        assert_eq!(StunirError::Validation("test".into()).code(), "E3000");
    }

    #[test]
    fn test_is_recoverable() {
        assert!(StunirError::Usage("test".into()).is_recoverable());
        assert!(!StunirError::Io("test".into()).is_recoverable());
    }

    #[test]
    fn test_helper_functions() {
        let err = io_error("file.txt", "not found");
        assert!(err.to_string().contains("file.txt"));

        let err = validation_error("name", "required");
        assert!(err.to_string().contains("name"));
    }
}
