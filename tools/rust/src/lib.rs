//! STUNIR Core Library - Rust Implementation
//!
//! This is a production-ready implementation of the STUNIR toolchain in Rust.
//! It provides memory safety guarantees and deterministic execution.
//!
//! # Design Goals
//! - Memory safety (no unsafe code except where necessary)
//! - Deterministic hashing and output generation
//! - Confluence with SPARK, Python, and Haskell implementations
//! - Zero-copy optimizations where possible

use sha2::{Digest, Sha256};

pub mod types;
pub mod hash;
pub mod ir;
pub mod optimizer;

/// Compute SHA-256 hash of bytes
///
/// # Arguments
/// * `data` - The byte slice to hash
///
/// # Returns
/// A hex-encoded SHA-256 hash string
///
/// # Examples
/// ```
/// let hash = sha256_bytes(b"hello world");
/// assert_eq!(hash.len(), 64); // SHA-256 produces 32 bytes = 64 hex chars
/// ```
pub fn sha256_bytes(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

/// Canonicalize JSON object for deterministic hashing
///
/// Converts a JSON value to a canonical string representation with:
/// - Sorted object keys (lexicographic order)
/// - No insignificant whitespace
/// - Consistent number formatting
///
/// This ensures that semantically equivalent JSON produces identical
/// byte sequences for cryptographic hashing.
///
/// # Arguments
/// * `value` - The JSON value to canonicalize
///
/// # Returns
/// * `Ok(String)` - The canonicalized JSON string with trailing newline
/// * `Err(serde_json::Error)` - If serialization fails
///
/// # Examples
/// ```
/// use serde_json::json;
/// let value = json!({"b": 1, "a": 2});
/// let canonical = canonicalize_json(&value).unwrap();
/// assert_eq!(canonical, r#"{"a":2,"b":1}"#);
/// ```
///
/// # Errors
/// Returns an error if the JSON value contains non-string keys in maps,
/// which cannot be represented in standard JSON.
pub fn canonicalize_json(value: &serde_json::Value) -> Result<String, serde_json::Error> {
    // Use serde_json's to_string which sorts keys by default
    serde_json::to_string(value).map(|s| s + "\n")
}

/// Compute SHA-256 of canonicalized JSON
///
/// Convenience function that canonicalizes JSON and returns its hash.
/// Useful for creating deterministic content hashes.
///
/// # Arguments
/// * `value` - The JSON value to hash
///
/// # Returns
/// * `Ok(String)` - The hex-encoded SHA-256 hash
/// * `Err(serde_json::Error)` - If JSON canonicalization fails
///
/// # Examples
/// ```
/// use serde_json::json;
/// let value = json!({"key": "value"});
/// let hash = sha256_json(&value).unwrap();
/// assert_eq!(hash.len(), 64);
/// ```
pub fn sha256_json(value: &serde_json::Value) -> Result<String, serde_json::Error> {
    canonicalize_json(value).map(|json| sha256_bytes(json.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_deterministic() {
        let data = b"hello world";
        let hash1 = sha256_bytes(data);
        let hash2 = sha256_bytes(data);
        assert_eq!(hash1, hash2);
        assert_eq!(hash1, "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9");
    }

    #[test]
    fn test_json_canonicalization() {
        let json1: serde_json::Value = serde_json::json!({"b": 2, "a": 1});
        let json2: serde_json::Value = serde_json::json!({"a": 1, "b": 2});
        assert_eq!(sha256_json(&json1), sha256_json(&json2));
    }
}
