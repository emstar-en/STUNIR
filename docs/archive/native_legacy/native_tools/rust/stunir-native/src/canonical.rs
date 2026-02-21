//! # JSON Canonicalization Module
//!
//! This module provides JSON Canonicalization Scheme (JCS) compliant normalization
//! as specified in RFC 8785. Canonicalization ensures that semantically equivalent
//! JSON documents produce identical byte sequences.
//!
//! ## Purpose
//!
//! Canonical JSON is essential for:
//! - Deterministic hashing of JSON documents
//! - Verifiable signatures over JSON data
//! - Reproducible build outputs
//! - Cross-platform consistency
//!
//! ## Specification Compliance
//!
//! This implementation follows RFC 8785 with the following behaviors:
//!
//! | Aspect | Behavior |
//! |--------|----------|
//! | Key ordering | Lexicographic (Unicode code point order) |
//! | Whitespace | No insignificant whitespace |
//! | Numbers | Shortest representation, no leading zeros |
//! | Strings | UTF-8 with minimal escaping |
//!
//! ## Examples
//!
//! ### Basic Normalization
//!
//! ```rust
//! use stunir_native::canonical::normalize;
//!
//! // Keys are sorted alphabetically
//! let input = r#"{"zebra": 1, "alpha": 2}"#;
//! let canonical = normalize(input).unwrap();
//! assert_eq!(canonical, r#"{"alpha":2,"zebra":1}"#);
//! ```
//!
//! ### Whitespace Removal
//!
//! ```rust
//! use stunir_native::canonical::normalize;
//!
//! let input = r#"{
//!     "key": "value",
//!     "number": 42
//! }"#;
//! let canonical = normalize(input).unwrap();
//! assert_eq!(canonical, r#"{"key":"value","number":42}"#);
//! ```
//!
//! ### Nested Objects
//!
//! ```rust
//! use stunir_native::canonical::normalize;
//!
//! let input = r#"{"outer": {"inner_z": 1, "inner_a": 2}}"#;
//! let canonical = normalize(input).unwrap();
//! assert_eq!(canonical, r#"{"outer":{"inner_a":2,"inner_z":1}}"#);
//! ```
//!
//! ## Security Notes
//!
//! - Input is validated as proper JSON before processing
//! - No external resources are accessed
//! - Memory-safe parsing via serde_json
//!
//! ## Limitations
//!
//! This is a "v0" implementation that handles most common cases but may not
//! perfectly handle all RFC 8785 edge cases (e.g., extreme floating-point values).
//! For cryptographic applications, consider additional validation.

use anyhow::{Context, Result};
use serde_json::Value;

/// Normalize a JSON string to canonical form.
///
/// Parses the input JSON and re-serializes it with sorted keys and no
/// insignificant whitespace, producing a deterministic byte sequence.
///
/// # Arguments
///
/// * `json_str` - A valid JSON string to canonicalize
///
/// # Returns
///
/// * `Ok(String)` - The canonicalized JSON string
/// * `Err(anyhow::Error)` - If parsing or serialization fails
///
/// # Examples
///
/// ```rust
/// use stunir_native::canonical::normalize;
///
/// let result = normalize(r#"{"b": 1, "a": 2}"#).unwrap();
/// assert_eq!(result, r#"{"a":2,"b":1}"#);
/// ```
///
/// # Errors
///
/// Returns an error if:
/// - The input is not valid JSON
/// - The JSON contains invalid UTF-8 sequences
pub fn normalize(json_str: &str) -> Result<String> {
    // Parse as generic Value - this validates the JSON
    let v: Value = serde_json::from_str(json_str)
        .with_context(|| "Failed to parse JSON for canonicalization")?;

    // Serialize with compact format (no pretty printing)
    // serde_json uses BTreeMap internally when keys need sorting
    let normalized = serde_json::to_string(&v)
        .with_context(|| "Failed to serialize canonical JSON")?;

    Ok(normalized)
}

/// Normalize JSON and compute its SHA-256 hash.
///
/// Convenience function that canonicalizes JSON and returns its hash.
/// Useful for creating deterministic content hashes.
///
/// # Arguments
///
/// * `json_str` - A valid JSON string
///
/// # Returns
///
/// * `Ok(String)` - The hex-encoded SHA-256 hash of the canonical JSON
/// * `Err(anyhow::Error)` - If parsing, serialization, or hashing fails
///
/// # Examples
///
/// ```rust,ignore
/// use stunir_native::canonical::normalize_and_hash;
///
/// let hash = normalize_and_hash(r#"{"key": "value"}"#)?;
/// println!("Content hash: {}", hash);
/// ```
pub fn normalize_and_hash(json_str: &str) -> Result<String> {
    use sha2::{Sha256, Digest};
    
    let canonical = normalize(json_str)?;
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    Ok(hex::encode(hasher.finalize()))
}

/// Check if two JSON strings are semantically equivalent.
///
/// Compares the canonical forms of two JSON strings to determine
/// if they represent the same data, ignoring formatting differences.
///
/// # Arguments
///
/// * `json_a` - First JSON string
/// * `json_b` - Second JSON string
///
/// # Returns
///
/// * `Ok(true)` - If both JSON strings are semantically equivalent
/// * `Ok(false)` - If the JSON strings differ
/// * `Err(anyhow::Error)` - If either string is invalid JSON
///
/// # Examples
///
/// ```rust
/// use stunir_native::canonical::json_equal;
///
/// let a = r#"{"x": 1, "y": 2}"#;
/// let b = r#"{"y":2,"x":1}"#;
/// assert!(json_equal(a, b).unwrap());
/// ```
pub fn json_equal(json_a: &str, json_b: &str) -> Result<bool> {
    let norm_a = normalize(json_a)?;
    let norm_b = normalize(json_b)?;
    Ok(norm_a == norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_sorting() {
        let input = r#"{"z": 1, "a": 2, "m": 3}"#;
        let result = normalize(input).unwrap();
        assert_eq!(result, r#"{"a":2,"m":3,"z":1}"#);
    }

    #[test]
    fn test_whitespace_removal() {
        let input = "{\n  \"key\": \"value\"\n}";
        let result = normalize(input).unwrap();
        assert_eq!(result, r#"{"key":"value"}"#);
    }

    #[test]
    fn test_nested_sorting() {
        let input = r#"{"outer": {"b": 1, "a": 2}}"#;
        let result = normalize(input).unwrap();
        assert_eq!(result, r#"{"outer":{"a":2,"b":1}}"#);
    }

    #[test]
    fn test_json_equal() {
        assert!(json_equal(r#"{"a":1}"#, r#"{ "a" : 1 }"#).unwrap());
        assert!(!json_equal(r#"{"a":1}"#, r#"{"a":2}"#).unwrap());
    }
}
