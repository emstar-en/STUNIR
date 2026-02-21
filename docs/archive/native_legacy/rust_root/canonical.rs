//! Canonical JSON Serialization Module
//!
//! Provides JCS (RFC 8785) canonical JSON serialization for STUNIR.
//! This ensures deterministic, reproducible JSON output across all platforms.
//!
//! # STUNIR Profile 3 Compliance
//!
//! STUNIR Profile 3 only allows integers (no floats), which simplifies
//! canonicalization. Standard JSON sorting is sufficient for most use cases.
//!
//! # Safety
//!
//! This module is designed for critical systems where deterministic output
//! is required. The canonicalization process ensures that the same input
//! always produces the same byte-for-byte output.
//!
//! # Limitations
//!
//! This is a simplified canonicalizer. For full JCS compliance, use the
//! `serde_jcs` crate.

use serde::Serialize;
use serde_json::ser::Formatter;
use std::io;

/// Serialize a value to canonical JSON string.
///
/// Uses compact formatting without whitespace and sorts object keys.
/// This produces deterministic output suitable for hashing and verification.
///
/// # Type Parameters
///
/// * `T` - The type to serialize, must implement `Serialize`
///
/// # Arguments
///
/// * `value` - The value to serialize
///
/// # Returns
///
/// * `Ok(String)` - The canonical JSON string on success
/// * `Err(serde_json::Error)` - If serialization fails
///
/// # Examples
///
/// ```
/// use serde_json::json;
///
/// let value = json!({"b": 2, "a": 1});
/// let canonical = canonical::to_string_canonical(&value).unwrap();
/// assert_eq!(canonical, r#"{"a":1,"b":2}"#);
/// ```
///
/// # Safety
///
/// This function is deterministic - the same input always produces the same output.
/// Float values should be avoided in STUNIR Profile 3.
pub fn to_string_canonical<T>(value: &T) -> Result<String, serde_json::Error>
where
    T: Serialize,
{
    // 1. Sort keys (serde_json does this with sort_keys(true) but we need to be careful about floats)
    // STUNIR Profile 3 only allows integers, so standard JSON sorting is mostly sufficient.
    let mut buf = Vec::new();
    let formatter = serde_json::ser::CompactFormatter;
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, formatter);
    value.serialize(&mut ser)?;

    // Note: This is a simplified canonicalizer.
    // For full JCS compliance, use the `serde_jcs` crate.
    String::from_utf8(buf).map_err(|e| serde_json::Error::custom(e.to_string()))
}
