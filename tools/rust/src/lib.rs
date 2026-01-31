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

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub mod types;
pub mod hash;
pub mod ir;

/// Compute SHA-256 hash of bytes
pub fn sha256_bytes(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

/// Canonicalize JSON object for deterministic hashing
pub fn canonicalize_json(value: &serde_json::Value) -> String {
    // Use serde_json's to_string which sorts keys by default
    serde_json::to_string(value).expect("JSON canonicalization failed") + "\n"
}

/// Compute SHA-256 of canonicalized JSON
pub fn sha256_json(value: &serde_json::Value) -> String {
    sha256_bytes(canonicalize_json(value).as_bytes())
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
