//! Hashing utilities for STUNIR

use crate::{sha256_bytes, sha256_json};
use serde_json::Value;

/// Compute deterministic hash for IR manifest
pub fn hash_ir_manifest(manifest: &Value) -> String {
    sha256_json(manifest)
}

/// Verify hash matches expected value
pub fn verify_hash(data: &[u8], expected: &str) -> bool {
    sha256_bytes(data) == expected
}
