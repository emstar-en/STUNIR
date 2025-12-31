use serde::Serialize;
use serde_json::{to_vec, Serializer, Value};
use sha2::{Sha256, Digest};
use std::fs;
use std::path::Path;
use std::io::Write;

pub fn to_string_canonical<T: Serialize>(value: &T) -> Result<String, serde_json::Error> {
    // For STUNIR, we want keys sorted. serde_json does this by default.
    // We also want no extra whitespace.
    serde_json::to_string(value)
}

pub fn write_json<T: Serialize>(path: &str, value: &T) -> std::io::Result<()> {
    if let Some(parent) = Path::new(path).parent() {
        fs::create_dir_all(parent)?;
    }
    let json = to_string_canonical(value)?;
    fs::write(path, json)
}

pub fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}
