//! STUNIR Canonical JSON Module
//!
//! Provides RFC 8785 / JCS subset canonicalization for deterministic JSON output.
//!
//! This module is part of the tools → native → rust pipeline stage.

use serde_json::{Value, Map};
use std::collections::BTreeMap;

/// Canonicalize a JSON value according to RFC 8785 / JCS subset.
/// 
/// Rules:
/// 1. Object keys are sorted lexicographically (Unicode code point order)
/// 2. No whitespace between tokens
/// 3. No trailing newline
pub fn canonicalize(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(b) => if *b { "true" } else { "false" }.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => format!("\"{}\"" , escape_string(s)),
        Value::Array(arr) => {
            let items: Vec<String> = arr.iter().map(|v| canonicalize(v)).collect();
            format!("[{}]", items.join(","))
        },
        Value::Object(obj) => {
            // Sort keys lexicographically
            let sorted: BTreeMap<&String, &Value> = obj.iter().collect();
            let items: Vec<String> = sorted.iter()
                .map(|(k, v)| format!("\"{}\":{}", escape_string(k), canonicalize(v)))
                .collect();
            format!("{{{}}}", items.join(","))
        }
    }
}

/// Escape a string for JSON output.
fn escape_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c < '\x20' => result.push_str(&format!("\\u{:04x}", c as u32)),
            c => result.push(c),
        }
    }
    result
}

/// Compute SHA-256 hash of canonical JSON.
pub fn canonical_hash(value: &Value) -> String {
    use sha2::{Sha256, Digest};
    let canonical = canonicalize(value);
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_canonicalize_object() {
        let value = json!({
            "z": 1,
            "a": 2,
            "m": 3
        });
        let canonical = canonicalize(&value);
        assert_eq!(canonical, r#"{"a":2,"m":3,"z":1}"#);
    }

    #[test]
    fn test_canonicalize_nested() {
        let value = json!({
            "b": {"y": 1, "x": 2},
            "a": [3, 2, 1]
        });
        let canonical = canonicalize(&value);
        assert_eq!(canonical, r#"{"a":[3,2,1],"b":{"x":2,"y":1}}"#);
    }
}
