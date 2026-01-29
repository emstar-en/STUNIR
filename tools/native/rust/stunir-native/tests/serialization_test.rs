//! Tests for STUNIR Serialization utilities
//!
//! Tests JSON serialization for deterministic output.

mod common;

use serde::{Deserialize, Serialize};
use common::sha256_str;

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct TestStruct {
    name: String,
    value: i32,
    tags: Vec<String>,
}

#[test]
fn test_json_serialize_deterministic() {
    let obj = TestStruct {
        name: "test".to_string(),
        value: 42,
        tags: vec!["a".to_string(), "b".to_string()],
    };

    let json1 = serde_json::to_string(&obj).expect("First serialize");
    let json2 = serde_json::to_string(&obj).expect("Second serialize");

    assert_eq!(json1, json2, "Serialization should be deterministic");
}

#[test]
fn test_json_pretty_format() {
    let obj = TestStruct {
        name: "pretty".to_string(),
        value: 100,
        tags: vec![],
    };

    let pretty = serde_json::to_string_pretty(&obj).expect("Pretty serialize");

    assert!(pretty.contains("\n"), "Pretty JSON should have newlines");
    assert!(pretty.contains("  "), "Pretty JSON should have indentation");
}

#[test]
fn test_json_roundtrip() {
    let original = TestStruct {
        name: "roundtrip".to_string(),
        value: 999,
        tags: vec!["tag1".to_string(), "tag2".to_string()],
    };

    let json = serde_json::to_string(&original).expect("Serialize");
    let restored: TestStruct = serde_json::from_str(&json).expect("Deserialize");

    assert_eq!(original, restored, "Roundtrip should preserve data");
}

#[test]
fn test_json_hash_consistency() {
    let obj = TestStruct {
        name: "hash-test".to_string(),
        value: 1,
        tags: vec![],
    };

    let json = serde_json::to_string(&obj).expect("Serialize");
    let hash1 = sha256_str(&json);
    let hash2 = sha256_str(&json);

    assert_eq!(hash1, hash2, "Same JSON should produce same hash");
}

#[test]
fn test_serialize_special_characters() {
    let obj = TestStruct {
        name: "test\"with\\escapes\n".to_string(),
        value: 0,
        tags: vec!["tab\there".to_string()],
    };

    let json = serde_json::to_string(&obj).expect("Serialize with escapes");
    let restored: TestStruct = serde_json::from_str(&json).expect("Deserialize");

    assert_eq!(obj.name, restored.name);
}

#[test]
fn test_serialize_empty_strings() {
    let obj = TestStruct {
        name: "".to_string(),
        value: 0,
        tags: vec!["".to_string()],
    };

    let json = serde_json::to_string(&obj).expect("Serialize empty strings");
    let restored: TestStruct = serde_json::from_str(&json).expect("Deserialize");

    assert_eq!(obj, restored);
}
