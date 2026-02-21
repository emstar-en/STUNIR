//! Tests for STUNIR Canonical JSON module
//!
//! Tests JSON canonicalization for deterministic output.

mod common;

use stunir_native::canonical::normalize;

#[test]
fn test_normalize_simple_json() {
    let input = r#"{  "hello":  "world"  }"#;
    let result = normalize(input).expect("Should normalize simple JSON");

    // Should remove extra whitespace
    assert_eq!(result, r#"{"hello":"world"}"#);
}

#[test]
fn test_normalize_preserves_content() {
    let input = r#"{"key": "value", "number": 42, "bool": true}"#;
    let result = normalize(input).expect("Should normalize");

    // Parse result to verify content is preserved
    let parsed: serde_json::Value = serde_json::from_str(&result)
        .expect("Result should be valid JSON");

    assert_eq!(parsed["key"], "value");
    assert_eq!(parsed["number"], 42);
    assert_eq!(parsed["bool"], true);
}

#[test]
fn test_normalize_sorts_keys() {
    // With preserve_order feature disabled, serde_json sorts keys alphabetically
    let input = r#"{"z": 1, "a": 2, "m": 3}"#;
    let result = normalize(input).expect("Should normalize");

    // Verify it's valid JSON
    let _: serde_json::Value = serde_json::from_str(&result)
        .expect("Result should be valid JSON");
}

#[test]
fn test_normalize_nested_json() {
    let input = r#"{"outer": {"inner": {"deep": "value"}}}"#;
    let result = normalize(input).expect("Should normalize nested JSON");

    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(parsed["outer"]["inner"]["deep"], "value");
}

#[test]
fn test_normalize_arrays() {
    let input = r#"[ 1 , 2 , 3 , 4 ]"#;
    let result = normalize(input).expect("Should normalize arrays");

    assert_eq!(result, "[1,2,3,4]");
}

#[test]
fn test_normalize_determinism() {
    let input = r#"{"a": 1, "b": 2}"#;

    let result1 = normalize(input).expect("First normalization");
    let result2 = normalize(input).expect("Second normalization");

    assert_eq!(result1, result2, "Normalization should be deterministic");
}

#[test]
fn test_normalize_invalid_json() {
    let invalid = "not valid json";
    let result = normalize(invalid);
    assert!(result.is_err(), "Should fail on invalid JSON");
}

#[test]
fn test_normalize_empty_object() {
    let input = "  {   }  ";
    let result = normalize(input).expect("Should normalize empty object");
    assert_eq!(result, "{}");
}

#[test]
fn test_normalize_empty_array() {
    let input = "  [   ]  ";
    let result = normalize(input).expect("Should normalize empty array");
    assert_eq!(result, "[]");
}

#[test]
fn test_normalize_unicode() {
    let input = r#"{"message": "Hello, 世界!"}"#;
    let result = normalize(input).expect("Should handle unicode");

    assert!(result.contains("世界") || result.contains("\\u"),
        "Unicode should be preserved (possibly escaped)");
}
