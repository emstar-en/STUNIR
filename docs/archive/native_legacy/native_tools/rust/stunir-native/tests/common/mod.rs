//! STUNIR Test Utilities
//!
//! Common utilities for testing STUNIR components.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

/// Create a temporary directory for tests
pub fn create_temp_dir() -> TempDir {
    tempfile::tempdir().expect("Failed to create temp directory")
}

/// Create a test file with specified content
pub fn create_test_file(dir: &Path, name: &str, content: &str) -> PathBuf {
    let path = dir.join(name);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("Failed to create parent dirs");
    }
    let mut file = File::create(&path).expect("Failed to create test file");
    file.write_all(content.as_bytes()).expect("Failed to write content");
    path
}

/// Create a nested directory structure for testing
pub fn create_test_tree(base: &Path) -> Vec<PathBuf> {
    let mut created = Vec::new();

    // Create files at root
    created.push(create_test_file(base, "file1.txt", "content1"));
    created.push(create_test_file(base, "file2.txt", "content2"));

    // Create nested directory
    let sub = base.join("subdir");
    fs::create_dir_all(&sub).expect("Failed to create subdir");
    created.push(create_test_file(&sub, "nested.txt", "nested content"));

    created
}

/// Sample valid spec JSON
pub const SAMPLE_SPEC_JSON: &str = r#"{
    "kind": "spec",
    "modules": [
        {
            "name": "hello",
            "source": "print('Hello, World!')",
            "lang": "python"
        }
    ],
    "metadata": {}
}"#;

/// Sample valid IR JSON
pub const SAMPLE_IR_JSON: &str = r#"{
    "kind": "ir",
    "generator": "stunir-native-rust",
    "ir_version": "v1",
    "module_name": "main",
    "functions": [
        {
            "name": "main",
            "body": [
                {"op": "print", "args": ["Hello"]}
            ]
        }
    ],
    "modules": [],
    "metadata": {
        "original_spec_kind": "spec",
        "source_modules": []
    }
}"#;

/// Sample receipt JSON
pub const SAMPLE_RECEIPT_JSON: &str = r#"{
    "id": "test-receipt-001",
    "tools": [
        {"name": "stunir-native", "version": "0.1.0"}
    ]
}"#;

/// Assert that a file exists and has expected content
pub fn assert_file_contains(path: &Path, expected: &str) {
    let content = fs::read_to_string(path).expect("Failed to read file");
    assert!(
        content.contains(expected),
        "File {:?} does not contain expected content: {}",
        path,
        expected
    );
}

/// Assert that JSON is valid
pub fn assert_valid_json(content: &str) -> serde_json::Value {
    serde_json::from_str(content).expect("Invalid JSON")
}

/// Compute SHA-256 hash of a string
pub fn sha256_str(s: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(s.as_bytes());
    hex::encode(hasher.finalize())
}
