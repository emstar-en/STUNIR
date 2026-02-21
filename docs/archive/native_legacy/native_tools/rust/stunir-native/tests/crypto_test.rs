//! Tests for STUNIR cryptographic utilities
//!
//! Tests directory hashing, Merkle trees, and SHA-256 operations.

mod common;

use common::{create_temp_dir, create_test_file, create_test_tree, sha256_str};
use std::fs;
use stunir_native::crypto::{hash_file, hash_path};

#[test]
fn test_hash_file_basic() {
    let temp = create_temp_dir();
    let content = "Hello, STUNIR!";
    let path = create_test_file(temp.path(), "test.txt", content);

    let hash = hash_file(&path).expect("Hash should succeed");

    // SHA-256 produces 64 hex characters
    assert_eq!(hash.len(), 64, "SHA-256 hash should be 64 hex chars");

    // Verify determinism - same content = same hash
    let hash2 = hash_file(&path).expect("Second hash should succeed");
    assert_eq!(hash, hash2, "Hashes should be deterministic");

    // Verify expected hash
    let expected = sha256_str(content);
    assert_eq!(hash, expected, "Hash should match expected value");
}

#[test]
fn test_hash_file_empty() {
    let temp = create_temp_dir();
    let path = create_test_file(temp.path(), "empty.txt", "");

    let hash = hash_file(&path).expect("Hash of empty file should succeed");

    // SHA-256 of empty string
    let expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    assert_eq!(hash, expected, "Empty file hash should match known value");
}

#[test]
fn test_hash_path_file() {
    let temp = create_temp_dir();
    let path = create_test_file(temp.path(), "file.txt", "test content");

    let hash = hash_path(&path).expect("hash_path should work for files");
    assert_eq!(hash.len(), 64);
}

#[test]
fn test_hash_path_directory() {
    let temp = create_temp_dir();
    create_test_tree(temp.path());

    let hash = hash_path(temp.path()).expect("hash_path should work for directories");
    assert_eq!(hash.len(), 64, "Directory hash should be 64 hex chars");

    // Verify determinism
    let hash2 = hash_path(temp.path()).expect("Second hash should succeed");
    assert_eq!(hash, hash2, "Directory hash should be deterministic");
}

#[test]
fn test_hash_directory_order_independent() {
    // Create two directories with same files but different creation order
    let temp1 = create_temp_dir();
    create_test_file(temp1.path(), "a.txt", "A content");
    create_test_file(temp1.path(), "b.txt", "B content");

    let temp2 = create_temp_dir();
    create_test_file(temp2.path(), "b.txt", "B content");
    create_test_file(temp2.path(), "a.txt", "A content");

    let hash1 = hash_path(temp1.path()).expect("First hash should succeed");
    let hash2 = hash_path(temp2.path()).expect("Second hash should succeed");

    assert_eq!(hash1, hash2, "Same content should produce same hash regardless of order");
}

#[test]
fn test_hash_nonexistent_file() {
    let result = hash_file(std::path::Path::new("/nonexistent/file.txt"));
    assert!(result.is_err(), "Hash of nonexistent file should fail");
}

#[test]
fn test_hash_different_content() {
    let temp = create_temp_dir();
    let path1 = create_test_file(temp.path(), "file1.txt", "content1");
    let path2 = create_test_file(temp.path(), "file2.txt", "content2");

    let hash1 = hash_file(&path1).expect("First hash should succeed");
    let hash2 = hash_file(&path2).expect("Second hash should succeed");

    assert_ne!(hash1, hash2, "Different content should produce different hashes");
}
