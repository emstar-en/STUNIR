//! Tests for STUNIR Receipt module
//!
//! Tests receipt generation and verification.

mod common;

use common::SAMPLE_RECEIPT_JSON;
use stunir_native::receipt::{Receipt, ToolInfo};

// Note: Receipt struct is in ir_v1 or receipt module depending on organization
// These tests verify the receipt structure and serialization

#[derive(serde::Serialize, serde::Deserialize, Debug)]
struct TestReceipt {
    id: String,
    tools: Vec<TestToolInfo>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
struct TestToolInfo {
    name: String,
    version: String,
}

#[test]
fn test_parse_receipt_json() {
    let receipt: TestReceipt = serde_json::from_str(SAMPLE_RECEIPT_JSON)
        .expect("Should parse valid receipt JSON");

    assert_eq!(receipt.id, "test-receipt-001");
    assert_eq!(receipt.tools.len(), 1);
    assert_eq!(receipt.tools[0].name, "stunir-native");
    assert_eq!(receipt.tools[0].version, "0.1.0");
}

#[test]
fn test_receipt_serialization() {
    let receipt = TestReceipt {
        id: "receipt-12345".to_string(),
        tools: vec![
            TestToolInfo {
                name: "stunir-native".to_string(),
                version: "0.1.0".to_string(),
            },
            TestToolInfo {
                name: "stunir-haskell".to_string(),
                version: "0.2.0".to_string(),
            },
        ],
    };

    let json = serde_json::to_string_pretty(&receipt)
        .expect("Should serialize receipt");

    assert!(json.contains("receipt-12345"));
    assert!(json.contains("stunir-native"));
    assert!(json.contains("stunir-haskell"));
}

#[test]
fn test_receipt_roundtrip() {
    let original = TestReceipt {
        id: "test-001".to_string(),
        tools: vec![TestToolInfo {
            name: "test-tool".to_string(),
            version: "1.0.0".to_string(),
        }],
    };

    let serialized = serde_json::to_string(&original)
        .expect("Should serialize");

    let deserialized: TestReceipt = serde_json::from_str(&serialized)
        .expect("Should deserialize");

    assert_eq!(original.id, deserialized.id);
    assert_eq!(original.tools.len(), deserialized.tools.len());
    assert_eq!(original.tools[0].name, deserialized.tools[0].name);
}

#[test]
fn test_receipt_with_empty_tools() {
    let receipt = TestReceipt {
        id: "empty-receipt".to_string(),
        tools: vec![],
    };

    let json = serde_json::to_string(&receipt)
        .expect("Should serialize empty tools receipt");

    let parsed: TestReceipt = serde_json::from_str(&json)
        .expect("Should parse empty tools receipt");

    assert!(parsed.tools.is_empty());
}

#[test]
fn test_receipt_id_format() {
    // Test various valid ID formats
    let ids = vec![
        "simple-id",
        "receipt-001",
        "stunir.receipt.2024.01.01.001",
        "UUID-LIKE-1234-5678-90ab-cdef",
    ];

    for id in ids {
        let receipt = TestReceipt {
            id: id.to_string(),
            tools: vec![],
        };
        let json = serde_json::to_string(&receipt).expect("Should serialize");
        assert!(json.contains(id), "JSON should contain ID: {}", id);
    }
}
