//! Basic Integration Tests for STUNIR
//!
//! Tests that combine multiple modules and verify end-to-end functionality.

mod common;

use common::{create_temp_dir, create_test_file, SAMPLE_SPEC_JSON, sha256_str};
use stunir_native::canonical::normalize;
use stunir_native::crypto::hash_file;
use stunir_native::ir_v1::{IrV1, Spec};

#[test]
fn test_spec_to_ir_pipeline() {
    // Parse a spec
    let spec: Spec = serde_json::from_str(SAMPLE_SPEC_JSON)
        .expect("Should parse spec");

    // Verify spec structure
    assert_eq!(spec.kind, "spec");
    assert!(!spec.modules.is_empty());

    // Create an IR from the spec (simulated)
    let ir = IrV1 {
        kind: "ir".to_string(),
        generator: "stunir-native-rust".to_string(),
        ir_version: "v1".to_string(),
        module_name: spec.modules[0].name.clone(),
        functions: vec![],
        modules: vec![],
        metadata: stunir_native::ir_v1::IrMetadata {
            original_spec_kind: spec.kind.clone(),
            source_modules: spec.modules.clone(),
        },
    };

    // Serialize and verify
    let ir_json = serde_json::to_string(&ir).expect("Should serialize IR");
    let parsed: IrV1 = serde_json::from_str(&ir_json).expect("Should parse IR JSON");

    assert_eq!(parsed.kind, "ir");
    assert_eq!(parsed.metadata.original_spec_kind, "spec");
}

#[test]
fn test_file_hash_to_manifest() {
    let temp = create_temp_dir();

    // Create test files
    let file1 = create_test_file(temp.path(), "module1.json", SAMPLE_SPEC_JSON);
    let file2 = create_test_file(temp.path(), "module2.json", r#"{"kind":"spec","modules":[]}"#);

    // Hash files
    let hash1 = hash_file(&file1).expect("Should hash file1");
    let hash2 = hash_file(&file2).expect("Should hash file2");

    // Create a manifest-like structure
    let manifest = serde_json::json!({
        "schema": "stunir.manifest.test.v1",
        "entries": [
            {"path": "module1.json", "hash": hash1},
            {"path": "module2.json", "hash": hash2}
        ]
    });

    let manifest_json = serde_json::to_string_pretty(&manifest).expect("Serialize manifest");

    // Verify manifest content
    assert!(manifest_json.contains(&hash1));
    assert!(manifest_json.contains(&hash2));
    assert!(manifest_json.contains("stunir.manifest.test.v1"));
}

#[test]
fn test_canonical_hash_determinism() {
    let input = r#"{  "z": 1,  "a": 2,   "m": 3  }"#;

    // Normalize multiple times
    let norm1 = normalize(input).expect("First normalize");
    let norm2 = normalize(input).expect("Second normalize");

    // Should be identical
    assert_eq!(norm1, norm2);

    // Hashes should match
    let hash1 = sha256_str(&norm1);
    let hash2 = sha256_str(&norm2);
    assert_eq!(hash1, hash2);
}

#[test]
fn test_end_to_end_workflow() {
    let temp = create_temp_dir();

    // Step 1: Write a spec file
    let spec_path = create_test_file(temp.path(), "spec.json", SAMPLE_SPEC_JSON);

    // Step 2: Read and parse
    let content = std::fs::read_to_string(&spec_path).expect("Read spec");
    let spec: Spec = serde_json::from_str(&content).expect("Parse spec");

    // Step 3: Transform to IR
    let ir = IrV1 {
        kind: "ir".to_string(),
        generator: "stunir-native-rust".to_string(),
        ir_version: "v1".to_string(),
        module_name: "main".to_string(),
        functions: spec.modules.iter().map(|m| {
            stunir_native::ir_v1::IrFunction {
                name: m.name.clone(),
                body: vec![stunir_native::ir_v1::IrInstruction {
                    op: "raw".to_string(),
                    args: vec![m.source.clone()],
                }],
            }
        }).collect(),
        modules: vec![],
        metadata: stunir_native::ir_v1::IrMetadata {
            original_spec_kind: spec.kind.clone(),
            source_modules: spec.modules.clone(),
        },
    };

    // Step 4: Serialize IR
    let ir_json = serde_json::to_string_pretty(&ir).expect("Serialize IR");
    let ir_path = create_test_file(temp.path(), "output.ir.json", &ir_json);

    // Step 5: Hash the output
    let ir_hash = hash_file(&ir_path).expect("Hash IR");

    // Verify the workflow completed
    assert!(!ir_json.is_empty());
    assert_eq!(ir_hash.len(), 64);
    assert!(ir_json.contains("stunir-native-rust"));
}
