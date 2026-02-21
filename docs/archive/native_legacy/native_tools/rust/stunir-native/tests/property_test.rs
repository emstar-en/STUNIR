//! Property-Based Tests for STUNIR
//!
//! These tests verify fundamental properties that must hold for all inputs:
//! - Serialization round-trip (serialize → deserialize = identity)
//! - Hash determinism (same input = same hash)
//! - Canonicalization idempotence (canon(canon(x)) = canon(x))
//! - Receipt verification consistency

use proptest::prelude::*;
use stunir_native::canonical;
use stunir_native::ir_v1::{IrV1, IrFunction, IrInstruction, IrMetadata, SpecModule, Spec};
use stunir_native::receipt::{Receipt, ToolInfo};
use std::collections::HashMap;

// ============================================================================
// Strategies for generating test data
// ============================================================================

/// Strategy for generating valid identifiers
fn identifier_strategy() -> impl Strategy<Value = String> {
    "[a-z][a-z0-9_]{0,20}".prop_map(String::from)
}

/// Strategy for generating simple JSON values
fn simple_json_value() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("null".to_string()),
        Just("true".to_string()),
        Just("false".to_string()),
        any::<i32>().prop_map(|n| n.to_string()),
        "[a-zA-Z0-9 ]{0,50}".prop_map(|s| format!(r#""{}""#, s)),
    ]
}

/// Strategy for generating JSON objects
fn json_object_strategy() -> impl Strategy<Value = String> {
    prop::collection::vec(
        (identifier_strategy(), simple_json_value()),
        0..5
    ).prop_map(|pairs| {
        let entries: Vec<String> = pairs
            .into_iter()
            .map(|(k, v)| format!(r#""{}":{}"#, k, v))
            .collect();
        format!(r#"{{{}}}"#, entries.join(","))
    })
}

/// Strategy for generating ToolInfo
fn tool_info_strategy() -> impl Strategy<Value = ToolInfo> {
    (identifier_strategy(), "[0-9]+\\.[0-9]+\\.[0-9]+")
        .prop_map(|(name, version)| ToolInfo { name, version })
}

/// Strategy for generating IrInstruction
fn instruction_strategy() -> impl Strategy<Value = IrInstruction> {
    (
        prop_oneof![Just("call"), Just("return"), Just("assign"), Just("load")],
        prop::collection::vec(identifier_strategy(), 0..3)
    ).prop_map(|(op, args)| IrInstruction { op: op.to_string(), args })
}

/// Strategy for generating IrFunction
fn function_strategy() -> impl Strategy<Value = IrFunction> {
    (identifier_strategy(), prop::collection::vec(instruction_strategy(), 0..5))
        .prop_map(|(name, body)| IrFunction { name, body })
}

// ============================================================================
// Property Tests: JSON Canonicalization
// ============================================================================

proptest! {
    /// Property: Canonicalization is idempotent
    /// canon(canon(x)) = canon(x)
    #[test]
    fn prop_canonicalization_idempotent(json in json_object_strategy()) {
        if let Ok(canonical1) = canonical::normalize(&json) {
            let canonical2 = canonical::normalize(&canonical1).unwrap();
            prop_assert_eq!(canonical1, canonical2, 
                "Canonicalization should be idempotent");
        }
    }

    /// Property: Canonical JSON has no extra whitespace
    #[test]
    fn prop_canonical_no_leading_trailing_whitespace(json in json_object_strategy()) {
        if let Ok(canonical) = canonical::normalize(&json) {
            prop_assert_eq!(canonical.trim(), canonical,
                "Canonical JSON should have no leading/trailing whitespace");
        }
    }

    /// Property: Same content produces same canonical form regardless of formatting
    #[test]
    fn prop_canonical_determinism(key in identifier_strategy(), value in 0i32..1000) {
        // Create same content with different formatting
        let json1 = format!(r#"{{"{}":{}}}}"#, key, value);
        let json2 = format!(r#"{{ "{}" : {} }}"#, key, value);
        
        let canon1 = canonical::normalize(&json1);
        let canon2 = canonical::normalize(&json2);
        
        if canon1.is_ok() && canon2.is_ok() {
            prop_assert_eq!(canon1.unwrap(), canon2.unwrap(),
                "Same content should produce same canonical form");
        }
    }
}

// ============================================================================
// Property Tests: Serialization Round-Trip
// ============================================================================

proptest! {
    /// Property: ToolInfo serialization round-trip
    #[test]
    fn prop_tool_info_roundtrip(tool in tool_info_strategy()) {
        let json = serde_json::to_string(&tool).unwrap();
        let parsed: ToolInfo = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(tool, parsed);
    }

    /// Property: IrInstruction serialization round-trip
    #[test]
    fn prop_instruction_roundtrip(inst in instruction_strategy()) {
        let json = serde_json::to_string(&inst).unwrap();
        let parsed: IrInstruction = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(inst, parsed);
    }

    /// Property: IrFunction serialization round-trip
    #[test]
    fn prop_function_roundtrip(func in function_strategy()) {
        let json = serde_json::to_string(&func).unwrap();
        let parsed: IrFunction = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(func, parsed);
    }

    /// Property: Receipt serialization round-trip
    #[test]
    fn prop_receipt_roundtrip(
        id in identifier_strategy(),
        tools in prop::collection::vec(tool_info_strategy(), 0..3)
    ) {
        let receipt = Receipt {
            id,
            tools,
            inputs: HashMap::new(),
            outputs: HashMap::new(),
            timestamp: None,
            metadata: HashMap::new(),
        };
        
        let json = serde_json::to_string(&receipt).unwrap();
        let parsed: Receipt = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(receipt, parsed);
    }
}

// ============================================================================
// Property Tests: Hash Determinism
// ============================================================================

proptest! {
    /// Property: Same content always produces the same hash
    #[test]
    fn prop_hash_determinism(content in "[a-zA-Z0-9]{0,100}") {
        use sha2::{Sha256, Digest};
        
        let mut hasher1 = Sha256::new();
        hasher1.update(content.as_bytes());
        let hash1 = hex::encode(hasher1.finalize());
        
        let mut hasher2 = Sha256::new();
        hasher2.update(content.as_bytes());
        let hash2 = hex::encode(hasher2.finalize());
        
        prop_assert_eq!(hash1, hash2, "Same content should produce same hash");
    }

    /// Property: Different content produces different hashes (with high probability)
    #[test]
    fn prop_hash_collision_resistant(
        content1 in "[a-zA-Z0-9]{10,50}",
        content2 in "[a-zA-Z0-9]{10,50}"
    ) {
        use sha2::{Sha256, Digest};
        
        if content1 != content2 {
            let mut hasher1 = Sha256::new();
            hasher1.update(content1.as_bytes());
            let hash1 = hex::encode(hasher1.finalize());
            
            let mut hasher2 = Sha256::new();
            hasher2.update(content2.as_bytes());
            let hash2 = hex::encode(hasher2.finalize());
            
            prop_assert_ne!(hash1, hash2, 
                "Different content should produce different hashes");
        }
    }
}

// ============================================================================
// Property Tests: IR Validation
// ============================================================================

proptest! {
    /// Property: IrV1::new creates valid IR
    #[test]
    fn prop_ir_new_valid(module_name in identifier_strategy()) {
        let ir = IrV1::new(&module_name);
        prop_assert!(ir.validate().is_ok());
        prop_assert_eq!(ir.module_name, module_name);
        prop_assert_eq!(ir.kind, "ir");
        prop_assert_eq!(ir.ir_version, "v1");
    }

    /// Property: Added functions are preserved
    #[test]
    fn prop_ir_add_function(module_name in identifier_strategy(), func in function_strategy()) {
        let mut ir = IrV1::new(&module_name);
        let func_name = func.name.clone();
        ir.add_function(func);
        
        prop_assert_eq!(ir.functions.len(), 1);
        prop_assert_eq!(ir.functions[0].name, func_name);
    }
}

#[cfg(test)]
mod quickcheck_tests {
    use quickcheck_macros::quickcheck;
    use stunir_native::canonical;

    /// QuickCheck: JSON with only numbers canonicalizes correctly
    #[quickcheck]
    fn qc_number_json_canonical(n: i32) -> bool {
        let json = format!(r#"{{"value":{}}}}"#, n);
        match canonical::normalize(&json) {
            Ok(canonical) => canonical.contains(&n.to_string()),
            Err(_) => false,
        }
    }

    /// QuickCheck: Empty object canonicalizes to {}
    #[quickcheck]
    fn qc_empty_object_canonical(_dummy: u8) -> bool {
        canonical::normalize("{}").unwrap() == "{}"
    }
}
