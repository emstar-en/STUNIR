//! Tests for STUNIR IR (Intermediate Representation) module
//!
//! Tests IR parsing, validation, and structure.

mod common;

use common::{SAMPLE_IR_JSON, SAMPLE_SPEC_JSON};
use stunir_native::ir_v1::{IrV1, Spec, IrFunction, IrInstruction};

#[test]
fn test_parse_spec_json() {
    let spec: Spec = serde_json::from_str(SAMPLE_SPEC_JSON)
        .expect("Should parse valid spec JSON");

    assert_eq!(spec.kind, "spec", "Kind should be 'spec'");
    assert_eq!(spec.modules.len(), 1, "Should have one module");
    assert_eq!(spec.modules[0].name, "hello", "Module name should be 'hello'");
    assert_eq!(spec.modules[0].lang, "python", "Language should be 'python'");
}

#[test]
fn test_parse_ir_json() {
    let ir: IrV1 = serde_json::from_str(SAMPLE_IR_JSON)
        .expect("Should parse valid IR JSON");

    assert_eq!(ir.kind, "ir", "Kind should be 'ir'");
    assert_eq!(ir.generator, "stunir-native-rust", "Generator should match");
    assert_eq!(ir.ir_version, "v1", "Version should be v1");
    assert_eq!(ir.module_name, "main", "Module name should be 'main'");
    assert_eq!(ir.functions.len(), 1, "Should have one function");
}

#[test]
fn test_ir_function_structure() {
    let func = IrFunction {
        name: "test_func".to_string(),
        body: vec![
            IrInstruction {
                op: "print".to_string(),
                args: vec!["Hello".to_string()],
            },
            IrInstruction {
                op: "call".to_string(),
                args: vec!["other_func".to_string()],
            },
        ],
    };

    assert_eq!(func.name, "test_func");
    assert_eq!(func.body.len(), 2);
    assert_eq!(func.body[0].op, "print");
    assert_eq!(func.body[1].op, "call");
}

#[test]
fn test_ir_serialization_roundtrip() {
    let original: IrV1 = serde_json::from_str(SAMPLE_IR_JSON)
        .expect("Should parse IR");

    let serialized = serde_json::to_string(&original)
        .expect("Should serialize IR");

    let deserialized: IrV1 = serde_json::from_str(&serialized)
        .expect("Should deserialize IR");

    assert_eq!(original.kind, deserialized.kind);
    assert_eq!(original.generator, deserialized.generator);
    assert_eq!(original.ir_version, deserialized.ir_version);
    assert_eq!(original.module_name, deserialized.module_name);
    assert_eq!(original.functions.len(), deserialized.functions.len());
}

#[test]
fn test_spec_with_multiple_modules() {
    let json = r#"{
        "kind": "spec",
        "modules": [
            {"name": "mod1", "source": "code1", "lang": "python"},
            {"name": "mod2", "source": "code2", "lang": "bash"},
            {"name": "mod3", "source": "code3", "lang": "javascript"}
        ],
        "metadata": {"author": "test"}
    }"#;

    let spec: Spec = serde_json::from_str(json)
        .expect("Should parse multi-module spec");

    assert_eq!(spec.modules.len(), 3);
    assert_eq!(spec.modules[0].lang, "python");
    assert_eq!(spec.modules[1].lang, "bash");
    assert_eq!(spec.modules[2].lang, "javascript");
    assert!(spec.metadata.contains_key("author"));
}

#[test]
fn test_invalid_spec_json() {
    let invalid = r#"{"kind": "spec"}"#;  // Missing required 'modules'
    let result: Result<Spec, _> = serde_json::from_str(invalid);
    assert!(result.is_err(), "Should fail on invalid spec");
}
