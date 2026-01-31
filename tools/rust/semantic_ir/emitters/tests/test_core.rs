//! Tests for core emitters

use stunir_emitters::base::{BaseEmitter, EmitterConfig, EmitterStatus};
use stunir_emitters::core::*;
use stunir_emitters::types::*;
use tempfile::TempDir;

#[test]
fn test_embedded_arm() {
    let temp_dir = TempDir::new().unwrap();
    let base_config = EmitterConfig::new(temp_dir.path(), "test_embedded");
    let config = EmbeddedConfig::new(base_config, Architecture::ARM);
    let emitter = EmbeddedEmitter::new(config);

    let ir_module = create_test_module("test_embedded");
    let result = emitter.emit(&ir_module).unwrap();

    assert_eq!(result.status, EmitterStatus::Success);
    assert!(result.files.len() >= 2); // header + source
}

#[test]
fn test_embedded_avr() {
    let temp_dir = TempDir::new().unwrap();
    let base_config = EmitterConfig::new(temp_dir.path(), "test_avr");
    let config = EmbeddedConfig::new(base_config, Architecture::AVR);
    let emitter = EmbeddedEmitter::new(config);

    let ir_module = create_test_module("test_avr");
    let result = emitter.emit(&ir_module).unwrap();

    assert_eq!(result.status, EmitterStatus::Success);
}

#[test]
fn test_gpu_cuda() {
    let temp_dir = TempDir::new().unwrap();
    let base_config = EmitterConfig::new(temp_dir.path(), "test_cuda");
    let config = GPUConfig::new(base_config, GPUPlatform::CUDA);
    let emitter = GPUEmitter::new(config);

    let ir_module = create_test_module("test_cuda");
    let result = emitter.emit(&ir_module).unwrap();

    assert_eq!(result.status, EmitterStatus::Success);
    assert_eq!(result.files.len(), 1);
}

#[test]
fn test_gpu_opencl() {
    let temp_dir = TempDir::new().unwrap();
    let base_config = EmitterConfig::new(temp_dir.path(), "test_opencl");
    let config = GPUConfig::new(base_config, GPUPlatform::OpenCL);
    let emitter = GPUEmitter::new(config);

    let ir_module = create_test_module("test_opencl");
    let result = emitter.emit(&ir_module).unwrap();

    assert_eq!(result.status, EmitterStatus::Success);
}

#[test]
fn test_wasm() {
    let temp_dir = TempDir::new().unwrap();
    let base_config = EmitterConfig::new(temp_dir.path(), "test_wasm");
    let config = WasmConfig::new(base_config, WasmTarget::Core);
    let emitter = WasmEmitter::new(config);

    let ir_module = create_test_module("test_wasm");
    let result = emitter.emit(&ir_module).unwrap();

    assert_eq!(result.status, EmitterStatus::Success);
    assert_eq!(result.files.len(), 1);
}

#[test]
fn test_assembly_x86() {
    let temp_dir = TempDir::new().unwrap();
    let base_config = EmitterConfig::new(temp_dir.path(), "test_asm");
    let config = AssemblyConfig::new(base_config, Architecture::X86);
    let emitter = AssemblyEmitter::new(config);

    let ir_module = create_test_module("test_asm");
    let result = emitter.emit(&ir_module).unwrap();

    assert_eq!(result.status, EmitterStatus::Success);
    assert_eq!(result.files.len(), 1);
}

#[test]
fn test_polyglot_c89() {
    let temp_dir = TempDir::new().unwrap();
    let base_config = EmitterConfig::new(temp_dir.path(), "test_c89");
    let config = PolyglotConfig::new(base_config, PolyglotLanguage::C89);
    let emitter = PolyglotEmitter::new(config);

    let ir_module = create_test_module("test_c89");
    let result = emitter.emit(&ir_module).unwrap();

    assert_eq!(result.status, EmitterStatus::Success);
    assert_eq!(result.files.len(), 2); // header + source
}

#[test]
fn test_polyglot_rust() {
    let temp_dir = TempDir::new().unwrap();
    let base_config = EmitterConfig::new(temp_dir.path(), "test_rust");
    let config = PolyglotConfig::new(base_config, PolyglotLanguage::Rust);
    let emitter = PolyglotEmitter::new(config);

    let ir_module = create_test_module("test_rust");
    let result = emitter.emit(&ir_module).unwrap();

    assert_eq!(result.status, EmitterStatus::Success);
    assert_eq!(result.files.len(), 1);
}

fn create_test_module(name: &str) -> IRModule {
    IRModule {
        ir_version: "1.0".to_string(),
        module_name: name.to_string(),
        types: vec![],
        functions: vec![IRFunction {
            name: "test_function".to_string(),
            return_type: IRDataType::I32,
            parameters: vec![IRParameter {
                name: "x".to_string(),
                param_type: IRDataType::I32,
            }],
            statements: vec![IRStatement {
                stmt_type: IRStatementType::Return,
                data_type: Some(IRDataType::I32),
                target: None,
                value: Some("42".to_string()),
                left_op: None,
                right_op: None,
            }],
            docstring: Some("Test function".to_string()),
        }],
        docstring: Some("Test module".to_string()),
    }
}
