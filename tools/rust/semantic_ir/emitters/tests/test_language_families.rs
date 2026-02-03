//! Tests for language family emitters

use stunir_emitters::base::{BaseEmitter, EmitterConfig, EmitterStatus};
use stunir_emitters::language_families::*;
use stunir_emitters::types::*;
use tempfile::TempDir;

#[test]
fn test_lisp_common_lisp() {
    let temp_dir = TempDir::new().unwrap();
    let base_config = EmitterConfig::new(temp_dir.path(), "test_cl");
    let config = LispConfig::new(base_config, LispDialect::CommonLisp);
    let emitter = LispEmitter::new(config);

    let ir_module = create_test_module("test_cl");
    let result = emitter.emit(&ir_module).unwrap();

    assert_eq!(result.status, EmitterStatus::Success);
    assert_eq!(result.files.len(), 1);
}

#[test]
fn test_lisp_scheme() {
    let temp_dir = TempDir::new().unwrap();
    let base_config = EmitterConfig::new(temp_dir.path(), "test_scheme");
    let config = LispConfig::new(base_config, LispDialect::Scheme);
    let emitter = LispEmitter::new(config);

    let ir_module = create_test_module("test_scheme");
    let result = emitter.emit(&ir_module).unwrap();

    assert_eq!(result.status, EmitterStatus::Success);
}

#[test]
fn test_lisp_clojure() {
    let temp_dir = TempDir::new().unwrap();
    let base_config = EmitterConfig::new(temp_dir.path(), "test_clojure");
    let config = LispConfig::new(base_config, LispDialect::Clojure);
    let emitter = LispEmitter::new(config);

    let ir_module = create_test_module("test_clojure");
    let result = emitter.emit(&ir_module).unwrap();

    assert_eq!(result.status, EmitterStatus::Success);
}

#[test]
fn test_prolog_swi() {
    let temp_dir = TempDir::new().unwrap();
    let base_config = EmitterConfig::new(temp_dir.path(), "test_swi");
    let config = PrologConfig::new(base_config, PrologDialect::SWI);
    let emitter = PrologEmitter::new(config);

    let ir_module = create_test_module("test_swi");
    let result = emitter.emit(&ir_module).unwrap();

    assert_eq!(result.status, EmitterStatus::Success);
    assert_eq!(result.files.len(), 1);
}

#[test]
fn test_prolog_gnu() {
    let temp_dir = TempDir::new().unwrap();
    let base_config = EmitterConfig::new(temp_dir.path(), "test_gnu");
    let config = PrologConfig::new(base_config, PrologDialect::GNU);
    let emitter = PrologEmitter::new(config);

    let ir_module = create_test_module("test_gnu");
    let result = emitter.emit(&ir_module).unwrap();

    assert_eq!(result.status, EmitterStatus::Success);
}

fn create_test_module(name: &str) -> IRModule {
    IRModule {
        ir_version: "1.0".to_string(),
        module_name: name.to_string(),
        types: vec![],
        functions: vec![IRFunction {
            name: "test_pred".to_string(),
            return_type: IRDataType::Bool,
            parameters: vec![],
            statements: vec![],
            docstring: Some("Test predicate".to_string()),
        }],
        docstring: None,
    }
}
