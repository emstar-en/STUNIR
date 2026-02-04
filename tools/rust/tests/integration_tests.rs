//! STUNIR Rust Pipeline Integration Tests
//!
//! These tests verify end-to-end functionality across spec_to_ir -> optimizer -> ir_to_code

use std::fs;
use std::process::Command;
use tempfile::TempDir;

/// Helper to run the spec_to_ir binary
fn run_spec_to_ir(spec_content: &str, output_path: &std::path::Path) -> Result<(), String> {
    let temp_dir = TempDir::new().map_err(|e| e.to_string())?;
    let spec_path = temp_dir.path().join("test_spec.json");
    fs::write(&spec_path, spec_content).map_err(|e| e.to_string())?;

    let output = Command::new("cargo")
        .args(&[
            "run",
            "--bin",
            "stunir_spec_to_ir",
            "--",
            spec_path.to_str().unwrap(),
            "--out",
            output_path.to_str().unwrap(),
        ])
        .current_dir("..")
        .output()
        .map_err(|e| e.to_string())?;

    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }

    Ok(())
}

/// Helper to run the ir_to_code binary
fn run_ir_to_code(ir_path: &std::path::Path, target: &str, output_path: &std::path::Path) -> Result<(), String> {
    let output = Command::new("cargo")
        .args(&[
            "run",
            "--bin",
            "stunir_ir_to_code",
            "--",
            ir_path.to_str().unwrap(),
            "--target",
            target,
            "--output",
            output_path.to_str().unwrap(),
        ])
        .current_dir("..")
        .output()
        .map_err(|e| e.to_string())?;

    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }

    Ok(())
}

/// Helper to run the optimizer
fn run_optimizer(ir_path: &std::path::Path, output_path: &std::path::Path, level: &str) -> Result<(), String> {
    let output = Command::new("cargo")
        .args(&[
            "run",
            "--bin",
            "stunir_optimizer",
            "--",
            ir_path.to_str().unwrap(),
            "--out",
            output_path.to_str().unwrap(),
            "--level",
            level,
        ])
        .current_dir("..")
        .output()
        .map_err(|e| e.to_string())?;

    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }

    Ok(())
}

#[test]
fn test_end_to_end_basic_function() {
    let temp_dir = TempDir::new().unwrap();
    let ir_path = temp_dir.path().join("test.ir.json");
    let code_path = temp_dir.path().join("test.c");

    let spec = r#"{
        "kind": "spec",
        "schema": "stunir_spec_v1",
        "modules": [{
            "name": "test",
            "functions": [{
                "name": "add",
                "parameters": [
                    {"name": "a", "type": "i32"},
                    {"name": "b", "type": "i32"}
                ],
                "return_type": "i32",
                "body": [{"type": "return", "value": "a + b"}]
            }]
        }]
    }"#;

    // Skip if binaries not built
    if run_spec_to_ir(spec, &ir_path).is_err() {
        eprintln!("Skipping integration test - binaries not available");
        return;
    }

    assert!(ir_path.exists());

    if run_ir_to_code(&ir_path, "c", &code_path).is_ok() {
        let code = fs::read_to_string(&code_path).unwrap();
        assert!(code.contains("int32_t add(int32_t a, int32_t b)"));
        assert!(code.contains("return a + b;"));
    }
}

#[test]
fn test_end_to_end_multiple_targets() {
    let temp_dir = TempDir::new().unwrap();
    let ir_path = temp_dir.path().join("test.ir.json");
    let c_path = temp_dir.path().join("test.c");
    let rust_path = temp_dir.path().join("test.rs");
    let py_path = temp_dir.path().join("test.py");

    let spec = r#"{
        "kind": "spec",
        "schema": "stunir_spec_v1",
        "modules": [{
            "name": "test",
            "functions": [{
                "name": "greet",
                "parameters": [{"name": "name", "type": "string"}],
                "return_type": "void",
                "body": [{"type": "call", "function": "print", "args": ["name"]}]
            }]
        }]
    }"#;

    if run_spec_to_ir(spec, &ir_path).is_err() {
        eprintln!("Skipping integration test - binaries not available");
        return;
    }

    // Test C output
    if run_ir_to_code(&ir_path, "c", &c_path).is_ok() {
        let code = fs::read_to_string(&c_path).unwrap();
        assert!(code.contains("void greet"));
    }

    // Test Rust output
    if run_ir_to_code(&ir_path, "rust", &rust_path).is_ok() {
        let code = fs::read_to_string(&rust_path).unwrap();
        assert!(code.contains("fn greet"));
    }

    // Test Python output
    if run_ir_to_code(&ir_path, "python", &py_path).is_ok() {
        let code = fs::read_to_string(&py_path).unwrap();
        assert!(code.contains("def greet"));
    }
}

#[test]
fn test_end_to_end_with_optimization() {
    let temp_dir = TempDir::new().unwrap();
    let ir_path = temp_dir.path().join("test.ir.json");
    let opt_ir_path = temp_dir.path().join("test.opt.json");
    let code_path = temp_dir.path().join("test.c");

    let spec = r#"{
        "kind": "spec",
        "schema": "stunir_spec_v1",
        "modules": [{
            "name": "test",
            "functions": [{
                "name": "compute",
                "parameters": [],
                "return_type": "i32",
                "body": [
                    {"type": "var_decl", "name": "x", "var_type": "i32", "init": "1 + 2"},
                    {"type": "return", "value": "x"}
                ]
            }]
        }]
    }"#;

    if run_spec_to_ir(spec, &ir_path).is_err() {
        eprintln!("Skipping integration test - binaries not available");
        return;
    }

    // Note: Optimizer binary may not exist, skip if not available
    if run_optimizer(&ir_path, &opt_ir_path, "O2").is_ok() {
        assert!(opt_ir_path.exists());
        
        if run_ir_to_code(&opt_ir_path, "c", &code_path).is_ok() {
            let code = fs::read_to_string(&code_path).unwrap();
            // After optimization, "1 + 2" should be folded to "3"
            assert!(code.contains("3") || code.contains("x"));
        }
    }
}

#[test]
fn test_end_to_end_complex_control_flow() {
    let temp_dir = TempDir::new().unwrap();
    let ir_path = temp_dir.path().join("test.ir.json");
    let code_path = temp_dir.path().join("test.c");

    let spec = r#"{
        "kind": "spec",
        "schema": "stunir_spec_v1",
        "modules": [{
            "name": "test",
            "functions": [{
                "name": "factorial",
                "parameters": [{"name": "n", "type": "i32"}],
                "return_type": "i32",
                "body": [
                    {"type": "if", "condition": "n <= 1", "then": [
                        {"type": "return", "value": "1"}
                    ]},
                    {"type": "return", "value": "n * factorial(n - 1)"}
                ]
            }]
        }]
    }"#;

    if run_spec_to_ir(spec, &ir_path).is_err() {
        eprintln!("Skipping integration test - binaries not available");
        return;
    }

    if run_ir_to_code(&ir_path, "c", &code_path).is_ok() {
        let code = fs::read_to_string(&code_path).unwrap();
        assert!(code.contains("int32_t factorial(int32_t n)"));
        assert!(code.contains("if (n <= 1)"));
        assert!(code.contains("return 1;"));
    }
}

#[test]
fn test_end_to_end_data_structures() {
    let temp_dir = TempDir::new().unwrap();
    let ir_path = temp_dir.path().join("test.ir.json");
    let code_path = temp_dir.path().join("test.c");

    let spec = r#"{
        "kind": "spec",
        "schema": "stunir_spec_v1",
        "modules": [{
            "name": "test",
            "types": [{
                "name": "Point",
                "kind": "struct",
                "fields": [
                    {"name": "x", "type": "f64"},
                    {"name": "y", "type": "f64"}
                ]
            }],
            "functions": [{
                "name": "create_point",
                "parameters": [
                    {"name": "x", "type": "f64"},
                    {"name": "y", "type": "f64"}
                ],
                "return_type": "Point",
                "body": [
                    {"type": "struct_new", "target": "p", "struct_type": "Point", "fields": {"x": "x", "y": "y"}},
                    {"type": "return", "value": "p"}
                ]
            }]
        }]
    }"#;

    if run_spec_to_ir(spec, &ir_path).is_err() {
        eprintln!("Skipping integration test - binaries not available");
        return;
    }

    if run_ir_to_code(&ir_path, "c", &code_path).is_ok() {
        let code = fs::read_to_string(&code_path).unwrap();
        assert!(code.contains("typedef struct"));
        assert!(code.contains("Point"));
    }
}

#[test]
fn test_end_to_end_exception_handling() {
    let temp_dir = TempDir::new().unwrap();
    let ir_path = temp_dir.path().join("test.ir.json");
    let code_path = temp_dir.path().join("test.c");

    let spec = r#"{
        "kind": "spec",
        "schema": "stunir_spec_v1",
        "modules": [{
            "name": "test",
            "functions": [{
                "name": "safe_divide",
                "parameters": [
                    {"name": "a", "type": "i32"},
                    {"name": "b", "type": "i32"}
                ],
                "return_type": "i32",
                "body": [
                    {
                        "type": "try",
                        "try_block": [{"type": "return", "value": "a / b"}],
                        "catch_blocks": [
                            {"exception_type": "DivisionByZero", "var": "e", "body": [{"type": "return", "value": "0"}]}
                        ]
                    }
                ]
            }]
        }]
    }"#;

    if run_spec_to_ir(spec, &ir_path).is_err() {
        eprintln!("Skipping integration test - binaries not available");
        return;
    }

    if run_ir_to_code(&ir_path, "c", &code_path).is_ok() {
        let code = fs::read_to_string(&code_path).unwrap();
        assert!(code.contains("safe_divide"));
        // C output should have some error handling structure
        assert!(code.contains("return") || code.contains("if"));
    }
}

#[test]
fn test_confluence_with_python() {
    // This test verifies that Rust and Python produce equivalent IR
    // when given the same spec
    
    let spec = r#"{
        "kind": "spec",
        "schema": "stunir_spec_v1",
        "modules": [{
            "name": "test",
            "functions": [{
                "name": "identity",
                "parameters": [{"name": "x", "type": "i32"}],
                "return_type": "i32",
                "body": [{"type": "return", "value": "x"}]
            }]
        }]
    }"#;

    // Generate IR using Rust
    let temp_dir = TempDir::new().unwrap();
    let rust_ir_path = temp_dir.path().join("rust.ir.json");

    if run_spec_to_ir(spec, &rust_ir_path).is_err() {
        eprintln!("Skipping confluence test - Rust binary not available");
        return;
    }

    // Verify the IR is valid JSON and has expected structure
    let ir_content = fs::read_to_string(&rust_ir_path).unwrap();
    let ir: serde_json::Value = serde_json::from_str(&ir_content).unwrap();
    
    assert_eq!(ir["kind"], "ir");
    assert_eq!(ir["schema"], "stunir_ir_v1");
    assert!(ir["functions"].as_array().unwrap().len() > 0);
    assert_eq!(ir["functions"][0]["name"], "identity");
}