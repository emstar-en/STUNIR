#!/usr/bin/env -S cargo +stable run --manifest-path /home/ubuntu/stunir_repo/tools/rust/Cargo.toml --bin stunir_spec_to_ir --
//!
//! STUNIR Spec to IR Converter - Rust Production Implementation
//!
//! This is a production-ready implementation providing memory safety
//! and deterministic execution guarantees.
//!
//! # Confluence
//! This implementation produces bitwise-identical outputs to:
//! - Ada SPARK implementation (reference)
//! - Python implementation
//! - Haskell implementation

use anyhow::{Context, Result};
use clap::Parser;
use serde_json::{json, Value};
use std::fs;
use std::path::PathBuf;
use stunir_tools::{sha256_json, types::*};

#[derive(Parser, Debug)]
#[command(name = "stunir_spec_to_ir")]
#[command(about = "Convert STUNIR specification to Intermediate Reference (IR)")]
struct Args {
    /// Input specification file (JSON) - for backward compatibility
    #[arg(value_name = "SPEC_FILE", required_unless_present = "spec_root")]
    spec_file: Option<PathBuf>,

    /// Spec root directory (for multi-file processing)
    #[arg(long = "spec-root", value_name = "SPEC_ROOT")]
    spec_root: Option<PathBuf>,

    /// Output IR file (JSON)
    #[arg(short, long = "out", value_name = "OUTPUT", required = true)]
    output: PathBuf,

    /// Print version and exit
    #[arg(long)]
    version: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.version {
        println!("STUNIR Spec to IR (Rust) v1.0.0");
        return Ok(());
    }

    let ir_module = if let Some(spec_root) = args.spec_root {
        // Multi-file mode: process all JSON files in directory
        eprintln!("[STUNIR][Rust] Processing specs from: {:?}", spec_root);
        
        if !spec_root.exists() {
            anyhow::bail!("Spec root not found: {:?}", spec_root);
        }

        // Collect all spec files
        let mut spec_files = Vec::new();
        collect_spec_files(&spec_root, &mut spec_files)?;
        spec_files.sort();

        if spec_files.is_empty() {
            anyhow::bail!("No spec files found in {:?}", spec_root);
        }

        eprintln!("[STUNIR][Rust] Found {} spec file(s)", spec_files.len());

        // Generate merged IR from all specs
        generate_merged_ir(&spec_files)?
    } else if let Some(spec_file) = args.spec_file {
        // Single-file mode (backward compatibility)
        let spec_contents = fs::read_to_string(&spec_file)
            .with_context(|| format!("Failed to read spec file: {:?}", spec_file))?;

        let spec: Value = serde_json::from_str(&spec_contents)
            .context("Failed to parse spec JSON")?;

        generate_ir(&spec)?
    } else {
        anyhow::bail!("Either SPEC_FILE or --spec-root must be provided");
    };

    // Output
    let output_json = serde_json::to_string_pretty(&ir_module)?;
    
    fs::write(&args.output, &output_json)
        .with_context(|| format!("Failed to write output: {:?}", args.output))?;
    eprintln!("[STUNIR][Rust] IR written to: {:?}", args.output);
    eprintln!("[STUNIR][Rust] Schema: {}", ir_module.schema);
    
    Ok(())
}

fn collect_spec_files(dir: &PathBuf, spec_files: &mut Vec<PathBuf>) -> Result<()> {
    if dir.is_dir() {
        let mut entries: Vec<_> = fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .collect();
        
        // Sort for determinism
        entries.sort_by_key(|e| e.path());

        for entry in entries {
            let path = entry.path();
            if path.is_dir() {
                collect_spec_files(&path, spec_files)?;
            } else if path.extension().and_then(|s| s.to_str()) == Some("json") {
                spec_files.push(path);
            }
        }
    }
    Ok(())
}

fn generate_merged_ir(spec_files: &[PathBuf]) -> Result<IRModule> {
    // Process first spec file to get metadata
    let first_spec_contents = fs::read_to_string(&spec_files[0])
        .with_context(|| format!("Failed to read spec file: {:?}", spec_files[0]))?;
    let first_spec: Value = serde_json::from_str(&first_spec_contents)?;
    
    let mut ir_module = generate_ir(&first_spec)?;
    
    // If there are more files, merge their functions
    if spec_files.len() > 1 {
        eprintln!("[STUNIR][Rust] Merging {} spec files...", spec_files.len());
        
        for spec_file in &spec_files[1..] {
            let spec_contents = fs::read_to_string(spec_file)
                .with_context(|| format!("Failed to read spec file: {:?}", spec_file))?;
            let spec: Value = serde_json::from_str(&spec_contents)?;
            
            let additional_ir = generate_ir(&spec)?;
            
            // Merge functions
            ir_module.functions.extend(additional_ir.functions);
        }
        
        eprintln!("[STUNIR][Rust] Total functions: {}", ir_module.functions.len());
    }
    
    Ok(ir_module)
}

fn generate_ir(spec: &Value) -> Result<IRModule> {
    // Extract module metadata
    // Try both "module" (stunir.spec.v1) and "name" (fallback)
    let module_name = spec["module"]
        .as_str()
        .or_else(|| spec["name"].as_str())
        .unwrap_or("unnamed_module")
        .to_string();
    
    let docstring = spec["description"]
        .as_str()
        .map(|s| s.to_string());

    // Extract types (simplified - for now empty)
    let types = vec![];

    // Extract functions
    let functions = if let Some(funcs) = spec["functions"].as_array() {
        funcs
            .iter()
            .map(|f| parse_function(f))
            .collect::<Result<Vec<_>>>()?        
    } else {
        vec![]
    };

    Ok(IRModule {
        schema: "stunir_ir_v1".to_string(),
        ir_version: "v1".to_string(),
        module_name,
        docstring,
        types,
        functions,
    })
}

fn parse_function(func: &Value) -> Result<IRFunction> {
    let name = func["name"]
        .as_str()
        .context("Function missing 'name'")?        
        .to_string();

    let docstring = func["description"]
        .as_str()
        .map(|s| s.to_string());

    // Try both "returns" (stunir.spec.v1) and "return_type" (fallback)
    let return_type = func["returns"]
        .as_str()
        .or_else(|| func["return_type"].as_str())
        .unwrap_or("void")
        .to_string();

    // Try both "params" (stunir.spec.v1) and "parameters" (fallback)
    let args = if let Some(params) = func["params"].as_array() {
        params
            .iter()
            .map(|p| parse_arg(p))
            .collect::<Result<Vec<_>>>()?        
    } else if let Some(params) = func["parameters"].as_array() {
        params
            .iter()
            .map(|p| parse_arg(p))
            .collect::<Result<Vec<_>>>()?        
    } else {
        vec![]
    };

    // Parse body statements into steps
    let steps = if let Some(body) = func["body"].as_array() {
        Some(
            body.iter()
                .map(|s| parse_statement(s))
                .collect::<Result<Vec<_>>>()?
        )
    } else {
        Some(vec![])
    };

    Ok(IRFunction {
        name,
        docstring,
        args,
        return_type,
        steps,
    })
}

fn parse_arg(param: &Value) -> Result<IRArg> {
    let name = param["name"]
        .as_str()
        .context("Parameter missing 'name'")?        
        .to_string();

    let arg_type = param["type"]
        .as_str()
        .unwrap_or("i32")
        .to_string();

    Ok(IRArg { name, arg_type })
}

fn parse_statement(stmt: &Value) -> Result<IRStep> {
    use stunir_tools::types::{IRStep, IRCase};
    
    let stmt_type = stmt["type"]
        .as_str()
        .context("Statement missing 'type'")?;
    
    match stmt_type {
        // Control flow statements (v0.6.1)
        "if" => {
            let condition = stmt["condition"]
                .as_str()
                .unwrap_or("true")
                .to_string();
            
            let then_block = if let Some(then_stmts) = stmt["then"].as_array() {
                Some(
                    then_stmts.iter()
                        .map(|s| parse_statement(s))
                        .collect::<Result<Vec<_>>>()?
                )
            } else {
                Some(vec![])
            };
            
            let else_block = if let Some(else_stmts) = stmt["else"].as_array() {
                Some(
                    else_stmts.iter()
                        .map(|s| parse_statement(s))
                        .collect::<Result<Vec<_>>>()?
                )
            } else {
                None
            };
            
            Ok(IRStep {
                op: "if".to_string(),
                target: None,
                value: None,
                condition: Some(condition),
                then_block,
                else_block,
                body: None,
                init: None,
                increment: None,
                expr: None,
                cases: None,
                default: None,
                try_block: None,
                catch_blocks: None,
                finally_block: None,
                exception_type: None,
                exception_message: None,
                ..Default::default()
            })
        },
        "while" => {
            let condition = stmt["condition"]
                .as_str()
                .unwrap_or("true")
                .to_string();
            
            let body = if let Some(body_stmts) = stmt["body"].as_array() {
                Some(
                    body_stmts.iter()
                        .map(|s| parse_statement(s))
                        .collect::<Result<Vec<_>>>()?
                )
            } else {
                Some(vec![])
            };
            
            Ok(IRStep {
                op: "while".to_string(),
                target: None,
                value: None,
                condition: Some(condition),
                then_block: None,
                else_block: None,
                body,
                init: None,
                increment: None,
                expr: None,
                cases: None,
                default: None,
                try_block: None,
                catch_blocks: None,
                finally_block: None,
                exception_type: None,
                exception_message: None,
                ..Default::default()
            })
        },
        "for" => {
            let init = stmt["init"]
                .as_str()
                .unwrap_or("")
                .to_string();
            let condition = stmt["condition"]
                .as_str()
                .unwrap_or("true")
                .to_string();
            let increment = stmt["increment"]
                .as_str()
                .unwrap_or("")
                .to_string();
            
            let body = if let Some(body_stmts) = stmt["body"].as_array() {
                Some(
                    body_stmts.iter()
                        .map(|s| parse_statement(s))
                        .collect::<Result<Vec<_>>>()?
                )
            } else {
                Some(vec![])
            };
            
            Ok(IRStep {
                op: "for".to_string(),
                target: None,
                value: None,
                condition: Some(condition),
                then_block: None,
                else_block: None,
                body,
                init: Some(init),
                increment: Some(increment),
                expr: None,
                cases: None,
                default: None,
                try_block: None,
                catch_blocks: None,
                finally_block: None,
                exception_type: None,
                exception_message: None,
                ..Default::default()
            })
        },
        
        // Switch/case statement (v0.9.0)
        "switch" => {
            let expr = stmt["expr"]
                .as_str()
                .unwrap_or("0")
                .to_string();
            
            let cases = if let Some(case_array) = stmt["cases"].as_array() {
                let parsed_cases: Result<Vec<IRCase>> = case_array.iter()
                    .map(|c| {
                        let value = c["value"].clone();
                        let body = if let Some(body_stmts) = c["body"].as_array() {
                            body_stmts.iter()
                                .map(|s| parse_statement(s))
                                .collect::<Result<Vec<_>>>()?
                        } else {
                            vec![]
                        };
                        Ok(IRCase { value, body })
                    })
                    .collect();
                Some(parsed_cases?)
            } else {
                None
            };
            
            let default = if let Some(default_stmts) = stmt["default"].as_array() {
                Some(
                    default_stmts.iter()
                        .map(|s| parse_statement(s))
                        .collect::<Result<Vec<_>>>()?
                )
            } else {
                None
            };
            
            Ok(IRStep {
                op: "switch".to_string(),
                target: None,
                value: None,
                condition: None,
                then_block: None,
                else_block: None,
                body: None,
                init: None,
                increment: None,
                expr: Some(expr),
                cases,
                default,
                try_block: None,
                catch_blocks: None,
                finally_block: None,
                exception_type: None,
                exception_message: None,
                ..Default::default()
            })
        },
        
        // Break/continue statements (v0.9.0)
        "break" => {
            Ok(IRStep {
                op: "break".to_string(),
                target: None,
                value: None,
                condition: None,
                then_block: None,
                else_block: None,
                body: None,
                init: None,
                increment: None,
                expr: None,
                cases: None,
                default: None,
                try_block: None,
                catch_blocks: None,
                finally_block: None,
                exception_type: None,
                exception_message: None,
                ..Default::default()
            })
        },
        "continue" => {
            Ok(IRStep {
                op: "continue".to_string(),
                target: None,
                value: None,
                condition: None,
                then_block: None,
                else_block: None,
                body: None,
                init: None,
                increment: None,
                expr: None,
                cases: None,
                default: None,
                try_block: None,
                catch_blocks: None,
                finally_block: None,
                exception_type: None,
                exception_message: None,
                ..Default::default()
            })
        },
        
        // Regular statements
        "call" => {
            let called_func = stmt["func"]
                .as_str()
                .unwrap_or("unknown");
            let args = stmt["args"].as_array()
                .map(|arr| {
                    arr.iter()
                        .map(|v| v.as_str().unwrap_or(""))
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();
            
            let value = format!("{}({})", called_func, args);
            
            let target = stmt["assign_to"]
                .as_str()
                .map(|s| s.to_string());
            
            Ok(IRStep {
                op: "call".to_string(),
                target,
                value: Some(serde_json::json!(value)),
                condition: None,
                then_block: None,
                else_block: None,
                body: None,
                init: None,
                increment: None,
                expr: None,
                cases: None,
                default: None,
                try_block: None,
                catch_blocks: None,
                finally_block: None,
                exception_type: None,
                exception_message: None,
                ..Default::default()
            })
        },
        "assign" => {
            let target = stmt["target"]
                .as_str()
                .context("assign missing 'target'")?
                .to_string();
            let value = stmt["value"]
                .as_str()
                .map(|v| serde_json::json!(v));
            
            Ok(IRStep {
                op: "assign".to_string(),
                target: Some(target),
                value,
                condition: None,
                then_block: None,
                else_block: None,
                body: None,
                init: None,
                increment: None,
                expr: None,
                cases: None,
                default: None,
                try_block: None,
                catch_blocks: None,
                finally_block: None,
                exception_type: None,
                exception_message: None,
                ..Default::default()
            })
        },
        "var_decl" => {
            // Map var_decl to assignment operation
            let target = stmt["var_name"]
                .as_str()
                .context("var_decl missing 'var_name'")?
                .to_string();
            let init = stmt["init"]
                .as_str()
                .unwrap_or("0");
            
            Ok(IRStep {
                op: "assign".to_string(),
                target: Some(target),
                value: Some(serde_json::json!(init)),
                condition: None,
                then_block: None,
                else_block: None,
                body: None,
                init: None,
                increment: None,
                expr: None,
                cases: None,
                default: None,
                try_block: None,
                catch_blocks: None,
                finally_block: None,
                exception_type: None,
                exception_message: None,
                ..Default::default()
            })
        },
        "return" => {
            let value = stmt["value"]
                .as_str()
                .map(|v| serde_json::json!(v));
            Ok(IRStep {
                op: "return".to_string(),
                target: None,
                value,
                condition: None,
                then_block: None,
                else_block: None,
                body: None,
                init: None,
                increment: None,
                expr: None,
                cases: None,
                default: None,
                try_block: None,
                catch_blocks: None,
                finally_block: None,
                exception_type: None,
                exception_message: None,
                ..Default::default()
            })
        },
        "comment" => {
            Ok(IRStep {
                op: "nop".to_string(),
                target: None,
                value: None,
                condition: None,
                then_block: None,
                else_block: None,
                body: None,
                init: None,
                increment: None,
                expr: None,
                cases: None,
                default: None,
                try_block: None,
                catch_blocks: None,
                finally_block: None,
                exception_type: None,
                exception_message: None,
                ..Default::default()
            })
        },
        
        // v0.8.7: try/catch/finally exception handling
        "try" => {
            use stunir_tools::types::IRCatch;
            
            // Parse try block
            let try_block = if let Some(try_stmts) = stmt["try"].as_array()
                .or_else(|| stmt["body"].as_array()) {
                Some(
                    try_stmts.iter()
                        .map(|s| parse_statement(s))
                        .collect::<Result<Vec<_>>>()?
                )
            } else {
                Some(vec![])
            };
            
            // Parse catch blocks
            let catches = stmt["catch"].as_array()
                .or_else(|| stmt["catches"].as_array());
            let catch_blocks = if let Some(catch_array) = catches {
                let parsed_catches: Result<Vec<IRCatch>> = catch_array.iter()
                    .map(|c| {
                        let exception_type = c["exception_type"]
                            .as_str()
                            .or_else(|| c["type"].as_str())
                            .unwrap_or("*")
                            .to_string();
                        let exception_var = c["exception_var"]
                            .as_str()
                            .or_else(|| c["var"].as_str())
                            .map(|s| s.to_string());
                        let body = if let Some(body_stmts) = c["body"].as_array() {
                            body_stmts.iter()
                                .map(|s| parse_statement(s))
                                .collect::<Result<Vec<_>>>()?
                        } else {
                            vec![]
                        };
                        Ok(IRCatch { exception_type, exception_var, body })
                    })
                    .collect();
                Some(parsed_catches?)
            } else {
                None
            };
            
            // Parse finally block
            let finally_block = if let Some(finally_stmts) = stmt["finally"].as_array() {
                Some(
                    finally_stmts.iter()
                        .map(|s| parse_statement(s))
                        .collect::<Result<Vec<_>>>()?
                )
            } else {
                None
            };
            
            Ok(IRStep {
                op: "try".to_string(),
                target: None,
                value: None,
                condition: None,
                then_block: None,
                else_block: None,
                body: None,
                init: None,
                increment: None,
                expr: None,
                cases: None,
                default: None,
                try_block,
                catch_blocks,
                finally_block,
                exception_type: None,
                exception_message: None,
                ..Default::default()
            })
        },
        
        // v0.8.7: throw exception
        "throw" => {
            let exception_type = stmt["exception_type"]
                .as_str()
                .or_else(|| stmt["type"].as_str())
                .map(|s| s.to_string());
            let exception_message = stmt["exception_message"]
                .as_str()
                .or_else(|| stmt["message"].as_str())
                .map(|s| s.to_string());
            
            Ok(IRStep {
                op: "throw".to_string(),
                exception_type,
                exception_message,
                ..Default::default()
            })
        },
        
        _ => {
            // Unknown statement type - return a noop
            eprintln!("[WARN] Unknown statement type: {}", stmt_type);
            Ok(IRStep {
                op: "nop".to_string(),
                ..Default::default()
            })
        }
    }
}
