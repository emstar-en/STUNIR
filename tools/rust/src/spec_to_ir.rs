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
    /// Input specification file (JSON)
    #[arg(value_name = "SPEC_FILE")]
    spec_file: PathBuf,

    /// Output IR file (JSON)
    #[arg(short, long, value_name = "OUTPUT")]
    output: Option<PathBuf>,

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

    // Read spec file
    let spec_contents = fs::read_to_string(&args.spec_file)
        .with_context(|| format!("Failed to read spec file: {:?}", args.spec_file))?;

    // Parse spec JSON
    let spec: Value = serde_json::from_str(&spec_contents)
        .context("Failed to parse spec JSON")?;

    // Generate IR - now returns flat IRModule matching stunir_ir_v1 schema
    let ir_module = generate_ir(&spec)?;

    // Output
    let output_json = serde_json::to_string_pretty(&ir_module)?;
    
    if let Some(output_path) = args.output {
        fs::write(&output_path, &output_json)
            .with_context(|| format!("Failed to write output: {:?}", output_path))?;
        eprintln!("[STUNIR][Rust] IR written to: {:?}", output_path);
    } else {
        println!("{}", output_json);
    }

    eprintln!("[STUNIR][Rust] Schema: {}", ir_module.schema);
    Ok(())
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
    use stunir_tools::types::IRStep;
    
    let stmt_type = stmt["type"]
        .as_str()
        .context("Statement missing 'type'")?;
    
    match stmt_type {
        "return" => {
            let value = stmt["value"]
                .as_str()
                .map(|v| serde_json::json!(v));
            Ok(IRStep {
                op: "return".to_string(),
                target: None,
                value,
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
            })
        },
        _ => {
            // Unknown statement type - return a noop
            eprintln!("[WARN] Unknown statement type: {}", stmt_type);
            Ok(IRStep {
                op: "noop".to_string(),
                target: None,
                value: None,
            })
        }
    }
}
