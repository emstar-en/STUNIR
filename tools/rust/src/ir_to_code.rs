#!/usr/bin/env -S cargo +stable run --manifest-path /home/ubuntu/stunir_repo/tools/rust/Cargo.toml --bin stunir_ir_to_code --
//!
//! STUNIR IR to Code Emitter - Rust Production Implementation
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
use std::fs;
use std::path::PathBuf;
use stunir_tools::types::*;

#[derive(Parser, Debug)]
#[command(name = "stunir_ir_to_code")]
#[command(about = "Emit code from STUNIR Intermediate Reference (IR)")]
struct Args {
    /// Input IR file (JSON)
    #[arg(value_name = "IR_FILE")]
    ir_file: PathBuf,

    /// Target language/platform (default: c)
    #[arg(short, long, value_name = "TARGET", default_value = "c")]
    target: String,

    /// Output code file
    #[arg(short, long, value_name = "OUTPUT")]
    output: Option<PathBuf>,

    /// Print version and exit
    #[arg(long)]
    version: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.version {
        println!("STUNIR IR to Code (Rust) v1.0.0");
        return Ok(());
    }

    // Read IR file
    let ir_contents = fs::read_to_string(&args.ir_file)
        .with_context(|| format!("Failed to read IR file: {:?}", args.ir_file))?;

    // Parse IR JSON - now expects flat stunir_ir_v1 format
    let module: IRModule = serde_json::from_str(&ir_contents)
        .context("Failed to parse IR module")?;

    // Emit code based on target
    let code = emit_code(&module, &args.target)?;

    // Output
    if let Some(output_path) = args.output {
        fs::write(&output_path, &code)
            .with_context(|| format!("Failed to write output: {:?}", output_path))?;
        eprintln!("[STUNIR][Rust] Code written to: {:?}", output_path);
    } else {
        print!("{}", code);
    }

    Ok(())
}

fn emit_code(module: &IRModule, target: &str) -> Result<String> {
    match target {
        "c" | "c99" => emit_c99(module),
        "rust" => emit_rust(module),
        "python" => emit_python(module),
        _ => anyhow::bail!("Unsupported target: {}", target),
    }
}

fn emit_c99(module: &IRModule) -> Result<String> {
    let mut code = String::new();
    
    code.push_str("/*\n");
    code.push_str(" * STUNIR Generated Code\n");
    code.push_str(" * Language: C99\n");
    code.push_str(&format!(" * Module: {}\n", module.module_name));
    code.push_str(" * Generator: Rust Pipeline\n");
    code.push_str(" */\n\n");
    
    code.push_str("#include <stdint.h>\n");
    code.push_str("#include <stdbool.h>\n\n");

    for func in &module.functions {
        let return_type = map_type_to_c(&func.return_type);
        code.push_str(&format!("{}\n", return_type));
        code.push_str(&format!("{}(", func.name));
        
        for (i, arg) in func.args.iter().enumerate() {
            if i > 0 {
                code.push_str(", ");
            }
            code.push_str(&format!("{} {}", map_type_to_c(&arg.arg_type), arg.name));
        }
        
        if func.args.is_empty() {
            code.push_str("void");
        }
        
        code.push_str(")\n{\n");
        
        // Generate function body from steps
        if let Some(steps) = &func.steps {
            let body = translate_steps_to_c(steps, &func.return_type, 1);
            code.push_str(&body);
            code.push_str("\n");
        } else {
            // No steps - generate stub
            let c_ret = map_type_to_c(&func.return_type);
            if c_ret == "void" {
                code.push_str("    /* TODO: implement */\n");
                code.push_str("    return;\n");
            } else {
                code.push_str("    /* TODO: implement */\n");
                code.push_str(&format!("    return {};\n", c_default_return(&func.return_type)));
            }
        }
        
        code.push_str("}\n\n");
    }

    Ok(code)
}

fn emit_rust(module: &IRModule) -> Result<String> {
    let mut code = String::new();
    
    code.push_str("//! STUNIR Generated Code\n");
    code.push_str("//! Language: Rust\n");
    code.push_str(&format!("//! Module: {}\n", module.module_name));
    code.push_str("//! Generator: Rust Pipeline\n\n");

    for func in &module.functions {
        let return_type = map_type_to_rust(&func.return_type);
        code.push_str("pub fn ");
        code.push_str(&func.name);
        code.push_str("(");
        
        for (i, arg) in func.args.iter().enumerate() {
            if i > 0 {
                code.push_str(", ");
            }
            code.push_str(&format!("{}: {}", arg.name, map_type_to_rust(&arg.arg_type)));
        }
        
        code.push_str(&format!(") -> {} {{\n", return_type));
        code.push_str("    // Function body\n");
        code.push_str("    unimplemented!()\n");
        code.push_str("}\n\n");
    }

    Ok(code)
}

fn emit_python(module: &IRModule) -> Result<String> {
    let mut code = String::new();
    
    code.push_str("\"\"\"\n");
    code.push_str("STUNIR Generated Code\n");
    code.push_str("Language: Python\n");
    code.push_str(&format!("Module: {}\n", module.module_name));
    code.push_str("Generator: Rust Pipeline\n");
    code.push_str("\"\"\"\n\n");

    for func in &module.functions {
        code.push_str(&format!("def {}(", func.name));
        
        for (i, arg) in func.args.iter().enumerate() {
            if i > 0 {
                code.push_str(", ");
            }
            code.push_str(&arg.name);
        }
        
        code.push_str("):\n");
        code.push_str("    \"\"\"Function body\"\"\"\n");
        code.push_str("    pass\n\n");
    }

    Ok(code)
}

/// Map IR type string to C type
fn map_type_to_c(type_str: &str) -> String {
    match type_str {
        "i8" => "int8_t".to_string(),
        "i16" => "int16_t".to_string(),
        "i32" => "int32_t".to_string(),
        "i64" => "int64_t".to_string(),
        "u8" => "uint8_t".to_string(),
        "u16" => "uint16_t".to_string(),
        "u32" => "uint32_t".to_string(),
        "u64" => "uint64_t".to_string(),
        "f32" => "float".to_string(),
        "f64" => "double".to_string(),
        "bool" => "bool".to_string(),
        "string" => "char*".to_string(),
        "byte[]" => "const uint8_t*".to_string(),
        "void" => "void".to_string(),
        // Pass through unknown types (e.g., custom structs, pointers)
        // This handles struct pointers correctly
        _ => type_str.to_string(),
    }
}

/// Map IR type string to Rust type
fn map_type_to_rust(type_str: &str) -> &str {
    match type_str {
        "i8" => "i8",
        "i16" => "i16",
        "i32" => "i32",
        "i64" => "i64",
        "u8" => "u8",
        "u16" => "u16",
        "u32" => "u32",
        "u64" => "u64",
        "f32" => "f32",
        "f64" => "f64",
        "bool" => "bool",
        "string" => "String",
        "void" => "()",
        _ => "()",
    }
}

/// Infer C type from a value string
fn infer_c_type_from_value(value: &str) -> &str {
    let value = value.trim();
    
    // Boolean values
    if value == "true" || value == "false" {
        return "bool";
    }
    
    // Floating point
    if value.contains('.') {
        return "double";
    }
    
    // Negative integer
    if value.starts_with('-') && value[1..].chars().all(|c| c.is_ascii_digit()) {
        return "int32_t";
    }
    
    // Positive integer
    if value.chars().all(|c| c.is_ascii_digit()) {
        if let Ok(num) = value.parse::<u32>() {
            if num <= 255 {
                return "uint8_t";
            }
        }
        return "int32_t";
    }
    
    // Default to int32_t for complex expressions
    "int32_t"
}

/// Get default return value for a type
fn c_default_return(type_str: &str) -> &str {
    match type_str {
        "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" => "0",
        "f32" | "f64" | "float" | "double" => "0.0",
        "bool" => "false",
        "string" | "char*" => "NULL",
        "void" => "",
        _ => "0",
    }
}

/// Translate IR steps to C code (with support for control flow)
fn translate_steps_to_c(steps: &[IRStep], ret_type: &str, indent: usize) -> String {
    translate_steps_to_c_internal(steps, ret_type, indent, &mut std::collections::HashMap::new())
}

/// Internal function to translate IR steps with shared variable tracking
fn translate_steps_to_c_internal(
    steps: &[IRStep], 
    ret_type: &str, 
    indent: usize,
    local_vars: &mut std::collections::HashMap<String, String>
) -> String {
    use serde_json::Value;
    
    let indent_str = "    ".repeat(indent);
    
    if steps.is_empty() {
        let c_ret = map_type_to_c(ret_type);
        if c_ret == "void" {
            return format!("{}/* Empty function body */\n{}return;", indent_str, indent_str);
        } else {
            return format!("{}/* Empty function body */\n{}return {};", indent_str, indent_str, c_default_return(ret_type));
        }
    }
    
    let mut lines = Vec::new();
    
    for step in steps {
        match step.op.as_str() {
            "assign" => {
                let target = step.target.as_deref().unwrap_or("");
                let value_str = match &step.value {
                    Some(Value::String(s)) => s.clone(),
                    Some(Value::Number(n)) => n.to_string(),
                    Some(Value::Bool(b)) => b.to_string(),
                    _ => "0".to_string(),
                };
                
                if !target.is_empty() && !local_vars.contains_key(target) {
                    let var_type = infer_c_type_from_value(&value_str);
                    local_vars.insert(target.to_string(), var_type.to_string());
                    lines.push(format!("{}{} {} = {};", indent_str, var_type, target, value_str));
                } else {
                    lines.push(format!("{}{} = {};", indent_str, target, value_str));
                }
            }
            "return" => {
                let value_str = match &step.value {
                    Some(Value::String(s)) => s.clone(),
                    Some(Value::Number(n)) => n.to_string(),
                    Some(Value::Bool(b)) => b.to_string(),
                    _ => String::new(),
                };
                
                let c_ret = map_type_to_c(ret_type);
                if !value_str.is_empty() {
                    lines.push(format!("{}return {};", indent_str, value_str));
                } else if c_ret == "void" {
                    lines.push(format!("{}return;", indent_str));
                } else {
                    lines.push(format!("{}return {};", indent_str, c_default_return(ret_type)));
                }
            }
            "call" => {
                // Handle function calls
                // Get the function call expression from value field
                // Format: "function_name(arg1, arg2, ...)"
                let call_expr = match &step.value {
                    Some(Value::String(s)) => s.as_str(),
                    _ => "unknown()"
                };
                let target = step.target.as_deref();
                
                if let Some(target_var) = target {
                    // Call with assignment
                    if !local_vars.contains_key(target_var) {
                        // Default to int32_t for function return values
                        local_vars.insert(target_var.to_string(), "int32_t".to_string());
                        lines.push(format!("{}int32_t {} = {};", indent_str, target_var, call_expr));
                    } else {
                        lines.push(format!("{}{} = {};", indent_str, target_var, call_expr));
                    }
                } else {
                    // Call without assignment
                    lines.push(format!("{}{};", indent_str, call_expr));
                }
            }
            "nop" => {
                lines.push(format!("{}/* nop */", indent_str));
            }
            "if" => {
                // If/else statement
                let condition = step.condition.as_deref().unwrap_or("true");
                lines.push(format!("{}if ({}) {{", indent_str, condition));
                
                if let Some(ref then_block) = step.then_block {
                    let then_body = translate_steps_to_c_internal(then_block, ret_type, indent + 1, local_vars);
                    lines.push(then_body);
                }
                
                if let Some(ref else_block) = step.else_block {
                    if !else_block.is_empty() {
                        lines.push(format!("{}}} else {{", indent_str));
                        let else_body = translate_steps_to_c_internal(else_block, ret_type, indent + 1, local_vars);
                        lines.push(else_body);
                    }
                }
                
                lines.push(format!("{}}}", indent_str));
            }
            "while" => {
                // While loop
                let condition = step.condition.as_deref().unwrap_or("true");
                lines.push(format!("{}while ({}) {{", indent_str, condition));
                
                if let Some(ref body) = step.body {
                    let loop_body = translate_steps_to_c_internal(body, ret_type, indent + 1, local_vars);
                    lines.push(loop_body);
                }
                
                lines.push(format!("{}}}", indent_str));
            }
            "for" => {
                // For loop
                let init = step.init.as_deref().unwrap_or("");
                let condition = step.condition.as_deref().unwrap_or("true");
                let increment = step.increment.as_deref().unwrap_or("");
                
                // Parse init to extract variable name and declare if needed
                // Format: "var = value"
                if !init.is_empty() {
                    if let Some(eq_pos) = init.find('=') {
                        let var_name = init[..eq_pos].trim();
                        if !var_name.is_empty() && !local_vars.contains_key(var_name) {
                            let init_value = init[eq_pos + 1..].trim();
                            let var_type = infer_c_type_from_value(init_value);
                            local_vars.insert(var_name.to_string(), var_type.to_string());
                            // Declare variable before for loop
                            lines.push(format!("{}{} {};", indent_str, var_type, var_name));
                        }
                    }
                }
                
                lines.push(format!("{}for ({}; {}; {}) {{", indent_str, init, condition, increment));
                
                if let Some(ref body) = step.body {
                    let loop_body = translate_steps_to_c_internal(body, ret_type, indent + 1, local_vars);
                    lines.push(loop_body);
                }
                
                lines.push(format!("{}}}", indent_str));
            }
            "break" => {
                // v0.9.0: Break statement
                lines.push(format!("{}break;", indent_str));
            }
            "continue" => {
                // v0.9.0: Continue statement
                lines.push(format!("{}continue;", indent_str));
            }
            "switch" => {
                // v0.9.0: Switch/case statement
                let expr = step.expr.as_deref().unwrap_or("0");
                lines.push(format!("{}switch ({}) {{", indent_str, expr));
                
                // Generate case labels
                if let Some(ref cases) = step.cases {
                    for case in cases {
                        let case_value = match &case.value {
                            Value::String(s) => s.clone(),
                            Value::Number(n) => n.to_string(),
                            Value::Bool(b) => b.to_string(),
                            _ => "0".to_string(),
                        };
                        lines.push(format!("{}  case {}:", indent_str, case_value));
                        
                        // Recursively translate case body with increased indentation
                        let case_code = translate_steps_to_c_internal(&case.body, ret_type, indent + 2, local_vars);
                        lines.push(case_code);
                    }
                }
                
                // Generate default case if present
                if let Some(ref default) = step.default {
                    lines.push(format!("{}  default:", indent_str));
                    let default_code = translate_steps_to_c_internal(default, ret_type, indent + 2, local_vars);
                    lines.push(default_code);
                }
                
                lines.push(format!("{}}}", indent_str));
            }
            _ => {
                lines.push(format!("{}/* UNKNOWN OP: {} */", indent_str, step.op));
            }
        }
    }
    
    // If no return statement was generated, add a default one
    // Only add at the top level (indent == 1)
    if indent == 1 && !lines.iter().any(|line| line.contains("return")) {
        let c_ret = map_type_to_c(ret_type);
        if c_ret == "void" {
            lines.push(format!("{}return;", indent_str));
        } else {
            lines.push(format!("{}return {};", indent_str, c_default_return(ret_type)));
        }
    }
    
    lines.join("\n")
}
