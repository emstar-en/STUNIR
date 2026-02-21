//! Bytecode emitters
//!
//! Supports: JVM bytecode, .NET IL, Python bytecode, WebAssembly

use crate::types::*;

/// Bytecode format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BytecodeFormat {
    JVM,
    DotNetIL,
    PythonBytecode,
    WebAssemblyBytecode,
}

impl std::fmt::Display for BytecodeFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            BytecodeFormat::JVM => write!(f, "JVM Bytecode"),
            BytecodeFormat::DotNetIL => write!(f, ".NET IL"),
            BytecodeFormat::PythonBytecode => write!(f, "Python Bytecode"),
            BytecodeFormat::WebAssemblyBytecode => write!(f, "WebAssembly Bytecode"),
        }
    }
}

/// Emit bytecode
pub fn emit(format: BytecodeFormat, class_name: &str) -> EmitterResult<String> {
    match format {
        BytecodeFormat::JVM => emit_jvm_bytecode(class_name),
        BytecodeFormat::DotNetIL => emit_dotnet_il(class_name),
        BytecodeFormat::PythonBytecode => emit_python_bytecode(class_name),
        BytecodeFormat::WebAssemblyBytecode => emit_wasm_bytecode(class_name),
    }
}

fn emit_jvm_bytecode(class_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("; STUNIR Generated JVM Bytecode\n");
    code.push_str(&format!("; Class: {}\n", class_name));
    code.push_str("; Generator: Rust Pipeline\n\n");
    
    code.push_str(&format!(".class public {}\n", class_name));
    code.push_str(".super java/lang/Object\n\n");
    
    code.push_str(".method public <init>()V\n");
    code.push_str("    .limit stack 1\n");
    code.push_str("    .limit locals 1\n");
    code.push_str("    aload_0\n");
    code.push_str("    invokespecial java/lang/Object/<init>()V\n");
    code.push_str("    return\n");
    code.push_str(".end method\n\n");
    
    code.push_str(".method public static main([Ljava/lang/String;)V\n");
    code.push_str("    .limit stack 2\n");
    code.push_str("    .limit locals 1\n");
    code.push_str("    getstatic java/lang/System/out Ljava/io/PrintStream;\n");
    code.push_str("    ldc \"STUNIR Generated\"\n");
    code.push_str("    invokevirtual java/io/PrintStream/println(Ljava/lang/String;)V\n");
    code.push_str("    return\n");
    code.push_str(".end method\n");
    
    Ok(code)
}

fn emit_dotnet_il(class_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("// STUNIR Generated .NET IL\n");
    code.push_str(&format!("// Class: {}\n", class_name));
    code.push_str("// Generator: Rust Pipeline\n\n");
    
    code.push_str(".assembly extern mscorlib {}\n");
    code.push_str(&format!(".assembly {} {{}}\n\n", class_name));
    
    code.push_str(&format!(".class public auto ansi beforefieldinit {}\n", class_name));
    code.push_str("       extends [mscorlib]System.Object\n");
    code.push_str("{\n");
    code.push_str("    .method public hidebysig static void Main() cil managed\n");
    code.push_str("    {\n");
    code.push_str("        .entrypoint\n");
    code.push_str("        .maxstack 1\n");
    code.push_str("        ldstr \"STUNIR Generated\"\n");
    code.push_str("        call void [mscorlib]System.Console::WriteLine(string)\n");
    code.push_str("        ret\n");
    code.push_str("    }\n");
    code.push_str("}\n");
    
    Ok(code)
}

fn emit_python_bytecode(class_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("# STUNIR Generated Python Bytecode (Human-readable)\n");
    code.push_str(&format!("# Module: {}\n", class_name));
    code.push_str("# Generator: Rust Pipeline\n\n");
    
    code.push_str("# Bytecode disassembly:\n");
    code.push_str("# 1           0 LOAD_CONST               0 ('STUNIR Generated')\n");
    code.push_str("#             2 PRINT_ITEM\n");
    code.push_str("#             3 PRINT_NEWLINE\n");
    code.push_str("#             4 LOAD_CONST               1 (None)\n");
    code.push_str("#             6 RETURN_VALUE\n");
    
    Ok(code)
}

fn emit_wasm_bytecode(class_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str(";; STUNIR Generated WebAssembly Bytecode\n");
    code.push_str(&format!(";; Module: {}\n", class_name));
    code.push_str(";; Generator: Rust Pipeline\n\n");
    
    code.push_str("(module\n");
    code.push_str("  (func $main (result i32)\n");
    code.push_str("    i32.const 42\n");
    code.push_str("  )\n");
    code.push_str("  (export \"main\" (func $main))\n");
    code.push_str(")\n");
    
    Ok(code)
}
