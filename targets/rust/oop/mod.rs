//! Object-Oriented Programming emitters
//!
//! Supports: Java, C++, C#, Python OOP, TypeScript

use crate::types::*;

/// OOP language
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OOPLanguage {
    Java,
    CPlusPlus,
    CSharp,
    PythonOOP,
    TypeScript,
}

impl std::fmt::Display for OOPLanguage {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            OOPLanguage::Java => write!(f, "Java"),
            OOPLanguage::CPlusPlus => write!(f, "C++"),
            OOPLanguage::CSharp => write!(f, "C#"),
            OOPLanguage::PythonOOP => write!(f, "Python OOP"),
            OOPLanguage::TypeScript => write!(f, "TypeScript"),
        }
    }
}

/// Emit OOP code
pub fn emit(language: OOPLanguage, class_name: &str) -> EmitterResult<String> {
    match language {
        OOPLanguage::Java => emit_java(class_name),
        OOPLanguage::CPlusPlus => emit_cpp(class_name),
        OOPLanguage::CSharp => emit_csharp(class_name),
        OOPLanguage::PythonOOP => emit_python_oop(class_name),
        OOPLanguage::TypeScript => emit_typescript(class_name),
    }
}

fn emit_java(class_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("// STUNIR Generated Java\n");
    code.push_str(&format!("// Class: {}\n", class_name));
    code.push_str("// Generator: Rust Pipeline\n\n");
    
    code.push_str(&format!("public class {} {{\n", class_name));
    code.push_str("    private int value;\n\n");
    
    code.push_str(&format!("    public {}(int value) {{\n", class_name));
    code.push_str("        this.value = value;\n");
    code.push_str("    }\n\n");
    
    code.push_str("    public int getValue() {\n");
    code.push_str("        return value;\n");
    code.push_str("    }\n\n");
    
    code.push_str("    public void setValue(int value) {\n");
    code.push_str("        this.value = value;\n");
    code.push_str("    }\n");
    code.push_str("}\n");
    
    Ok(code)
}

fn emit_cpp(class_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("// STUNIR Generated C++\n");
    code.push_str(&format!("// Class: {}\n", class_name));
    code.push_str("// Generator: Rust Pipeline\n\n");
    
    code.push_str("#include <iostream>\n\n");
    
    code.push_str(&format!("class {} {{\n", class_name));
    code.push_str("private:\n");
    code.push_str("    int value;\n\n");
    
    code.push_str("public:\n");
    code.push_str(&format!("    {}(int val) : value(val) {{}}\n\n", class_name));
    
    code.push_str("    int getValue() const {\n");
    code.push_str("        return value;\n");
    code.push_str("    }\n\n");
    
    code.push_str("    void setValue(int val) {\n");
    code.push_str("        value = val;\n");
    code.push_str("    }\n");
    code.push_str("};\n");
    
    Ok(code)
}

fn emit_csharp(class_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("// STUNIR Generated C#\n");
    code.push_str(&format!("// Class: {}\n", class_name));
    code.push_str("// Generator: Rust Pipeline\n\n");
    
    code.push_str(&format!("public class {}\n", class_name));
    code.push_str("{\n");
    code.push_str("    private int value;\n\n");
    
    code.push_str(&format!("    public {}(int value)\n", class_name));
    code.push_str("    {\n");
    code.push_str("        this.value = value;\n");
    code.push_str("    }\n\n");
    
    code.push_str("    public int Value\n");
    code.push_str("    {\n");
    code.push_str("        get { return value; }\n");
    code.push_str("        set { this.value = value; }\n");
    code.push_str("    }\n");
    code.push_str("}\n");
    
    Ok(code)
}

fn emit_python_oop(class_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("# STUNIR Generated Python OOP\n");
    code.push_str(&format!("# Class: {}\n", class_name));
    code.push_str("# Generator: Rust Pipeline\n\n");
    
    code.push_str(&format!("class {}:\n", class_name));
    code.push_str("    def __init__(self, value: int):\n");
    code.push_str("        self._value = value\n\n");
    
    code.push_str("    @property\n");
    code.push_str("    def value(self) -> int:\n");
    code.push_str("        return self._value\n\n");
    
    code.push_str("    @value.setter\n");
    code.push_str("    def value(self, value: int):\n");
    code.push_str("        self._value = value\n");
    
    Ok(code)
}

fn emit_typescript(class_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("// STUNIR Generated TypeScript\n");
    code.push_str(&format!("// Class: {}\n", class_name));
    code.push_str("// Generator: Rust Pipeline\n\n");
    
    code.push_str(&format!("class {} {{\n", class_name));
    code.push_str("    private value: number;\n\n");
    
    code.push_str("    constructor(value: number) {\n");
    code.push_str("        this.value = value;\n");
    code.push_str("    }\n\n");
    
    code.push_str("    getValue(): number {\n");
    code.push_str("        return this.value;\n");
    code.push_str("    }\n\n");
    
    code.push_str("    setValue(value: number): void {\n");
    code.push_str("        this.value = value;\n");
    code.push_str("    }\n");
    code.push_str("}\n");
    
    Ok(code)
}
