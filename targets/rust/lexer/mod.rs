//! Lexer code emitters
//!
//! Supports: Python lexers, Rust lexers, C lexers

use crate::types::*;

/// Lexer target language
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LexerTarget {
    Python,
    Rust,
    C,
    TableDriven,
}

impl std::fmt::Display for LexerTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            LexerTarget::Python => write!(f, "Python"),
            LexerTarget::Rust => write!(f, "Rust"),
            LexerTarget::C => write!(f, "C"),
            LexerTarget::TableDriven => write!(f, "Table-Driven"),
        }
    }
}

/// Emit lexer code
pub fn emit(target: LexerTarget, lexer_name: &str) -> EmitterResult<String> {
    match target {
        LexerTarget::Python => emit_python_lexer(lexer_name),
        LexerTarget::Rust => emit_rust_lexer(lexer_name),
        LexerTarget::C => emit_c_lexer(lexer_name),
        LexerTarget::TableDriven => emit_table_driven(lexer_name),
    }
}

fn emit_python_lexer(lexer_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("#!/usr/bin/env python3\n");
    code.push_str("# STUNIR Generated Python Lexer\n");
    code.push_str(&format!("# Lexer: {}\n", lexer_name));
    code.push_str("# Generator: Rust Pipeline\n\n");
    
    code.push_str("import re\n");
    code.push_str("from dataclasses import dataclass\n");
    code.push_str("from typing import List, Tuple\n\n");
    
    code.push_str("@dataclass\n");
    code.push_str("class Token:\n");
    code.push_str("    type: str\n");
    code.push_str("    value: str\n");
    code.push_str("    line: int\n");
    code.push_str("    column: int\n\n");
    
    code.push_str(&format!("class {}:\n", lexer_name));
    code.push_str("    def __init__(self, text: str):\n");
    code.push_str("        self.text = text\n");
    code.push_str("        self.pos = 0\n");
    code.push_str("        self.tokens = []\n\n");
    
    code.push_str("    def tokenize(self) -> List[Token]:\n");
    code.push_str("        # Tokenization logic here\n");
    code.push_str("        return self.tokens\n");
    
    Ok(code)
}

fn emit_rust_lexer(lexer_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("// STUNIR Generated Rust Lexer\n");
    code.push_str(&format!("// Lexer: {}\n", lexer_name));
    code.push_str("// Generator: Rust Pipeline\n\n");
    
    code.push_str("#[derive(Debug, Clone)]\n");
    code.push_str("pub enum TokenType {\n");
    code.push_str("    Int,\n");
    code.push_str("    Id,\n");
    code.push_str("    Operator,\n");
    code.push_str("    Eof,\n");
    code.push_str("}\n\n");
    
    code.push_str("#[derive(Debug, Clone)]\n");
    code.push_str("pub struct Token {\n");
    code.push_str("    pub token_type: TokenType,\n");
    code.push_str("    pub value: String,\n");
    code.push_str("    pub line: usize,\n");
    code.push_str("    pub column: usize,\n");
    code.push_str("}\n\n");
    
    code.push_str(&format!("pub struct {} {{\n", lexer_name));
    code.push_str("    text: String,\n");
    code.push_str("    pos: usize,\n");
    code.push_str("    tokens: Vec<Token>,\n");
    code.push_str("}\n\n");
    
    code.push_str(&format!("impl {} {{\n", lexer_name));
    code.push_str("    pub fn new(text: String) -> Self {\n");
    code.push_str(&format!("        {} {{ text, pos: 0, tokens: Vec::new() }}\n", lexer_name));
    code.push_str("    }\n\n");
    
    code.push_str("    pub fn tokenize(&mut self) -> Result<Vec<Token>, String> {\n");
    code.push_str("        // Tokenization logic here\n");
    code.push_str("        Ok(self.tokens.clone())\n");
    code.push_str("    }\n");
    code.push_str("}\n");
    
    Ok(code)
}

fn emit_c_lexer(lexer_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("/* STUNIR Generated C Lexer */\n");
    code.push_str(&format!("/* Lexer: {} */\n", lexer_name));
    code.push_str("/* Generator: Rust Pipeline */\n\n");
    
    code.push_str("#include <stdint.h>\n");
    code.push_str("#include <stdbool.h>\n\n");
    
    code.push_str("typedef enum {\n");
    code.push_str("    TOKEN_INT,\n");
    code.push_str("    TOKEN_ID,\n");
    code.push_str("    TOKEN_OPERATOR,\n");
    code.push_str("    TOKEN_EOF\n");
    code.push_str("} TokenType;\n\n");
    
    code.push_str("typedef struct {\n");
    code.push_str("    TokenType type;\n");
    code.push_str("    const char *value;\n");
    code.push_str("    uint32_t line;\n");
    code.push_str("    uint32_t column;\n");
    code.push_str("} Token;\n\n");
    
    code.push_str("typedef struct {\n");
    code.push_str("    const char *text;\n");
    code.push_str("    uint32_t pos;\n");
    code.push_str("    Token *tokens;\n");
    code.push_str("    uint32_t token_count;\n");
    code.push_str(&format!("}} {};\n", lexer_name));
    
    Ok(code)
}

fn emit_table_driven(lexer_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("{\n");
    code.push_str(&format!("  \"lexer\": \"{}\",\n", lexer_name));
    code.push_str("  \"generator\": \"Rust Pipeline\",\n");
    code.push_str("  \"states\": [\n");
    code.push_str("    {\"id\": 0, \"name\": \"start\", \"initial\": true},\n");
    code.push_str("    {\"id\": 1, \"name\": \"in_number\"},\n");
    code.push_str("    {\"id\": 2, \"name\": \"in_identifier\"}\n");
    code.push_str("  ],\n");
    code.push_str("  \"transitions\": [\n");
    code.push_str("    {\"from\": 0, \"to\": 1, \"condition\": \"digit\"},\n");
    code.push_str("    {\"from\": 0, \"to\": 2, \"condition\": \"letter\"}\n");
    code.push_str("  ],\n");
    code.push_str("  \"tokens\": [\n");
    code.push_str("    {\"type\": \"INT\", \"pattern\": \"[0-9]+\"},\n");
    code.push_str("    {\"type\": \"ID\", \"pattern\": \"[a-zA-Z_][a-zA-Z0-9_]*\"}\n");
    code.push_str("  ]\n");
    code.push_str("}\n");
    
    Ok(code)
}
