//! Grammar definition emitters
//!
//! Supports: ANTLR, Yacc/Bison, PEG, EBNF

use crate::types::*;

/// Grammar format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrammarFormat {
    ANTLR,
    Yacc,
    Bison,
    PEG,
    EBNF,
}

impl std::fmt::Display for GrammarFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            GrammarFormat::ANTLR => write!(f, "ANTLR"),
            GrammarFormat::Yacc => write!(f, "Yacc"),
            GrammarFormat::Bison => write!(f, "Bison"),
            GrammarFormat::PEG => write!(f, "PEG"),
            GrammarFormat::EBNF => write!(f, "EBNF"),
        }
    }
}

/// Emit grammar code
pub fn emit(format: GrammarFormat, grammar_name: &str) -> EmitterResult<String> {
    match format {
        GrammarFormat::ANTLR => emit_antlr(grammar_name),
        GrammarFormat::Yacc => emit_yacc(grammar_name),
        GrammarFormat::Bison => emit_bison(grammar_name),
        GrammarFormat::PEG => emit_peg(grammar_name),
        GrammarFormat::EBNF => emit_ebnf(grammar_name),
    }
}

fn emit_antlr(grammar_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("// STUNIR Generated ANTLR Grammar\n");
    code.push_str(&format!("// Grammar: {}\n", grammar_name));
    code.push_str("// Generator: Rust Pipeline\n\n");
    
    code.push_str(&format!("grammar {};\n\n", grammar_name));
    
    code.push_str("// Parser rules\n");
    code.push_str("program: statement+ EOF;\n\n");
    code.push_str("statement\n");
    code.push_str("    : expr SEMICOLON\n");
    code.push_str("    | IF LPAREN expr RPAREN block\n");
    code.push_str("    ;\n\n");
    
    code.push_str("expr\n");
    code.push_str("    : expr (MUL | DIV) expr\n");
    code.push_str("    | expr (ADD | SUB) expr\n");
    code.push_str("    | INT\n");
    code.push_str("    | ID\n");
    code.push_str("    ;\n\n");
    
    code.push_str("block: LBRACE statement* RBRACE;\n\n");
    
    code.push_str("// Lexer rules\n");
    code.push_str("IF: 'if';\n");
    code.push_str("INT: [0-9]+;\n");
    code.push_str("ID: [a-zA-Z_][a-zA-Z0-9_]*;\n");
    code.push_str("ADD: '+'; SUB: '-'; MUL: '*'; DIV: '/';\n");
    code.push_str("LPAREN: '('; RPAREN: ')';\n");
    code.push_str("LBRACE: '{'; RBRACE: '}';\n");
    code.push_str("SEMICOLON: ';';\n");
    code.push_str("WS: [ \\t\\r\\n]+ -> skip;\n");
    
    Ok(code)
}

fn emit_yacc(grammar_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("/* STUNIR Generated Yacc Grammar */\n");
    code.push_str(&format!("/* Grammar: {} */\n", grammar_name));
    code.push_str("/* Generator: Rust Pipeline */\n\n");
    
    code.push_str("%{\n");
    code.push_str("#include <stdio.h>\n");
    code.push_str("int yylex();\n");
    code.push_str("void yyerror(const char *s);\n");
    code.push_str("%}\n\n");
    
    code.push_str("%token INT ID IF\n");
    code.push_str("%left '+' '-'\n");
    code.push_str("%left '*' '/'\n\n");
    
    code.push_str("%%\n\n");
    
    code.push_str("program: statement_list\n");
    code.push_str("    ;\n\n");
    
    code.push_str("statement_list\n");
    code.push_str("    : statement\n");
    code.push_str("    | statement_list statement\n");
    code.push_str("    ;\n\n");
    
    code.push_str("statement\n");
    code.push_str("    : expr ';'\n");
    code.push_str("    ;\n\n");
    
    code.push_str("expr\n");
    code.push_str("    : expr '+' expr\n");
    code.push_str("    | expr '-' expr\n");
    code.push_str("    | expr '*' expr\n");
    code.push_str("    | expr '/' expr\n");
    code.push_str("    | INT\n");
    code.push_str("    | ID\n");
    code.push_str("    ;\n\n");
    
    code.push_str("%%\n");
    
    Ok(code)
}

fn emit_bison(grammar_name: &str) -> EmitterResult<String> {
    // Bison is very similar to Yacc
    emit_yacc(grammar_name)
}

fn emit_peg(grammar_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("# STUNIR Generated PEG Grammar\n");
    code.push_str(&format!("# Grammar: {}\n", grammar_name));
    code.push_str("# Generator: Rust Pipeline\n\n");
    
    code.push_str("program = statement+\n\n");
    code.push_str("statement = expr \";\";\n\n");
    code.push_str("expr = term ((\"\" + / \"-\") term)*\n\n");
    code.push_str("term = factor ((\"*\" / \"/\") factor)*\n\n");
    code.push_str("factor = number / identifier / \"(\" expr \")\"\n\n");
    code.push_str("number = [0-9]+\n\n");
    code.push_str("identifier = [a-zA-Z_][a-zA-Z0-9_]*\n\n");
    code.push_str("whitespace = [ \\t\\n\\r]*\n");
    
    Ok(code)
}

fn emit_ebnf(grammar_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("(* STUNIR Generated EBNF Grammar *)\n");
    code.push_str(&format!("(* Grammar: {} *)\n", grammar_name));
    code.push_str("(* Generator: Rust Pipeline *)\n\n");
    
    code.push_str("program = { statement } ;\n\n");
    code.push_str("statement = expression \";\" ;\n\n");
    code.push_str("expression = term { ( \"+\" | \"-\" ) term } ;\n\n");
    code.push_str("term = factor { ( \"*\" | \"/\" ) factor } ;\n\n");
    code.push_str("factor = number | identifier | \"(\" expression \")\" ;\n\n");
    code.push_str("number = digit { digit } ;\n\n");
    code.push_str("identifier = letter { letter | digit | \"_\" } ;\n\n");
    code.push_str("letter = \"A\" | \"B\" | \"C\" | \"D\" | \"E\" | \"F\" | \"G\"\n");
    code.push_str("       | \"H\" | \"I\" | \"J\" | \"K\" | \"L\" | \"M\" | \"N\"\n");
    code.push_str("       | \"O\" | \"P\" | \"Q\" | \"R\" | \"S\" | \"T\" | \"U\"\n");
    code.push_str("       | \"V\" | \"W\" | \"X\" | \"Y\" | \"Z\"\n");
    code.push_str("       | \"a\" | \"b\" | \"c\" | \"d\" | \"e\" | \"f\" | \"g\"\n");
    code.push_str("       | \"h\" | \"i\" | \"j\" | \"k\" | \"l\" | \"m\" | \"n\"\n");
    code.push_str("       | \"o\" | \"p\" | \"q\" | \"r\" | \"s\" | \"t\" | \"u\"\n");
    code.push_str("       | \"v\" | \"w\" | \"x\" | \"y\" | \"z\" ;\n\n");
    code.push_str("digit = \"0\" | \"1\" | \"2\" | \"3\" | \"4\" | \"5\" | \"6\" | \"7\" | \"8\" | \"9\" ;\n");
    
    Ok(code)
}
