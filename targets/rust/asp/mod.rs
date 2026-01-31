//! Answer Set Programming emitters
//!
//! Supports: Clingo, DLV, ASP-Core-2

use crate::types::*;

/// ASP dialect
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ASPDialect {
    Clingo,
    DLV,
    ASPCore2,
}

impl std::fmt::Display for ASPDialect {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ASPDialect::Clingo => write!(f, "Clingo"),
            ASPDialect::DLV => write!(f, "DLV"),
            ASPDialect::ASPCore2 => write!(f, "ASP-Core-2"),
        }
    }
}

/// Emit ASP code
pub fn emit(dialect: ASPDialect, program_name: &str) -> EmitterResult<String> {
    match dialect {
        ASPDialect::Clingo => emit_clingo(program_name),
        ASPDialect::DLV => emit_dlv(program_name),
        ASPDialect::ASPCore2 => emit_aspcore2(program_name),
    }
}

fn emit_clingo(program_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("% STUNIR Generated Clingo Program\n");
    code.push_str(&format!("% Program: {}\n", program_name));
    code.push_str("% Generator: Rust Pipeline\n\n");
    
    code.push_str("% Domain\n");
    code.push_str("node(1..5).\n");
    code.push_str("edge(1,2). edge(2,3). edge(3,4). edge(4,5).\n\n");
    
    code.push_str("% Graph coloring\n");
    code.push_str("color(red; blue; green).\n\n");
    
    code.push_str("% Choice rule: each node gets exactly one color\n");
    code.push_str("1 { colored(N, C) : color(C) } 1 :- node(N).\n\n");
    
    code.push_str("% Constraint: adjacent nodes must have different colors\n");
    code.push_str(":- edge(N1, N2), colored(N1, C), colored(N2, C).\n\n");
    
    code.push_str("#show colored/2.\n");
    
    Ok(code)
}

fn emit_dlv(program_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("% STUNIR Generated DLV Program\n");
    code.push_str(&format!("% Program: {}\n", program_name));
    code.push_str("% Generator: Rust Pipeline\n\n");
    
    code.push_str("% Facts\n");
    code.push_str("node(1). node(2). node(3). node(4). node(5).\n");
    code.push_str("edge(1,2). edge(2,3). edge(3,4). edge(4,5).\n\n");
    
    code.push_str("% Colors\n");
    code.push_str("color(red). color(blue). color(green).\n\n");
    
    code.push_str("% Guess coloring\n");
    code.push_str("colored(N, C) v not_colored(N, C) :- node(N), color(C).\n\n");
    
    code.push_str("% Exactly one color per node\n");
    code.push_str(":- node(N), #count{C: colored(N, C)} != 1.\n\n");
    
    code.push_str("% Adjacent nodes have different colors\n");
    code.push_str(":- edge(N1, N2), colored(N1, C), colored(N2, C).\n");
    
    Ok(code)
}

fn emit_aspcore2(program_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("% STUNIR Generated ASP-Core-2 Program\n");
    code.push_str(&format!("% Program: {}\n", program_name));
    code.push_str("% Generator: Rust Pipeline\n\n");
    
    code.push_str("% Input\n");
    code.push_str("#const n=5.\n");
    code.push_str("node(1..n).\n");
    code.push_str("edge(1,2; 2,3; 3,4; 4,5).\n\n");
    
    code.push_str("% Colors\n");
    code.push_str("color(red; blue; green).\n\n");
    
    code.push_str("% Generate: assign colors\n");
    code.push_str("{colored(N,C): color(C)} = 1 :- node(N).\n\n");
    
    code.push_str("% Test: no adjacent nodes with same color\n");
    code.push_str(":- edge(N1,N2), colored(N1,C), colored(N2,C).\n\n");
    
    code.push_str("% Output\n");
    code.push_str("#show colored/2.\n");
    
    Ok(code)
}
