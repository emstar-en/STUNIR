//! Expert system emitters
//!
//! Supports: CLIPS, Jess, Drools, rule-based systems

use crate::types::*;

/// Expert system type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertSystemType {
    CLIPS,
    Jess,
    Drools,
    Generic,
}

impl std::fmt::Display for ExpertSystemType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ExpertSystemType::CLIPS => write!(f, "CLIPS"),
            ExpertSystemType::Jess => write!(f, "Jess"),
            ExpertSystemType::Drools => write!(f, "Drools"),
            ExpertSystemType::Generic => write!(f, "Generic Rule System"),
        }
    }
}

/// Emit expert system code
pub fn emit(system_type: ExpertSystemType, module_name: &str) -> EmitterResult<String> {
    match system_type {
        ExpertSystemType::CLIPS => emit_clips(module_name),
        ExpertSystemType::Jess => emit_jess(module_name),
        ExpertSystemType::Drools => emit_drools(module_name),
        ExpertSystemType::Generic => emit_generic(module_name),
    }
}

fn emit_clips(module_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("; STUNIR Generated CLIPS Rules\n");
    code.push_str(&format!("; Module: {}\n", module_name));
    code.push_str("; Generator: Rust Pipeline\n\n");
    
    code.push_str(&format!("(defmodule {}\n", module_name));
    code.push_str("    \"STUNIR generated expert system\")\n\n");
    
    code.push_str("(deftemplate fact\n");
    code.push_str("    (slot name (type STRING))\n");
    code.push_str("    (slot value (type NUMBER)))\n\n");
    
    code.push_str("(defrule example-rule\n");
    code.push_str("    \"Example rule\"\n");
    code.push_str("    (fact (name ?name) (value ?v&:(> ?v 100)))\n");
    code.push_str("    =>\n");
    code.push_str("    (printout t \"High value detected: \" ?v crlf))\n");
    
    Ok(code)
}

fn emit_jess(module_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("; STUNIR Generated Jess Rules\n");
    code.push_str(&format!("; Module: {}\n", module_name));
    code.push_str("; Generator: Rust Pipeline\n\n");
    
    code.push_str("(deftemplate fact\n");
    code.push_str("    (slot name)\n");
    code.push_str("    (slot value))\n\n");
    
    code.push_str("(defrule example-rule\n");
    code.push_str("    (fact (name ?name) (value ?v&:(> ?v 100)))\n");
    code.push_str("    =>\n");
    code.push_str("    (printout t \"High value detected: \" ?v crlf))\n");
    
    Ok(code)
}

fn emit_drools(module_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("// STUNIR Generated Drools Rules\n");
    code.push_str(&format!("// Module: {}\n", module_name));
    code.push_str("// Generator: Rust Pipeline\n\n");
    
    code.push_str(&format!("package com.stunir.{};\n\n", module_name.to_lowercase()));
    
    code.push_str("import com.stunir.model.Fact;\n\n");
    
    code.push_str("rule \"Example Rule\"\n");
    code.push_str("    when\n");
    code.push_str("        $fact : Fact( value > 100 )\n");
    code.push_str("    then\n");
    code.push_str("        System.out.println(\"High value detected: \" + $fact.getValue());\n");
    code.push_str("end\n");
    
    Ok(code)
}

fn emit_generic(module_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("# STUNIR Generated Rule System\n");
    code.push_str(&format!("# Module: {}\n", module_name));
    code.push_str("# Generator: Rust Pipeline\n\n");
    
    code.push_str("rules:\n");
    code.push_str("  - name: example_rule\n");
    code.push_str("    condition:\n");
    code.push_str("      field: value\n");
    code.push_str("      operator: greater_than\n");
    code.push_str("      threshold: 100\n");
    code.push_str("    action:\n");
    code.push_str("      type: alert\n");
    code.push_str("      message: High value detected\n");
    
    Ok(code)
}
