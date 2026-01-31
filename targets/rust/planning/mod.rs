//! Planning and scheduling emitters
//!
//! Supports: PDDL, STRIPS, HTN, Timeline planning

use crate::types::*;

/// Planning language
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanningLanguage {
    PDDL,
    STRIPS,
    HTN,
    Timeline,
}

impl std::fmt::Display for PlanningLanguage {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            PlanningLanguage::PDDL => write!(f, "PDDL"),
            PlanningLanguage::STRIPS => write!(f, "STRIPS"),
            PlanningLanguage::HTN => write!(f, "HTN"),
            PlanningLanguage::Timeline => write!(f, "Timeline"),
        }
    }
}

/// Emit planning code
pub fn emit(language: PlanningLanguage, domain_name: &str) -> EmitterResult<String> {
    match language {
        PlanningLanguage::PDDL => emit_pddl(domain_name),
        PlanningLanguage::STRIPS => emit_strips(domain_name),
        PlanningLanguage::HTN => emit_htn(domain_name),
        PlanningLanguage::Timeline => emit_timeline(domain_name),
    }
}

fn emit_pddl(domain_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("; STUNIR Generated PDDL\n");
    code.push_str(&format!("; Domain: {}\n", domain_name));
    code.push_str("; Generator: Rust Pipeline\n\n");
    
    code.push_str(&format!("(define (domain {})\n", domain_name));
    code.push_str("  (:requirements :strips :typing)\n\n");
    
    code.push_str("  (:types\n");
    code.push_str("    location object - object\n");
    code.push_str("  )\n\n");
    
    code.push_str("  (:predicates\n");
    code.push_str("    (at ?obj - object ?loc - location)\n");
    code.push_str("    (connected ?from ?to - location)\n");
    code.push_str("  )\n\n");
    
    code.push_str("  (:action move\n");
    code.push_str("    :parameters (?obj - object ?from ?to - location)\n");
    code.push_str("    :precondition (and\n");
    code.push_str("      (at ?obj ?from)\n");
    code.push_str("      (connected ?from ?to)\n");
    code.push_str("    )\n");
    code.push_str("    :effect (and\n");
    code.push_str("      (not (at ?obj ?from))\n");
    code.push_str("      (at ?obj ?to)\n");
    code.push_str("    )\n");
    code.push_str("  )\n");
    code.push_str(")\n");
    
    Ok(code)
}

fn emit_strips(domain_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("; STUNIR Generated STRIPS\n");
    code.push_str(&format!("; Domain: {}\n", domain_name));
    code.push_str("; Generator: Rust Pipeline\n\n");
    
    code.push_str(&format!("(define-domain {}\n", domain_name));
    code.push_str("  (operators\n");
    code.push_str("    (move\n");
    code.push_str("      (params ?obj ?from ?to)\n");
    code.push_str("      (preconds (at ?obj ?from) (connected ?from ?to))\n");
    code.push_str("      (effects (at ?obj ?to) (not (at ?obj ?from)))\n");
    code.push_str("    )\n");
    code.push_str("  )\n");
    code.push_str(")\n");
    
    Ok(code)
}

fn emit_htn(domain_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("; STUNIR Generated HTN\n");
    code.push_str(&format!("; Domain: {}\n", domain_name));
    code.push_str("; Generator: Rust Pipeline\n\n");
    
    code.push_str(&format!("(defdomain {}\n", domain_name));
    code.push_str("  (:method transport\n");
    code.push_str("    :parameters (?obj ?from ?to)\n");
    code.push_str("    :task (deliver ?obj ?to)\n");
    code.push_str("    :ordered-subtasks (\n");
    code.push_str("      (pickup ?obj ?from)\n");
    code.push_str("      (move ?from ?to)\n");
    code.push_str("      (putdown ?obj ?to)\n");
    code.push_str("    )\n");
    code.push_str("  )\n");
    code.push_str(")\n");
    
    Ok(code)
}

fn emit_timeline(domain_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("# STUNIR Generated Timeline Planning\n");
    code.push_str(&format!("# Domain: {}\n", domain_name));
    code.push_str("# Generator: Rust Pipeline\n\n");
    
    code.push_str(&format!("domain: {}\n\n", domain_name));
    code.push_str("timelines:\n");
    code.push_str("  - name: robot_location\n");
    code.push_str("    type: state_variable\n");
    code.push_str("    values: [loc_a, loc_b, loc_c]\n\n");
    
    code.push_str("transitions:\n");
    code.push_str("  - from: loc_a\n");
    code.push_str("    to: loc_b\n");
    code.push_str("    duration: 10\n");
    code.push_str("    cost: 5\n");
    
    Ok(code)
}
