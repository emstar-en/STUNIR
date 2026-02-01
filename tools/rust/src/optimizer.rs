//! STUNIR IR Optimizer - Rust implementation (v0.8.9+)
//!
//! Provides optimization passes for STUNIR IR:
//! - Dead code elimination
//! - Constant folding
//! - Constant propagation
//! - Unreachable code elimination

use crate::types::{IRFunction, IRModule, IRStep, OptimizationHint};
use regex::Regex;
use std::collections::HashMap;

/// Optimization pass trait
pub trait OptimizationPass {
    fn name(&self) -> &'static str;
    fn optimize_module(&self, module: &mut IRModule) -> usize;
}

/// Dead code elimination pass
pub struct DeadCodeElimination;

impl OptimizationPass for DeadCodeElimination {
    fn name(&self) -> &'static str {
        "dead_code_elimination"
    }

    fn optimize_module(&self, module: &mut IRModule) -> usize {
        let mut changes = 0;
        for func in &mut module.functions {
            if let Some(ref mut steps) = func.steps {
                let (new_steps, c) = eliminate_dead_code(steps.clone());
                *steps = new_steps;
                changes += c;
            }
        }
        changes
    }
}

fn eliminate_dead_code(steps: Vec<IRStep>) -> (Vec<IRStep>, usize) {
    let mut changes = 0;
    let mut new_steps = Vec::new();
    let mut seen_return = false;

    for step in steps {
        // Skip steps after unconditional return
        if seen_return {
            changes += 1;
            continue;
        }

        // Skip nop operations
        if step.op == "nop" {
            changes += 1;
            continue;
        }

        // Skip steps marked as dead code
        if let Some(ref opt) = step.optimization {
            if opt.dead_code.unwrap_or(false) {
                changes += 1;
                continue;
            }
        }

        // Process nested blocks
        let mut new_step = step.clone();

        if let Some(ref then_block) = new_step.then_block {
            let (new_then, c) = eliminate_dead_code(then_block.clone());
            new_step.then_block = Some(new_then);
            changes += c;
        }

        if let Some(ref else_block) = new_step.else_block {
            let (new_else, c) = eliminate_dead_code(else_block.clone());
            new_step.else_block = Some(new_else);
            changes += c;
        }

        if let Some(ref body) = new_step.body {
            let (new_body, c) = eliminate_dead_code(body.clone());
            new_step.body = Some(new_body);
            changes += c;
        }

        if let Some(ref try_block) = new_step.try_block {
            let (new_try, c) = eliminate_dead_code(try_block.clone());
            new_step.try_block = Some(new_try);
            changes += c;
        }

        if let Some(ref catch_blocks) = new_step.catch_blocks {
            let mut new_catches = Vec::new();
            for catch in catch_blocks {
                let (new_body, c) = eliminate_dead_code(catch.body.clone());
                let mut new_catch = catch.clone();
                new_catch.body = new_body;
                new_catches.push(new_catch);
                changes += c;
            }
            new_step.catch_blocks = Some(new_catches);
        }

        if let Some(ref finally_block) = new_step.finally_block {
            let (new_finally, c) = eliminate_dead_code(finally_block.clone());
            new_step.finally_block = Some(new_finally);
            changes += c;
        }

        new_steps.push(new_step);

        // Mark that we've seen an unconditional return
        if step.op == "return" {
            seen_return = true;
        }
    }

    (new_steps, changes)
}

/// Constant folding pass
pub struct ConstantFolding;

impl OptimizationPass for ConstantFolding {
    fn name(&self) -> &'static str {
        "constant_folding"
    }

    fn optimize_module(&self, module: &mut IRModule) -> usize {
        let mut changes = 0;
        for func in &mut module.functions {
            if let Some(ref mut steps) = func.steps {
                let (new_steps, c) = fold_constants(steps.clone());
                *steps = new_steps;
                changes += c;
            }
        }
        changes
    }
}

fn fold_constants(steps: Vec<IRStep>) -> (Vec<IRStep>, usize) {
    let mut changes = 0;
    let mut new_steps = Vec::new();

    for step in steps {
        let mut new_step = step.clone();

        // Fold constant assignments
        if step.op == "assign" {
            if let Some(ref value) = step.value {
                if let Some(s) = value.as_str() {
                    if let Some(folded) = try_fold_expression(s) {
                        new_step.value = Some(serde_json::json!(folded));
                        let mut opt = new_step.optimization.unwrap_or_default();
                        opt.const_eval = Some(true);
                        opt.constant_value = Some(serde_json::json!(folded));
                        new_step.optimization = Some(opt);
                        changes += 1;
                    }
                }
            }
        }

        // Fold constant conditions
        if let Some(ref cond) = step.condition {
            if let Some(folded) = try_fold_expression(cond) {
                if folded == "true" || folded == "false" {
                    new_step.condition = Some(folded);
                    changes += 1;
                }
            }
        }

        // Process nested blocks
        if let Some(ref then_block) = new_step.then_block {
            let (new_then, c) = fold_constants(then_block.clone());
            new_step.then_block = Some(new_then);
            changes += c;
        }

        if let Some(ref else_block) = new_step.else_block {
            let (new_else, c) = fold_constants(else_block.clone());
            new_step.else_block = Some(new_else);
            changes += c;
        }

        if let Some(ref body) = new_step.body {
            let (new_body, c) = fold_constants(body.clone());
            new_step.body = Some(new_body);
            changes += c;
        }

        new_steps.push(new_step);
    }

    (new_steps, changes)
}

fn try_fold_expression(expr: &str) -> Option<String> {
    let expr = expr.trim();

    // Try simple arithmetic: "1 + 2", "3 * 4", etc.
    let arith_re = Regex::new(r"^(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)$").ok()?;
    if let Some(caps) = arith_re.captures(expr) {
        let left: f64 = caps[1].parse().ok()?;
        let op = &caps[2];
        let right: f64 = caps[3].parse().ok()?;

        let result = match op {
            "+" => left + right,
            "-" => left - right,
            "*" => left * right,
            "/" if right != 0.0 => left / right,
            _ => return None,
        };

        // Return int if possible
        if result == result.floor() && result.abs() < i64::MAX as f64 {
            return Some((result as i64).to_string());
        }
        return Some(result.to_string());
    }

    // Try boolean expressions: "true && false", etc.
    let bool_re = Regex::new(r"^(true|false)\s*(&&|\|\|)\s*(true|false)$").ok()?;
    if let Some(caps) = bool_re.captures(&expr.to_lowercase()) {
        let left = &caps[1] == "true";
        let op = &caps[2];
        let right = &caps[3] == "true";

        let result = match op {
            "&&" => left && right,
            "||" => left || right,
            _ => return None,
        };

        return Some(result.to_string());
    }

    // Try comparison: "1 > 0", "2 == 2", etc.
    let cmp_re = Regex::new(r"^(-?\d+)\s*(==|!=|<|>|<=|>=)\s*(-?\d+)$").ok()?;
    if let Some(caps) = cmp_re.captures(expr) {
        let left: i64 = caps[1].parse().ok()?;
        let op = &caps[2];
        let right: i64 = caps[3].parse().ok()?;

        let result = match op {
            "==" => left == right,
            "!=" => left != right,
            "<" => left < right,
            ">" => left > right,
            "<=" => left <= right,
            ">=" => left >= right,
            _ => return None,
        };

        return Some(result.to_string());
    }

    None
}

/// Unreachable code elimination pass
pub struct UnreachableCodeElimination;

impl OptimizationPass for UnreachableCodeElimination {
    fn name(&self) -> &'static str {
        "unreachable_code_elimination"
    }

    fn optimize_module(&self, module: &mut IRModule) -> usize {
        let mut changes = 0;
        for func in &mut module.functions {
            if let Some(ref mut steps) = func.steps {
                let (new_steps, c) = eliminate_unreachable(steps.clone());
                *steps = new_steps;
                changes += c;
            }
        }
        changes
    }
}

fn eliminate_unreachable(steps: Vec<IRStep>) -> (Vec<IRStep>, usize) {
    let mut changes = 0;
    let mut new_steps = Vec::new();

    for step in steps {
        let mut new_step = step.clone();

        // Check for if with constant false condition
        if step.op == "if" {
            if let Some(ref cond) = step.condition {
                if cond.to_lowercase() == "false" {
                    // Replace with else block if exists
                    if let Some(ref else_block) = step.else_block {
                        if !else_block.is_empty() {
                            new_steps.extend(else_block.clone());
                        }
                    }
                    changes += 1;
                    continue;
                } else if cond.to_lowercase() == "true" {
                    // Replace with then block
                    if let Some(ref then_block) = step.then_block {
                        if !then_block.is_empty() {
                            new_steps.extend(then_block.clone());
                        }
                    }
                    changes += 1;
                    continue;
                }
            }
        }

        // Check for while with constant false condition
        if step.op == "while" {
            if let Some(ref cond) = step.condition {
                if cond.to_lowercase() == "false" {
                    changes += 1;
                    continue;
                }
            }
        }

        // Process nested blocks
        if let Some(ref then_block) = new_step.then_block {
            let (new_then, c) = eliminate_unreachable(then_block.clone());
            new_step.then_block = Some(new_then);
            changes += c;
        }

        if let Some(ref else_block) = new_step.else_block {
            let (new_else, c) = eliminate_unreachable(else_block.clone());
            new_step.else_block = Some(new_else);
            changes += c;
        }

        if let Some(ref body) = new_step.body {
            let (new_body, c) = eliminate_unreachable(body.clone());
            new_step.body = Some(new_body);
            changes += c;
        }

        new_steps.push(new_step);
    }

    (new_steps, changes)
}

/// Main optimizer that runs multiple passes
pub struct Optimizer {
    passes: Vec<Box<dyn OptimizationPass>>,
    max_iterations: usize,
}

impl Optimizer {
    /// Create optimizer with default passes
    pub fn new() -> Self {
        Self {
            passes: vec![
                Box::new(DeadCodeElimination),
                Box::new(ConstantFolding),
                Box::new(UnreachableCodeElimination),
            ],
            max_iterations: 10,
        }
    }

    /// Create optimizer with specific passes by name
    pub fn with_passes(pass_names: &[&str]) -> Self {
        let mut passes: Vec<Box<dyn OptimizationPass>> = Vec::new();
        
        for name in pass_names {
            match *name {
                "dead_code_elimination" => passes.push(Box::new(DeadCodeElimination)),
                "constant_folding" => passes.push(Box::new(ConstantFolding)),
                "unreachable_code_elimination" => passes.push(Box::new(UnreachableCodeElimination)),
                _ => {}
            }
        }

        Self {
            passes,
            max_iterations: 10,
        }
    }

    /// Create optimizer based on optimization level
    pub fn for_level(level: i32) -> Self {
        match level {
            0 => Self { passes: vec![], max_iterations: 0 },
            1 => Self::with_passes(&["dead_code_elimination", "constant_folding"]),
            2 => Self::with_passes(&["dead_code_elimination", "constant_folding", "unreachable_code_elimination"]),
            _ => Self::new(),
        }
    }

    /// Run all optimization passes until fixed point
    pub fn optimize(&self, module: &mut IRModule) -> usize {
        let mut total_changes = 0;

        for iteration in 0..self.max_iterations {
            let mut iteration_changes = 0;

            for pass in &self.passes {
                let changes = pass.optimize_module(module);
                iteration_changes += changes;
                if changes > 0 {
                    eprintln!("[STUNIR][Optimizer] Pass {}: {} changes", pass.name(), changes);
                }
            }

            total_changes += iteration_changes;

            if iteration_changes == 0 {
                eprintln!("[STUNIR][Optimizer] Converged after {} iterations", iteration + 1);
                break;
            }
        }

        eprintln!("[STUNIR][Optimizer] Total optimizations applied: {}", total_changes);
        total_changes
    }
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_folding_arithmetic() {
        assert_eq!(try_fold_expression("1 + 2"), Some("3".to_string()));
        assert_eq!(try_fold_expression("10 - 3"), Some("7".to_string()));
        assert_eq!(try_fold_expression("4 * 5"), Some("20".to_string()));
        assert_eq!(try_fold_expression("20 / 4"), Some("5".to_string()));
    }

    #[test]
    fn test_constant_folding_boolean() {
        assert_eq!(try_fold_expression("true && true"), Some("true".to_string()));
        assert_eq!(try_fold_expression("true && false"), Some("false".to_string()));
        assert_eq!(try_fold_expression("false || true"), Some("true".to_string()));
    }

    #[test]
    fn test_constant_folding_comparison() {
        assert_eq!(try_fold_expression("1 > 0"), Some("true".to_string()));
        assert_eq!(try_fold_expression("2 == 2"), Some("true".to_string()));
        assert_eq!(try_fold_expression("3 != 3"), Some("false".to_string()));
    }
}
