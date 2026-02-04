//! STUNIR IR Optimizer - Rust implementation (v0.8.9+)
//!
//! Provides optimization passes for STUNIR IR:
//! - Dead code elimination
//! - Constant folding
//! - Constant propagation
//! - Unreachable code elimination

use crate::types::{IRModule, IRStep, OptimizationHint};
use regex::Regex;
use std::collections::HashMap;

/// Optimization pass trait for implementing IR optimizations
pub trait OptimizationPass {
    /// Returns the name of this optimization pass
    fn name(&self) -> &'static str;
    /// Optimizes the given module and returns the number of changes made
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

/// Constant propagation pass
/// Propagates constant values through variables to enable further optimizations
pub struct ConstantPropagation;

impl OptimizationPass for ConstantPropagation {
    fn name(&self) -> &'static str {
        "constant_propagation"
    }

    fn optimize_module(&self, module: &mut IRModule) -> usize {
        let mut changes = 0;
        for func in &mut module.functions {
            if let Some(ref mut steps) = func.steps {
                let (new_steps, c) = propagate_constants(steps.clone());
                *steps = new_steps;
                changes += c;
            }
        }
        changes
    }
}

/// Propagate constant values through a sequence of steps
/// Returns the optimized steps and the number of changes made
fn propagate_constants(steps: Vec<IRStep>) -> (Vec<IRStep>, usize) {
    let mut changes = 0;
    let mut new_steps = Vec::new();
    let mut constants: HashMap<String, serde_json::Value> = HashMap::new();

    for step in steps {
        let mut new_step = step.clone();

        // Replace variable references with constant values
        if let Some(ref target) = step.target {
            if let Some(value) = step.value.clone() {
                // Track this assignment
                if is_constant_value(&value) {
                    constants.insert(target.clone(), value);
                } else {
                    // Remove from constants if reassigned with non-constant
                    constants.remove(target);
                }
            }
        }

        // Replace references in value field
        if let Some(ref value) = step.value {
            if let Some(s) = value.as_str() {
                if let Some(const_val) = constants.get(s) {
                    new_step.value = Some(const_val.clone());
                    changes += 1;
                }
            }
        }

        // Replace references in condition field
        if let Some(ref cond) = step.condition {
            if let Some(const_val) = constants.get(cond) {
                if let Some(s) = const_val.as_str() {
                    new_step.condition = Some(s.to_string());
                    changes += 1;
                }
            }
        }

        // Process nested blocks with their own constant scope
        if let Some(ref then_block) = new_step.then_block {
            let (new_then, c) = propagate_constants(then_block.clone());
            new_step.then_block = Some(new_then);
            changes += c;
        }

        if let Some(ref else_block) = new_step.else_block {
            let (new_else, c) = propagate_constants(else_block.clone());
            new_step.else_block = Some(new_else);
            changes += c;
        }

        if let Some(ref body) = new_step.body {
            let (new_body, c) = propagate_constants(body.clone());
            new_step.body = Some(new_body);
            changes += c;
        }

        if let Some(ref try_block) = new_step.try_block {
            let (new_try, c) = propagate_constants(try_block.clone());
            new_step.try_block = Some(new_try);
            changes += c;
        }

        if let Some(ref catch_blocks) = new_step.catch_blocks {
            let mut new_catches = Vec::new();
            for catch in catch_blocks {
                let (new_body, c) = propagate_constants(catch.body.clone());
                let mut new_catch = catch.clone();
                new_catch.body = new_body;
                new_catches.push(new_catch);
                changes += c;
            }
            new_step.catch_blocks = Some(new_catches);
        }

        if let Some(ref finally_block) = new_step.finally_block {
            let (new_finally, c) = propagate_constants(finally_block.clone());
            new_step.finally_block = Some(new_finally);
            changes += c;
        }

        new_steps.push(new_step);
    }

    (new_steps, changes)
}

/// Check if a value is a constant (number, string literal, boolean)
fn is_constant_value(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Number(_) => true,
        serde_json::Value::String(s) => {
            // Check if it's a literal (quoted) or boolean/null
            s.parse::<f64>().is_ok()
                || s == "true"
                || s == "false"
                || s == "null"
                || (s.starts_with('"') && s.ends_with('"'))
        }
        serde_json::Value::Bool(_) => true,
        serde_json::Value::Null => true,
        _ => false,
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

/// Optimization level for controlling optimization aggressiveness
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// No optimization
    O0,
    /// Basic optimization (dead code elimination, constant folding)
    O1,
    /// Standard optimization (all basic passes)
    O2,
    /// Aggressive optimization (same as O2 for now)
    O3,
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
                Box::new(ConstantPropagation),
                Box::new(UnreachableCodeElimination),
            ],
            max_iterations: 10,
        }
    }

    /// Create optimizer with specific optimization level
    pub fn with_level(level: OptimizationLevel) -> Self {
        match level {
            OptimizationLevel::O0 => Self { passes: vec![], max_iterations: 0 },
            OptimizationLevel::O1 => Self::with_passes(&["dead_code_elimination", "constant_folding"]),
            OptimizationLevel::O2 => Self::with_passes(&["dead_code_elimination", "constant_folding", "constant_propagation", "unreachable_code_elimination"]),
            OptimizationLevel::O3 => Self::with_passes(&["dead_code_elimination", "constant_folding", "constant_propagation", "unreachable_code_elimination"]),
        }
    }

    /// Create optimizer with specific passes by name
    pub fn with_passes(pass_names: &[&str]) -> Self {
        let mut passes: Vec<Box<dyn OptimizationPass>> = Vec::new();

        for name in pass_names {
            match *name {
                "dead_code_elimination" => passes.push(Box::new(DeadCodeElimination)),
                "constant_folding" => passes.push(Box::new(ConstantFolding)),
                "constant_propagation" => passes.push(Box::new(ConstantPropagation)),
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
            2 => Self::with_passes(&["dead_code_elimination", "constant_folding", "constant_propagation", "unreachable_code_elimination"]),
            3 => Self::with_passes(&["dead_code_elimination", "constant_folding", "constant_propagation", "unreachable_code_elimination"]),
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
    use crate::types::{IRFunction, IRModule, IRStep};

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

    // === Dead Code Elimination Tests ===

    fn create_test_step(op: &str) -> IRStep {
        IRStep {
            op: op.to_string(),
            target: None,
            value: None,
            condition: None,
            then_block: None,
            else_block: None,
            body: None,
            args: None,
            fields: None,
            error: None,
            catch_blocks: None,
            finally_block: None,
            try_block: None,
            throw_value: None,
            exception_type: None,
            exception_var: None,
            optimization: None,
            ..Default::default()
        }
    }

    fn create_return_step(value: &str) -> IRStep {
        IRStep {
            op: "return".to_string(),
            target: None,
            value: Some(serde_json::Value::String(value.to_string())),
            condition: None,
            then_block: None,
            else_block: None,
            body: None,
            args: None,
            fields: None,
            error: None,
            catch_blocks: None,
            finally_block: None,
            try_block: None,
            throw_value: None,
            exception_type: None,
            exception_var: None,
            optimization: None,
            ..Default::default()
        }
    }

    #[test]
    fn test_dead_code_elimination_after_return() {
        let steps = vec![
            create_return_step("42"),
            create_test_step("assign"),
            create_test_step("call"),
        ];

        let (result, changes) = eliminate_dead_code(steps);
        assert_eq!(changes, 2);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].op, "return");
    }

    #[test]
    fn test_dead_code_elimination_nop() {
        let mut nop_step = create_test_step("nop");
        nop_step.optimization = Some(OptimizationHint {
            pure: None,
            inline: None,
            const_eval: None,
            dead_code: Some(true),
            constant_value: None,
        });

        let steps = vec![
            create_test_step("assign"),
            nop_step,
            create_test_step("call"),
        ];

        let (result, changes) = eliminate_dead_code(steps);
        assert_eq!(changes, 1);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].op, "assign");
        assert_eq!(result[1].op, "call");
    }

    #[test]
    fn test_dead_code_elimination_nested_blocks() {
        let inner_then = vec![
            create_test_step("inner_assign"),
            create_return_step("1"),
            create_test_step("dead_after_return"),
        ];

        let mut if_step = create_test_step("if");
        if_step.then_block = Some(inner_then);

        let steps = vec![if_step];

        let (result, changes) = eliminate_dead_code(steps);
        assert_eq!(changes, 1);
        assert!(result[0].then_block.is_some());
        assert_eq!(result[0].then_block.as_ref().unwrap().len(), 2);
    }

    // === Constant Folding Module Tests ===

    #[test]
    fn test_constant_folding_pass_module() {
        let pass = ConstantFolding;
        let mut module = IRModule {
            schema: "stunir_ir_v1".to_string(),
            ir_version: "v1".to_string(),
            module_name: "test".to_string(),
            docstring: None,
            type_params: None,
            optimization_level: None,
            functions: vec![IRFunction {
                name: "test_func".to_string(),
                docstring: None,
                type_params: None,
                optimization: None,
                generic_instantiations: None,
                params: None,
                args: vec![],
                return_type: None,
                steps: Some(vec![IRStep {
                    op: "assign".to_string(),
                    target: Some("x".to_string()),
                    value: Some(serde_json::Value::String("1 + 2".to_string())),
                    condition: None,
                    then_block: None,
                    else_block: None,
                    body: None,
                    args: None,
                    fields: None,
                    error: None,
                    catch_blocks: None,
                    finally_block: None,
                    try_block: None,
                    throw_value: None,
                    exception_type: None,
                    exception_var: None,
                    optimization: None,
                    ..Default::default()
                }]),
            }],
            types: None,
            generic_types: None,
            imports: None,
            generic_instantiations: None,
        };

        let changes = pass.optimize_module(&mut module);
        assert!(changes > 0);
    }

    // === Unreachable Code Elimination Tests ===

    #[test]
    fn test_unreachable_code_if_false() {
        let mut if_step = IRStep {
            op: "if".to_string(),
            target: None,
            value: None,
            condition: Some("false".to_string()),
            then_block: Some(vec![create_test_step("then_body")]),
            else_block: Some(vec![create_test_step("else_body")]),
            body: None,
            args: None,
            fields: None,
            error: None,
            catch_blocks: None,
            finally_block: None,
            try_block: None,
            throw_value: None,
            exception_type: None,
            exception_var: None,
            optimization: None,
            ..Default::default()
        };

        let steps = vec![if_step];
        let (result, changes) = eliminate_unreachable(steps);
        assert_eq!(changes, 1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].op, "else_body");
    }

    #[test]
    fn test_unreachable_code_if_true() {
        let mut if_step = IRStep {
            op: "if".to_string(),
            target: None,
            value: None,
            condition: Some("true".to_string()),
            then_block: Some(vec![create_test_step("then_body")]),
            else_block: Some(vec![create_test_step("else_body")]),
            body: None,
            args: None,
            fields: None,
            error: None,
            catch_blocks: None,
            finally_block: None,
            try_block: None,
            throw_value: None,
            exception_type: None,
            exception_var: None,
            optimization: None,
            ..Default::default()
        };

        let steps = vec![if_step];
        let (result, changes) = eliminate_unreachable(steps);
        assert_eq!(changes, 1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].op, "then_body");
    }

    #[test]
    fn test_unreachable_code_while_false() {
        let mut while_step = IRStep {
            op: "while".to_string(),
            target: None,
            value: None,
            condition: Some("false".to_string()),
            then_block: None,
            else_block: None,
            body: Some(vec![create_test_step("loop_body")]),
            args: None,
            fields: None,
            error: None,
            catch_blocks: None,
            finally_block: None,
            try_block: None,
            throw_value: None,
            exception_type: None,
            exception_var: None,
            optimization: None,
            ..Default::default()
        };

        let steps = vec![while_step];
        let (result, changes) = eliminate_unreachable(steps);
        assert_eq!(changes, 1);
        assert_eq!(result.len(), 0);
    }

    // === Optimizer Integration Tests ===

    #[test]
    fn test_optimizer_default_passes() {
        let optimizer = Optimizer::new();
        assert_eq!(optimizer.passes.len(), 3); // DCE, CF, UCE
    }

    #[test]
    fn test_optimizer_with_level_o0() {
        let optimizer = Optimizer::with_level(OptimizationLevel::O0);
        assert_eq!(optimizer.passes.len(), 0);
    }

    #[test]
    fn test_optimizer_with_level_o1() {
        let optimizer = Optimizer::with_level(OptimizationLevel::O1);
        assert_eq!(optimizer.passes.len(), 2); // DCE, CF
    }

    #[test]
    fn test_optimizer_with_level_o2() {
        let optimizer = Optimizer::with_level(OptimizationLevel::O2);
        assert_eq!(optimizer.passes.len(), 3); // DCE, CF, UCE
    }

    #[test]
    fn test_optimizer_full_optimization() {
        let optimizer = Optimizer::new();
        let mut module = IRModule {
            schema: "stunir_ir_v1".to_string(),
            ir_version: "v1".to_string(),
            module_name: "test".to_string(),
            docstring: None,
            type_params: None,
            optimization_level: None,
            functions: vec![IRFunction {
                name: "test_func".to_string(),
                docstring: None,
                type_params: None,
                optimization: None,
                generic_instantiations: None,
                params: None,
                args: vec![],
                return_type: None,
                steps: Some(vec![
                    IRStep {
                        op: "assign".to_string(),
                        target: Some("x".to_string()),
                        value: Some(serde_json::Value::String("1 + 2".to_string())),
                        condition: None,
                        then_block: None,
                        else_block: None,
                        body: None,
                        args: None,
                        fields: None,
                        error: None,
                        catch_blocks: None,
                        finally_block: None,
                        try_block: None,
                        throw_value: None,
                        exception_type: None,
                        exception_var: None,
                        optimization: None,
                        ..Default::default()
                    },
                    create_return_step("x"),
                    create_test_step("dead_code"),
                ]),
            }],
            types: None,
            generic_types: None,
            imports: None,
            generic_instantiations: None,
        };

        let changes = optimizer.optimize(&mut module);
        assert!(changes >= 2); // Constant folding + dead code elimination
    }

    // === Edge Case Tests ===

    #[test]
    fn test_constant_folding_division_by_zero() {
        // Should not fold division by zero
        assert_eq!(try_fold_expression("5 / 0"), None);
    }

    #[test]
    fn test_constant_folding_float_arithmetic() {
        assert_eq!(try_fold_expression("3.5 + 2.5"), Some("6".to_string()));
        assert_eq!(try_fold_expression("10.0 / 4.0"), Some("2.5".to_string()));
    }

    #[test]
    fn test_constant_folding_complex_boolean() {
        assert_eq!(try_fold_expression("true && (false || true)"), None); // Not supported yet
    }

    #[test]
    fn test_dead_code_empty_steps() {
        let steps: Vec<IRStep> = vec![];
        let (result, changes) = eliminate_dead_code(steps);
        assert_eq!(changes, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_unreachable_nested_if() {
        let inner_if = IRStep {
            op: "if".to_string(),
            target: None,
            value: None,
            condition: Some("true".to_string()),
            then_block: Some(vec![create_test_step("inner_then")]),
            else_block: Some(vec![create_test_step("inner_else")]),
            body: None,
            args: None,
            fields: None,
            error: None,
            catch_blocks: None,
            finally_block: None,
            try_block: None,
            throw_value: None,
            exception_type: None,
            exception_var: None,
            optimization: None,
            ..Default::default()
        };

        let steps = vec![inner_if];
        let (result, changes) = eliminate_unreachable(steps);
        assert_eq!(changes, 1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].op, "inner_then");
    }

    // === Constant Propagation Tests ===

    #[test]
    fn test_constant_propagation_basic() {
        let steps = vec![
            IRStep {
                op: "assign".to_string(),
                target: Some("x".to_string()),
                value: Some(serde_json::json!(42)),
                condition: None,
                then_block: None,
                else_block: None,
                body: None,
                args: None,
                fields: None,
                error: None,
                catch_blocks: None,
                finally_block: None,
                try_block: None,
                throw_value: None,
                exception_type: None,
                exception_var: None,
                optimization: None,
                ..Default::default()
            },
            IRStep {
                op: "assign".to_string(),
                target: Some("y".to_string()),
                value: Some(serde_json::json!("x")),
                condition: None,
                then_block: None,
                else_block: None,
                body: None,
                args: None,
                fields: None,
                error: None,
                catch_blocks: None,
                finally_block: None,
                try_block: None,
                throw_value: None,
                exception_type: None,
                exception_var: None,
                optimization: None,
                ..Default::default()
            },
        ];

        let (result, changes) = propagate_constants(steps);
        assert_eq!(changes, 1);
        assert_eq!(result[1].value, Some(serde_json::json!(42)));
    }

    #[test]
    fn test_constant_propagation_in_condition() {
        let steps = vec![
            IRStep {
                op: "assign".to_string(),
                target: Some("flag".to_string()),
                value: Some(serde_json::json!("true")),
                condition: None,
                then_block: None,
                else_block: None,
                body: None,
                args: None,
                fields: None,
                error: None,
                catch_blocks: None,
                finally_block: None,
                try_block: None,
                throw_value: None,
                exception_type: None,
                exception_var: None,
                optimization: None,
                ..Default::default()
            },
            IRStep {
                op: "if".to_string(),
                target: None,
                value: None,
                condition: Some("flag".to_string()),
                then_block: Some(vec![create_test_step("then")]),
                else_block: Some(vec![create_test_step("else")]),
                body: None,
                args: None,
                fields: None,
                error: None,
                catch_blocks: None,
                finally_block: None,
                try_block: None,
                throw_value: None,
                exception_type: None,
                exception_var: None,
                optimization: None,
                ..Default::default()
            },
        ];

        let (result, changes) = propagate_constants(steps);
        assert_eq!(changes, 1);
        assert_eq!(result[1].condition, Some("true".to_string()));
    }

    #[test]
    fn test_constant_propagation_reassignment() {
        let steps = vec![
            IRStep {
                op: "assign".to_string(),
                target: Some("x".to_string()),
                value: Some(serde_json::json!(10)),
                condition: None,
                then_block: None,
                else_block: None,
                body: None,
                fields: None,
                error: None,
                catch_blocks: None,
                finally_block: None,
                try_block: None,
                throw_value: None,
                exception_type: None,
                exception_var: None,
                optimization: None,
                ..Default::default()
            },
            IRStep {
                op: "assign".to_string(),
                target: Some("x".to_string()),
                value: Some(serde_json::json!("y")),
                condition: None,
                then_block: None,
                else_block: None,
                body: None,
                fields: None,
                error: None,
                catch_blocks: None,
                finally_block: None,
                try_block: None,
                throw_value: None,
                exception_type: None,
                exception_var: None,
                optimization: None,
                ..Default::default()
            },
            IRStep {
                op: "assign".to_string(),
                target: Some("z".to_string()),
                value: Some(serde_json::json!("x")),
                condition: None,
                then_block: None,
                else_block: None,
                body: None,
                fields: None,
                error: None,
                catch_blocks: None,
                finally_block: None,
                try_block: None,
                throw_value: None,
                exception_type: None,
                exception_var: None,
                optimization: None,
                ..Default::default()
            },
        ];

        let (result, changes) = propagate_constants(steps);
        // x is reassigned with non-constant, so z should not be propagated
        assert_eq!(changes, 0);
        assert_eq!(result[2].value, Some(serde_json::json!("x")));
    }

    #[test]
    fn test_is_constant_value() {
        assert!(is_constant_value(&serde_json::json!(42)));
        assert!(is_constant_value(&serde_json::json!(3.14)));
        assert!(is_constant_value(&serde_json::json!("true")));
        assert!(is_constant_value(&serde_json::json!("false")));
        assert!(is_constant_value(&serde_json::json!("null")));
        assert!(is_constant_value(&serde_json::json!("\"hello\"")));
        assert!(!is_constant_value(&serde_json::json!("variable")));
        assert!(!is_constant_value(&serde_json::json!([])));
    }

    #[test]
    fn test_constant_propagation_pass() {
        let mut module = IRModule {
            schema: "stunir_ir_v1".to_string(),
            ir_version: "v1".to_string(),
            module_name: "test".to_string(),
            docstring: None,
            type_params: None,
            optimization_level: None,
            functions: vec![IRFunction {
                name: "test".to_string(),
                docstring: None,
                type_params: None,
                optimization: None,
                generic_instantiations: None,
                params: None,
                args: vec![],
                return_type: None,
                steps: Some(vec![
                    IRStep {
                        op: "assign".to_string(),
                        target: Some("x".to_string()),
                        value: Some(serde_json::json!(42)),
                        condition: None,
                        then_block: None,
                        else_block: None,
                        body: None,
                        args: None,
                        fields: None,
                        error: None,
                        catch_blocks: None,
                        finally_block: None,
                        try_block: None,
                        throw_value: None,
                        exception_type: None,
                        exception_var: None,
                        optimization: None,
                        ..Default::default()
                    },
                    IRStep {
                        op: "assign".to_string(),
                        target: Some("y".to_string()),
                        value: Some(serde_json::json!("x")),
                        condition: None,
                        then_block: None,
                        else_block: None,
                        body: None,
                        args: None,
                        fields: None,
                        error: None,
                        catch_blocks: None,
                        finally_block: None,
                        try_block: None,
                        throw_value: None,
                        exception_type: None,
                        exception_var: None,
                        optimization: None,
                        ..Default::default()
                    },
                ]),
            }],
            types: None,
            generic_types: None,
            imports: None,
            generic_instantiations: None,
        };

        let pass = ConstantPropagation;
        let changes = pass.optimize_module(&mut module);
        assert_eq!(changes, 1);

        let func = &module.functions[0];
        let steps = func.steps.as_ref().unwrap();
        assert_eq!(steps[1].value, Some(serde_json::json!(42)));
    }
}