//! STUNIR IR Visitor - Rust Implementation
//!
//! Visitor pattern for traversing and processing IR structures.
//! Based on Ada SPARK visitor patterns.

use crate::types::{IRFunction, IRModule, IRParameter, IRStatement, IRType, IRTypeField};

/// IR visitor trait
///
/// Implements the Visitor pattern for walking the IR tree.
/// Implementers can override specific visit methods to customize behavior.
pub trait IRVisitor {
    /// Visit result type
    type Result;

    /// Visit an IR module (entry point)
    fn visit_module(&mut self, module: &IRModule) -> Self::Result {
        self.enter_module(module);

        // Visit all types
        for ir_type in &module.types {
            self.visit_type(ir_type);
        }

        // Visit all functions
        for function in &module.functions {
            self.visit_function(function);
        }

        self.exit_module(module)
    }

    /// Visit a type definition
    fn visit_type(&mut self, ir_type: &IRType) {
        self.enter_type(ir_type);

        // Visit all fields
        for field in &ir_type.fields {
            self.visit_field(field);
        }

        self.exit_type(ir_type);
    }

    /// Visit a type field
    fn visit_field(&mut self, field: &IRTypeField) {
        self.process_field(field);
    }

    /// Visit a function definition
    fn visit_function(&mut self, function: &IRFunction) {
        self.enter_function(function);

        // Visit all parameters
        for param in &function.parameters {
            self.visit_parameter(param);
        }

        // Visit all statements
        for stmt in &function.statements {
            self.visit_statement(stmt);
        }

        self.exit_function(function);
    }

    /// Visit a function parameter
    fn visit_parameter(&mut self, param: &IRParameter) {
        self.process_parameter(param);
    }

    /// Visit a statement
    fn visit_statement(&mut self, stmt: &IRStatement) {
        self.process_statement(stmt);
    }

    // Methods to be implemented

    /// Called when entering a module
    fn enter_module(&mut self, module: &IRModule);

    /// Called when exiting a module
    fn exit_module(&mut self, module: &IRModule) -> Self::Result;

    /// Called when entering a type definition
    fn enter_type(&mut self, ir_type: &IRType);

    /// Called when exiting a type definition
    fn exit_type(&mut self, ir_type: &IRType);

    /// Process a type field
    fn process_field(&mut self, field: &IRTypeField);

    /// Called when entering a function
    fn enter_function(&mut self, function: &IRFunction);

    /// Called when exiting a function
    fn exit_function(&mut self, function: &IRFunction);

    /// Process a function parameter
    fn process_parameter(&mut self, param: &IRParameter);

    /// Process a statement
    fn process_statement(&mut self, stmt: &IRStatement);
}

/// Code generation visitor
///
/// Extends IRVisitor to accumulate generated code as a string.
pub struct CodeGenVisitor {
    /// Generated code lines
    pub code: Vec<String>,
    /// Current indentation level
    pub indent_level: usize,
}

impl CodeGenVisitor {
    /// Create new code generation visitor
    pub fn new() -> Self {
        Self {
            code: Vec::new(),
            indent_level: 0,
        }
    }

    /// Emit a line of code with current indentation
    pub fn emit(&mut self, line: impl Into<String>) {
        let line = line.into();
        if line.is_empty() {
            self.code.push(String::new());
        } else {
            self.code
                .push(format!("{}{}", "    ".repeat(self.indent_level), line));
        }
    }

    /// Get accumulated code as string
    pub fn get_code(&self) -> String {
        self.code.join("\n")
    }

    /// Increase indentation level
    pub fn increase_indent(&mut self) {
        self.indent_level += 1;
    }

    /// Decrease indentation level
    pub fn decrease_indent(&mut self) {
        self.indent_level = self.indent_level.saturating_sub(1);
    }
}

impl Default for CodeGenVisitor {
    fn default() -> Self {
        Self::new()
    }
}
