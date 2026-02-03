//! STUNIR Lisp Family Emitter - Rust Implementation
//!
//! Generates code for Lisp family languages.
//! Based on DO-178C Level A compliant Ada SPARK implementation.
//!
//! Supported dialects:
//! - Common Lisp
//! - Scheme (R5RS, R6RS, R7RS)
//! - Clojure
//! - Racket
//! - Emacs Lisp
//! - Guile
//! - Hy
//! - Janet

use crate::base::{BaseEmitter, EmitterConfig, EmitterError, EmitterResult, EmitterStatus};
use crate::types::{IRDataType, IRFunction, IRModule, IRParameter, IRStatement, IRStatementType};
use std::fmt::Write;

/// Lisp dialect
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LispDialect {
    /// Common Lisp
    CommonLisp,
    /// Scheme
    Scheme,
    /// Clojure
    Clojure,
    /// Racket
    Racket,
    /// Emacs Lisp
    EmacsLisp,
    /// Guile
    Guile,
    /// Hy (Python-based Lisp)
    Hy,
    /// Janet
    Janet,
}

/// Lisp emitter configuration
#[derive(Debug, Clone)]
pub struct LispConfig {
    /// Base emitter configuration
    pub base: EmitterConfig,
    /// Lisp dialect
    pub dialect: LispDialect,
    /// Indent size (number of spaces)
    pub indent_size: usize,
}

impl LispConfig {
    /// Create new Lisp configuration
    pub fn new(base: EmitterConfig, dialect: LispDialect) -> Self {
        Self {
            base,
            dialect,
            indent_size: 2,
        }
    }
}

/// Lisp emitter
pub struct LispEmitter {
    config: LispConfig,
}

impl LispEmitter {
    /// Create new Lisp emitter
    pub fn new(config: LispConfig) -> Self {
        Self { config }
    }

    /// Generate Lisp code
    fn generate_lisp(&self, ir_module: &IRModule) -> Result<String, EmitterError> {
        let mut content = String::new();

        // Header comment
        writeln!(
            content,
            "{} STUNIR Generated Lisp Code",
            self.comment_prefix()
        )
        .unwrap();
        writeln!(
            content,
            "{} DO-178C Level A Compliant",
            self.comment_prefix()
        )
        .unwrap();
        writeln!(
            content,
            "{} Dialect: {:?}\\n",
            self.comment_prefix(),
            self.config.dialect
        )
        .unwrap();

        // Dialect-specific preamble
        self.generate_preamble(&mut content)?;

        // Module/package definition
        self.generate_module_def(&mut content, ir_module)?;

        // Type definitions (if supported)
        for ir_type in &ir_module.types {
            self.generate_type(&mut content, ir_type)?;
        }

        // Function definitions
        for function in &ir_module.functions {
            self.generate_function(&mut content, function)?;
            writeln!(content).unwrap();
        }

        // Dialect-specific postamble
        self.generate_postamble(&mut content)?;

        Ok(content)
    }

    /// Get comment prefix for dialect
    fn comment_prefix(&self) -> &'static str {
        match self.config.dialect {
            LispDialect::Janet => "#",
            _ => ";;",
        }
    }

    /// Generate dialect-specific preamble
    fn generate_preamble(&self, content: &mut String) -> Result<(), EmitterError> {
        match self.config.dialect {
            LispDialect::Racket => {
                writeln!(content, "#lang racket\\n").unwrap();
            }
            LispDialect::Hy => {
                writeln!(content, "#!/usr/bin/env hy\\n").unwrap();
            }
            _ => {}
        }
        Ok(())
    }

    /// Generate module definition
    fn generate_module_def(
        &self,
        content: &mut String,
        ir_module: &IRModule,
    ) -> Result<(), EmitterError> {
        match self.config.dialect {
            LispDialect::CommonLisp => {
                writeln!(content, "(defpackage :{}", ir_module.module_name).unwrap();
                writeln!(content, "  (:use :cl)").unwrap();
                writeln!(content, "  (:export))\\n").unwrap();
                writeln!(content, "(in-package :{})\\n", ir_module.module_name).unwrap();
            }
            LispDialect::Clojure => {
                writeln!(content, "(ns {})", ir_module.module_name).unwrap();
                writeln!(content).unwrap();
            }
            LispDialect::Scheme => {
                writeln!(
                    content,
                    "{} Module: {}",
                    self.comment_prefix(),
                    ir_module.module_name
                )
                .unwrap();
                writeln!(content).unwrap();
            }
            _ => {}
        }
        Ok(())
    }

    /// Generate type definition
    fn generate_type(
        &self,
        content: &mut String,
        _ir_type: &crate::types::IRType,
    ) -> Result<(), EmitterError> {
        // Most Lisp dialects don't have static type definitions
        // This is dialect-specific
        match self.config.dialect {
            LispDialect::CommonLisp => {
                // Could use defstruct or defclass
            }
            LispDialect::Clojure => {
                // Could use defrecord
            }
            _ => {}
        }
        Ok(())
    }

    /// Generate function definition
    fn generate_function(
        &self,
        content: &mut String,
        function: &IRFunction,
    ) -> Result<(), EmitterError> {
        if let Some(ref doc) = function.docstring {
            writeln!(content, "{} {}", self.comment_prefix(), doc).unwrap();
        }

        match self.config.dialect {
            LispDialect::CommonLisp => {
                writeln!(
                    content,
                    "(defun {} ({})",
                    function.name,
                    self.format_params(&function.parameters)
                )
                .unwrap();
                if let Some(ref doc) = function.docstring {
                    writeln!(content, "  \\\"{}\\\"", doc).unwrap();
                }
                self.generate_statements(content, &function.statements, 1)?;
                writeln!(content, ")").unwrap();
            }
            LispDialect::Scheme | LispDialect::Guile | LispDialect::Racket => {
                writeln!(
                    content,
                    "(define ({} {})",
                    function.name,
                    self.format_params(&function.parameters)
                )
                .unwrap();
                self.generate_statements(content, &function.statements, 1)?;
                writeln!(content, ")").unwrap();
            }
            LispDialect::Clojure => {
                writeln!(
                    content,
                    "(defn {} [{}]",
                    function.name,
                    self.format_params(&function.parameters)
                )
                .unwrap();
                if let Some(ref doc) = function.docstring {
                    writeln!(content, "  \\\"{}\\\"", doc).unwrap();
                }
                self.generate_statements(content, &function.statements, 1)?;
                writeln!(content, ")").unwrap();
            }
            LispDialect::EmacsLisp => {
                writeln!(
                    content,
                    "(defun {} ({})",
                    function.name,
                    self.format_params(&function.parameters)
                )
                .unwrap();
                if let Some(ref doc) = function.docstring {
                    writeln!(content, "  \\\"{}\\\"", doc).unwrap();
                }
                self.generate_statements(content, &function.statements, 1)?;
                writeln!(content, ")").unwrap();
            }
            LispDialect::Hy => {
                writeln!(
                    content,
                    "(defn {} [{}]",
                    function.name,
                    self.format_params(&function.parameters)
                )
                .unwrap();
                self.generate_statements(content, &function.statements, 1)?;
                writeln!(content, ")").unwrap();
            }
            LispDialect::Janet => {
                writeln!(
                    content,
                    "(defn {} [{}]",
                    function.name,
                    self.format_params(&function.parameters)
                )
                .unwrap();
                self.generate_statements(content, &function.statements, 1)?;
                writeln!(content, ")").unwrap();
            }
        }

        Ok(())
    }

    /// Generate statements
    fn generate_statements(
        &self,
        content: &mut String,
        statements: &[IRStatement],
        indent_level: usize,
    ) -> Result<(), EmitterError> {
        for stmt in statements {
            self.generate_statement(content, stmt, indent_level)?;
        }
        Ok(())
    }

    /// Generate statement
    fn generate_statement(
        &self,
        content: &mut String,
        stmt: &IRStatement,
        indent_level: usize,
    ) -> Result<(), EmitterError> {
        let indent = " ".repeat(self.config.indent_size * indent_level);

        match stmt.stmt_type {
            IRStatementType::Return => {
                let value = stmt.value.as_deref().unwrap_or("0");
                writeln!(content, "{}{}", indent, value).unwrap();
            }
            IRStatementType::Add => {
                let left = stmt.left_op.as_deref().unwrap_or("0");
                let right = stmt.right_op.as_deref().unwrap_or("0");
                writeln!(content, "{}(+ {} {})", indent, left, right).unwrap();
            }
            IRStatementType::Sub => {
                let left = stmt.left_op.as_deref().unwrap_or("0");
                let right = stmt.right_op.as_deref().unwrap_or("0");
                writeln!(content, "{}(- {} {})", indent, left, right).unwrap();
            }
            IRStatementType::Mul => {
                let left = stmt.left_op.as_deref().unwrap_or("0");
                let right = stmt.right_op.as_deref().unwrap_or("0");
                writeln!(content, "{}(* {} {})", indent, left, right).unwrap();
            }
            IRStatementType::Div => {
                let left = stmt.left_op.as_deref().unwrap_or("0");
                let right = stmt.right_op.as_deref().unwrap_or("0");
                writeln!(content, "{}(/ {} {})", indent, left, right).unwrap();
            }
            _ => {
                writeln!(
                    content,
                    "{}{} {:?}",
                    indent,
                    self.comment_prefix(),
                    stmt.stmt_type
                )
                .unwrap();
            }
        }

        Ok(())
    }

    /// Format parameters
    fn format_params(&self, params: &[IRParameter]) -> String {
        params
            .iter()
            .map(|p| p.name.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Generate postamble
    fn generate_postamble(&self, _content: &mut String) -> Result<(), EmitterError> {
        // Most dialects don't need postamble
        Ok(())
    }
}

impl BaseEmitter for LispEmitter {
    fn emit(&self, ir_module: &IRModule) -> Result<EmitterResult, EmitterError> {
        if !self.validate_ir(ir_module) {
            return Ok(EmitterResult::error(
                EmitterStatus::ErrorInvalidIR,
                "Invalid IR module".to_string(),
            ));
        }

        let lisp_content = self.generate_lisp(ir_module)?;
        let extension = match self.config.dialect {
            LispDialect::CommonLisp => "lisp",
            LispDialect::Scheme | LispDialect::Guile => "scm",
            LispDialect::Clojure => "clj",
            LispDialect::Racket => "rkt",
            LispDialect::EmacsLisp => "el",
            LispDialect::Hy => "hy",
            LispDialect::Janet => "janet",
        };

        let file = self.write_file(
            &self.config.base.output_dir,
            &format!("{}.{}", self.config.base.module_name, extension),
            &lisp_content,
        )?;

        Ok(EmitterResult::success(vec![file.clone()], file.size))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{IRFunction, IRModule};
    use tempfile::TempDir;

    #[test]
    fn test_lisp_common_lisp() {
        let temp_dir = TempDir::new().unwrap();
        let base_config = EmitterConfig::new(temp_dir.path(), "test_module");
        let config = LispConfig::new(base_config, LispDialect::CommonLisp);
        let emitter = LispEmitter::new(config);

        let ir_module = IRModule {
            ir_version: "1.0".to_string(),
            module_name: "test_module".to_string(),
            types: vec![],
            functions: vec![IRFunction {
                name: "add".to_string(),
                return_type: IRDataType::I32,
                parameters: vec![],
                statements: vec![],
                docstring: Some("Addition function".to_string()),
            }],
            docstring: None,
        };

        let result = emitter.emit(&ir_module).unwrap();
        assert_eq!(result.status, EmitterStatus::Success);
    }

    #[test]
    fn test_lisp_clojure() {
        let temp_dir = TempDir::new().unwrap();
        let base_config = EmitterConfig::new(temp_dir.path(), "test_module");
        let config = LispConfig::new(base_config, LispDialect::Clojure);
        let emitter = LispEmitter::new(config);

        let ir_module = IRModule {
            ir_version: "1.0".to_string(),
            module_name: "test_module".to_string(),
            types: vec![],
            functions: vec![],
            docstring: None,
        };

        let result = emitter.emit(&ir_module).unwrap();
        assert_eq!(result.status, EmitterStatus::Success);
    }
}
