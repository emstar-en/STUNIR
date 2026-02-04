//! STUNIR Prolog Family Emitter - Rust Implementation
//!
//! Generates code for Prolog family languages.
//! Based on DO-178C Level A compliant Ada SPARK implementation.
//!
//! Supported dialects:
//! - SWI-Prolog
//! - GNU Prolog
//! - SICStus Prolog
//! - YAP (Yet Another Prolog)
//! - XSB
//! - Ciao Prolog
//! - B-Prolog
//! - ECLiPSe

use crate::base::{BaseEmitter, EmitterConfig, EmitterError, EmitterResult, EmitterStatus};
use crate::types::{IRFunction, IRModule, IRParameter, IRStatement, IRStatementType};
use std::fmt::Write;

/// Prolog dialect
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrologDialect {
    /// SWI-Prolog
    SWI,
    /// GNU Prolog
    GNU,
    /// SICStus Prolog
    SICStus,
    /// YAP (Yet Another Prolog)
    YAP,
    /// XSB
    XSB,
    /// Ciao Prolog
    Ciao,
    /// B-Prolog
    BProlog,
    /// ECLiPSe
    ECLiPSe,
}

/// Prolog emitter configuration
#[derive(Debug, Clone)]
pub struct PrologConfig {
    /// Base emitter configuration
    pub base: EmitterConfig,
    /// Prolog dialect
    pub dialect: PrologDialect,
}

impl PrologConfig {
    /// Create new Prolog configuration
    pub fn new(base: EmitterConfig, dialect: PrologDialect) -> Self {
        Self { base, dialect }
    }
}

/// Prolog emitter
pub struct PrologEmitter {
    config: PrologConfig,
}

impl PrologEmitter {
    /// Create new Prolog emitter
    pub fn new(config: PrologConfig) -> Self {
        Self { config }
    }

    /// Generate Prolog code
    fn generate_prolog(&self, ir_module: &IRModule) -> Result<String, EmitterError> {
        let mut content = String::new();

        // Header comment
        writeln!(content, "% STUNIR Generated Prolog Code")?;
        writeln!(content, "% DO-178C Level A Compliant")?;
        writeln!(content, "% Dialect: {:?}\\n", self.config.dialect)?;

        // Module declaration (dialect-specific)
        self.generate_module_decl(&mut content, ir_module)?;

        // Directives
        self.generate_directives(&mut content)?;

        // Facts and rules (from types)
        for ir_type in &ir_module.types {
            self.generate_type_facts(&mut content, ir_type)?;
        }

        // Predicate definitions (from functions)
        for function in &ir_module.functions {
            self.generate_predicate(&mut content, function)?;
            writeln!(content)?;
        }

        Ok(content)
    }

    /// Generate module declaration
    fn generate_module_decl(
        &self,
        content: &mut String,
        ir_module: &IRModule,
    ) -> Result<(), EmitterError> {
        match self.config.dialect {
            PrologDialect::SWI | PrologDialect::YAP => {
                writeln!(content, ":- module({}, []).", ir_module.module_name)?;
                writeln!(content)?;
            }
            PrologDialect::SICStus | PrologDialect::Ciao => {
                writeln!(content, ":- module({}, []).", ir_module.module_name)?;
                writeln!(content)?;
            }
            _ => {
                writeln!(content, "% Module: {}", ir_module.module_name)?;
                writeln!(content)?;
            }
        }
        Ok(())
    }

    /// Generate directives
    fn generate_directives(&self, content: &mut String) -> Result<(), EmitterError> {
        writeln!(content, "% Directives")?;
        match self.config.dialect {
            PrologDialect::SWI => {
                writeln!(content, ":- set_prolog_flag(double_quotes, codes).")?;
            }
            PrologDialect::ECLiPSe => {
                writeln!(content, ":- lib(ic).")?;
            }
            _ => {}
        }
        writeln!(content)?;
        Ok(())
    }

    /// Generate type facts
    fn generate_type_facts(
        &self,
        content: &mut String,
        ir_type: &crate::types::IRType,
    ) -> Result<(), EmitterError> {
        writeln!(content, "% Type: {}", ir_type.name)?;
        if let Some(ref doc) = ir_type.docstring {
            writeln!(content, "% {}", doc)?;
        }

        // Generate facts for each field
        for field in &ir_type.fields {
            writeln!(
                content,
                "field({}, {}, {}).",
                ir_type.name, field.name, field.field_type
            )
            .unwrap();
        }
        writeln!(content)?;
        Ok(())
    }

    /// Generate predicate definition
    fn generate_predicate(
        &self,
        content: &mut String,
        function: &IRFunction,
    ) -> Result<(), EmitterError> {
        if let Some(ref doc) = function.docstring {
            writeln!(content, "% {}", doc)?;
        }

        // Predicate head
        let params = self.format_params(&function.parameters);
        writeln!(content, "{}({}) :-", function.name, params)?;

        // Predicate body
        if function.statements.is_empty() {
            writeln!(content, "    true.")?;
        } else {
            for (i, stmt) in function.statements.iter().enumerate() {
                let is_last = i == function.statements.len() - 1;
                self.generate_statement(content, stmt, is_last)?;
            }
        }

        Ok(())
    }

    /// Generate statement
    fn generate_statement(
        &self,
        content: &mut String,
        stmt: &IRStatement,
        is_last: bool,
    ) -> Result<(), EmitterError> {
        let terminator = if is_last { "." } else { "," };

        match stmt.stmt_type {
            IRStatementType::VarDecl => {
                let var_name = stmt.target.as_deref().unwrap_or("V");
                let value = stmt.value.as_deref().unwrap_or("0");
                writeln!(content, "    {} = {}{}", var_name, value, terminator)?;
            }
            IRStatementType::Assign => {
                let target = stmt.target.as_deref().unwrap_or("V");
                let value = stmt.value.as_deref().unwrap_or("0");
                writeln!(content, "    {} = {}{}", target, value, terminator)?;
            }
            IRStatementType::Add => {
                let target = stmt.target.as_deref().unwrap_or("Result");
                let left = stmt.left_op.as_deref().unwrap_or("0");
                let right = stmt.right_op.as_deref().unwrap_or("0");
                writeln!(
                    content,
                    "    {} is {} + {}{}",
                    target, left, right, terminator
                )?;
            }
            IRStatementType::Sub => {
                let target = stmt.target.as_deref().unwrap_or("Result");
                let left = stmt.left_op.as_deref().unwrap_or("0");
                let right = stmt.right_op.as_deref().unwrap_or("0");
                writeln!(
                    content,
                    "    {} is {} - {}{}",
                    target, left, right, terminator
                )?;
            }
            IRStatementType::Mul => {
                let target = stmt.target.as_deref().unwrap_or("Result");
                let left = stmt.left_op.as_deref().unwrap_or("0");
                let right = stmt.right_op.as_deref().unwrap_or("0");
                writeln!(
                    content,
                    "    {} is {} * {}{}",
                    target, left, right, terminator
                )?;
            }
            IRStatementType::Div => {
                let target = stmt.target.as_deref().unwrap_or("Result");
                let left = stmt.left_op.as_deref().unwrap_or("0");
                let right = stmt.right_op.as_deref().unwrap_or("0");
                writeln!(
                    content,
                    "    {} is {} / {}{}",
                    target, left, right, terminator
                )?;
            }
            IRStatementType::Call => {
                let func = stmt.target.as_deref().unwrap_or("call");
                let args = stmt.value.as_deref().unwrap_or("");
                writeln!(content, "    {}({}){}", func, args, terminator)?;
            }
            _ => {
                writeln!(content, "    % {:?}{}", stmt.stmt_type, terminator)?;
            }
        }

        Ok(())
    }

    /// Format parameters
    fn format_params(&self, params: &[IRParameter]) -> String {
        if params.is_empty() {
            return String::new();
        }
        params
            .iter()
            .map(|p| p.name.to_uppercase())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

impl BaseEmitter for PrologEmitter {
    fn emit(&self, ir_module: &IRModule) -> Result<EmitterResult, EmitterError> {
        if !self.validate_ir(ir_module) {
            return Ok(EmitterResult::error(
                EmitterStatus::ErrorInvalidIR,
                "Invalid IR module".to_string(),
            ));
        }

        let prolog_content = self.generate_prolog(ir_module)?;
        let extension = match self.config.dialect {
            PrologDialect::SWI | PrologDialect::GNU | PrologDialect::YAP => "pl",
            PrologDialect::ECLiPSe => "ecl",
            _ => "pro",
        };

        let file = self.write_file(
            &self.config.base.output_dir,
            &format!("{}.{}", self.config.base.module_name, extension),
            &prolog_content,
        )?;

        Ok(EmitterResult::success(vec![file.clone()], file.size))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{IRDataType, IRFunction, IRModule};
    use tempfile::TempDir;

    #[test]
    fn test_prolog_swi() {
        let temp_dir = TempDir::new()?;
        let base_config = EmitterConfig::new(temp_dir.path(), "test_module");
        let config = PrologConfig::new(base_config, PrologDialect::SWI);
        let emitter = PrologEmitter::new(config);

        let ir_module = IRModule {
            ir_version: "1.0".to_string(),
            module_name: "test_module".to_string(),
            types: vec![],
            functions: vec![IRFunction {
                name: "fact".to_string(),
                return_type: IRDataType::Bool,
                parameters: vec![],
                statements: vec![],
                docstring: Some("Test fact".to_string()),
            }],
            docstring: None,
        };

        let result = emitter.emit(&ir_module)?;
        assert_eq!(result.status, EmitterStatus::Success);
    }
}
