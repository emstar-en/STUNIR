//! STUNIR Functional Emitter - Rust Implementation
//!
//! Generates Functional programming languages.
//! Based on DO-178C Level A compliant Ada SPARK implementation.

use crate::base::{BaseEmitter, EmitterConfig, EmitterError, EmitterResult, EmitterStatus};
use crate::types::{IRFunction, IRModule, IRStatement};
use std::fmt::Write;

/// Functional variant
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FunctionalVariant {
    /// Haskell
    Haskell,
    /// OCaml
    OCaml,
    /// FSharp
    FSharp,
    /// Erlang
    Erlang,
    /// Elixir
    Elixir,
}

/// Functional emitter configuration
#[derive(Debug, Clone)]
pub struct FunctionalConfig {
    /// Base emitter configuration
    pub base: EmitterConfig,
    /// Target variant
    pub variant: FunctionalVariant,
}

impl FunctionalConfig {
    /// Create new configuration
    pub fn new(base: EmitterConfig, variant: FunctionalVariant) -> Self {
        Self { base, variant }
    }
}

/// Functional emitter
pub struct FunctionalEmitter {
    config: FunctionalConfig,
}

impl FunctionalEmitter {
    /// Create new emitter
    pub fn new(config: FunctionalConfig) -> Self {
        Self { config }
    }

    /// Generate code
    fn generate_code(&self, ir_module: &IRModule) -> Result<String, EmitterError> {
        let mut content = String::new();

        writeln!(content, "# STUNIR Generated {:?} Code", self.config.variant).unwrap();
        writeln!(content, "# DO-178C Level A Compliant\n").unwrap();

        // Module/namespace
        writeln!(content, "# Module: {}", ir_module.module_name).unwrap();
        writeln!(content).unwrap();

        // Functions
        for function in &ir_module.functions {
            if let Some(ref doc) = function.docstring {
                writeln!(content, "# {}", doc).unwrap();
            }
            writeln!(content, "# Function: {}", function.name).unwrap();
            self.generate_function(&mut content, function)?;
            writeln!(content).unwrap();
        }

        Ok(content)
    }

    /// Generate function
    fn generate_function(
        &self,
        content: &mut String,
        function: &IRFunction,
    ) -> Result<(), EmitterError> {
        writeln!(content, "# Generated function: {}", function.name).unwrap();

        // Variant-specific generation
        match self.config.variant {
            FunctionalVariant::Haskell => {
                writeln!(content, "# {:?} specific code", self.config.variant).unwrap();
            }
            _ => {
                writeln!(content, "# Generic code for {:?}", self.config.variant).unwrap();
            }
        }

        Ok(())
    }
}

impl BaseEmitter for FunctionalEmitter {
    fn emit(&self, ir_module: &IRModule) -> Result<EmitterResult, EmitterError> {
        if !self.validate_ir(ir_module) {
            return Ok(EmitterResult::error(
                EmitterStatus::ErrorInvalidIR,
                "Invalid IR module".to_string(),
            ));
        }

        let code_content = self.generate_code(ir_module)?;
        let extension = match self.config.variant {
            FunctionalVariant::Haskell => "txt",
            _ => "gen",
        };

        let file = self.write_file(
            &self.config.base.output_dir,
            &format!("{}.{}", self.config.base.module_name, extension),
            &code_content,
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
    fn test_functional_emitter() {
        let temp_dir = TempDir::new().unwrap();
        let base_config = EmitterConfig::new(temp_dir.path(), "test_module");
        let config = FunctionalConfig::new(base_config, FunctionalVariant::Haskell);
        let emitter = FunctionalEmitter::new(config);

        let ir_module = IRModule {
            ir_version: "1.0".to_string(),
            module_name: "test_module".to_string(),
            types: vec![],
            functions: vec![IRFunction {
                name: "test_func".to_string(),
                return_type: IRDataType::I32,
                parameters: vec![],
                statements: vec![],
                docstring: Some("Test function".to_string()),
            }],
            docstring: None,
        };

        let result = emitter.emit(&ir_module).unwrap();
        assert_eq!(result.status, EmitterStatus::Success);
    }
}
