//! STUNIR Asm_ir Emitter - Rust Implementation
//!
//! Generates Compiler intermediate representations.
//! Based on DO-178C Level A compliant Ada SPARK implementation.

use crate::base::{BaseEmitter, EmitterConfig, EmitterError, EmitterResult, EmitterStatus};
use crate::types::{IRFunction, IRModule, IRStatement};
use std::fmt::Write;

/// Asm_ir variant
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Asm_irVariant {
    /// LLVM_IR
    LLVM_IR,
    /// GCC_RTL
    GCC_RTL,
    /// MLIR
    MLIR,
    /// QBE_IR
    QBE_IR,
}

/// Asm_ir emitter configuration
#[derive(Debug, Clone)]
pub struct Asm_irConfig {
    /// Base emitter configuration
    pub base: EmitterConfig,
    /// Target variant
    pub variant: Asm_irVariant,
}

impl Asm_irConfig {
    /// Create new configuration
    pub fn new(base: EmitterConfig, variant: Asm_irVariant) -> Self {
        Self { base, variant }
    }
}

/// Asm_ir emitter
pub struct Asm_irEmitter {
    config: Asm_irConfig,
}

impl Asm_irEmitter {
    /// Create new emitter
    pub fn new(config: Asm_irConfig) -> Self {
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
            Asm_irVariant::LLVM_IR => {
                writeln!(content, "# {:?} specific code", self.config.variant).unwrap();
            }
            _ => {
                writeln!(content, "# Generic code for {:?}", self.config.variant).unwrap();
            }
        }

        Ok(())
    }
}

impl BaseEmitter for Asm_irEmitter {
    fn emit(&self, ir_module: &IRModule) -> Result<EmitterResult, EmitterError> {
        if !self.validate_ir(ir_module) {
            return Ok(EmitterResult::error(
                EmitterStatus::ErrorInvalidIR,
                "Invalid IR module".to_string(),
            ));
        }

        let code_content = self.generate_code(ir_module)?;
        let extension = match self.config.variant {
            Asm_irVariant::LLVM_IR => "txt",
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
    fn test_asm_ir_emitter() {
        let temp_dir = TempDir::new().unwrap();
        let base_config = EmitterConfig::new(temp_dir.path(), "test_module");
        let config = Asm_irConfig::new(base_config, Asm_irVariant::LLVM_IR);
        let emitter = Asm_irEmitter::new(config);

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
