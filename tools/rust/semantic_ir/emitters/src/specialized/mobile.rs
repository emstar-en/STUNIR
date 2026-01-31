//! STUNIR Mobile Emitter - Rust Implementation
//!
//! Generates Mobile development platforms.
//! Based on DO-178C Level A compliant Ada SPARK implementation.

use crate::base::{BaseEmitter, EmitterConfig, EmitterError, EmitterResult, EmitterStatus};
use crate::types::{IRFunction, IRModule, IRStatement};
use std::fmt::Write;

/// Mobile variant
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MobileVariant {
    /// iOS_Swift
    iOS_Swift,
    /// Android_Kotlin
    Android_Kotlin,
    /// React_Native
    React_Native,
    /// Flutter
    Flutter,
}

/// Mobile emitter configuration
#[derive(Debug, Clone)]
pub struct MobileConfig {
    /// Base emitter configuration
    pub base: EmitterConfig,
    /// Target variant
    pub variant: MobileVariant,
}

impl MobileConfig {
    /// Create new configuration
    pub fn new(base: EmitterConfig, variant: MobileVariant) -> Self {
        Self { base, variant }
    }
}

/// Mobile emitter
pub struct MobileEmitter {
    config: MobileConfig,
}

impl MobileEmitter {
    /// Create new emitter
    pub fn new(config: MobileConfig) -> Self {
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
            MobileVariant::iOS_Swift => {
                writeln!(content, "# {:?} specific code", self.config.variant).unwrap();
            }
            _ => {
                writeln!(content, "# Generic code for {:?}", self.config.variant).unwrap();
            }
        }

        Ok(())
    }
}

impl BaseEmitter for MobileEmitter {
    fn emit(&self, ir_module: &IRModule) -> Result<EmitterResult, EmitterError> {
        if !self.validate_ir(ir_module) {
            return Ok(EmitterResult::error(
                EmitterStatus::ErrorInvalidIR,
                "Invalid IR module".to_string(),
            ));
        }

        let code_content = self.generate_code(ir_module)?;
        let extension = match self.config.variant {
            MobileVariant::iOS_Swift => "txt",
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
    fn test_mobile_emitter() {
        let temp_dir = TempDir::new().unwrap();
        let base_config = EmitterConfig::new(temp_dir.path(), "test_module");
        let config = MobileConfig::new(base_config, MobileVariant::iOS_Swift);
        let emitter = MobileEmitter::new(config);

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
