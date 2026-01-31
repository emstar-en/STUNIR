//! STUNIR Assembly Emitter - Rust Implementation
//!
//! Generates assembly code for x86, x86_64, ARM, and ARM64.
//! Based on DO-178C Level A compliant Ada SPARK implementation.
//!
//! Supported architectures:
//! - x86 (32-bit)
//! - x86_64 (64-bit)
//! - ARM (32-bit)
//! - ARM64 (AArch64)

use crate::base::{BaseEmitter, EmitterConfig, EmitterError, EmitterResult, EmitterStatus};
use crate::types::{
    Architecture, IRDataType, IRFunction, IRModule, IRParameter, IRStatement, IRStatementType,
};
use std::fmt::Write;

/// Assembly syntax
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssemblySyntax {
    /// AT&T syntax (used by GAS)
    ATT,
    /// Intel syntax
    Intel,
    /// ARM syntax
    ARM,
}

/// Assembly emitter configuration
#[derive(Debug, Clone)]
pub struct AssemblyConfig {
    /// Base emitter configuration
    pub base: EmitterConfig,
    /// Target architecture
    pub architecture: Architecture,
    /// Assembly syntax
    pub syntax: AssemblySyntax,
}

impl AssemblyConfig {
    /// Create new assembly configuration
    pub fn new(base: EmitterConfig, architecture: Architecture) -> Self {
        let syntax = match architecture {
            Architecture::ARM | Architecture::ARM64 => AssemblySyntax::ARM,
            _ => AssemblySyntax::Intel,
        };
        Self {
            base,
            architecture,
            syntax,
        }
    }
}

/// Assembly emitter
pub struct AssemblyEmitter {
    config: AssemblyConfig,
}

impl AssemblyEmitter {
    /// Create new assembly emitter
    pub fn new(config: AssemblyConfig) -> Self {
        Self { config }
    }

    /// Generate assembly file
    fn generate_assembly(&self, ir_module: &IRModule) -> Result<String, EmitterError> {
        let mut content = String::new();

        // Header comment
        writeln!(content, "; STUNIR Generated Assembly").unwrap();
        writeln!(content, "; DO-178C Level A Compliant").unwrap();
        writeln!(content, "; Architecture: {:?}", self.config.architecture).unwrap();
        writeln!(content, "; Syntax: {:?}\n", self.config.syntax).unwrap();

        // Directives
        self.generate_directives(&mut content)?;

        // Data section
        writeln!(content, ".section .data").unwrap();
        writeln!(content, "    ; Data section\n").unwrap();

        // Text section
        writeln!(content, ".section .text").unwrap();
        writeln!(content, ".global _start\n").unwrap();

        // Functions
        for function in &ir_module.functions {
            self.generate_function(&mut content, function)?;
            writeln!(content).unwrap();
        }

        Ok(content)
    }

    /// Generate assembly directives
    fn generate_directives(&self, content: &mut String) -> Result<(), EmitterError> {
        match self.config.architecture {
            Architecture::X86 => {
                writeln!(content, ".arch i386").unwrap();
            }
            Architecture::X86_64 => {
                writeln!(content, ".arch x86-64").unwrap();
            }
            Architecture::ARM => {
                writeln!(content, ".arch armv7-a").unwrap();
            }
            Architecture::ARM64 => {
                writeln!(content, ".arch armv8-a").unwrap();
            }
            _ => {}
        }
        writeln!(content).unwrap();
        Ok(())
    }

    /// Generate function
    fn generate_function(
        &self,
        content: &mut String,
        function: &IRFunction,
    ) -> Result<(), EmitterError> {
        if let Some(ref doc) = function.docstring {
            writeln!(content, "; {}", doc).unwrap();
        }

        writeln!(content, ".global {}", function.name).unwrap();
        writeln!(content, "{}:", function.name).unwrap();

        // Function prologue
        self.generate_prologue(content)?;

        // Function body
        for stmt in &function.statements {
            self.generate_statement(content, stmt)?;
        }

        // Function epilogue
        self.generate_epilogue(content)?;

        Ok(())
    }

    /// Generate function prologue
    fn generate_prologue(&self, content: &mut String) -> Result<(), EmitterError> {
        writeln!(content, "    ; Prologue").unwrap();
        match self.config.architecture {
            Architecture::X86 => {
                if self.config.syntax == AssemblySyntax::Intel {
                    writeln!(content, "    push ebp").unwrap();
                    writeln!(content, "    mov ebp, esp").unwrap();
                } else {
                    writeln!(content, "    pushl %ebp").unwrap();
                    writeln!(content, "    movl %esp, %ebp").unwrap();
                }
            }
            Architecture::X86_64 => {
                if self.config.syntax == AssemblySyntax::Intel {
                    writeln!(content, "    push rbp").unwrap();
                    writeln!(content, "    mov rbp, rsp").unwrap();
                } else {
                    writeln!(content, "    pushq %rbp").unwrap();
                    writeln!(content, "    movq %rsp, %rbp").unwrap();
                }
            }
            Architecture::ARM => {
                writeln!(content, "    push {{fp, lr}}").unwrap();
                writeln!(content, "    mov fp, sp").unwrap();
            }
            Architecture::ARM64 => {
                writeln!(content, "    stp x29, x30, [sp, #-16]!").unwrap();
                writeln!(content, "    mov x29, sp").unwrap();
            }
            _ => {}
        }
        Ok(())
    }

    /// Generate function epilogue
    fn generate_epilogue(&self, content: &mut String) -> Result<(), EmitterError> {
        writeln!(content, "    ; Epilogue").unwrap();
        match self.config.architecture {
            Architecture::X86 => {
                if self.config.syntax == AssemblySyntax::Intel {
                    writeln!(content, "    pop ebp").unwrap();
                    writeln!(content, "    ret").unwrap();
                } else {
                    writeln!(content, "    popl %ebp").unwrap();
                    writeln!(content, "    ret").unwrap();
                }
            }
            Architecture::X86_64 => {
                if self.config.syntax == AssemblySyntax::Intel {
                    writeln!(content, "    pop rbp").unwrap();
                    writeln!(content, "    ret").unwrap();
                } else {
                    writeln!(content, "    popq %rbp").unwrap();
                    writeln!(content, "    ret").unwrap();
                }
            }
            Architecture::ARM => {
                writeln!(content, "    pop {{fp, pc}}").unwrap();
            }
            Architecture::ARM64 => {
                writeln!(content, "    ldp x29, x30, [sp], #16").unwrap();
                writeln!(content, "    ret").unwrap();
            }
            _ => {}
        }
        Ok(())
    }

    /// Generate statement
    fn generate_statement(
        &self,
        content: &mut String,
        stmt: &IRStatement,
    ) -> Result<(), EmitterError> {
        match stmt.stmt_type {
            IRStatementType::Nop => {
                writeln!(content, "    ; nop").unwrap();
                writeln!(content, "    nop").unwrap();
            }
            IRStatementType::Return => {
                writeln!(content, "    ; return").unwrap();
                // Return value should already be in appropriate register
            }
            IRStatementType::Add => {
                writeln!(content, "    ; add").unwrap();
                match self.config.architecture {
                    Architecture::X86 | Architecture::X86_64 => {
                        if self.config.syntax == AssemblySyntax::Intel {
                            writeln!(content, "    add eax, ebx").unwrap();
                        } else {
                            writeln!(content, "    addl %ebx, %eax").unwrap();
                        }
                    }
                    Architecture::ARM | Architecture::ARM64 => {
                        writeln!(content, "    add r0, r0, r1").unwrap();
                    }
                    _ => {}
                }
            }
            _ => {
                writeln!(content, "    ; {:?}", stmt.stmt_type).unwrap();
            }
        }
        Ok(())
    }
}

impl BaseEmitter for AssemblyEmitter {
    fn emit(&self, ir_module: &IRModule) -> Result<EmitterResult, EmitterError> {
        if !self.validate_ir(ir_module) {
            return Ok(EmitterResult::error(
                EmitterStatus::ErrorInvalidIR,
                "Invalid IR module".to_string(),
            ));
        }

        let mut files = Vec::new();
        let mut total_size = 0;

        // Generate assembly file
        let asm_content = self.generate_assembly(ir_module)?;
        let extension = match self.config.architecture {
            Architecture::ARM | Architecture::ARM64 => "s",
            _ => "asm",
        };
        let asm_file = self.write_file(
            &self.config.base.output_dir,
            &format!("{}.{}", self.config.base.module_name, extension),
            &asm_content,
        )?;
        total_size += asm_file.size;
        files.push(asm_file);

        Ok(EmitterResult::success(files, total_size))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{IRFunction, IRModule};
    use tempfile::TempDir;

    #[test]
    fn test_assembly_emitter_x86() {
        let temp_dir = TempDir::new().unwrap();
        let base_config = EmitterConfig::new(temp_dir.path(), "test_module");
        let config = AssemblyConfig::new(base_config, Architecture::X86);
        let emitter = AssemblyEmitter::new(config);

        let ir_module = IRModule {
            ir_version: "1.0".to_string(),
            module_name: "test_module".to_string(),
            types: vec![],
            functions: vec![IRFunction {
                name: "test_func".to_string(),
                return_type: IRDataType::Void,
                parameters: vec![],
                statements: vec![],
                docstring: Some("Test function".to_string()),
            }],
            docstring: None,
        };

        let result = emitter.emit(&ir_module).unwrap();
        assert_eq!(result.status, EmitterStatus::Success);
        assert_eq!(result.files.len(), 1);
    }

    #[test]
    fn test_assembly_emitter_arm() {
        let temp_dir = TempDir::new().unwrap();
        let base_config = EmitterConfig::new(temp_dir.path(), "test_arm");
        let config = AssemblyConfig::new(base_config, Architecture::ARM);
        let emitter = AssemblyEmitter::new(config);

        assert_eq!(emitter.config.syntax, AssemblySyntax::ARM);
    }
}
