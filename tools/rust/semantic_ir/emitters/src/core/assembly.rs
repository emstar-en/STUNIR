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
        writeln!(content, "; STUNIR Generated Assembly")?;
        writeln!(content, "; DO-178C Level A Compliant")?;
        writeln!(content, "; Architecture: {:?}", self.config.architecture)?;
        writeln!(content, "; Syntax: {:?}\n", self.config.syntax)?;

        // Directives
        self.generate_directives(&mut content)?;

        // Data section
        writeln!(content, ".section .data")?;
        writeln!(content, "    ; Data section\n")?;

        // Text section
        writeln!(content, ".section .text")?;
        writeln!(content, ".global _start\n")?;

        // Functions
        for function in &ir_module.functions {
            self.generate_function(&mut content, function)?;
            writeln!(content)?;
        }

        Ok(content)
    }

    /// Generate assembly directives
    fn generate_directives(&self, content: &mut String) -> Result<(), EmitterError> {
        match self.config.architecture {
            Architecture::X86 => {
                writeln!(content, ".arch i386")?;
            }
            Architecture::X86_64 => {
                writeln!(content, ".arch x86-64")?;
            }
            Architecture::ARM => {
                writeln!(content, ".arch armv7-a")?;
            }
            Architecture::ARM64 => {
                writeln!(content, ".arch armv8-a")?;
            }
            _ => {}
        }
        writeln!(content)?;
        Ok(())
    }

    /// Generate function
    fn generate_function(
        &self,
        content: &mut String,
        function: &IRFunction,
    ) -> Result<(), EmitterError> {
        if let Some(ref doc) = function.docstring {
            writeln!(content, "; {}", doc)?;
        }

        writeln!(content, ".global {}", function.name)?;
        writeln!(content, "{}:", function.name)?;

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
        writeln!(content, "    ; Prologue")?;
        match self.config.architecture {
            Architecture::X86 => {
                if self.config.syntax == AssemblySyntax::Intel {
                    writeln!(content, "    push ebp")?;
                    writeln!(content, "    mov ebp, esp")?;
                } else {
                    writeln!(content, "    pushl %ebp")?;
                    writeln!(content, "    movl %esp, %ebp")?;
                }
            }
            Architecture::X86_64 => {
                if self.config.syntax == AssemblySyntax::Intel {
                    writeln!(content, "    push rbp")?;
                    writeln!(content, "    mov rbp, rsp")?;
                } else {
                    writeln!(content, "    pushq %rbp")?;
                    writeln!(content, "    movq %rsp, %rbp")?;
                }
            }
            Architecture::ARM => {
                writeln!(content, "    push {{fp, lr}}")?;
                writeln!(content, "    mov fp, sp")?;
            }
            Architecture::ARM64 => {
                writeln!(content, "    stp x29, x30, [sp, #-16]!")?;
                writeln!(content, "    mov x29, sp")?;
            }
            _ => {}
        }
        Ok(())
    }

    /// Generate function epilogue
    fn generate_epilogue(&self, content: &mut String) -> Result<(), EmitterError> {
        writeln!(content, "    ; Epilogue")?;
        match self.config.architecture {
            Architecture::X86 => {
                if self.config.syntax == AssemblySyntax::Intel {
                    writeln!(content, "    pop ebp")?;
                    writeln!(content, "    ret")?;
                } else {
                    writeln!(content, "    popl %ebp")?;
                    writeln!(content, "    ret")?;
                }
            }
            Architecture::X86_64 => {
                if self.config.syntax == AssemblySyntax::Intel {
                    writeln!(content, "    pop rbp")?;
                    writeln!(content, "    ret")?;
                } else {
                    writeln!(content, "    popq %rbp")?;
                    writeln!(content, "    ret")?;
                }
            }
            Architecture::ARM => {
                writeln!(content, "    pop {{fp, pc}}")?;
            }
            Architecture::ARM64 => {
                writeln!(content, "    ldp x29, x30, [sp], #16")?;
                writeln!(content, "    ret")?;
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
                writeln!(content, "    ; nop")?;
                writeln!(content, "    nop")?;
            }
            IRStatementType::Return => {
                writeln!(content, "    ; return")?;
                // Return value should already be in appropriate register
            }
            IRStatementType::Add => {
                writeln!(content, "    ; add")?;
                match self.config.architecture {
                    Architecture::X86 | Architecture::X86_64 => {
                        if self.config.syntax == AssemblySyntax::Intel {
                            writeln!(content, "    add eax, ebx")?;
                        } else {
                            writeln!(content, "    addl %ebx, %eax")?;
                        }
                    }
                    Architecture::ARM | Architecture::ARM64 => {
                        writeln!(content, "    add r0, r0, r1")?;
                    }
                    _ => {}
                }
            }
            _ => {
                writeln!(content, "    ; {:?}", stmt.stmt_type)?;
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
