//! STUNIR Embedded Emitter - Rust Implementation
//!
//! Generates bare-metal C code for embedded systems (ARM, ARM64, RISC-V, MIPS, AVR, x86).
//! Based on DO-178C Level A compliant Ada SPARK implementation.
//!
//! Supports architectures:
//! - ARM (Cortex-M, Cortex-A)
//! - ARM64 (AArch64)
//! - RISC-V (RV32, RV64)
//! - MIPS
//! - AVR (ATmega, ATtiny)
//! - x86 (32/64-bit)

use crate::base::{BaseEmitter, EmitterConfig, EmitterError, EmitterResult, EmitterStatus};
use crate::types::{
    get_arch_config, map_ir_type_to_c, Architecture, GeneratedFile, IRDataType, IRFunction,
    IRModule, IRParameter, IRStatement, IRStatementType,
};
use std::fmt::Write;

/// Embedded emitter configuration
#[derive(Debug, Clone)]
pub struct EmbeddedConfig {
    /// Base emitter configuration
    pub base: EmitterConfig,
    /// Target architecture
    pub architecture: Architecture,
    /// Stack size in bytes
    pub stack_size: usize,
    /// Heap size in bytes (0 = no heap)
    pub heap_size: usize,
    /// Use fixed-width integer types
    pub use_fixed_width_types: bool,
    /// Generate linker script
    pub generate_linker_script: bool,
}

impl EmbeddedConfig {
    /// Create new embedded configuration
    pub fn new(base: EmitterConfig, architecture: Architecture) -> Self {
        Self {
            base,
            architecture,
            stack_size: 1024,
            heap_size: 0,
            use_fixed_width_types: true,
            generate_linker_script: true,
        }
    }
}

/// Embedded emitter for bare-metal C code
pub struct EmbeddedEmitter {
    config: EmbeddedConfig,
}

impl EmbeddedEmitter {
    /// Create new embedded emitter
    pub fn new(config: EmbeddedConfig) -> Self {
        Self { config }
    }

    /// Generate C header file
    fn generate_header(&self, ir_module: &IRModule) -> Result<String, EmitterError> {
        let mut content = String::new();

        // DO-178C header
        content.push_str(&self.get_do178c_header(
            &self.config.base,
            &format!(
                "Embedded C Header - {} Architecture",
                arch_name(&self.config.architecture)
            ),
        ));

        // Include guards
        let guard = format!("STUNIR_{}_H", self.config.base.module_name.to_uppercase());
        writeln!(content, "#ifndef {}", guard).unwrap();
        writeln!(content, "#define {}", guard).unwrap();
        writeln!(content).unwrap();

        // Standard includes for embedded
        writeln!(content, "/* Embedded System Includes */").unwrap();
        if self.config.use_fixed_width_types {
            writeln!(content, "#include <stdint.h>").unwrap();
        }
        writeln!(content, "#include <stddef.h>").unwrap();
        writeln!(content).unwrap();

        // Architecture info comment
        let arch_config = get_arch_config(self.config.architecture);
        writeln!(
            content,
            "/* Architecture: {} */",
            arch_name(&self.config.architecture)
        )
        .unwrap();
        writeln!(content, "/* Word Size: {} bits */", arch_config.word_size).unwrap();
        writeln!(content, "/* Endianness: {:?} */", arch_config.endianness).unwrap();
        writeln!(content, "/* Alignment: {} bytes */", arch_config.alignment).unwrap();
        writeln!(content).unwrap();

        // Type definitions
        if !ir_module.types.is_empty() {
            writeln!(content, "/* Type Definitions */").unwrap();
            for ir_type in &ir_module.types {
                if let Some(ref doc) = ir_type.docstring {
                    writeln!(content, "/* {} */", doc).unwrap();
                }
                writeln!(content, "typedef struct {} {{", ir_type.name).unwrap();
                for field in &ir_type.fields {
                    writeln!(content, "    {} {};", field.field_type, field.name).unwrap();
                }
                writeln!(content, "}} {};\n", ir_type.name).unwrap();
            }
            writeln!(content).unwrap();
        }

        // Function declarations
        if !ir_module.functions.is_empty() {
            writeln!(content, "/* Function Declarations */").unwrap();
            for function in &ir_module.functions {
                if let Some(ref doc) = function.docstring {
                    writeln!(content, "/* {} */", doc).unwrap();
                }
                let ret_type = map_ir_type_to_c(function.return_type);
                let params = self.format_parameters(&function.parameters);
                writeln!(content, "{} {}({});", ret_type, function.name, params).unwrap();
            }
            writeln!(content).unwrap();
        }

        // Close include guard
        writeln!(content, "#endif /* {} */", guard).unwrap();

        Ok(content)
    }

    /// Generate C source file
    fn generate_source(&self, ir_module: &IRModule) -> Result<String, EmitterError> {
        let mut content = String::new();

        // DO-178C header
        content.push_str(&self.get_do178c_header(
            &self.config.base,
            &format!(
                "Embedded C Source - {} Architecture",
                arch_name(&self.config.architecture)
            ),
        ));

        // Include header
        writeln!(content, "#include \"{}.h\"\n", self.config.base.module_name).unwrap();

        // Function implementations
        for function in &ir_module.functions {
            self.generate_function(&mut content, function)?;
            writeln!(content).unwrap();
        }

        Ok(content)
    }

    /// Generate function implementation
    fn generate_function(
        &self,
        content: &mut String,
        function: &IRFunction,
    ) -> Result<(), EmitterError> {
        if let Some(ref doc) = function.docstring {
            writeln!(content, "/* {} */", doc).unwrap();
        }

        let ret_type = map_ir_type_to_c(function.return_type);
        let params = self.format_parameters(&function.parameters);
        writeln!(content, "{} {}({}) {{", ret_type, function.name, params).unwrap();

        // Generate statements
        for stmt in &function.statements {
            self.generate_statement(content, stmt, 1)?;
        }

        writeln!(content, "}}").unwrap();
        Ok(())
    }

    /// Generate statement
    fn generate_statement(
        &self,
        content: &mut String,
        stmt: &IRStatement,
        indent_level: usize,
    ) -> Result<(), EmitterError> {
        let indent = self.indent(&self.config.base, indent_level);

        match stmt.stmt_type {
            IRStatementType::Nop => {
                writeln!(content, "{}/* nop */", indent).unwrap();
            }
            IRStatementType::VarDecl => {
                let c_type = stmt.data_type.map(map_ir_type_to_c).unwrap_or("int32_t");
                let var_name = stmt.target.as_deref().unwrap_or("v0");
                let init = stmt.value.as_deref().unwrap_or("0");
                writeln!(content, "{}{} {} = {};", indent, c_type, var_name, init).unwrap();
            }
            IRStatementType::Assign => {
                let target = stmt.target.as_deref().unwrap_or("v0");
                let value = stmt.value.as_deref().unwrap_or("0");
                writeln!(content, "{}{} = {};", indent, target, value).unwrap();
            }
            IRStatementType::Return => {
                let value = stmt.value.as_deref().unwrap_or("0");
                writeln!(content, "{}return {};", indent, value).unwrap();
            }
            IRStatementType::Add
            | IRStatementType::Sub
            | IRStatementType::Mul
            | IRStatementType::Div => {
                let target = stmt.target.as_deref().unwrap_or("v0");
                let left = stmt.left_op.as_deref().unwrap_or("0");
                let right = stmt.right_op.as_deref().unwrap_or("0");
                let op = match stmt.stmt_type {
                    IRStatementType::Add => "+",
                    IRStatementType::Sub => "-",
                    IRStatementType::Mul => "*",
                    IRStatementType::Div => "/",
                    _ => "+",
                };
                writeln!(content, "{}{} = {} {} {};", indent, target, left, op, right).unwrap();
            }
            IRStatementType::Call => {
                let func = stmt.target.as_deref().unwrap_or("noop");
                let args = stmt.value.as_deref().unwrap_or("");
                writeln!(content, "{}{}({});", indent, func, args).unwrap();
            }
            IRStatementType::If => {
                let condition = stmt.value.as_deref().unwrap_or("1");
                writeln!(content, "{}if ({}) {{", indent, condition).unwrap();
                writeln!(content, "{}    /* if body */", indent).unwrap();
                writeln!(content, "{}}}", indent).unwrap();
            }
            IRStatementType::Loop => {
                writeln!(content, "{}while (1) {{", indent).unwrap();
                writeln!(content, "{}    /* loop body */", indent).unwrap();
                writeln!(content, "{}}}", indent).unwrap();
            }
            IRStatementType::Break => {
                writeln!(content, "{}break;", indent).unwrap();
            }
            IRStatementType::Continue => {
                writeln!(content, "{}continue;", indent).unwrap();
            }
            IRStatementType::Block => {
                writeln!(content, "{}{{", indent).unwrap();
                writeln!(content, "{}    /* block */", indent).unwrap();
                writeln!(content, "{}}}", indent).unwrap();
            }
        }

        Ok(())
    }

    /// Format function parameters
    fn format_parameters(&self, params: &[IRParameter]) -> String {
        if params.is_empty() {
            return "void".to_string();
        }
        params
            .iter()
            .map(|p| format!("{} {}", map_ir_type_to_c(p.param_type), p.name))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Generate linker script
    fn generate_linker_script(&self) -> Result<String, EmitterError> {
        let mut content = String::new();

        writeln!(content, "/* STUNIR Generated Linker Script */").unwrap();
        writeln!(
            content,
            "/* Architecture: {} */\n",
            arch_name(&self.config.architecture)
        )
        .unwrap();

        writeln!(content, "MEMORY {{").unwrap();
        writeln!(
            content,
            "    FLASH (rx) : ORIGIN = 0x00000000, LENGTH = 256K"
        )
        .unwrap();
        writeln!(
            content,
            "    RAM (rwx)  : ORIGIN = 0x20000000, LENGTH = 64K"
        )
        .unwrap();
        writeln!(content, "}}\n").unwrap();

        writeln!(content, "SECTIONS {{").unwrap();
        writeln!(content, "    .text : {{").unwrap();
        writeln!(content, "        *(.text*)").unwrap();
        writeln!(content, "    }} > FLASH\n").unwrap();

        writeln!(content, "    .data : {{").unwrap();
        writeln!(content, "        *(.data*)").unwrap();
        writeln!(content, "    }} > RAM\n").unwrap();

        writeln!(content, "    .bss : {{").unwrap();
        writeln!(content, "        *(.bss*)").unwrap();
        writeln!(content, "    }} > RAM\n").unwrap();

        writeln!(content, "    .stack : {{").unwrap();
        writeln!(content, "        . = . + {};", self.config.stack_size).unwrap();
        writeln!(content, "        _stack_top = .;").unwrap();
        writeln!(content, "    }} > RAM").unwrap();
        writeln!(content, "}}").unwrap();

        Ok(content)
    }
}

impl BaseEmitter for EmbeddedEmitter {
    fn emit(&self, ir_module: &IRModule) -> Result<EmitterResult, EmitterError> {
        if !self.validate_ir(ir_module) {
            return Ok(EmitterResult::error(
                EmitterStatus::ErrorInvalidIR,
                "Invalid IR module".to_string(),
            ));
        }

        let mut files = Vec::new();
        let mut total_size = 0;

        // Generate header
        let header_content = self.generate_header(ir_module)?;
        let header_file = self.write_file(
            &self.config.base.output_dir,
            &format!("{}.h", self.config.base.module_name),
            &header_content,
        )?;
        total_size += header_file.size;
        files.push(header_file);

        // Generate source
        let source_content = self.generate_source(ir_module)?;
        let source_file = self.write_file(
            &self.config.base.output_dir,
            &format!("{}.c", self.config.base.module_name),
            &source_content,
        )?;
        total_size += source_file.size;
        files.push(source_file);

        // Generate linker script
        if self.config.generate_linker_script {
            let linker_content = self.generate_linker_script()?;
            let linker_file =
                self.write_file(&self.config.base.output_dir, "linker.ld", &linker_content)?;
            total_size += linker_file.size;
            files.push(linker_file);
        }

        Ok(EmitterResult::success(files, total_size))
    }
}

/// Get architecture name as string
fn arch_name(arch: &Architecture) -> &'static str {
    match arch {
        Architecture::ARM => "ARM",
        Architecture::ARM64 => "ARM64",
        Architecture::AVR => "AVR",
        Architecture::MIPS => "MIPS",
        Architecture::RISCV => "RISC-V",
        Architecture::X86 => "x86",
        Architecture::X86_64 => "x86_64",
        Architecture::PowerPC => "PowerPC",
        Architecture::Generic => "Generic",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{IRFunction, IRModule, IRParameter};
    use tempfile::TempDir;

    #[test]
    fn test_embedded_emitter_arm() {
        let temp_dir = TempDir::new().unwrap();
        let base_config = EmitterConfig::new(temp_dir.path(), "test_module");
        let config = EmbeddedConfig::new(base_config, Architecture::ARM);
        let emitter = EmbeddedEmitter::new(config);

        let ir_module = IRModule {
            ir_version: "1.0".to_string(),
            module_name: "test_module".to_string(),
            types: vec![],
            functions: vec![IRFunction {
                name: "test_func".to_string(),
                return_type: IRDataType::I32,
                parameters: vec![IRParameter {
                    name: "x".to_string(),
                    param_type: IRDataType::I32,
                }],
                statements: vec![],
                docstring: Some("Test function".to_string()),
            }],
            docstring: None,
        };

        let result = emitter.emit(&ir_module).unwrap();
        assert_eq!(result.status, EmitterStatus::Success);
        assert_eq!(result.files.len(), 3); // header, source, linker
    }

    #[test]
    fn test_arch_name() {
        assert_eq!(arch_name(&Architecture::ARM), "ARM");
        assert_eq!(arch_name(&Architecture::RISCV), "RISC-V");
        assert_eq!(arch_name(&Architecture::AVR), "AVR");
    }
}
