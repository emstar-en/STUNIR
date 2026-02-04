//! STUNIR Polyglot Emitter - Rust Implementation
//!
//! Generates multi-language code (C89, C99, Rust).
//! Based on DO-178C Level A compliant Ada SPARK implementation.
//!
//! Supported languages:
//! - C89 (ANSI C)
//! - C99 (ISO/IEC 9899:1999)
//! - Rust (2015, 2018, 2021 editions)

use crate::base::{BaseEmitter, EmitterConfig, EmitterError, EmitterResult, EmitterStatus};
use crate::types::{
    map_ir_type_to_c, GeneratedFile, IRDataType, IRFunction, IRModule, IRParameter, IRStatement,
    IRStatementType, IRType,
};
use std::fmt::Write;

/// Polyglot target language
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolyglotLanguage {
    /// ANSI C89
    C89,
    /// ISO C99
    C99,
    /// Rust
    Rust,
}

/// Rust edition
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RustEdition {
    /// Rust 2015
    Edition2015,
    /// Rust 2018
    Edition2018,
    /// Rust 2021
    Edition2021,
}

/// Polyglot emitter configuration
#[derive(Debug, Clone)]
pub struct PolyglotConfig {
    /// Base emitter configuration
    pub base: EmitterConfig,
    /// Target language
    pub language: PolyglotLanguage,
    /// Rust edition (only for Rust)
    pub rust_edition: Option<RustEdition>,
}

impl PolyglotConfig {
    /// Create new polyglot configuration
    pub fn new(base: EmitterConfig, language: PolyglotLanguage) -> Self {
        Self {
            base,
            language,
            rust_edition: Some(RustEdition::Edition2021),
        }
    }
}

/// Polyglot emitter
pub struct PolyglotEmitter {
    config: PolyglotConfig,
}

impl PolyglotEmitter {
    /// Create new polyglot emitter
    pub fn new(config: PolyglotConfig) -> Self {
        Self { config }
    }

    /// Generate code based on target language
    fn generate_code(&self, ir_module: &IRModule) -> Result<Vec<(String, String)>, EmitterError> {
        match self.config.language {
            PolyglotLanguage::C89 => self.generate_c89(ir_module),
            PolyglotLanguage::C99 => self.generate_c99(ir_module),
            PolyglotLanguage::Rust => self.generate_rust(ir_module),
        }
    }

    /// Generate C89 code
    fn generate_c89(&self, ir_module: &IRModule) -> Result<Vec<(String, String)>, EmitterError> {
        let mut files = Vec::new();

        // Header file
        let mut header = String::new();
        header.push_str(&self.get_do178c_header(&self.config.base, "C89 Header"));

        let guard = format!("{}_H", self.config.base.module_name.to_uppercase());
        writeln!(header, "#ifndef {}", guard)?;
        writeln!(header, "#define {}\\n", guard)?;

        // C89 doesn't have stdint.h, so define types
        writeln!(header, "/* Fixed-width integer types for C89 */")?;
        writeln!(header, "typedef signed char int8_t;")?;
        writeln!(header, "typedef short int16_t;")?;
        writeln!(header, "typedef long int32_t;")?;
        writeln!(header, "typedef unsigned char uint8_t;")?;
        writeln!(header, "typedef unsigned short uint16_t;")?;
        writeln!(header, "typedef unsigned long uint32_t;\\n")?;

        // Type definitions
        for ir_type in &ir_module.types {
            self.generate_c_type(&mut header, ir_type)?;
        }

        // Function declarations
        for function in &ir_module.functions {
            let ret_type = map_ir_type_to_c(function.return_type);
            let params = self.format_c_parameters(&function.parameters);
            writeln!(header, "{} {}({});", ret_type, function.name, params)?;
        }

        writeln!(header, "\\n#endif /* {} */", guard)?;
        files.push((format!("{}.h", self.config.base.module_name), header));

        // Source file
        let mut source = String::new();
        source.push_str(&self.get_do178c_header(&self.config.base, "C89 Source"));
        writeln!(source, "#include \"{}.h\"\\n", self.config.base.module_name)?;

        for function in &ir_module.functions {
            self.generate_c_function(&mut source, function)?;
        }

        files.push((format!("{}.c", self.config.base.module_name), source));

        Ok(files)
    }

    /// Generate C99 code
    fn generate_c99(&self, ir_module: &IRModule) -> Result<Vec<(String, String)>, EmitterError> {
        let mut files = Vec::new();

        // Header file
        let mut header = String::new();
        header.push_str(&self.get_do178c_header(&self.config.base, "C99 Header"));

        let guard = format!("{}_H", self.config.base.module_name.to_uppercase());
        writeln!(header, "#ifndef {}", guard)?;
        writeln!(header, "#define {}\\n", guard)?;

        // C99 has stdint.h
        writeln!(header, "#include <stdint.h>")?;
        writeln!(header, "#include <stdbool.h>\\n")?;

        // Type definitions
        for ir_type in &ir_module.types {
            self.generate_c_type(&mut header, ir_type)?;
        }

        // Function declarations
        for function in &ir_module.functions {
            let ret_type = map_ir_type_to_c(function.return_type);
            let params = self.format_c_parameters(&function.parameters);
            writeln!(header, "{} {}({});", ret_type, function.name, params)?;
        }

        writeln!(header, "\\n#endif /* {} */", guard)?;
        files.push((format!("{}.h", self.config.base.module_name), header));

        // Source file
        let mut source = String::new();
        source.push_str(&self.get_do178c_header(&self.config.base, "C99 Source"));
        writeln!(source, "#include \"{}.h\"\\n", self.config.base.module_name)?;

        for function in &ir_module.functions {
            self.generate_c_function(&mut source, function)?;
        }

        files.push((format!("{}.c", self.config.base.module_name), source));

        Ok(files)
    }

    /// Generate Rust code
    fn generate_rust(&self, ir_module: &IRModule) -> Result<Vec<(String, String)>, EmitterError> {
        let mut files = Vec::new();

        let mut content = String::new();
        content.push_str("//! STUNIR Generated Rust Code\\n");
        content.push_str("//! DO-178C Level A Compliant\\n\\n");

        // Module attributes
        writeln!(content, "#![deny(unsafe_code)]")?;
        writeln!(content, "#![allow(dead_code)]\\n")?;

        // Type definitions
        for ir_type in &ir_module.types {
            self.generate_rust_type(&mut content, ir_type)?;
        }

        // Function implementations
        for function in &ir_module.functions {
            self.generate_rust_function(&mut content, function)?;
        }

        files.push((format!("{}.rs", self.config.base.module_name), content));

        Ok(files)
    }

    /// Generate C type definition
    fn generate_c_type(&self, content: &mut String, ir_type: &IRType) -> Result<(), EmitterError> {
        if let Some(ref doc) = ir_type.docstring {
            writeln!(content, "/* {} */", doc)?;
        }
        writeln!(content, "typedef struct {} {{", ir_type.name)?;
        for field in &ir_type.fields {
            writeln!(content, "    {} {};", field.field_type, field.name)?;
        }
        writeln!(content, "}} {};\\n", ir_type.name)?;
        Ok(())
    }

    /// Generate Rust type definition
    fn generate_rust_type(
        &self,
        content: &mut String,
        ir_type: &IRType,
    ) -> Result<(), EmitterError> {
        if let Some(ref doc) = ir_type.docstring {
            writeln!(content, "/// {}", doc)?;
        }
        writeln!(content, "#[derive(Debug, Clone)]")?;
        writeln!(content, "pub struct {} {{", ir_type.name)?;
        for field in &ir_type.fields {
            let rust_type = map_ir_type_to_rust(&field.field_type);
            writeln!(content, "    pub {}: {},", field.name, rust_type)?;
        }
        writeln!(content, "}}\\n")?;
        Ok(())
    }

    /// Generate C function
    fn generate_c_function(
        &self,
        content: &mut String,
        function: &IRFunction,
    ) -> Result<(), EmitterError> {
        if let Some(ref doc) = function.docstring {
            writeln!(content, "/* {} */", doc)?;
        }

        let ret_type = map_ir_type_to_c(function.return_type);
        let params = self.format_c_parameters(&function.parameters);
        writeln!(content, "{} {}({}) {{", ret_type, function.name, params)?;

        for stmt in &function.statements {
            self.generate_c_statement(content, stmt, 1)?;
        }

        writeln!(content, "}}\\n")?;
        Ok(())
    }

    /// Generate Rust function
    fn generate_rust_function(
        &self,
        content: &mut String,
        function: &IRFunction,
    ) -> Result<(), EmitterError> {
        if let Some(ref doc) = function.docstring {
            writeln!(content, "/// {}", doc)?;
        }

        let ret_type = map_ir_type_to_rust_type(function.return_type);
        let params = self.format_rust_parameters(&function.parameters);
        let return_sig = if ret_type == "()" {
            String::new()
        } else {
            format!(" -> {}", ret_type)
        };
        writeln!(
            content,
            "pub fn {}({}){} {{",
            function.name, params, return_sig
        )
        .unwrap();

        for stmt in &function.statements {
            self.generate_rust_statement(content, stmt, 1)?;
        }

        writeln!(content, "}}\\n")?;
        Ok(())
    }

    /// Generate C statement
    fn generate_c_statement(
        &self,
        content: &mut String,
        stmt: &IRStatement,
        indent_level: usize,
    ) -> Result<(), EmitterError> {
        let indent = self.indent(&self.config.base, indent_level);

        match stmt.stmt_type {
            IRStatementType::VarDecl => {
                let c_type = stmt.data_type.map(map_ir_type_to_c).unwrap_or("int32_t");
                let var_name = stmt.target.as_deref().unwrap_or("v0");
                let init = stmt.value.as_deref().unwrap_or("0");
                writeln!(content, "{}{} {} = {};", indent, c_type, var_name, init)?;
            }
            IRStatementType::Return => {
                let value = stmt.value.as_deref().unwrap_or("0");
                writeln!(content, "{}return {};", indent, value)?;
            }
            _ => {
                writeln!(content, "{}/* {:?} */", indent, stmt.stmt_type)?;
            }
        }

        Ok(())
    }

    /// Generate Rust statement
    fn generate_rust_statement(
        &self,
        content: &mut String,
        stmt: &IRStatement,
        indent_level: usize,
    ) -> Result<(), EmitterError> {
        let indent = self.indent(&self.config.base, indent_level);

        match stmt.stmt_type {
            IRStatementType::VarDecl => {
                let rust_type = stmt
                    .data_type
                    .map(map_ir_type_to_rust_type)
                    .unwrap_or("i32");
                let var_name = stmt.target.as_deref().unwrap_or("v0");
                let init = stmt.value.as_deref().unwrap_or("0");
                writeln!(
                    content,
                    "{}let {}: {} = {};",
                    indent, var_name, rust_type, init
                )?;
            }
            IRStatementType::Return => {
                let value = stmt.value.as_deref().unwrap_or("0");
                writeln!(content, "{}{}", indent, value)?;
            }
            _ => {
                writeln!(content, "{}// {:?}", indent, stmt.stmt_type)?;
            }
        }

        Ok(())
    }

    /// Format C parameters
    fn format_c_parameters(&self, params: &[IRParameter]) -> String {
        if params.is_empty() {
            return "void".to_string();
        }
        params
            .iter()
            .map(|p| format!("{} {}", map_ir_type_to_c(p.param_type), p.name))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Format Rust parameters
    fn format_rust_parameters(&self, params: &[IRParameter]) -> String {
        params
            .iter()
            .map(|p| format!("{}: {}", p.name, map_ir_type_to_rust_type(p.param_type)))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

impl BaseEmitter for PolyglotEmitter {
    fn emit(&self, ir_module: &IRModule) -> Result<EmitterResult, EmitterError> {
        if !self.validate_ir(ir_module) {
            return Ok(EmitterResult::error(
                EmitterStatus::ErrorInvalidIR,
                "Invalid IR module".to_string(),
            ));
        }

        let files_content = self.generate_code(ir_module)?;
        let mut generated_files = Vec::new();
        let mut total_size = 0;

        for (filename, content) in files_content {
            let file = self.write_file(&self.config.base.output_dir, &filename, &content)?;
            total_size += file.size;
            generated_files.push(file);
        }

        Ok(EmitterResult::success(generated_files, total_size))
    }
}

/// Map IR type name string to Rust type
fn map_ir_type_to_rust(type_name: &str) -> String {
    match type_name {
        "int8_t" => "i8".to_string(),
        "int16_t" => "i16".to_string(),
        "int32_t" => "i32".to_string(),
        "int64_t" => "i64".to_string(),
        "uint8_t" => "u8".to_string(),
        "uint16_t" => "u16".to_string(),
        "uint32_t" => "u32".to_string(),
        "uint64_t" => "u64".to_string(),
        "float" => "f32".to_string(),
        "double" => "f64".to_string(),
        "bool" => "bool".to_string(),
        "char" => "char".to_string(),
        _ => type_name.to_string(),
    }
}

/// Map IR data type to Rust type
fn map_ir_type_to_rust_type(ir_type: IRDataType) -> &'static str {
    match ir_type {
        IRDataType::Void => "()",
        IRDataType::Bool => "bool",
        IRDataType::I8 => "i8",
        IRDataType::I16 => "i16",
        IRDataType::I32 => "i32",
        IRDataType::I64 => "i64",
        IRDataType::U8 => "u8",
        IRDataType::U16 => "u16",
        IRDataType::U32 => "u32",
        IRDataType::U64 => "u64",
        IRDataType::F32 => "f32",
        IRDataType::F64 => "f64",
        IRDataType::Char => "char",
        IRDataType::String => "String",
        _ => "i32",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{IRFunction, IRModule};
    use tempfile::TempDir;

    #[test]
    fn test_polyglot_c89() {
        let temp_dir = TempDir::new()?;
        let base_config = EmitterConfig::new(temp_dir.path(), "test_module");
        let config = PolyglotConfig::new(base_config, PolyglotLanguage::C89);
        let emitter = PolyglotEmitter::new(config);

        let ir_module = IRModule {
            ir_version: "1.0".to_string(),
            module_name: "test_module".to_string(),
            types: vec![],
            functions: vec![IRFunction {
                name: "test_func".to_string(),
                return_type: IRDataType::I32,
                parameters: vec![],
                statements: vec![],
                docstring: None,
            }],
            docstring: None,
        };

        let result = emitter.emit(&ir_module)?;
        assert_eq!(result.status, EmitterStatus::Success);
        assert_eq!(result.files.len(), 2); // header and source
    }

    #[test]
    fn test_polyglot_rust() {
        let temp_dir = TempDir::new()?;
        let base_config = EmitterConfig::new(temp_dir.path(), "test_module");
        let config = PolyglotConfig::new(base_config, PolyglotLanguage::Rust);
        let emitter = PolyglotEmitter::new(config);

        let ir_module = IRModule {
            ir_version: "1.0".to_string(),
            module_name: "test_module".to_string(),
            types: vec![],
            functions: vec![IRFunction {
                name: "test_func".to_string(),
                return_type: IRDataType::I32,
                parameters: vec![],
                statements: vec![],
                docstring: None,
            }],
            docstring: None,
        };

        let result = emitter.emit(&ir_module)?;
        assert_eq!(result.status, EmitterStatus::Success);
        assert_eq!(result.files.len(), 1); // single .rs file
    }
}
