//! STUNIR Code Generator Utilities - Rust Implementation
//!
//! Utility functions for code generation across all emitters.
//! Based on Ada SPARK code generation utilities.

use crate::types::IRDataType;
use regex::Regex;

/// Code generation utilities
pub struct CodeGenerator;

impl CodeGenerator {
    /// Sanitize identifier to be safe for code generation
    pub fn sanitize_identifier(name: &str) -> String {
        // Remove invalid characters
        let re = Regex::new(r"[^a-zA-Z0-9_]").unwrap();
        let sanitized = re.replace_all(name, "_");

        // Ensure starts with letter or underscore
        if sanitized.chars().next().map_or(false, |c| c.is_numeric()) {
            format!("_{}", sanitized)
        } else if sanitized.is_empty() {
            "unnamed".to_string()
        } else {
            sanitized.to_string()
        }
    }

    /// Escape string for code generation
    pub fn escape_string(s: &str, style: &str) -> String {
        match style {
            "c" => s
                .replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\t", "\\t"),
            "python" | "rust" => s.replace("\\", "\\\\").replace("\"", "\\\""),
            _ => s.to_string(),
        }
    }

    /// Generate C/C++ include guard macros
    pub fn generate_include_guard(filename: &str) -> (String, String) {
        let guard = filename
            .to_uppercase()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect::<String>();
        let guard = format!("STUNIR_{}_", guard);

        let start = format!("#ifndef {}\n#define {}", guard, guard);
        let end = format!("#endif /* {} */", guard);

        (start, end)
    }

    /// Map IR type to language-specific type
    pub fn map_type_to_language(ir_type: IRDataType, language: &str) -> String {
        match language {
            "c" => Self::map_type_to_c(ir_type),
            "python" => Self::map_type_to_python(ir_type),
            "rust" => Self::map_type_to_rust(ir_type),
            "haskell" => Self::map_type_to_haskell(ir_type),
            _ => "unknown".to_string(),
        }
    }

    /// Map IR type to C type
    pub fn map_type_to_c(ir_type: IRDataType) -> String {
        match ir_type {
            IRDataType::Void => "void",
            IRDataType::Bool => "bool",
            IRDataType::I8 => "int8_t",
            IRDataType::I16 => "int16_t",
            IRDataType::I32 => "int32_t",
            IRDataType::I64 => "int64_t",
            IRDataType::U8 => "uint8_t",
            IRDataType::U16 => "uint16_t",
            IRDataType::U32 => "uint32_t",
            IRDataType::U64 => "uint64_t",
            IRDataType::F32 => "float",
            IRDataType::F64 => "double",
            IRDataType::Char => "char",
            IRDataType::String => "char*",
            IRDataType::Pointer => "void*",
            _ => "void",
        }
        .to_string()
    }

    /// Map IR type to Python type
    pub fn map_type_to_python(ir_type: IRDataType) -> String {
        match ir_type {
            IRDataType::Bool => "bool",
            IRDataType::I32 | IRDataType::I64 => "int",
            IRDataType::F32 | IRDataType::F64 => "float",
            IRDataType::String => "str",
            _ => "Any",
        }
        .to_string()
    }

    /// Map IR type to Rust type
    pub fn map_type_to_rust(ir_type: IRDataType) -> String {
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
            _ => "()",
        }
        .to_string()
    }

    /// Map IR type to Haskell type
    pub fn map_type_to_haskell(ir_type: IRDataType) -> String {
        match ir_type {
            IRDataType::Void => "()",
            IRDataType::Bool => "Bool",
            IRDataType::I32 => "Int32",
            IRDataType::I64 => "Int64",
            IRDataType::F32 => "Float",
            IRDataType::F64 => "Double",
            IRDataType::String => "String",
            _ => "()",
        }
        .to_string()
    }

    /// Generate function signature for target language
    pub fn generate_function_signature(
        name: &str,
        params: &[(String, String)],
        return_type: &str,
        language: &str,
    ) -> String {
        match language {
            "c" => {
                let param_str = if params.is_empty() {
                    "void".to_string()
                } else {
                    params
                        .iter()
                        .map(|(n, t)| format!("{} {}", t, n))
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                format!("{} {}({})", return_type, name, param_str)
            }
            "python" => {
                let param_str = params
                    .iter()
                    .map(|(n, t)| format!("{}: {}", n, t))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("def {}({}) -> {}:", name, param_str, return_type)
            }
            "rust" => {
                let param_str = params
                    .iter()
                    .map(|(n, t)| format!("{}: {}", n, t))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("fn {}({}) -> {}", name, param_str, return_type)
            }
            "haskell" => {
                let type_sig = params
                    .iter()
                    .map(|(_, t)| t.as_str())
                    .chain(std::iter::once(return_type))
                    .collect::<Vec<_>>()
                    .join(" -> ");
                format!("{} :: {}", name, type_sig)
            }
            _ => format!("{}(...)", name),
        }
    }
}
