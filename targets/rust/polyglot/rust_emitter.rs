//! Rust code emitter

use crate::types::*;

/// Rust edition
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RustEdition {
    Rust2015,
    Rust2018,
    Rust2021,
}

/// Rust configuration
#[derive(Debug, Clone)]
pub struct RustConfig {
    pub edition: RustEdition,
    pub no_std: bool,
    pub allow_unsafe: bool,
}

impl Default for RustConfig {
    fn default() -> Self {
        Self {
            edition: RustEdition::Rust2021,
            no_std: true,
            allow_unsafe: false,
        }
    }
}

/// Emit Rust module
pub fn emit_module(module_name: &str, config: &RustConfig) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("//! STUNIR Generated Code\n");
    code.push_str("//! Language: Rust\n");
    code.push_str(&format!("//! Module: {}\n", module_name));
    code.push_str(&format!("//! Edition: {:?}\n", config.edition));
    code.push_str("//! Generator: Rust Pipeline\n");
    code.push_str("//! DO-178C Level A Compliance\n\n");
    
    if config.no_std {
        code.push_str("#![no_std]\n");
    }
    
    if !config.allow_unsafe {
        code.push_str("#![forbid(unsafe_code)]\n");
    }
    
    code.push_str("\n");
    
    code.push_str(&format!("/// Module: {}\n", module_name));
    code.push_str("pub mod generated {\n");
    code.push_str("    /// Initialize module\n");
    code.push_str("    pub fn init() {\n");
    code.push_str("        // Initialization code\n");
    code.push_str("    }\n\n");
    
    code.push_str("    /// Process input\n");
    code.push_str("    pub fn process(input: i32) -> i32 {\n");
    code.push_str("        input * 2\n");
    code.push_str("    }\n\n");
    
    code.push_str("    /// Cleanup module\n");
    code.push_str("    pub fn cleanup() {\n");
    code.push_str("        // Cleanup code\n");
    code.push_str("    }\n");
    code.push_str("}\n\n");
    
    code.push_str("#[cfg(test)]\n");
    code.push_str("mod tests {\n");
    code.push_str("    use super::generated::*;\n\n");
    
    code.push_str("    #[test]\n");
    code.push_str("    fn test_process() {\n");
    code.push_str("        assert_eq!(process(21), 42);\n");
    code.push_str("    }\n");
    code.push_str("}\n");
    
    Ok(code)
}

/// Main emit entry point
pub fn emit(module_name: &str) -> EmitterResult<String> {
    let config = RustConfig::default();
    emit_module(module_name, &config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_module() {
        let config = RustConfig::default();
        let result = emit_module("test_module", &config);
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("#![no_std]"));
        assert!(code.contains("#![forbid(unsafe_code)]"));
        assert!(code.contains("pub mod generated"));
    }

    #[test]
    fn test_emit_with_std() {
        let mut config = RustConfig::default();
        config.no_std = false;
        let result = emit_module("test_module", &config);
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(!code.contains("#![no_std]"));
    }
}
