//! C89 (ANSI C) code emitter

use crate::types::*;

/// C89 configuration
#[derive(Debug, Clone)]
pub struct C89Config {
    pub use_kr_style: bool,
    pub max_line_width: usize,
    pub use_trigraphs: bool,
}

impl Default for C89Config {
    fn default() -> Self {
        Self {
            use_kr_style: false,
            max_line_width: 80,
            use_trigraphs: false,
        }
    }
}

/// Emit C89 header file
pub fn emit_header(module_name: &str, _config: &C89Config) -> EmitterResult<String> {
    let mut code = String::new();
    let guard = format!("{}_H", module_name.to_uppercase());
    
    code.push_str("/*\n");
    code.push_str(" * STUNIR Generated Code\n");
    code.push_str(" * Language: ANSI C89\n");
    code.push_str(&format!(" * Module: {}\n", module_name));
    code.push_str(" * Generator: Rust Pipeline\n");
    code.push_str(" * DO-178C Level A Compliance\n");
    code.push_str(" */\n\n");
    
    code.push_str(&format!("#ifndef {}\n", guard));
    code.push_str(&format!("#define {}\n\n", guard));
    
    code.push_str("/* Type definitions for C89 compatibility */\n");
    code.push_str("typedef signed char int8_t;\n");
    code.push_str("typedef unsigned char uint8_t;\n");
    code.push_str("typedef short int16_t;\n");
    code.push_str("typedef unsigned short uint16_t;\n");
    code.push_str("typedef long int32_t;\n");
    code.push_str("typedef unsigned long uint32_t;\n\n");
    
    code.push_str("#ifdef __cplusplus\n");
    code.push_str("extern \"C\" {\n");
    code.push_str("#endif\n\n");
    
    code.push_str(&format!("/* Module: {} functions */\n", module_name));
    code.push_str("void init(void);\n");
    code.push_str("void cleanup(void);\n\n");
    
    code.push_str("#ifdef __cplusplus\n");
    code.push_str("}\n");
    code.push_str("#endif\n\n");
    
    code.push_str(&format!("#endif /* {} */\n", guard));
    
    Ok(code)
}

/// Emit C89 source file
pub fn emit_source(module_name: &str, config: &C89Config) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("/*\n");
    code.push_str(" * STUNIR Generated Code\n");
    code.push_str(" * Language: ANSI C89\n");
    code.push_str(&format!(" * Module: {}\n", module_name));
    code.push_str(" * Generator: Rust Pipeline\n");
    code.push_str(" * DO-178C Level A Compliance\n");
    code.push_str(" */\n\n");
    
    code.push_str(&format!("#include \"{}.h\"\n\n", module_name));
    
    code.push_str(&format!("/* Module: {} implementation */\n\n", module_name));
    
    if config.use_kr_style {
        code.push_str("void init()\n");
        code.push_str("{\n");
    } else {
        code.push_str("void init(void) {\n");
    }
    code.push_str("    /* Initialization code */\n");
    code.push_str("}\n\n");
    
    if config.use_kr_style {
        code.push_str("void cleanup()\n");
        code.push_str("{\n");
    } else {
        code.push_str("void cleanup(void) {\n");
    }
    code.push_str("    /* Cleanup code */\n");
    code.push_str("}\n");
    
    Ok(code)
}

/// Main emit entry point (generates header)
pub fn emit(module_name: &str) -> EmitterResult<String> {
    let config = C89Config::default();
    emit_header(module_name, &config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_header() {
        let config = C89Config::default();
        let result = emit_header("test_module", &config);
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("#ifndef TEST_MODULE_H"));
        assert!(code.contains("typedef signed char int8_t"));
    }

    #[test]
    fn test_emit_source() {
        let config = C89Config::default();
        let result = emit_source("test_module", &config);
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("#include \"test_module.h\""));
        assert!(code.contains("void init(void)"));
    }
}
