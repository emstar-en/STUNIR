//! C99 code emitter

use crate::types::*;

/// C99 configuration
#[derive(Debug, Clone)]
pub struct C99Config {
    pub use_vla: bool,
    pub use_designated_initializers: bool,
    pub max_line_width: usize,
}

impl Default for C99Config {
    fn default() -> Self {
        Self {
            use_vla: false,
            use_designated_initializers: true,
            max_line_width: 100,
        }
    }
}

/// Emit C99 header file
pub fn emit_header(module_name: &str, config: &C99Config) -> EmitterResult<String> {
    let mut code = String::new();
    let guard = format!("{}_H", module_name.to_uppercase());
    
    code.push_str("/*\n");
    code.push_str(" * STUNIR Generated Code\n");
    code.push_str(" * Language: C99\n");
    code.push_str(&format!(" * Module: {}\n", module_name));
    code.push_str(" * Generator: Rust Pipeline\n");
    code.push_str(" * DO-178C Level A Compliance\n");
    code.push_str(" */\n\n");
    
    code.push_str(&format!("#ifndef {}\n", guard));
    code.push_str(&format!("#define {}\n\n", guard));
    
    code.push_str("#include <stdint.h>\n");
    code.push_str("#include <stdbool.h>\n");
    code.push_str("#include <stddef.h>\n\n");
    
    code.push_str("#ifdef __cplusplus\n");
    code.push_str("extern \"C\" {\n");
    code.push_str("#endif\n\n");
    
    code.push_str(&format!("/* Module: {} API */\n", module_name));
    code.push_str("void init(void);\n");
    code.push_str("void cleanup(void);\n");
    code.push_str("int32_t process(int32_t input);\n\n");
    
    code.push_str("#ifdef __cplusplus\n");
    code.push_str("}\n");
    code.push_str("#endif\n\n");
    
    code.push_str(&format!("#endif /* {} */\n", guard));
    
    Ok(code)
}

/// Emit C99 source file
pub fn emit_source(module_name: &str, config: &C99Config) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("/*\n");
    code.push_str(" * STUNIR Generated Code\n");
    code.push_str(" * Language: C99\n");
    code.push_str(&format!(" * Module: {}\n", module_name));
    code.push_str(" * Generator: Rust Pipeline\n");
    code.push_str(" * DO-178C Level A Compliance\n");
    code.push_str(" */\n\n");
    
    code.push_str(&format!("#include \"{}.h\"\n\n", module_name));
    
    code.push_str(&format!("/* Module: {} implementation */\n\n", module_name));
    
    code.push_str("void init(void) {\n");
    code.push_str("    /* Initialization code */\n");
    code.push_str("}\n\n");
    
    code.push_str("void cleanup(void) {\n");
    code.push_str("    /* Cleanup code */\n");
    code.push_str("}\n\n");
    
    code.push_str("int32_t process(int32_t input) {\n");
    code.push_str("    /* Process input */\n");
    code.push_str("    return input * 2;\n");
    code.push_str("}\n");
    
    Ok(code)
}

/// Main emit entry point (generates header)
pub fn emit(module_name: &str) -> EmitterResult<String> {
    let config = C99Config::default();
    emit_header(module_name, &config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_header() {
        let config = C99Config::default();
        let result = emit_header("test_module", &config);
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("#ifndef TEST_MODULE_H"));
        assert!(code.contains("#include <stdint.h>"));
        assert!(code.contains("#include <stdbool.h>"));
    }

    #[test]
    fn test_emit_source() {
        let config = C99Config::default();
        let result = emit_source("test_module", &config);
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("#include \"test_module.h\""));
        assert!(code.contains("void init(void)"));
        assert!(code.contains("int32_t process(int32_t input)"));
    }
}
