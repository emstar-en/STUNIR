//! Embedded system emitters
//!
//! Supports: ARM Cortex-M, AVR, RISC-V, bare-metal

use crate::types::*;
use std::fmt;

/// Embedded configuration
#[derive(Debug, Clone)]
pub struct EmbeddedConfig {
    pub arch: Architecture,
    pub optimize_size: bool,
    pub use_hal: bool,
}

impl Default for EmbeddedConfig {
    fn default() -> Self {
        Self {
            arch: Architecture::ARM,
            optimize_size: true,
            use_hal: false,
        }
    }
}

/// Get architecture-specific includes
fn get_arch_includes(arch: &Architecture) -> &'static str {
    match arch {
        Architecture::ARM => "#include <stdint.h>\n#include <stdbool.h>\n",
        Architecture::X86 => "#include <stdint.h>\n#include <stdbool.h>\n",
        Architecture::X86_64 => "#include <stdint.h>\n#include <stdbool.h>\n",
        _ => "#include <stdint.h>\n",
    }
}

/// Get architecture-specific definitions
fn get_arch_defines(arch: &Architecture) -> String {
    match arch {
        Architecture::ARM => {
            "#define CORTEX_M\n\
             #define __ARM_ARCH 7\n\
             #define LITTLE_ENDIAN 1\n".to_string()
        },
        _ => String::new(),
    }
}

/// Emit startup code
fn emit_startup(arch: &Architecture, module_name: &str) -> String {
    let mut code = String::new();
    
    code.push_str(&format!("/* Startup code for {} */\n", module_name));
    code.push_str("void system_init(void) {\n");
    code.push_str("    /* Initialize system clock */\n");
    code.push_str("    /* Configure peripherals */\n");
    match arch {
        Architecture::ARM => {
            code.push_str("    /* ARM Cortex-M specific initialization */\n");
            code.push_str("    __disable_irq();\n");
            code.push_str("    /* Setup vector table */\n");
            code.push_str("    __enable_irq();\n");
        },
        _ => {
            code.push_str("    /* Generic initialization */\n");
        }
    }
    code.push_str("}\n\n");
    
    code
}

/// Emit embedded code
pub fn emit(arch: Architecture, module_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    // Header comments
    code.push_str("/* STUNIR Generated Embedded Code */\n");
    code.push_str(&format!("/* Architecture: {} */\n", arch));
    code.push_str(&format!("/* Module: {} */\n", module_name));
    code.push_str("/* Generator: Rust Pipeline */\n");
    code.push_str("/* Bare-metal embedded system */\n\n");
    
    // Includes
    code.push_str(get_arch_includes(&arch));
    code.push_str("\n");
    
    // Architecture-specific defines
    let defines = get_arch_defines(&arch);
    if !defines.is_empty() {
        code.push_str(&defines);
        code.push_str("\n");
    }
    
    // Type definitions
    code.push_str("/* Type definitions */\n");
    code.push_str("typedef uint8_t  u8;\n");
    code.push_str("typedef uint16_t u16;\n");
    code.push_str("typedef uint32_t u32;\n");
    code.push_str("typedef int8_t   i8;\n");
    code.push_str("typedef int16_t  i16;\n");
    code.push_str("typedef int32_t  i32;\n\n");
    
    // Memory sections
    code.push_str("/* Memory sections */\n");
    code.push_str("#define ROM_START 0x08000000\n");
    code.push_str("#define RAM_START 0x20000000\n\n");
    
    // Startup code
    code.push_str(&emit_startup(&arch, module_name));
    
    // Main function
    code.push_str("int main(void) {\n");
    code.push_str("    system_init();\n");
    code.push_str("    \n");
    code.push_str("    /* Main loop */\n");
    code.push_str("    while (1) {\n");
    code.push_str("        /* Application code */\n");
    code.push_str("    }\n");
    code.push_str("    \n");
    code.push_str("    return 0;\n");
    code.push_str("}\n");
    
    Ok(code)
}

/// Emit with configuration
pub fn emit_with_config(module_name: &str, config: &EmbeddedConfig) -> EmitterResult<String> {
    emit(config.arch.clone(), module_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_embedded() {
        let result = emit(Architecture::ARM, "test_module");
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("STUNIR Generated"));
        assert!(code.contains("system_init"));
        assert!(code.contains("main"));
    }

    #[test]
    fn test_arch_includes() {
        let includes = get_arch_includes(&Architecture::ARM);
        assert!(includes.contains("stdint.h"));
    }
}
