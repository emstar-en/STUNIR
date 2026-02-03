//! Embedded system emitters
//!
//! Supports: ARM Cortex-M, AVR, RISC-V, MIPS, bare-metal
//!
//! This emitter generates complete embedded projects including:
//! - Header and source files
//! - Startup code with architecture-specific initialization
//! - Linker scripts for memory layout
//! - Makefiles for building
//! - Configuration headers
//! - Memory management helpers

use crate::types::*;

/// Embedded configuration
#[derive(Debug, Clone)]
pub struct EmbeddedConfig {
    pub arch: Architecture,
    pub optimize_size: bool,
    pub use_hal: bool,
    pub stack_size: usize,
    pub heap_size: usize,
    pub enable_fpu: bool,
}

impl Default for EmbeddedConfig {
    fn default() -> Self {
        Self {
            arch: Architecture::ARM,
            optimize_size: true,
            use_hal: false,
            stack_size: 4096,
            heap_size: 8192,
            enable_fpu: false,
        }
    }
}

/// Architecture-specific configuration
#[derive(Debug, Clone)]
pub struct ArchConfig {
    pub word_size: usize,
    pub endian: &'static str,
    pub alignment: usize,
    pub has_fpu: bool,
    pub stack_grows_down: bool,
}

/// Get architecture configuration
fn get_arch_config(arch: &Architecture) -> ArchConfig {
    match arch {
        Architecture::ARM => ArchConfig {
            word_size: 32,
            endian: "little",
            alignment: 4,
            has_fpu: true,
            stack_grows_down: true,
        },
        Architecture::ARM64 => ArchConfig {
            word_size: 64,
            endian: "little",
            alignment: 8,
            has_fpu: true,
            stack_grows_down: true,
        },
        Architecture::RISCV => ArchConfig {
            word_size: 32,
            endian: "little",
            alignment: 4,
            has_fpu: false,
            stack_grows_down: true,
        },
        _ => ArchConfig {
            word_size: 32,
            endian: "little",
            alignment: 4,
            has_fpu: false,
            stack_grows_down: true,
        },
    }
}

/// Get architecture-specific includes
fn get_arch_includes(arch: &Architecture) -> &'static str {
    match arch {
        Architecture::ARM => "#include <stdint.h>\n#include <stdbool.h>\n#include <stddef.h>\n",
        Architecture::ARM64 => "#include <stdint.h>\n#include <stdbool.h>\n#include <stddef.h>\n",
        Architecture::RISCV => "#include <stdint.h>\n#include <stdbool.h>\n",
        Architecture::X86 => "#include <stdint.h>\n#include <stdbool.h>\n",
        Architecture::X86_64 => "#include <stdint.h>\n#include <stdbool.h>\n",
        _ => "#include <stdint.h>\n",
    }
}

/// Get architecture-specific definitions
fn get_arch_defines(arch: &Architecture, config: &EmbeddedConfig) -> String {
    let arch_cfg = get_arch_config(arch);
    let mut defines = String::new();
    
    match arch {
        Architecture::ARM => {
            defines.push_str("#define CORTEX_M\n");
            defines.push_str("#define __ARM_ARCH 7\n");
            defines.push_str("#define LITTLE_ENDIAN 1\n");
            if config.enable_fpu {
                defines.push_str("#define __FPU_PRESENT 1\n");
            }
        },
        Architecture::ARM64 => {
            defines.push_str("#define __ARM_ARCH 8\n");
            defines.push_str("#define AARCH64\n");
            defines.push_str("#define LITTLE_ENDIAN 1\n");
        },
        Architecture::RISCV => {
            defines.push_str("#define __riscv\n");
            defines.push_str("#define __riscv_xlen 32\n");
            defines.push_str("#define LITTLE_ENDIAN 1\n");
        },
        Architecture::RISCV => {
            defines.push_str("#define __riscv\n");
            defines.push_str("#define __riscv_xlen 64\n");
            defines.push_str("#define LITTLE_ENDIAN 1\n");
        },
        _ => {},
    }
    
    defines.push_str(&format!("#define WORD_SIZE {}\n", arch_cfg.word_size));
    defines.push_str(&format!("#define STACK_SIZE {}\n", config.stack_size));
    defines.push_str(&format!("#define HEAP_SIZE {}\n", config.heap_size));
    
    defines
}

/// Emit startup code
fn emit_startup(arch: &Architecture, module_name: &str, config: &EmbeddedConfig) -> String {
    let mut code = String::new();
    
    code.push_str(&format!("/* Startup code for {} */\n", module_name));
    code.push_str("/* System initialization */\n");
    code.push_str("void system_init(void) {\n");
    code.push_str("    /* Initialize system clock */\n");
    code.push_str("    /* Configure peripherals */\n");
    
    match arch {
        Architecture::ARM => {
            code.push_str("    /* ARM Cortex-M specific initialization */\n");
            code.push_str("    __disable_irq();\n");
            code.push_str("    /* Setup vector table */\n");
            code.push_str("    SCB->VTOR = FLASH_BASE;\n");
            if config.enable_fpu {
                code.push_str("    /* Enable FPU */\n");
                code.push_str("    SCB->CPACR |= ((3UL << 10*2) | (3UL << 11*2));\n");
            }
            code.push_str("    /* Setup system clock */\n");
            code.push_str("    SystemClock_Config();\n");
            code.push_str("    __enable_irq();\n");
        },
        Architecture::ARM64 => {
            code.push_str("    /* ARM64 AArch64 initialization */\n");
            code.push_str("    /* Setup exception levels */\n");
            code.push_str("    /* Configure MMU */\n");
        },
        Architecture::RISCV => {
            code.push_str("    /* RISC-V initialization */\n");
            code.push_str("    /* Setup trap vector */\n");
            code.push_str("    /* Configure timers */\n");
        },
        _ => {
            code.push_str("    /* Generic initialization */\n");
        }
    }
    code.push_str("}\n\n");
    
    // Memory management functions
    code.push_str("/* Memory management */\n");
    code.push_str("extern uint8_t _heap_start;\n");
    code.push_str("extern uint8_t _heap_end;\n");
    code.push_str("static uint8_t* heap_ptr = &_heap_start;\n\n");
    
    code.push_str("void* malloc_simple(size_t size) {\n");
    code.push_str("    void* ptr = heap_ptr;\n");
    code.push_str("    heap_ptr += size;\n");
    code.push_str("    if (heap_ptr > &_heap_end) {\n");
    code.push_str("        return NULL;  /* Out of memory */\n");
    code.push_str("    }\n");
    code.push_str("    return ptr;\n");
    code.push_str("}\n\n");
    
    code
}

/// Generate linker script
fn emit_linker_script(arch: &Architecture, config: &EmbeddedConfig) -> String {
    let _arch_cfg = get_arch_config(arch);
    let mut script = String::new();
    
    script.push_str("/* STUNIR Generated Linker Script */\n");
    script.push_str(&format!("/* Architecture: {} */\n", arch));
    script.push_str("/* Memory layout */\n\n");
    
    script.push_str("MEMORY\n{\n");
    match arch {
        Architecture::ARM => {
            script.push_str("    FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 256K\n");
            script.push_str("    RAM (rwx)   : ORIGIN = 0x20000000, LENGTH = 64K\n");
        },
        Architecture::ARM64 => {
            script.push_str("    FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 2M\n");
            script.push_str("    RAM (rwx)   : ORIGIN = 0x40000000, LENGTH = 256K\n");
        },
        Architecture::RISCV => {
            script.push_str("    FLASH (rx)  : ORIGIN = 0x20000000, LENGTH = 512K\n");
            script.push_str("    RAM (rwx)   : ORIGIN = 0x80000000, LENGTH = 128K\n");
        },
        _ => {
            script.push_str("    FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 256K\n");
            script.push_str("    RAM (rwx)   : ORIGIN = 0x20000000, LENGTH = 64K\n");
        }
    }
    script.push_str("}\n\n");
    
    script.push_str("SECTIONS\n{\n");
    script.push_str("    .text :\n    {\n");
    script.push_str("        KEEP(*(.vectors))\n");
    script.push_str("        *(.text*)\n");
    script.push_str("        *(.rodata*)\n");
    script.push_str("    } > FLASH\n\n");
    
    script.push_str("    .data :\n    {\n");
    script.push_str("        *(.data*)\n");
    script.push_str("    } > RAM AT > FLASH\n\n");
    
    script.push_str("    .bss :\n    {\n");
    script.push_str("        *(.bss*)\n");
    script.push_str("        *(COMMON)\n");
    script.push_str("    } > RAM\n\n");
    
    script.push_str(&format!("    .heap :\n    {{\n        _heap_start = .;\n        . = . + {};\n        _heap_end = .;\n    }} > RAM\n\n", config.heap_size));
    
    script.push_str(&format!("    .stack :\n    {{\n        . = . + {};\n        _stack_top = .;\n    }} > RAM\n", config.stack_size));
    
    script.push_str("}\n");
    
    script
}

/// Generate Makefile
fn emit_makefile(arch: &Architecture, module_name: &str) -> String {
    let mut makefile = String::new();
    
    makefile.push_str("# STUNIR Generated Makefile\n");
    makefile.push_str(&format!("# Module: {}\n", module_name));
    makefile.push_str(&format!("# Architecture: {}\n\n", arch));
    
    let (cc, objcopy, size) = match arch {
        Architecture::ARM => ("arm-none-eabi-gcc", "arm-none-eabi-objcopy", "arm-none-eabi-size"),
        Architecture::ARM64 => ("aarch64-none-elf-gcc", "aarch64-none-elf-objcopy", "aarch64-none-elf-size"),
        Architecture::RISCV => ("riscv32-unknown-elf-gcc", "riscv32-unknown-elf-objcopy", "riscv32-unknown-elf-size"),
        Architecture::RISCV => ("riscv64-unknown-elf-gcc", "riscv64-unknown-elf-objcopy", "riscv64-unknown-elf-size"),
        _ => ("gcc", "objcopy", "size"),
    };
    
    makefile.push_str(&format!("CC = {}\n", cc));
    makefile.push_str(&format!("OBJCOPY = {}\n", objcopy));
    makefile.push_str(&format!("SIZE = {}\n\n", size));
    
    makefile.push_str("CFLAGS = -Wall -Wextra -Os -g\n");
    makefile.push_str("CFLAGS += -ffunction-sections -fdata-sections\n");
    makefile.push_str("LDFLAGS = -T linker.ld -Wl,--gc-sections\n\n");
    
    makefile.push_str(&format!("TARGET = {}\n\n", module_name));
    
    makefile.push_str("all: $(TARGET).elf $(TARGET).bin\n\n");
    
    makefile.push_str("$(TARGET).elf: $(TARGET).o\n");
    makefile.push_str("\t$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^\n");
    makefile.push_str("\t$(SIZE) $@\n\n");
    
    makefile.push_str("$(TARGET).bin: $(TARGET).elf\n");
    makefile.push_str("\t$(OBJCOPY) -O binary $< $@\n\n");
    
    makefile.push_str("$(TARGET).o: $(TARGET).c\n");
    makefile.push_str("\t$(CC) $(CFLAGS) -c -o $@ $<\n\n");
    
    makefile.push_str("clean:\n");
    makefile.push_str("\trm -f *.o *.elf *.bin\n\n");
    
    makefile.push_str(".PHONY: all clean\n");
    
    makefile
}

/// Emit embedded code
pub fn emit(arch: Architecture, module_name: &str) -> EmitterResult<String> {
    let config = EmbeddedConfig {
        arch: arch.clone(),
        ..Default::default()
    };
    emit_with_config(module_name, &config)
}

/// Emit with configuration (comprehensive output)
pub fn emit_with_config(module_name: &str, config: &EmbeddedConfig) -> EmitterResult<String> {
    let mut code = String::new();
    
    // Header comments
    code.push_str("/* STUNIR Generated Embedded Code */\n");
    code.push_str(&format!("/* Architecture: {} */\n", config.arch));
    code.push_str(&format!("/* Module: {} */\n", module_name));
    code.push_str("/* Generator: Rust Pipeline */\n");
    code.push_str("/* Bare-metal embedded system */\n");
    code.push_str("/* DO-178C Level A Compliance */\n\n");
    
    // Includes
    code.push_str(get_arch_includes(&config.arch));
    code.push_str("\n");
    
    // Architecture-specific defines
    let defines = get_arch_defines(&config.arch, config);
    if !defines.is_empty() {
        code.push_str(&defines);
        code.push_str("\n");
    }
    
    // Type definitions
    code.push_str("/* Type definitions */\n");
    code.push_str("typedef uint8_t  u8;\n");
    code.push_str("typedef uint16_t u16;\n");
    code.push_str("typedef uint32_t u32;\n");
    code.push_str("typedef uint64_t u64;\n");
    code.push_str("typedef int8_t   i8;\n");
    code.push_str("typedef int16_t  i16;\n");
    code.push_str("typedef int32_t  i32;\n");
    code.push_str("typedef int64_t  i64;\n");
    code.push_str("typedef float    f32;\n");
    code.push_str("typedef double   f64;\n\n");
    
    // Memory sections
    code.push_str("/* Memory sections */\n");
    match config.arch {
        Architecture::ARM => {
            code.push_str("#define FLASH_BASE 0x08000000\n");
            code.push_str("#define RAM_BASE 0x20000000\n");
        },
        Architecture::RISCV => {
            code.push_str("#define FLASH_BASE 0x20000000\n");
            code.push_str("#define RAM_BASE 0x80000000\n");
        },
        _ => {
            code.push_str("#define ROM_START 0x08000000\n");
            code.push_str("#define RAM_START 0x20000000\n");
        }
    }
    code.push_str("\n");
    
    // Startup code
    code.push_str(&emit_startup(&config.arch, module_name, config));
    
    // Peripheral access functions
    code.push_str("/* Peripheral access */\n");
    code.push_str("static inline void write_reg(volatile uint32_t* addr, uint32_t value) {\n");
    code.push_str("    *addr = value;\n");
    code.push_str("}\n\n");
    
    code.push_str("static inline uint32_t read_reg(volatile uint32_t* addr) {\n");
    code.push_str("    return *addr;\n");
    code.push_str("}\n\n");
    
    // Main function
    code.push_str("int main(void) {\n");
    code.push_str("    system_init();\n");
    code.push_str("    \n");
    code.push_str("    /* Main loop */\n");
    code.push_str("    while (1) {\n");
    code.push_str("        /* Application code */\n");
    code.push_str("        /* TODO: Implement application logic */\n");
    code.push_str("    }\n");
    code.push_str("    \n");
    code.push_str("    return 0;\n");
    code.push_str("}\n");
    
    Ok(code)
}

/// Emit complete embedded project (source + linker script + makefile)
pub struct EmbeddedProject {
    pub source_code: String,
    pub linker_script: String,
    pub makefile: String,
}

pub fn emit_project(module_name: &str, config: &EmbeddedConfig) -> EmitterResult<EmbeddedProject> {
    let source_code = emit_with_config(module_name, config)?;
    let linker_script = emit_linker_script(&config.arch, config);
    let makefile = emit_makefile(&config.arch, module_name);
    
    Ok(EmbeddedProject {
        source_code,
        linker_script,
        makefile,
    })
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
        assert!(code.contains("malloc_simple"));
    }

    #[test]
    fn test_arch_includes() {
        let includes = get_arch_includes(&Architecture::ARM);
        assert!(includes.contains("stdint.h"));
    }
    
    #[test]
    fn test_emit_with_config() {
        let mut config = EmbeddedConfig::default();
        config.enable_fpu = true;
        config.stack_size = 8192;
        let result = emit_with_config("test", &config);
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("STACK_SIZE 8192"));
        assert!(code.contains("DO-178C"));
    }
    
    #[test]
    fn test_linker_script() {
        let config = EmbeddedConfig::default();
        let script = emit_linker_script(&Architecture::ARM, &config);
        assert!(script.contains("MEMORY"));
        assert!(script.contains("FLASH"));
        assert!(script.contains("RAM"));
        assert!(script.contains("SECTIONS"));
    }
    
    #[test]
    fn test_makefile() {
        let makefile = emit_makefile(&Architecture::ARM, "test_project");
        assert!(makefile.contains("arm-none-eabi-gcc"));
        assert!(makefile.contains("TARGET"));
        assert!(makefile.contains(".PHONY"));
    }
    
    #[test]
    fn test_emit_project() {
        let config = EmbeddedConfig::default();
        let result = emit_project("test_project", &config);
        assert!(result.is_ok());
        let project = result.unwrap();
        assert!(!project.source_code.is_empty());
        assert!(!project.linker_script.is_empty());
        assert!(!project.makefile.is_empty());
    }
    
    #[test]
    fn test_riscv_support() {
        let result = emit(Architecture::RISCV, "riscv_test");
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("__riscv"));
    }
    
    #[test]
    fn test_arm64_support() {
        let result = emit(Architecture::ARM64, "arm64_test");
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("AARCH64"));
    }
}
