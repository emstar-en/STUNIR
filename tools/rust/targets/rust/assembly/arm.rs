//! ARM/ARM64 assembly emitter

use crate::assembly::AssemblyEmitter;
use crate::types::Architecture;

pub struct ARMEmitter {
    arch: Architecture,
}

impl ARMEmitter {
    pub fn new(arch: Architecture) -> Self {
        Self { arch }
    }
}

impl AssemblyEmitter for ARMEmitter {
    fn emit_prologue(&self, function_name: &str, stack_size: usize) -> String {
        let mut code = String::new();
        code.push_str(&format!("@ STUNIR Generated Code - {} Assembly\n", self.arch));
        code.push_str(&format!("@ Function: {}\n", function_name));
        code.push_str(&format!("@ DO-178C Level A Compliance\n\n"));
        
        match self.arch {
            Architecture::ARM => {
                code.push_str(&format!("{}:\n", function_name));
                code.push_str("    push {fp, lr}\n");
                code.push_str("    mov fp, sp\n");
                if stack_size > 0 {
                    code.push_str(&format!("    sub sp, sp, #{}\n", stack_size));
                }
            }
            Architecture::ARM64 => {
                code.push_str(&format!("{}:\n", function_name));
                code.push_str("    stp x29, x30, [sp, #-16]!\n");
                code.push_str("    mov x29, sp\n");
                if stack_size > 0 {
                    code.push_str(&format!("    sub sp, sp, #{}\n", stack_size));
                }
            }
            _ => {}
        }
        
        code
    }

    fn emit_epilogue(&self, _function_name: &str) -> String {
        let mut code = String::new();
        
        match self.arch {
            Architecture::ARM => {
                code.push_str("    mov sp, fp\n");
                code.push_str("    pop {fp, pc}\n");
            }
            Architecture::ARM64 => {
                code.push_str("    mov sp, x29\n");
                code.push_str("    ldp x29, x30, [sp], #16\n");
                code.push_str("    ret\n");
            }
            _ => {}
        }
        
        code
    }

    fn emit_load(&self, reg: &str, offset: i32) -> String {
        match self.arch {
            Architecture::ARM => format!("    ldr {}, [fp, #{}]\n", reg, offset),
            Architecture::ARM64 => format!("    ldr {}, [x29, #{}]\n", reg, offset),
            _ => String::new(),
        }
    }

    fn emit_store(&self, reg: &str, offset: i32) -> String {
        match self.arch {
            Architecture::ARM => format!("    str {}, [fp, #{}]\n", reg, offset),
            Architecture::ARM64 => format!("    str {}, [x29, #{}]\n", reg, offset),
            _ => String::new(),
        }
    }
}
