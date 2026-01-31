//! x86/x86_64 assembly emitter

use crate::assembly::AssemblyEmitter;
use crate::types::Architecture;

pub struct X86Emitter {
    arch: Architecture,
}

impl X86Emitter {
    pub fn new(arch: Architecture) -> Self {
        Self { arch }
    }
}

impl AssemblyEmitter for X86Emitter {
    fn emit_prologue(&self, function_name: &str, stack_size: usize) -> String {
        let mut code = String::new();
        code.push_str(&format!("# STUNIR Generated Code - {} Assembly\n", self.arch));
        code.push_str(&format!("# Function: {}\n", function_name));
        code.push_str(&format!("# DO-178C Level A Compliance\n\n"));
        
        match self.arch {
            Architecture::X86 => {
                code.push_str(&format!("{}:\n", function_name));
                code.push_str("    pushl %ebp\n");
                code.push_str("    movl %esp, %ebp\n");
                if stack_size > 0 {
                    code.push_str(&format!("    subl ${}, %esp\n", stack_size));
                }
            }
            Architecture::X86_64 => {
                code.push_str(&format!("{}:\n", function_name));
                code.push_str("    pushq %rbp\n");
                code.push_str("    movq %rsp, %rbp\n");
                if stack_size > 0 {
                    code.push_str(&format!("    subq ${}, %rsp\n", stack_size));
                }
            }
            _ => {}
        }
        
        code
    }

    fn emit_epilogue(&self, _function_name: &str) -> String {
        let mut code = String::new();
        
        match self.arch {
            Architecture::X86 => {
                code.push_str("    leave\n");
                code.push_str("    ret\n");
            }
            Architecture::X86_64 => {
                code.push_str("    leave\n");
                code.push_str("    ret\n");
            }
            _ => {}
        }
        
        code
    }

    fn emit_load(&self, reg: &str, offset: i32) -> String {
        match self.arch {
            Architecture::X86 => format!("    movl {}(%ebp), {}\n", offset, reg),
            Architecture::X86_64 => format!("    movq {}(%rbp), {}\n", offset, reg),
            _ => String::new(),
        }
    }

    fn emit_store(&self, reg: &str, offset: i32) -> String {
        match self.arch {
            Architecture::X86 => format!("    movl {}, {}(%ebp)\n", reg, offset),
            Architecture::X86_64 => format!("    movq {}, {}(%rbp)\n", reg, offset),
            _ => String::new(),
        }
    }
}
