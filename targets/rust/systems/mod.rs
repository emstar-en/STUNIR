//! Systems programming emitters
//!
//! Supports: C, C++, Rust, Zig, systems-level code

use crate::types::*;

/// Systems language
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemsLanguage {
    C,
    CPlusPlus,
    Rust,
    Zig,
}

impl std::fmt::Display for SystemsLanguage {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            SystemsLanguage::C => write!(f, "C"),
            SystemsLanguage::CPlusPlus => write!(f, "C++"),
            SystemsLanguage::Rust => write!(f, "Rust"),
            SystemsLanguage::Zig => write!(f, "Zig"),
        }
    }
}

/// Emit systems code
pub fn emit(language: SystemsLanguage, module_name: &str) -> EmitterResult<String> {
    match language {
        SystemsLanguage::C => emit_c_systems(module_name),
        SystemsLanguage::CPlusPlus => emit_cpp_systems(module_name),
        SystemsLanguage::Rust => emit_rust_systems(module_name),
        SystemsLanguage::Zig => emit_zig(module_name),
    }
}

fn emit_c_systems(module_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("/* STUNIR Generated C Systems Code */\n");
    code.push_str(&format!("/* Module: {} */\n", module_name));
    code.push_str("/* Generator: Rust Pipeline */\n\n");
    
    code.push_str("#include <stdint.h>\n");
    code.push_str("#include <stdbool.h>\n");
    code.push_str("#include <string.h>\n\n");
    
    code.push_str("/* Memory management */\n");
    code.push_str("typedef struct {\n");
    code.push_str("    uint8_t *data;\n");
    code.push_str("    size_t size;\n");
    code.push_str("    size_t capacity;\n");
    code.push_str("} Buffer;\n\n");
    
    code.push_str("void buffer_init(Buffer *buf, size_t capacity) {\n");
    code.push_str("    buf->data = (uint8_t *)malloc(capacity);\n");
    code.push_str("    buf->size = 0;\n");
    code.push_str("    buf->capacity = capacity;\n");
    code.push_str("}\n\n");
    
    code.push_str("void buffer_free(Buffer *buf) {\n");
    code.push_str("    if (buf->data) {\n");
    code.push_str("        free(buf->data);\n");
    code.push_str("        buf->data = NULL;\n");
    code.push_str("    }\n");
    code.push_str("}\n");
    
    Ok(code)
}

fn emit_cpp_systems(module_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("// STUNIR Generated C++ Systems Code\n");
    code.push_str(&format!("// Module: {}\n", module_name));
    code.push_str("// Generator: Rust Pipeline\n\n");
    
    code.push_str("#include <cstdint>\n");
    code.push_str("#include <memory>\n");
    code.push_str("#include <vector>\n\n");
    
    code.push_str("// RAII Buffer class\n");
    code.push_str("class Buffer {\n");
    code.push_str("private:\n");
    code.push_str("    std::vector<uint8_t> data_;\n\n");
    
    code.push_str("public:\n");
    code.push_str("    explicit Buffer(size_t capacity) : data_(capacity) {}\n\n");
    
    code.push_str("    uint8_t* data() { return data_.data(); }\n");
    code.push_str("    size_t size() const { return data_.size(); }\n");
    code.push_str("    size_t capacity() const { return data_.capacity(); }\n");
    code.push_str("};\n");
    
    Ok(code)
}

fn emit_rust_systems(module_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("// STUNIR Generated Rust Systems Code\n");
    code.push_str(&format!("// Module: {}\n", module_name));
    code.push_str("// Generator: Rust Pipeline\n\n");
    
    code.push_str("use std::alloc::{alloc, dealloc, Layout};\n");
    code.push_str("use std::ptr;\n\n");
    
    code.push_str("/// Safe buffer abstraction\n");
    code.push_str("pub struct Buffer {\n");
    code.push_str("    data: Vec<u8>,\n");
    code.push_str("}\n\n");
    
    code.push_str("impl Buffer {\n");
    code.push_str("    pub fn new(capacity: usize) -> Self {\n");
    code.push_str("        Buffer {\n");
    code.push_str("            data: Vec::with_capacity(capacity),\n");
    code.push_str("        }\n");
    code.push_str("    }\n\n");
    
    code.push_str("    pub fn as_ptr(&self) -> *const u8 {\n");
    code.push_str("        self.data.as_ptr()\n");
    code.push_str("    }\n\n");
    
    code.push_str("    pub fn len(&self) -> usize {\n");
    code.push_str("        self.data.len()\n");
    code.push_str("    }\n");
    code.push_str("}\n");
    
    Ok(code)
}

fn emit_zig(module_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("// STUNIR Generated Zig Systems Code\n");
    code.push_str(&format!("// Module: {}\n", module_name));
    code.push_str("// Generator: Rust Pipeline\n\n");
    
    code.push_str("const std = @import(\"std\");\n\n");
    
    code.push_str("pub const Buffer = struct {\n");
    code.push_str("    data: []u8,\n");
    code.push_str("    allocator: std.mem.Allocator,\n\n");
    
    code.push_str("    pub fn init(allocator: std.mem.Allocator, capacity: usize) !Buffer {\n");
    code.push_str("        return Buffer{\n");
    code.push_str("            .data = try allocator.alloc(u8, capacity),\n");
    code.push_str("            .allocator = allocator,\n");
    code.push_str("        };\n");
    code.push_str("    }\n\n");
    
    code.push_str("    pub fn deinit(self: *Buffer) void {\n");
    code.push_str("        self.allocator.free(self.data);\n");
    code.push_str("    }\n");
    code.push_str("};\n");
    
    Ok(code)
}
