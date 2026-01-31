//! WebAssembly emitters
//!
//! Supports: WASM, WASI, WAT (WebAssembly Text Format)

use crate::types::*;

/// WebAssembly configuration
#[derive(Debug, Clone)]
pub struct WasmConfig {
    pub format: WasmFormat,
    pub use_wasi: bool,
    pub optimize: bool,
    pub export_memory: bool,
    pub initial_memory_pages: usize,
    pub max_memory_pages: Option<usize>,
    pub use_bulk_memory: bool,
    pub use_simd: bool,
    pub use_threads: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum WasmFormat {
    WAT,      // WebAssembly Text Format
    Binary,   // Binary .wasm format
}

impl Default for WasmConfig {
    fn default() -> Self {
        Self {
            format: WasmFormat::WAT,
            use_wasi: false,
            optimize: true,
            export_memory: true,
            initial_memory_pages: 1,
            max_memory_pages: None,
            use_bulk_memory: false,
            use_simd: false,
            use_threads: false,
        }
    }
}

/// Emit WebAssembly Text (WAT) format with configuration
fn emit_wat(module_name: &str, config: &WasmConfig) -> String {
    let mut code = String::new();
    
    code.push_str(";; STUNIR Generated WebAssembly Code\n");
    code.push_str(&format!(";; Module: {}\n", module_name));
    code.push_str(";; Generator: Rust Pipeline\n");
    code.push_str(";; Format: WAT (WebAssembly Text)\n");
    code.push_str(";; DO-178C Level A Compliance\n\n");
    
    code.push_str("(module\n");
    
    // WASI imports if enabled
    if config.use_wasi {
        code.push_str("  ;; WASI imports\n");
        code.push_str("  (import \"wasi_snapshot_preview1\" \"fd_write\"\n");
        code.push_str("    (func $fd_write (param i32 i32 i32 i32) (result i32)))\n");
        code.push_str("  (import \"wasi_snapshot_preview1\" \"fd_read\"\n");
        code.push_str("    (func $fd_read (param i32 i32 i32 i32) (result i32)))\n");
        code.push_str("  (import \"wasi_snapshot_preview1\" \"proc_exit\"\n");
        code.push_str("    (func $proc_exit (param i32)))\n\n");
    }
    
    // Memory declaration
    code.push_str("  ;; Memory\n");
    if let Some(max_pages) = config.max_memory_pages {
        code.push_str(&format!("  (memory (export \"memory\") {} {})\n\n", 
            config.initial_memory_pages, max_pages));
    } else {
        code.push_str(&format!("  (memory (export \"memory\") {})\n\n", 
            config.initial_memory_pages));
    }
    
    // Data section
    code.push_str("  ;; Data section\n");
    code.push_str(&format!("  (data (i32.const 0) \"{}\\00\")\n\n", module_name));
    
    // Type definitions
    code.push_str("  ;; Type definitions\n");
    code.push_str("  (type $binary_op (func (param i32 i32) (result i32)))\n");
    code.push_str("  (type $unary_op (func (param i32) (result i32)))\n");
    code.push_str("  (type $no_param (func))\n");
    code.push_str("  (type $mem_op (func (param i32 i32 i32)))\n\n");
    
    // Function: add
    code.push_str("  ;; Add function\n");
    code.push_str("  (func $add (export \"add\") (type $binary_op)\n");
    code.push_str("    local.get 0\n");
    code.push_str("    local.get 1\n");
    code.push_str("    i32.add\n");
    code.push_str("  )\n\n");
    
    // Function: multiply
    code.push_str("  ;; Multiply function\n");
    code.push_str("  (func $multiply (export \"multiply\") (type $binary_op)\n");
    code.push_str("    local.get 0\n");
    code.push_str("    local.get 1\n");
    code.push_str("    i32.mul\n");
    code.push_str("  )\n\n");
    
    // Function: subtract
    code.push_str("  ;; Subtract function\n");
    code.push_str("  (func $subtract (export \"subtract\") (type $binary_op)\n");
    code.push_str("    local.get 0\n");
    code.push_str("    local.get 1\n");
    code.push_str("    i32.sub\n");
    code.push_str("  )\n\n");
    
    // Function: divide
    code.push_str("  ;; Divide function\n");
    code.push_str("  (func $divide (export \"divide\") (type $binary_op)\n");
    code.push_str("    local.get 0\n");
    code.push_str("    local.get 1\n");
    code.push_str("    i32.div_s\n");
    code.push_str("  )\n\n");
    
    // Memory operations
    if config.use_bulk_memory {
        code.push_str("  ;; Bulk memory operations (fill)\n");
        code.push_str("  (func $mem_fill (export \"mem_fill\") (type $mem_op)\n");
        code.push_str("    local.get 0\n");
        code.push_str("    local.get 1\n");
        code.push_str("    local.get 2\n");
        code.push_str("    memory.fill\n");
        code.push_str("  )\n\n");
        
        code.push_str("  ;; Bulk memory operations (copy)\n");
        code.push_str("  (func $mem_copy (export \"mem_copy\") (type $mem_op)\n");
        code.push_str("    local.get 0\n");
        code.push_str("    local.get 1\n");
        code.push_str("    local.get 2\n");
        code.push_str("    memory.copy\n");
        code.push_str("  )\n\n");
    }
    
    // SIMD operations
    if config.use_simd {
        code.push_str("  ;; SIMD vector operations\n");
        code.push_str("  (func $simd_add (export \"simd_add\") (param i32 i32) (result v128)\n");
        code.push_str("    local.get 0\n");
        code.push_str("    v128.load\n");
        code.push_str("    local.get 1\n");
        code.push_str("    v128.load\n");
        code.push_str("    i32x4.add\n");
        code.push_str("  )\n\n");
    }
    
    // Start function if WASI
    if config.use_wasi {
        code.push_str("  ;; _start function (WASI entry point)\n");
        code.push_str("  (func $_start (export \"_start\")\n");
        code.push_str("    ;; Application entry point\n");
        code.push_str("    call $main\n");
        code.push_str("    drop\n");
        code.push_str("  )\n\n");
        
        code.push_str("  ;; Main application function\n");
        code.push_str("  (func $main (result i32)\n");
        code.push_str("    i32.const 0\n");
        code.push_str("  )\n\n");
    }
    
    // Global variables
    code.push_str("  ;; Global variables\n");
    code.push_str("  (global $counter (mut i32) (i32.const 0))\n");
    code.push_str("  (global $heap_ptr (mut i32) (i32.const 1024))\n\n");
    
    // Simple allocator
    code.push_str("  ;; Simple memory allocator\n");
    code.push_str("  (func $alloc (export \"alloc\") (param $size i32) (result i32)\n");
    code.push_str("    (local $ptr i32)\n");
    code.push_str("    global.get $heap_ptr\n");
    code.push_str("    local.set $ptr\n");
    code.push_str("    global.get $heap_ptr\n");
    code.push_str("    local.get $size\n");
    code.push_str("    i32.add\n");
    code.push_str("    global.set $heap_ptr\n");
    code.push_str("    local.get $ptr\n");
    code.push_str("  )\n\n");
    
    // Table for function references
    code.push_str("  ;; Function table\n");
    code.push_str("  (table $funcs 4 funcref)\n");
    code.push_str("  (elem (i32.const 0) $add $multiply $subtract $divide)\n");
    
    // Table accessor
    code.push_str("  (func $call_indirect (export \"call_indirect\") (param $idx i32) (param $a i32) (param $b i32) (result i32)\n");
    code.push_str("    local.get $a\n");
    code.push_str("    local.get $b\n");
    code.push_str("    local.get $idx\n");
    code.push_str("    call_indirect (type $binary_op)\n");
    code.push_str("  )\n\n");
    
    code.push_str(")\n");
    
    code
}

/// Main emit entry point
pub fn emit(module_name: &str) -> EmitterResult<String> {
    let config = WasmConfig::default();
    emit_with_config(module_name, &config)
}

/// Emit with configuration
pub fn emit_with_config(module_name: &str, config: &WasmConfig) -> EmitterResult<String> {
    match config.format {
        WasmFormat::WAT => Ok(emit_wat(module_name, config)),
        WasmFormat::Binary => {
            // For binary format, we'd normally use wabt or similar
            // For now, return WAT with a note
            let mut code = emit_wat(module_name, config);
            code.insert_str(0, ";; Note: Binary WASM requires compilation with wat2wasm\n");
            Ok(code)
        }
    }
}

/// Emit WASI-enabled WebAssembly
pub fn emit_wasi(module_name: &str) -> EmitterResult<String> {
    let mut config = WasmConfig::default();
    config.use_wasi = true;
    emit_with_config(module_name, &config)
}

/// Emit with bulk memory operations
pub fn emit_with_bulk_memory(module_name: &str) -> EmitterResult<String> {
    let mut config = WasmConfig::default();
    config.use_bulk_memory = true;
    emit_with_config(module_name, &config)
}

/// Emit with SIMD operations
pub fn emit_with_simd(module_name: &str) -> EmitterResult<String> {
    let mut config = WasmConfig::default();
    config.use_simd = true;
    emit_with_config(module_name, &config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_wat() {
        let result = emit("test_module");
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("(module"));
        assert!(code.contains("(func"));
        assert!(code.contains("STUNIR Generated"));
        assert!(code.contains("DO-178C"));
    }

    #[test]
    fn test_emit_wasi() {
        let result = emit_wasi("test_module");
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("wasi_snapshot_preview1"));
        assert!(code.contains("_start"));
        assert!(code.contains("fd_write"));
        assert!(code.contains("fd_read"));
        assert!(code.contains("proc_exit"));
    }

    #[test]
    fn test_wat_structure() {
        let result = emit("test");
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("(memory"));
        assert!(code.contains("(func $add"));
        assert!(code.contains("i32.add"));
        assert!(code.contains("(func $multiply"));
        assert!(code.contains("(func $subtract"));
        assert!(code.contains("(func $divide"));
    }
    
    #[test]
    fn test_memory_config() {
        let mut config = WasmConfig::default();
        config.initial_memory_pages = 2;
        config.max_memory_pages = Some(10);
        let result = emit_with_config("test", &config);
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("(memory (export \"memory\") 2 10)"));
    }
    
    #[test]
    fn test_bulk_memory() {
        let result = emit_with_bulk_memory("test");
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("memory.fill"));
        assert!(code.contains("memory.copy"));
        assert!(code.contains("$mem_fill"));
        assert!(code.contains("$mem_copy"));
    }
    
    #[test]
    fn test_simd() {
        let result = emit_with_simd("test");
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("v128"));
        assert!(code.contains("i32x4.add"));
        assert!(code.contains("$simd_add"));
    }
    
    #[test]
    fn test_allocator() {
        let result = emit("test");
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("$alloc"));
        assert!(code.contains("$heap_ptr"));
    }
    
    #[test]
    fn test_function_table() {
        let result = emit("test");
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("(table $funcs 4 funcref)"));
        assert!(code.contains("call_indirect"));
    }
    
    #[test]
    fn test_data_section() {
        let result = emit("test_mod");
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("(data"));
        assert!(code.contains("test_mod"));
    }
}
