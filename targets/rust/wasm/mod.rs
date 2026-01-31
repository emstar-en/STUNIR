//! WebAssembly emitters
//!
//! Supports: WASM, WASI, WAT (WebAssembly Text Format)

use crate::types::*;
use std::fmt;

/// WebAssembly configuration
#[derive(Debug, Clone)]
pub struct WasmConfig {
    pub format: WasmFormat,
    pub use_wasi: bool,
    pub optimize: bool,
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
        }
    }
}

/// Emit WebAssembly Text (WAT) format
fn emit_wat(module_name: &str, use_wasi: bool) -> String {
    let mut code = String::new();
    
    code.push_str(";; STUNIR Generated WebAssembly Code\n");
    code.push_str(&format!(";; Module: {}\n", module_name));
    code.push_str(";; Generator: Rust Pipeline\n");
    code.push_str(";; Format: WAT (WebAssembly Text)\n\n");
    
    code.push_str("(module\n");
    
    // WASI imports if enabled
    if use_wasi {
        code.push_str("  ;; WASI imports\n");
        code.push_str("  (import \"wasi_snapshot_preview1\" \"fd_write\"\n");
        code.push_str("    (func $fd_write (param i32 i32 i32 i32) (result i32)))\n\n");
    }
    
    // Memory declaration
    code.push_str("  ;; Memory\n");
    code.push_str("  (memory (export \"memory\") 1)\n\n");
    
    // Type definitions
    code.push_str("  ;; Type definitions\n");
    code.push_str("  (type $binary_op (func (param i32 i32) (result i32)))\n\n");
    
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
    
    // Start function if WASI
    if use_wasi {
        code.push_str("  ;; _start function (WASI entry point)\n");
        code.push_str("  (func $_start (export \"_start\")\n");
        code.push_str("    ;; Application entry point\n");
        code.push_str("    nop\n");
        code.push_str("  )\n\n");
    }
    
    // Global variables
    code.push_str("  ;; Global variables\n");
    code.push_str("  (global $counter (mut i32) (i32.const 0))\n\n");
    
    // Table for function references
    code.push_str("  ;; Function table\n");
    code.push_str("  (table $funcs 2 funcref)\n");
    code.push_str("  (elem (i32.const 0) $add $multiply)\n\n");
    
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
        WasmFormat::WAT => Ok(emit_wat(module_name, config.use_wasi)),
        WasmFormat::Binary => {
            // For binary format, we'd normally use wabt or similar
            // For now, return WAT with a note
            let mut code = emit_wat(module_name, config.use_wasi);
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
    }

    #[test]
    fn test_emit_wasi() {
        let result = emit_wasi("test_module");
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("wasi_snapshot_preview1"));
        assert!(code.contains("_start"));
    }

    #[test]
    fn test_wat_structure() {
        let result = emit("test");
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("(memory"));
        assert!(code.contains("(func $add"));
        assert!(code.contains("i32.add"));
    }
}
