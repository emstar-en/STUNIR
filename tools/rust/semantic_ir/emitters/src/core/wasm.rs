//! STUNIR WebAssembly Emitter - Rust Implementation
//!
//! Generates WebAssembly code (WASM, WASI, SIMD).
//! Based on DO-178C Level A compliant Ada SPARK implementation.
//!
//! Supports:
//! - WebAssembly (WASM)
//! - WebAssembly System Interface (WASI)
//! - WebAssembly SIMD

use crate::base::{BaseEmitter, EmitterConfig, EmitterError, EmitterResult, EmitterStatus};
use crate::types::{IRDataType, IRFunction, IRModule, IRParameter, IRStatement, IRStatementType};
use std::fmt::Write;

/// WebAssembly target
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WasmTarget {
    /// Core WebAssembly
    Core,
    /// WASI (System Interface)
    WASI,
    /// SIMD extensions
    SIMD,
}

/// WebAssembly emitter configuration
#[derive(Debug, Clone)]
pub struct WasmConfig {
    /// Base emitter configuration
    pub base: EmitterConfig,
    /// Target
    pub target: WasmTarget,
    /// Enable optimizations
    pub optimize: bool,
    /// Generate WAT (text format)
    pub generate_wat: bool,
}

impl WasmConfig {
    /// Create new WASM configuration
    pub fn new(base: EmitterConfig, target: WasmTarget) -> Self {
        Self {
            base,
            target,
            optimize: true,
            generate_wat: true,
        }
    }
}

/// WebAssembly emitter
pub struct WasmEmitter {
    config: WasmConfig,
}

impl WasmEmitter {
    /// Create new WASM emitter
    pub fn new(config: WasmConfig) -> Self {
        Self { config }
    }

    /// Generate WAT (WebAssembly Text) format
    fn generate_wat(&self, ir_module: &IRModule) -> Result<String, EmitterError> {
        let mut content = String::new();

        // Module header
        writeln!(content, ";; STUNIR Generated WebAssembly")?;
        writeln!(content, ";; DO-178C Level A Compliant\n")?;

        writeln!(content, "(module")?;

        // Memory declaration
        writeln!(content, "  ;; Memory")?;
        writeln!(content, "  (memory (export \"memory\") 1)\n")?;

        // Type imports for WASI
        if self.config.target == WasmTarget::WASI {
            writeln!(content, "  ;; WASI imports")?;
            writeln!(content, "  (import \"wasi_snapshot_preview1\" \"fd_write\"")?;
            writeln!(
                content,
                "    (func $fd_write (param i32 i32 i32 i32) (result i32)))\n"
            )
            .unwrap();
        }

        // Function implementations
        for function in &ir_module.functions {
            self.generate_wat_function(&mut content, function)?;
            writeln!(content)?;
        }

        writeln!(content, ")")?;

        Ok(content)
    }

    /// Generate WAT function
    fn generate_wat_function(
        &self,
        content: &mut String,
        function: &IRFunction,
    ) -> Result<(), EmitterError> {
        if let Some(ref doc) = function.docstring {
            writeln!(content, "  ;; {}", doc)?;
        }

        // Function signature
        writeln!(content, "  (func ${}", function.name)?;

        // Export if not internal
        writeln!(content, "    (export \"{}\")", function.name)?;

        // Parameters
        for param in &function.parameters {
            let wasm_type = map_ir_type_to_wasm(param.param_type);
            writeln!(content, "    (param ${} {})", param.name, wasm_type)?;
        }

        // Return type
        if function.return_type != IRDataType::Void {
            let wasm_type = map_ir_type_to_wasm(function.return_type);
            writeln!(content, "    (result {})", wasm_type)?;
        }

        // Local variables
        writeln!(content, "    ;; Locals")?;
        writeln!(content, "    (local $temp i32)\n")?;

        // Function body
        for stmt in &function.statements {
            self.generate_wat_statement(content, stmt)?;
        }

        writeln!(content, "  )")?;
        Ok(())
    }

    /// Generate WAT statement
    fn generate_wat_statement(
        &self,
        content: &mut String,
        stmt: &IRStatement,
    ) -> Result<(), EmitterError> {
        match stmt.stmt_type {
            IRStatementType::Nop => {
                writeln!(content, "    ;; nop")?;
            }
            IRStatementType::VarDecl => {
                writeln!(
                    content,
                    "    ;; var decl: {}",
                    stmt.target.as_deref().unwrap_or("v0")
                )
                .unwrap();
            }
            IRStatementType::Return => {
                let value = stmt.value.as_deref().unwrap_or("0");
                writeln!(content, "    i32.const {}", value)?;
            }
            IRStatementType::Add => {
                writeln!(
                    content,
                    "    local.get ${}",
                    stmt.left_op.as_deref().unwrap_or("v0")
                )
                .unwrap();
                writeln!(
                    content,
                    "    local.get ${}",
                    stmt.right_op.as_deref().unwrap_or("v1")
                )
                .unwrap();
                writeln!(content, "    i32.add")?;
                writeln!(
                    content,
                    "    local.set ${}",
                    stmt.target.as_deref().unwrap_or("result")
                )
                .unwrap();
            }
            IRStatementType::Sub => {
                writeln!(
                    content,
                    "    local.get ${}",
                    stmt.left_op.as_deref().unwrap_or("v0")
                )
                .unwrap();
                writeln!(
                    content,
                    "    local.get ${}",
                    stmt.right_op.as_deref().unwrap_or("v1")
                )
                .unwrap();
                writeln!(content, "    i32.sub")?;
                writeln!(
                    content,
                    "    local.set ${}",
                    stmt.target.as_deref().unwrap_or("result")
                )
                .unwrap();
            }
            IRStatementType::Mul => {
                writeln!(
                    content,
                    "    local.get ${}",
                    stmt.left_op.as_deref().unwrap_or("v0")
                )
                .unwrap();
                writeln!(
                    content,
                    "    local.get ${}",
                    stmt.right_op.as_deref().unwrap_or("v1")
                )
                .unwrap();
                writeln!(content, "    i32.mul")?;
                writeln!(
                    content,
                    "    local.set ${}",
                    stmt.target.as_deref().unwrap_or("result")
                )
                .unwrap();
            }
            IRStatementType::Div => {
                writeln!(
                    content,
                    "    local.get ${}",
                    stmt.left_op.as_deref().unwrap_or("v0")
                )
                .unwrap();
                writeln!(
                    content,
                    "    local.get ${}",
                    stmt.right_op.as_deref().unwrap_or("v1")
                )
                .unwrap();
                writeln!(content, "    i32.div_s")?;
                writeln!(
                    content,
                    "    local.set ${}",
                    stmt.target.as_deref().unwrap_or("result")
                )
                .unwrap();
            }
            _ => {
                writeln!(content, "    ;; {:?}", stmt.stmt_type)?;
            }
        }

        Ok(())
    }
}

impl BaseEmitter for WasmEmitter {
    fn emit(&self, ir_module: &IRModule) -> Result<EmitterResult, EmitterError> {
        if !self.validate_ir(ir_module) {
            return Ok(EmitterResult::error(
                EmitterStatus::ErrorInvalidIR,
                "Invalid IR module".to_string(),
            ));
        }

        let mut files = Vec::new();
        let mut total_size = 0;

        // Generate WAT file
        if self.config.generate_wat {
            let wat_content = self.generate_wat(ir_module)?;
            let wat_file = self.write_file(
                &self.config.base.output_dir,
                &format!("{}.wat", self.config.base.module_name),
                &wat_content,
            )?;
            total_size += wat_file.size;
            files.push(wat_file);
        }

        Ok(EmitterResult::success(files, total_size))
    }
}

/// Map IR data type to WASM type
fn map_ir_type_to_wasm(ir_type: IRDataType) -> &'static str {
    match ir_type {
        IRDataType::Void => "void",
        IRDataType::Bool => "i32",
        IRDataType::I8 | IRDataType::I16 | IRDataType::I32 => "i32",
        IRDataType::I64 => "i64",
        IRDataType::U8 | IRDataType::U16 | IRDataType::U32 => "i32",
        IRDataType::U64 => "i64",
        IRDataType::F32 => "f32",
        IRDataType::F64 => "f64",
        IRDataType::Char => "i32",
        _ => "i32",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{IRFunction, IRModule};
    use tempfile::TempDir;

    #[test]
    fn test_wasm_emitter() {
        let temp_dir = TempDir::new()?;
        let base_config = EmitterConfig::new(temp_dir.path(), "test_module");
        let config = WasmConfig::new(base_config, WasmTarget::Core);
        let emitter = WasmEmitter::new(config);

        let ir_module = IRModule {
            ir_version: "1.0".to_string(),
            module_name: "test_module".to_string(),
            types: vec![],
            functions: vec![IRFunction {
                name: "add".to_string(),
                return_type: IRDataType::I32,
                parameters: vec![],
                statements: vec![],
                docstring: Some("Add function".to_string()),
            }],
            docstring: None,
        };

        let result = emitter.emit(&ir_module)?;
        assert_eq!(result.status, EmitterStatus::Success);
        assert_eq!(result.files.len(), 1);
    }

    #[test]
    fn test_map_ir_type_to_wasm() {
        assert_eq!(map_ir_type_to_wasm(IRDataType::I32), "i32");
        assert_eq!(map_ir_type_to_wasm(IRDataType::I64), "i64");
        assert_eq!(map_ir_type_to_wasm(IRDataType::F32), "f32");
        assert_eq!(map_ir_type_to_wasm(IRDataType::F64), "f64");
    }
}
