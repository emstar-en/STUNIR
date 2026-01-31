//! STUNIR GPU Emitter - Rust Implementation
//!
//! Generates GPU computing code for CUDA, OpenCL, Metal, ROCm, and Vulkan.
//! Based on DO-178C Level A compliant Ada SPARK implementation.
//!
//! Supported platforms:
//! - CUDA (NVIDIA)
//! - OpenCL (cross-platform)
//! - Metal (Apple)
//! - ROCm (AMD)
//! - Vulkan Compute Shaders

use crate::base::{BaseEmitter, EmitterConfig, EmitterError, EmitterResult, EmitterStatus};
use crate::types::{
    map_ir_type_to_c, GeneratedFile, IRDataType, IRFunction, IRModule, IRParameter, IRStatement,
    IRStatementType,
};
use std::fmt::Write;

/// GPU platform enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GPUPlatform {
    /// NVIDIA CUDA
    CUDA,
    /// OpenCL
    OpenCL,
    /// Apple Metal
    Metal,
    /// AMD ROCm
    ROCm,
    /// Vulkan Compute
    Vulkan,
}

/// GPU emitter configuration
#[derive(Debug, Clone)]
pub struct GPUConfig {
    /// Base emitter configuration
    pub base: EmitterConfig,
    /// Target GPU platform
    pub platform: GPUPlatform,
    /// Compute capability (for CUDA)
    pub compute_capability: Option<String>,
    /// Work group size
    pub work_group_size: (usize, usize, usize),
    /// Enable optimizations
    pub optimize: bool,
}

impl GPUConfig {
    /// Create new GPU configuration
    pub fn new(base: EmitterConfig, platform: GPUPlatform) -> Self {
        Self {
            base,
            platform,
            compute_capability: None,
            work_group_size: (256, 1, 1),
            optimize: true,
        }
    }
}

/// GPU emitter for parallel computing code
pub struct GPUEmitter {
    config: GPUConfig,
}

impl GPUEmitter {
    /// Create new GPU emitter
    pub fn new(config: GPUConfig) -> Self {
        Self { config }
    }

    /// Generate kernel/shader code
    fn generate_kernel(&self, ir_module: &IRModule) -> Result<String, EmitterError> {
        let mut content = String::new();

        // DO-178C header
        content.push_str(&self.get_do178c_header(
            &self.config.base,
            &format!("{} Compute Kernel", platform_name(&self.config.platform)),
        ));

        // Platform-specific headers
        self.generate_platform_headers(&mut content)?;

        // Kernel implementations
        for function in &ir_module.functions {
            self.generate_kernel_function(&mut content, function)?;
            writeln!(content).unwrap();
        }

        Ok(content)
    }

    /// Generate platform-specific headers
    fn generate_platform_headers(&self, content: &mut String) -> Result<(), EmitterError> {
        match self.config.platform {
            GPUPlatform::CUDA => {
                writeln!(content, "#include <cuda_runtime.h>").unwrap();
                writeln!(content, "#include <device_launch_parameters.h>\n").unwrap();
            }
            GPUPlatform::OpenCL => {
                writeln!(content, "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n").unwrap();
            }
            GPUPlatform::Metal => {
                writeln!(content, "#include <metal_stdlib>").unwrap();
                writeln!(content, "using namespace metal;\n").unwrap();
            }
            GPUPlatform::ROCm => {
                writeln!(content, "#include <hip/hip_runtime.h>\n").unwrap();
            }
            GPUPlatform::Vulkan => {
                writeln!(content, "#version 450\n").unwrap();
            }
        }
        Ok(())
    }

    /// Generate kernel function
    fn generate_kernel_function(
        &self,
        content: &mut String,
        function: &IRFunction,
    ) -> Result<(), EmitterError> {
        if let Some(ref doc) = function.docstring {
            writeln!(content, "/* {} */", doc).unwrap();
        }

        // Platform-specific function qualifiers
        let qualifier = match self.config.platform {
            GPUPlatform::CUDA => "__global__",
            GPUPlatform::OpenCL => "__kernel",
            GPUPlatform::Metal => "kernel",
            GPUPlatform::ROCm => "__global__",
            GPUPlatform::Vulkan => "void",
        };

        let ret_type = map_ir_type_to_c(function.return_type);
        let params = self.format_parameters(&function.parameters);
        writeln!(
            content,
            "{} {} {}({}) {{",
            qualifier, ret_type, function.name, params
        )
        .unwrap();

        // Thread indexing
        self.generate_thread_indexing(content)?;

        // Generate statements
        for stmt in &function.statements {
            self.generate_statement(content, stmt, 1)?;
        }

        writeln!(content, "}}").unwrap();
        Ok(())
    }

    /// Generate thread indexing code
    fn generate_thread_indexing(&self, content: &mut String) -> Result<(), EmitterError> {
        let indent = self.indent(&self.config.base, 1);
        writeln!(content, "{}/* Thread indexing */", indent).unwrap();

        match self.config.platform {
            GPUPlatform::CUDA | GPUPlatform::ROCm => {
                writeln!(
                    content,
                    "{}int idx = blockIdx.x * blockDim.x + threadIdx.x;",
                    indent
                )
                .unwrap();
                writeln!(
                    content,
                    "{}int idy = blockIdx.y * blockDim.y + threadIdx.y;",
                    indent
                )
                .unwrap();
                writeln!(
                    content,
                    "{}int idz = blockIdx.z * blockDim.z + threadIdx.z;",
                    indent
                )
                .unwrap();
            }
            GPUPlatform::OpenCL => {
                writeln!(content, "{}int idx = get_global_id(0);", indent).unwrap();
                writeln!(content, "{}int idy = get_global_id(1);", indent).unwrap();
                writeln!(content, "{}int idz = get_global_id(2);", indent).unwrap();
            }
            GPUPlatform::Metal => {
                writeln!(content, "{}uint idx = gid.x;", indent).unwrap();
                writeln!(content, "{}uint idy = gid.y;", indent).unwrap();
                writeln!(content, "{}uint idz = gid.z;", indent).unwrap();
            }
            GPUPlatform::Vulkan => {
                writeln!(content, "{}uint idx = gl_GlobalInvocationID.x;", indent).unwrap();
                writeln!(content, "{}uint idy = gl_GlobalInvocationID.y;", indent).unwrap();
                writeln!(content, "{}uint idz = gl_GlobalInvocationID.z;", indent).unwrap();
            }
        }
        writeln!(content).unwrap();
        Ok(())
    }

    /// Generate statement
    fn generate_statement(
        &self,
        content: &mut String,
        stmt: &IRStatement,
        indent_level: usize,
    ) -> Result<(), EmitterError> {
        let indent = self.indent(&self.config.base, indent_level);

        match stmt.stmt_type {
            IRStatementType::Nop => {
                writeln!(content, "{}/* nop */", indent).unwrap();
            }
            IRStatementType::VarDecl => {
                let c_type = stmt.data_type.map(map_ir_type_to_c).unwrap_or("int");
                let var_name = stmt.target.as_deref().unwrap_or("v0");
                let init = stmt.value.as_deref().unwrap_or("0");
                writeln!(content, "{}{} {} = {};", indent, c_type, var_name, init).unwrap();
            }
            IRStatementType::Assign => {
                let target = stmt.target.as_deref().unwrap_or("v0");
                let value = stmt.value.as_deref().unwrap_or("0");
                writeln!(content, "{}{} = {};", indent, target, value).unwrap();
            }
            IRStatementType::Return => {
                let value = stmt.value.as_deref().unwrap_or("0");
                writeln!(content, "{}return {};", indent, value).unwrap();
            }
            IRStatementType::Add
            | IRStatementType::Sub
            | IRStatementType::Mul
            | IRStatementType::Div => {
                let target = stmt.target.as_deref().unwrap_or("v0");
                let left = stmt.left_op.as_deref().unwrap_or("0");
                let right = stmt.right_op.as_deref().unwrap_or("0");
                let op = match stmt.stmt_type {
                    IRStatementType::Add => "+",
                    IRStatementType::Sub => "-",
                    IRStatementType::Mul => "*",
                    IRStatementType::Div => "/",
                    _ => "+",
                };
                writeln!(content, "{}{} = {} {} {};", indent, target, left, op, right).unwrap();
            }
            _ => {
                writeln!(
                    content,
                    "{}/* {} */",
                    indent,
                    format!("{:?}", stmt.stmt_type)
                )
                .unwrap();
            }
        }

        Ok(())
    }

    /// Format function parameters
    fn format_parameters(&self, params: &[IRParameter]) -> String {
        if params.is_empty() {
            return String::new();
        }
        params
            .iter()
            .map(|p| {
                let type_str = map_ir_type_to_c(p.param_type);
                // Add pointer qualifier for GPU memory
                let qualifier = match self.config.platform {
                    GPUPlatform::CUDA | GPUPlatform::ROCm => "*",
                    GPUPlatform::OpenCL => "__global *",
                    GPUPlatform::Metal => "device *",
                    GPUPlatform::Vulkan => "*",
                };
                format!("{} {}{}_{}", type_str, qualifier, p.name, "ptr")
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

impl BaseEmitter for GPUEmitter {
    fn emit(&self, ir_module: &IRModule) -> Result<EmitterResult, EmitterError> {
        if !self.validate_ir(ir_module) {
            return Ok(EmitterResult::error(
                EmitterStatus::ErrorInvalidIR,
                "Invalid IR module".to_string(),
            ));
        }

        let mut files = Vec::new();
        let mut total_size = 0;

        // Generate kernel file
        let kernel_content = self.generate_kernel(ir_module)?;
        let extension = match self.config.platform {
            GPUPlatform::CUDA | GPUPlatform::ROCm => "cu",
            GPUPlatform::OpenCL => "cl",
            GPUPlatform::Metal => "metal",
            GPUPlatform::Vulkan => "comp",
        };
        let kernel_file = self.write_file(
            &self.config.base.output_dir,
            &format!("{}.{}", self.config.base.module_name, extension),
            &kernel_content,
        )?;
        total_size += kernel_file.size;
        files.push(kernel_file);

        Ok(EmitterResult::success(files, total_size))
    }
}

/// Get platform name as string
fn platform_name(platform: &GPUPlatform) -> &'static str {
    match platform {
        GPUPlatform::CUDA => "CUDA",
        GPUPlatform::OpenCL => "OpenCL",
        GPUPlatform::Metal => "Metal",
        GPUPlatform::ROCm => "ROCm",
        GPUPlatform::Vulkan => "Vulkan",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{IRFunction, IRModule};
    use tempfile::TempDir;

    #[test]
    fn test_gpu_emitter_cuda() {
        let temp_dir = TempDir::new().unwrap();
        let base_config = EmitterConfig::new(temp_dir.path(), "test_kernel");
        let config = GPUConfig::new(base_config, GPUPlatform::CUDA);
        let emitter = GPUEmitter::new(config);

        let ir_module = IRModule {
            ir_version: "1.0".to_string(),
            module_name: "test_kernel".to_string(),
            types: vec![],
            functions: vec![IRFunction {
                name: "vector_add".to_string(),
                return_type: IRDataType::Void,
                parameters: vec![],
                statements: vec![],
                docstring: Some("CUDA kernel".to_string()),
            }],
            docstring: None,
        };

        let result = emitter.emit(&ir_module).unwrap();
        assert_eq!(result.status, EmitterStatus::Success);
        assert_eq!(result.files.len(), 1);
    }

    #[test]
    fn test_platform_name() {
        assert_eq!(platform_name(&GPUPlatform::CUDA), "CUDA");
        assert_eq!(platform_name(&GPUPlatform::OpenCL), "OpenCL");
        assert_eq!(platform_name(&GPUPlatform::Metal), "Metal");
    }
}
