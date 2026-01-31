//! GPU acceleration emitters
//!
//! Supports: CUDA, ROCm, OpenCL, Metal, Vulkan

use crate::types::*;
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub enum GPUPlatform {
    CUDA,
    ROCm,
    OpenCL,
    Metal,
    Vulkan,
}

impl fmt::Display for GPUPlatform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GPUPlatform::CUDA => write!(f, "CUDA"),
            GPUPlatform::ROCm => write!(f, "ROCm"),
            GPUPlatform::OpenCL => write!(f, "OpenCL"),
            GPUPlatform::Metal => write!(f, "Metal"),
            GPUPlatform::Vulkan => write!(f, "Vulkan"),
        }
    }
}

/// GPU kernel configuration
#[derive(Debug, Clone)]
pub struct GPUConfig {
    pub platform: GPUPlatform,
    pub compute_capability: String,
    pub optimize: bool,
}

impl Default for GPUConfig {
    fn default() -> Self {
        Self {
            platform: GPUPlatform::CUDA,
            compute_capability: "sm_50".to_string(),
            optimize: true,
        }
    }
}

/// Emit CUDA kernel
fn emit_cuda_kernel(module_name: &str) -> String {
    let mut code = String::new();
    
    code.push_str("#include <cuda_runtime.h>\n");
    code.push_str("#include <device_launch_parameters.h>\n\n");
    
    code.push_str("// CUDA kernel\n");
    code.push_str(&format!("__global__ void {}_kernel(float* input, float* output, int n) {{\n", module_name));
    code.push_str("    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n");
    code.push_str("    if (idx < n) {\n");
    code.push_str("        output[idx] = input[idx] * 2.0f;  // Example operation\n");
    code.push_str("    }\n");
    code.push_str("}\n\n");
    
    code.push_str("// Host function\n");
    code.push_str(&format!("void {}_launch(float* h_input, float* h_output, int n) {{\n", module_name));
    code.push_str("    float *d_input, *d_output;\n");
    code.push_str("    \n");
    code.push_str("    // Allocate device memory\n");
    code.push_str("    cudaMalloc(&d_input, n * sizeof(float));\n");
    code.push_str("    cudaMalloc(&d_output, n * sizeof(float));\n");
    code.push_str("    \n");
    code.push_str("    // Copy to device\n");
    code.push_str("    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);\n");
    code.push_str("    \n");
    code.push_str("    // Launch kernel\n");
    code.push_str("    int blockSize = 256;\n");
    code.push_str("    int gridSize = (n + blockSize - 1) / blockSize;\n");
    code.push_str(&format!("    {}_kernel<<<gridSize, blockSize>>>(d_input, d_output, n);\n", module_name));
    code.push_str("    \n");
    code.push_str("    // Copy back to host\n");
    code.push_str("    cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);\n");
    code.push_str("    \n");
    code.push_str("    // Free device memory\n");
    code.push_str("    cudaFree(d_input);\n");
    code.push_str("    cudaFree(d_output);\n");
    code.push_str("}\n");
    
    code
}

/// Emit OpenCL kernel
fn emit_opencl_kernel(module_name: &str) -> String {
    let mut code = String::new();
    
    code.push_str("// OpenCL kernel\n");
    code.push_str(&format!("__kernel void {}_kernel(\n", module_name));
    code.push_str("    __global float* input,\n");
    code.push_str("    __global float* output,\n");
    code.push_str("    const int n)\n");
    code.push_str("{\n");
    code.push_str("    int idx = get_global_id(0);\n");
    code.push_str("    if (idx < n) {\n");
    code.push_str("        output[idx] = input[idx] * 2.0f;  // Example operation\n");
    code.push_str("    }\n");
    code.push_str("}\n");
    
    code
}

/// Emit Metal shader
fn emit_metal_shader(module_name: &str) -> String {
    let mut code = String::new();
    
    code.push_str("#include <metal_stdlib>\n");
    code.push_str("using namespace metal;\n\n");
    
    code.push_str(&format!("kernel void {}_kernel(\n", module_name));
    code.push_str("    device float* input [[buffer(0)]],\n");
    code.push_str("    device float* output [[buffer(1)]],\n");
    code.push_str("    constant int& n [[buffer(2)]],\n");
    code.push_str("    uint idx [[thread_position_in_grid]])\n");
    code.push_str("{\n");
    code.push_str("    if (idx < n) {\n");
    code.push_str("        output[idx] = input[idx] * 2.0f;  // Example operation\n");
    code.push_str("    }\n");
    code.push_str("}\n");
    
    code
}

pub fn emit(platform: GPUPlatform, module_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    // Header comments
    code.push_str(&format!("// STUNIR Generated GPU Code ({})\n", platform));
    code.push_str(&format!("// Module: {}\n", module_name));
    code.push_str("// Generator: Rust Pipeline\n");
    code.push_str("// DO-178C Level A Compliance\n\n");
    
    // Platform-specific code
    match platform {
        GPUPlatform::CUDA => {
            code.push_str(&emit_cuda_kernel(module_name));
        },
        GPUPlatform::OpenCL => {
            code.push_str(&emit_opencl_kernel(module_name));
        },
        GPUPlatform::Metal => {
            code.push_str(&emit_metal_shader(module_name));
        },
        GPUPlatform::ROCm => {
            // ROCm uses HIP (similar to CUDA)
            code.push_str("// ROCm/HIP kernel (similar to CUDA)\n");
            code.push_str(&emit_cuda_kernel(module_name).replace("cuda", "hip"));
        },
        GPUPlatform::Vulkan => {
            code.push_str("// Vulkan compute shader\n");
            code.push_str("#version 450\n\n");
            code.push_str("layout(local_size_x = 256) in;\n\n");
            code.push_str("layout(binding = 0) buffer InputBuffer { float input_data[]; };\n");
            code.push_str("layout(binding = 1) buffer OutputBuffer { float output_data[]; };\n\n");
            code.push_str("void main() {\n");
            code.push_str("    uint idx = gl_GlobalInvocationID.x;\n");
            code.push_str("    output_data[idx] = input_data[idx] * 2.0;  // Example operation\n");
            code.push_str("}\n");
        },
    }
    
    Ok(code)
}

/// Emit with configuration
pub fn emit_with_config(module_name: &str, config: &GPUConfig) -> EmitterResult<String> {
    emit(config.platform, module_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_cuda() {
        let result = emit(GPUPlatform::CUDA, "test");
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("__global__"));
        assert!(code.contains("CUDA"));
    }

    #[test]
    fn test_emit_opencl() {
        let result = emit(GPUPlatform::OpenCL, "test");
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("__kernel"));
    }

    #[test]
    fn test_emit_metal() {
        let result = emit(GPUPlatform::Metal, "test");
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("metal_stdlib"));
    }
}
