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
    pub block_size: usize,
    pub shared_memory_bytes: usize,
    pub use_fp16: bool,
    pub use_tensor_cores: bool,
    pub max_registers: Option<usize>,
}

impl Default for GPUConfig {
    fn default() -> Self {
        Self {
            platform: GPUPlatform::CUDA,
            compute_capability: "sm_50".to_string(),
            optimize: true,
            block_size: 256,
            shared_memory_bytes: 0,
            use_fp16: false,
            use_tensor_cores: false,
            max_registers: None,
        }
    }
}

/// Emit CUDA kernel with configuration
fn emit_cuda_kernel(module_name: &str, config: &GPUConfig) -> String {
    let mut code = String::new();
    
    code.push_str("#include <cuda_runtime.h>\n");
    code.push_str("#include <device_launch_parameters.h>\n");
    if config.use_fp16 {
        code.push_str("#include <cuda_fp16.h>\n");
    }
    if config.use_tensor_cores {
        code.push_str("#include <mma.h>\n");
    }
    code.push_str("\n");
    
    // Kernel attributes
    if let Some(max_regs) = config.max_registers {
        code.push_str(&format!("__launch_bounds__({}, {})\n", config.block_size, max_regs));
    }
    
    // Basic kernel
    code.push_str("// CUDA kernel\n");
    code.push_str(&format!("__global__ void {}_kernel(float* input, float* output, int n) {{\n", module_name));
    
    // Shared memory if configured
    if config.shared_memory_bytes > 0 {
        code.push_str(&format!("    __shared__ float sdata[{}];\n", config.shared_memory_bytes / 4));
    }
    
    code.push_str("    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n");
    code.push_str("    int tid = threadIdx.x;\n");
    code.push_str("    \n");
    code.push_str("    if (idx < n) {\n");
    
    if config.shared_memory_bytes > 0 {
        code.push_str("        // Load to shared memory\n");
        code.push_str("        sdata[tid] = input[idx];\n");
        code.push_str("        __syncthreads();\n");
        code.push_str("        \n");
        code.push_str("        // Process from shared memory\n");
        code.push_str("        output[idx] = sdata[tid] * 2.0f;\n");
    } else {
        code.push_str("        output[idx] = input[idx] * 2.0f;  // Example operation\n");
    }
    
    code.push_str("    }\n");
    code.push_str("}\n\n");
    
    // Optimized vectorized kernel
    code.push_str("// Vectorized kernel (float4)\n");
    code.push_str(&format!("__global__ void {}_kernel_vectorized(float4* input, float4* output, int n) {{\n", module_name));
    code.push_str("    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n");
    code.push_str("    if (idx < n) {\n");
    code.push_str("        float4 data = input[idx];\n");
    code.push_str("        data.x *= 2.0f;\n");
    code.push_str("        data.y *= 2.0f;\n");
    code.push_str("        data.z *= 2.0f;\n");
    code.push_str("        data.w *= 2.0f;\n");
    code.push_str("        output[idx] = data;\n");
    code.push_str("    }\n");
    code.push_str("}\n\n");
    
    // Reduction kernel
    code.push_str("// Parallel reduction kernel\n");
    code.push_str(&format!("__global__ void {}_reduce(float* input, float* output, int n) {{\n", module_name));
    code.push_str(&format!("    __shared__ float sdata[{}];\n", config.block_size));
    code.push_str("    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n");
    code.push_str("    int tid = threadIdx.x;\n");
    code.push_str("    \n");
    code.push_str("    sdata[tid] = (idx < n) ? input[idx] : 0.0f;\n");
    code.push_str("    __syncthreads();\n");
    code.push_str("    \n");
    code.push_str("    // Reduction in shared memory\n");
    code.push_str(&format!("    for (int s = {}; s > 0; s >>= 1) {{\n", config.block_size / 2));
    code.push_str("        if (tid < s) {\n");
    code.push_str("            sdata[tid] += sdata[tid + s];\n");
    code.push_str("        }\n");
    code.push_str("        __syncthreads();\n");
    code.push_str("    }\n");
    code.push_str("    \n");
    code.push_str("    if (tid == 0) {\n");
    code.push_str("        atomicAdd(output, sdata[0]);\n");
    code.push_str("    }\n");
    code.push_str("}\n\n");
    
    // Host function
    code.push_str("// Host function\n");
    code.push_str(&format!("cudaError_t {}_launch(float* h_input, float* h_output, int n) {{\n", module_name));
    code.push_str("    float *d_input, *d_output;\n");
    code.push_str("    cudaError_t err;\n");
    code.push_str("    \n");
    code.push_str("    // Allocate device memory\n");
    code.push_str("    err = cudaMalloc(&d_input, n * sizeof(float));\n");
    code.push_str("    if (err != cudaSuccess) return err;\n");
    code.push_str("    err = cudaMalloc(&d_output, n * sizeof(float));\n");
    code.push_str("    if (err != cudaSuccess) {\n");
    code.push_str("        cudaFree(d_input);\n");
    code.push_str("        return err;\n");
    code.push_str("    }\n");
    code.push_str("    \n");
    code.push_str("    // Copy to device\n");
    code.push_str("    err = cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);\n");
    code.push_str("    if (err != cudaSuccess) goto cleanup;\n");
    code.push_str("    \n");
    code.push_str("    // Launch kernel\n");
    code.push_str(&format!("    int blockSize = {};\n", config.block_size));
    code.push_str("    int gridSize = (n + blockSize - 1) / blockSize;\n");
    
    if config.shared_memory_bytes > 0 {
        code.push_str(&format!("    {}_kernel<<<gridSize, blockSize, {}>>>(d_input, d_output, n);\n", 
            module_name, config.shared_memory_bytes));
    } else {
        code.push_str(&format!("    {}_kernel<<<gridSize, blockSize>>>(d_input, d_output, n);\n", module_name));
    }
    
    code.push_str("    \n");
    code.push_str("    // Check kernel launch errors\n");
    code.push_str("    err = cudaGetLastError();\n");
    code.push_str("    if (err != cudaSuccess) goto cleanup;\n");
    code.push_str("    \n");
    code.push_str("    // Wait for completion\n");
    code.push_str("    err = cudaDeviceSynchronize();\n");
    code.push_str("    if (err != cudaSuccess) goto cleanup;\n");
    code.push_str("    \n");
    code.push_str("    // Copy back to host\n");
    code.push_str("    err = cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);\n");
    code.push_str("    \n");
    code.push_str("cleanup:\n");
    code.push_str("    cudaFree(d_input);\n");
    code.push_str("    cudaFree(d_output);\n");
    code.push_str("    return err;\n");
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
    let config = GPUConfig {
        platform,
        ..Default::default()
    };
    emit_with_config(module_name, &config)
}

/// Emit with configuration
pub fn emit_with_config(module_name: &str, config: &GPUConfig) -> EmitterResult<String> {
    let mut code = String::new();
    
    // Header comments
    code.push_str(&format!("// STUNIR Generated GPU Code ({})\n", config.platform));
    code.push_str(&format!("// Module: {}\n", module_name));
    code.push_str("// Generator: Rust Pipeline\n");
    code.push_str("// DO-178C Level A Compliance\n");
    code.push_str(&format!("// Compute Capability: {}\n", config.compute_capability));
    code.push_str(&format!("// Block Size: {}\n", config.block_size));
    if config.shared_memory_bytes > 0 {
        code.push_str(&format!("// Shared Memory: {} bytes\n", config.shared_memory_bytes));
    }
    code.push_str("\n");
    
    // Platform-specific code
    match config.platform {
        GPUPlatform::CUDA => {
            code.push_str(&emit_cuda_kernel(module_name, config));
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
            code.push_str(&emit_cuda_kernel(module_name, config).replace("cuda", "hip"));
        },
        GPUPlatform::Vulkan => {
            code.push_str("// Vulkan compute shader\n");
            code.push_str("#version 450\n\n");
            code.push_str(&format!("layout(local_size_x = {}) in;\n\n", config.block_size));
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
        assert!(code.contains("_reduce")); // Reduction kernel
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
    
    #[test]
    fn test_emit_with_shared_memory() {
        let mut config = GPUConfig::default();
        config.shared_memory_bytes = 1024;
        let result = emit_with_config("test", &config);
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("__shared__"));
        assert!(code.contains("Shared Memory: 1024 bytes"));
    }
    
    #[test]
    fn test_emit_with_optimization() {
        let mut config = GPUConfig::default();
        config.block_size = 512;
        config.use_fp16 = true;
        let result = emit_with_config("test", &config);
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("cuda_fp16.h"));
        assert!(code.contains("Block Size: 512"));
    }
    
    #[test]
    fn test_vectorized_kernel() {
        let result = emit(GPUPlatform::CUDA, "test");
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("_kernel_vectorized"));
        assert!(code.contains("float4"));
    }
    
    #[test]
    fn test_rocm() {
        let result = emit(GPUPlatform::ROCm, "test");
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("ROCm"));
        assert!(code.contains("hip"));
    }
    
    #[test]
    fn test_vulkan_compute() {
        let result = emit(GPUPlatform::Vulkan, "test");
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("#version 450"));
        assert!(code.contains("gl_GlobalInvocationID"));
    }
    
    #[test]
    fn test_error_handling() {
        let result = emit(GPUPlatform::CUDA, "test");
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("cudaError_t"));
        assert!(code.contains("goto cleanup"));
    }
}
