#!/usr/bin/env python3
"""STUNIR GPU Emitter - Emit CUDA/OpenCL/Metal/ROCm kernels.

This tool is part of the targets → gpu pipeline stage.
It converts STUNIR IR to GPU compute kernels.

Enhanced ROCm support includes:
- Advanced kernel patterns (conv2d, fft, sparse_matvec, transpose)
- hipBLAS/hipSPARSE wrapper generation
- Memory pool management
- Performance benchmarking
- Multi-GPU code generation

Usage:
    emitter.py <ir.json> --output=<dir> [--backend=cuda|opencl|metal|rocm]
    emitter.py <ir.json> --output=<dir> --backend=rocm --kernel=conv2d|fft|sparse|transpose
    emitter.py <ir.json> --output=<dir> --backend=rocm --wrapper=hipblas|hipsparse
    emitter.py <ir.json> --output=<dir> --backend=rocm --benchmark
    emitter.py <ir.json> --output=<dir> --backend=rocm --multi-gpu
    emitter.py --help

Schema: stunir.gpu.emitter.v2
"""

import json
import hashlib
import time
import sys
from pathlib import Path


def canonical_json(data):
    """Generate RFC 8785 / JCS subset canonical JSON."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'))


def compute_sha256(content):
    """Compute SHA256 hash of content."""
    if isinstance(content, str):
        content = content.encode('utf-8')
    return hashlib.sha256(content).hexdigest()


class GpuEmitter:
    """Emitter for GPU compute kernels (CUDA/OpenCL/Metal/ROCm)."""
    
    TYPE_MAP_CUDA = {
        'i32': 'int', 'i64': 'long long', 'f32': 'float', 'f64': 'double',
        'int': 'int', 'long': 'long long', 'float': 'float', 'double': 'double',
        'void': 'void', 'bool': 'bool', 'byte': 'unsigned char'
    }
    
    TYPE_MAP_OPENCL = {
        'i32': 'int', 'i64': 'long', 'f32': 'float', 'f64': 'double',
        'int': 'int', 'long': 'long', 'float': 'float', 'double': 'double',
        'void': 'void', 'bool': 'int', 'byte': 'uchar'
    }
    
    TYPE_MAP_METAL = {
        'i32': 'int', 'i64': 'int64_t', 'f32': 'float', 'f64': 'double',
        'int': 'int', 'long': 'int64_t', 'float': 'float', 'double': 'double',
        'void': 'void', 'bool': 'bool', 'byte': 'uint8_t'
    }
    
    TYPE_MAP_ROCM = {
        'i32': 'int', 'i64': 'long long', 'f32': 'float', 'f64': 'double',
        'int': 'int', 'long': 'long long', 'float': 'float', 'double': 'double',
        'void': 'void', 'bool': 'bool', 'byte': 'unsigned char',
        'half': '__half', 'f16': '__half'
    }
    
    BACKEND_CONFIG = {
        'cuda': {'ext': 'cu', 'type_map': TYPE_MAP_CUDA},
        'opencl': {'ext': 'cl', 'type_map': TYPE_MAP_OPENCL},
        'metal': {'ext': 'metal', 'type_map': TYPE_MAP_METAL},
        'rocm': {'ext': 'hip', 'type_map': TYPE_MAP_ROCM}
    }
    
    def __init__(self, ir_data, out_dir, options=None):
        """Initialize GPU emitter."""
        self.ir_data = ir_data
        self.out_dir = Path(out_dir)
        self.options = options or {}
        self.backend = options.get('backend', 'cuda') if options else 'cuda'
        self.generated_files = []
        self.epoch = int(time.time())
        
        config = self.BACKEND_CONFIG.get(self.backend, self.BACKEND_CONFIG['cuda'])
        self.ext = config['ext']
        self.type_map = config['type_map']
    
    def _write_file(self, path, content):
        """Write content to file."""
        full_path = self.out_dir / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding='utf-8', newline='\n')
        self.generated_files.append({
            'path': str(path),
            'sha256': compute_sha256(content),
            'size': len(content.encode('utf-8'))
        })
        return full_path
    
    def _map_type(self, ir_type):
        """Map IR type to GPU type."""
        return self.type_map.get(ir_type, 'int')
    
    def _emit_statement(self, stmt, indent='    '):
        """Convert IR statement to GPU code."""
        if isinstance(stmt, dict):
            stmt_type = stmt.get('type', 'nop')
            if stmt_type == 'var_decl':
                var_type = self._map_type(stmt.get('var_type', 'i32'))
                var_name = stmt.get('var_name', 'v0')
                init = stmt.get('init', '0')
                return f'{indent}{var_type} {var_name} = {init};'
            elif stmt_type == 'return':
                value = stmt.get('value', '0')
                return f'{indent}return {value};'
            elif stmt_type == 'assign':
                return f'{indent}{stmt.get("target", "v0")} = {stmt.get("value", "0")};'
            elif stmt_type in ('add', 'sub', 'mul'):
                op = {'+': '+', 'add': '+', 'sub': '-', 'mul': '*'}.get(stmt_type, '+')
                return f'{indent}{stmt.get("dest", "v0")} = {stmt.get("left", "0")} {op} {stmt.get("right", "0")};'
            elif stmt_type == 'call':
                func = stmt.get('func', 'noop')
                args = ', '.join(stmt.get('args', []))
                return f'{indent}{func}({args});'
            else:
                return f'{indent}// {stmt_type}: not implemented'
        return f'{indent}// nop'
    
    def _emit_kernel_cuda(self, func):
        """Emit CUDA kernel."""
        name = func.get('name', 'kernel')
        params = func.get('params', [])
        body = func.get('body', [])
        
        param_str = ', '.join([
            f"{self._map_type(p.get('type', 'i32'))} {p.get('name', f'arg{i}')}"
            if isinstance(p, dict) else f'int arg{i}'
            for i, p in enumerate(params)
        ])
        
        lines = [
            f'__global__ void {name}({param_str}) {{',
            '    // Thread indexing',
            '    int idx = blockIdx.x * blockDim.x + threadIdx.x;',
            ''
        ]
        
        for stmt in body:
            lines.append(self._emit_statement(stmt))
        
        lines.append('}')
        return '\n'.join(lines)
    
    def _emit_kernel_opencl(self, func):
        """Emit OpenCL kernel."""
        name = func.get('name', 'kernel')
        params = func.get('params', [])
        body = func.get('body', [])
        
        param_str = ', '.join([
            f"__global {self._map_type(p.get('type', 'i32'))}* {p.get('name', f'arg{i}')}"
            if isinstance(p, dict) else f'__global int* arg{i}'
            for i, p in enumerate(params)
        ])
        
        lines = [
            f'__kernel void {name}({param_str}) {{',
            '    // Work-item indexing',
            '    int idx = get_global_id(0);',
            ''
        ]
        
        for stmt in body:
            lines.append(self._emit_statement(stmt))
        
        lines.append('}')
        return '\n'.join(lines)
    
    def _emit_kernel_metal(self, func):
        """Emit Metal kernel."""
        name = func.get('name', 'kernel')
        params = func.get('params', [])
        body = func.get('body', [])
        
        param_str = ', '.join([
            f"device {self._map_type(p.get('type', 'i32'))}* {p.get('name', f'arg{i}')} [[buffer({i})]]"
            if isinstance(p, dict) else f'device int* arg{i} [[buffer({i})]]'
            for i, p in enumerate(params)
        ])
        
        lines = [
            f'kernel void {name}({param_str}, uint idx [[thread_position_in_grid]]) {{',
            ''
        ]
        
        for stmt in body:
            lines.append(self._emit_statement(stmt))
        
        lines.append('}')
        return '\n'.join(lines)
    
    def _emit_kernel_rocm(self, func):
        """Emit ROCm/HIP kernel."""
        name = func.get('name', 'kernel')
        params = func.get('params', [])
        body = func.get('body', [])
        
        param_str = ', '.join([
            f"{self._map_type(p.get('type', 'i32'))}* {p.get('name', f'arg{i}')}"
            if isinstance(p, dict) else f'int* arg{i}'
            for i, p in enumerate(params)
        ])
        
        lines = [
            f'__global__ void {name}({param_str}) {{',
            '    // HIP thread indexing (compatible with CUDA)',
            '    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;',
            ''
        ]
        
        for stmt in body:
            lines.append(self._emit_statement(stmt))
        
        lines.append('}')
        return '\n'.join(lines)
    
    def emit(self):
        """Emit GPU kernel files."""
        module_name = self.ir_data.get('ir_module', self.ir_data.get('module', 'module'))
        functions = self.ir_data.get('ir_functions', self.ir_data.get('functions', []))
        
        emit_funcs = {
            'cuda': (self._emit_kernel_cuda, self._emit_header_cuda),
            'opencl': (self._emit_kernel_opencl, self._emit_header_opencl),
            'metal': (self._emit_kernel_metal, self._emit_header_metal),
            'rocm': (self._emit_kernel_rocm, self._emit_header_rocm)
        }
        
        emit_kernel, emit_header = emit_funcs.get(self.backend, emit_funcs['cuda'])
        
        header = emit_header(module_name)
        kernel_code = header
        
        for func in functions:
            kernel_code += emit_kernel(func) + '\n\n'
        
        self._write_file(f'{module_name}.{self.ext}', kernel_code)
        
        # Build script
        build_script = self._emit_build_script(module_name)
        self._write_file('build.sh', build_script)
        
        # Host wrapper
        host_code = self._emit_host_wrapper(module_name, functions)
        self._write_file(f'{module_name}_host.cpp', host_code)
        
        # README
        self._write_file('README.md', self._emit_readme(module_name, len(functions)))
        
        return kernel_code
    
    def _emit_header_cuda(self, module_name):
        return f"""// STUNIR CUDA Kernels
// Module: {module_name}
// Schema: stunir.gpu.cuda.v1
// Epoch: {self.epoch}

#include <cuda_runtime.h>
#include <stdio.h>

"""
    
    def _emit_header_opencl(self, module_name):
        return f"""// STUNIR OpenCL Kernels
// Module: {module_name}
// Schema: stunir.gpu.opencl.v1
// Epoch: {self.epoch}

"""
    
    def _emit_header_metal(self, module_name):
        return f"""// STUNIR Metal Kernels
// Module: {module_name}
// Schema: stunir.gpu.metal.v1
// Epoch: {self.epoch}

#include <metal_stdlib>
using namespace metal;

"""
    
    def _emit_header_rocm(self, module_name):
        return f"""// STUNIR ROCm/HIP Kernels
// Module: {module_name}
// Schema: stunir.gpu.rocm.v1
// Epoch: {self.epoch}
//
// HIP provides portable GPU computing across AMD and NVIDIA GPUs.
// Compile with: hipcc -o <output> <source>.hip
// For AMD GPUs: hipcc defaults to HIP-Clang targeting AMD GPUs
// For NVIDIA: hipcc -platform nvidia (requires CUDA)

#include <hip/hip_runtime.h>
#include <stdio.h>

"""
    
    def _emit_build_script(self, module_name):
        """Generate build script."""
        scripts = {
            'cuda': f"""#!/bin/bash
# STUNIR CUDA Build Script
set -e

if command -v nvcc &> /dev/null; then
    echo "Compiling CUDA kernels..."
    nvcc -o {module_name} {module_name}.cu {module_name}_host.cpp
    echo "Generated: {module_name}"
else
    echo "Error: nvcc not found. Install CUDA toolkit."
    exit 1
fi
""",
            'opencl': f"""#!/bin/bash
# STUNIR OpenCL Build Script
set -e

echo "Compiling OpenCL host..."
g++ -o {module_name} {module_name}_host.cpp -lOpenCL
echo "Generated: {module_name}"
echo "Kernel: {module_name}.cl (loaded at runtime)"
""",
            'metal': f"""#!/bin/bash
# STUNIR Metal Build Script
set -e

echo "Compiling Metal shaders..."
xcrun -sdk macosx metal -c {module_name}.metal -o {module_name}.air
xcrun -sdk macosx metallib {module_name}.air -o {module_name}.metallib
echo "Generated: {module_name}.metallib"

echo "Compiling host..."
clang++ -framework Metal -framework Foundation {module_name}_host.cpp -o {module_name}
echo "Generated: {module_name}"
""",
            'rocm': f"""#!/bin/bash
# STUNIR ROCm/HIP Build Script
# Supports AMD GPUs (default) and NVIDIA GPUs (with --nvidia flag)
set -e

NVIDIA_MODE=0
DEBUG_MODE=0

for arg in "$@"; do
    case $arg in
        --nvidia)
            NVIDIA_MODE=1
            ;;
        --debug)
            DEBUG_MODE=1
            ;;
    esac
done

if ! command -v hipcc &> /dev/null; then
    echo "Error: hipcc not found."
    echo "Install ROCm: https://rocm.docs.amd.com/en/latest/"
    echo "Or install HIP on NVIDIA: https://rocm.docs.amd.com/en/latest/install/hip.html"
    exit 1
fi

echo "=== STUNIR ROCm/HIP Build ==="

BUILD_FLAGS="-O3"
if [ $DEBUG_MODE -eq 1 ]; then
    BUILD_FLAGS="-g -O0"
fi

if [ $NVIDIA_MODE -eq 1 ]; then
    echo "Target: NVIDIA GPU (CUDA backend)"
    hipcc --platform nvidia $BUILD_FLAGS -o {module_name} {module_name}.hip {module_name}_host.cpp
else
    echo "Target: AMD GPU (ROCm backend)"
    hipcc $BUILD_FLAGS -o {module_name} {module_name}.hip {module_name}_host.cpp
fi

echo "Generated: {module_name}"
echo ""
echo "Run with: ./{module_name}"
echo "For device info: hipinfo"
"""
        }
        return scripts.get(self.backend, scripts['cuda'])
    
    def _emit_host_wrapper(self, module_name, functions):
        """Generate host-side wrapper code."""
        kernel_name = functions[0].get('name', 'kernel') if functions else 'kernel'
        
        wrappers = {
            'cuda': f"""// STUNIR CUDA Host Wrapper
// Module: {module_name}

#include <stdio.h>
#include <cuda_runtime.h>

extern void {kernel_name}();

int main(int argc, char** argv) {{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("CUDA devices: %d\\n", deviceCount);
    
    if (deviceCount > 0) {{
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("Device: %s\\n", prop.name);
    }}
    
    return 0;
}}
""",
            'opencl': f"""// STUNIR OpenCL Host Wrapper
// Module: {module_name}

#include <stdio.h>
#include <CL/cl.h>

int main(int argc, char** argv) {{
    cl_uint platformCount;
    clGetPlatformIDs(0, NULL, &platformCount);
    printf("OpenCL platforms: %u\\n", platformCount);
    return 0;
}}
""",
            'metal': f"""// STUNIR Metal Host Wrapper
// Module: {module_name}

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

int main(int argc, char** argv) {{
    @autoreleasepool {{
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device) {{
            NSLog(@"Metal device: %@", device.name);
        }} else {{
            NSLog(@"Metal not available");
        }}
    }}
    return 0;
}}
""",
            'rocm': f"""// STUNIR ROCm/HIP Host Wrapper
// Module: {module_name}
// 
// HIP provides source-level portability between AMD and NVIDIA GPUs.
// This code compiles with both hipcc (AMD) and nvcc (NVIDIA).

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

void checkHipError(hipError_t err, const char* context) {{
    if (err != hipSuccess) {{
        fprintf(stderr, "HIP Error at %s: %s\\n", context, hipGetErrorString(err));
        exit(1);
    }}
}}

int main(int argc, char** argv) {{
    int deviceCount = 0;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    checkHipError(err, "hipGetDeviceCount");
    
    printf("=== STUNIR ROCm/HIP Runtime ===");
    printf("\\nHIP devices: %d\\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {{
        hipDeviceProp_t prop;
        err = hipGetDeviceProperties(&prop, i);
        checkHipError(err, "hipGetDeviceProperties");
        
        printf("\\nDevice %d: %s\\n", i, prop.name);
        printf("  Compute Capability: %d.%d\\n", prop.major, prop.minor);
        printf("  Total Memory: %.2f GB\\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Multiprocessors: %d\\n", prop.multiProcessorCount);
        printf("  Clock Rate: %.2f GHz\\n", prop.clockRate / 1e6);
        printf("  Max Threads/Block: %d\\n", prop.maxThreadsPerBlock);
        printf("  Warp Size: %d\\n", prop.warpSize);
    }}
    
    return 0;
}}
"""
        }
        return wrappers.get(self.backend, wrappers['cuda'])
    
    def _emit_readme(self, module_name, func_count):
        """Generate README."""
        backend_info = {
            'cuda': ('CUDA Toolkit (nvcc)', 'NVIDIA GPU'),
            'opencl': ('OpenCL SDK', 'Any GPU with OpenCL support'),
            'metal': ('Xcode Command Line Tools', 'Apple GPU (macOS/iOS)'),
            'rocm': ('ROCm/HIP (hipcc)', 'AMD GPU (native) or NVIDIA GPU (portable)')
        }
        
        req, gpu = backend_info.get(self.backend, backend_info['cuda'])
        
        extra_info = ''
        if self.backend == 'rocm':
            extra_info = """\n## ROCm/HIP Notes

- HIP kernels are portable across AMD and NVIDIA GPUs
- Use `--nvidia` flag to compile for NVIDIA: `./build.sh --nvidia`
- Install ROCm: https://rocm.docs.amd.com/en/latest/
- Check GPU info: `hipinfo` or `rocm-smi`
"""
        
        return f"""# {module_name} (GPU)

Generated by STUNIR GPU Emitter.

## Backend

{self.backend.upper()}

## Files

- `{module_name}.{self.ext}` - Kernel code
- `{module_name}_host.cpp` - Host wrapper
- `build.sh` - Build script

## Build

```bash
chmod +x build.sh
./build.sh
```

## Requirements

- {req}
- {gpu}
{extra_info}
## Statistics

- Kernels: {func_count}
- Epoch: {self.epoch}

## Schema

stunir.gpu.{self.backend}.v1
"""
    
    def emit_manifest(self):
        """Generate target manifest."""
        return {
            'schema': f'stunir.target.gpu.{self.backend}.manifest.v1',
            'epoch': self.epoch,
            'backend': self.backend,
            'files': sorted(self.generated_files, key=lambda f: f['path']),
            'file_count': len(self.generated_files)
        }
    
    def emit_receipt(self):
        """Generate target receipt."""
        manifest = self.emit_manifest()
        manifest_json = canonical_json(manifest)
        return {
            'schema': f'stunir.target.gpu.{self.backend}.receipt.v1',
            'epoch': self.epoch,
            'manifest_sha256': compute_sha256(manifest_json),
            'file_count': len(self.generated_files)
        }


class RocmAdvancedEmitter:
    """Advanced ROCm emitter for specialized kernel patterns."""
    
    KERNEL_PATTERNS = {
        'conv2d': 'conv2d.hip',
        'fft': 'fft.hip',
        'sparse': 'sparse_matvec.hip',
        'transpose': 'transpose.hip'
    }
    
    WRAPPER_TYPES = {
        'hipblas': 'hipblas_wrapper.hip',
        'hipsparse': 'hipsparse_wrapper.hip'
    }
    
    def __init__(self, ir_data, out_dir):
        self.ir_data = ir_data
        self.out_dir = Path(out_dir)
        self.epoch = int(time.time())
        self.generated_files = []
    
    def _write_file(self, path, content):
        """Write content to file."""
        full_path = self.out_dir / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding='utf-8', newline='\n')
        self.generated_files.append({
            'path': str(path),
            'sha256': compute_sha256(content),
            'size': len(content.encode('utf-8'))
        })
        return full_path
    
    def emit_kernel_pattern(self, pattern_type):
        """Generate specialized kernel pattern based on IR."""
        module = self.ir_data.get('ir_module', 'module')
        
        if pattern_type == 'conv2d':
            return self._emit_conv2d(module)
        elif pattern_type == 'fft':
            return self._emit_fft(module)
        elif pattern_type == 'sparse':
            return self._emit_sparse(module)
        elif pattern_type == 'transpose':
            return self._emit_transpose(module)
        return None
    
    def _emit_conv2d(self, module):
        """Generate 2D convolution kernel."""
        code = f'''// STUNIR ROCm Conv2D Kernel (Generated)
// Module: {module}
// Schema: stunir.gpu.rocm.kernel.conv2d.v1
// Epoch: {self.epoch}

#include <hip/hip_runtime.h>

#define TILE_WIDTH 16
#define TILE_HEIGHT 16

__global__ void conv2d_{module}(const float* __restrict__ input,
                                 const float* __restrict__ filter,
                                 float* __restrict__ output,
                                 int height, int width,
                                 int filter_h, int filter_w) {{
    __shared__ float tile[TILE_HEIGHT + 6][TILE_WIDTH + 6];
    __shared__ float filt[7][7];
    
    int tx = hipThreadIdx_x, ty = hipThreadIdx_y;
    int out_y = hipBlockIdx_y * TILE_HEIGHT + ty;
    int out_x = hipBlockIdx_x * TILE_WIDTH + tx;
    
    // Load filter
    if (tx < filter_w && ty < filter_h) {{
        filt[ty][tx] = filter[ty * filter_w + tx];
    }}
    __syncthreads();
    
    // Compute convolution
    if (out_y < height && out_x < width) {{
        float sum = 0.0f;
        for (int fy = 0; fy < filter_h; fy++) {{
            for (int fx = 0; fx < filter_w; fx++) {{
                int iy = out_y - filter_h/2 + fy;
                int ix = out_x - filter_w/2 + fx;
                if (iy >= 0 && iy < height && ix >= 0 && ix < width) {{
                    sum += input[iy * width + ix] * filt[fy][fx];
                }}
            }}
        }}
        output[out_y * width + out_x] = sum;
    }}
}}
'''
        self._write_file(f'{module}_conv2d.hip', code)
        return code
    
    def _emit_fft(self, module):
        """Generate FFT kernel."""
        code = f'''// STUNIR ROCm FFT Kernel (Generated)
// Module: {module}
// Schema: stunir.gpu.rocm.kernel.fft.v1
// Epoch: {self.epoch}

#include <hip/hip_runtime.h>
#include <math.h>

#define PI 3.14159265358979323846f

struct Complex {{
    float real, imag;
    __device__ Complex operator+(const Complex& b) const {{
        return {{real + b.real, imag + b.imag}};
    }}
    __device__ Complex operator-(const Complex& b) const {{
        return {{real - b.real, imag - b.imag}};
    }}
    __device__ Complex operator*(const Complex& b) const {{
        return {{real*b.real - imag*b.imag, real*b.imag + imag*b.real}};
    }}
}};

__device__ Complex twiddle(int k, int N, int dir) {{
    float angle = dir * 2.0f * PI * k / N;
    return {{cosf(angle), sinf(angle)}};
}}

__global__ void fft_{module}(Complex* data, int N, int log2N, int direction) {{
    extern __shared__ Complex shared[];
    int tid = hipThreadIdx_x;
    
    // Bit-reverse load
    unsigned int rev = 0;
    unsigned int idx = tid;
    for (int i = 0; i < log2N; i++) {{
        rev = (rev << 1) | (idx & 1);
        idx >>= 1;
    }}
    shared[tid] = data[hipBlockIdx_x * N + rev];
    __syncthreads();
    
    // FFT butterflies
    for (int s = 1; s <= log2N; s++) {{
        int m = 1 << s;
        int m2 = m >> 1;
        int j = tid & (m2 - 1);
        int i = (tid / m2) * m + j;
        
        Complex w = twiddle(j, m, direction);
        Complex t = w * shared[i + m2];
        Complex u = shared[i];
        shared[i] = u + t;
        shared[i + m2] = u - t;
        __syncthreads();
    }}
    
    data[hipBlockIdx_x * N + tid] = shared[tid];
}}
'''
        self._write_file(f'{module}_fft.hip', code)
        return code
    
    def _emit_sparse(self, module):
        """Generate sparse matrix-vector kernel."""
        code = f'''// STUNIR ROCm SpMV Kernel (Generated)
// Module: {module}
// Schema: stunir.gpu.rocm.kernel.sparse.v1
// Epoch: {self.epoch}

#include <hip/hip_runtime.h>

#define WARP_SIZE 64

__global__ void spmv_{module}(const float* __restrict__ values,
                               const int* __restrict__ col_indices,
                               const int* __restrict__ row_ptr,
                               const float* __restrict__ x,
                               float* __restrict__ y,
                               int num_rows) {{
    int lane = hipThreadIdx_x & (WARP_SIZE - 1);
    int warp = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) / WARP_SIZE;
    
    if (warp < num_rows) {{
        int start = row_ptr[warp];
        int end = row_ptr[warp + 1];
        float sum = 0.0f;
        
        for (int j = start + lane; j < end; j += WARP_SIZE) {{
            sum += values[j] * x[col_indices[j]];
        }}
        
        // Warp reduction
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {{
            sum += __shfl_down(sum, offset, WARP_SIZE);
        }}
        
        if (lane == 0) {{
            y[warp] = sum;
        }}
    }}
}}
'''
        self._write_file(f'{module}_spmv.hip', code)
        return code
    
    def _emit_transpose(self, module):
        """Generate optimized transpose kernel."""
        code = f'''// STUNIR ROCm Transpose Kernel (Generated)
// Module: {module}
// Schema: stunir.gpu.rocm.kernel.transpose.v1
// Epoch: {self.epoch}

#include <hip/hip_runtime.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose_{module}(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int rows, int cols) {{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    
    int x = hipBlockIdx_x * TILE_DIM + hipThreadIdx_x;
    int y = hipBlockIdx_y * TILE_DIM + hipThreadIdx_y;
    
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {{
        if (x < cols && (y + j) < rows) {{
            tile[hipThreadIdx_y + j][hipThreadIdx_x] = input[(y + j) * cols + x];
        }}
    }}
    __syncthreads();
    
    x = hipBlockIdx_y * TILE_DIM + hipThreadIdx_x;
    y = hipBlockIdx_x * TILE_DIM + hipThreadIdx_y;
    
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {{
        if (x < rows && (y + j) < cols) {{
            output[(y + j) * rows + x] = tile[hipThreadIdx_x][hipThreadIdx_y + j];
        }}
    }}
}}
'''
        self._write_file(f'{module}_transpose.hip', code)
        return code
    
    def emit_wrapper(self, wrapper_type):
        """Generate library wrapper stubs."""
        module = self.ir_data.get('ir_module', 'module')
        
        if wrapper_type == 'hipblas':
            return self._emit_hipblas_wrapper(module)
        elif wrapper_type == 'hipsparse':
            return self._emit_hipsparse_wrapper(module)
        return None
    
    def _emit_hipblas_wrapper(self, module):
        """Generate hipBLAS wrapper."""
        code = f'''// STUNIR hipBLAS Wrapper (Generated)
// Module: {module}
// Schema: stunir.gpu.rocm.wrapper.hipblas.v1
// Epoch: {self.epoch}
// Link with: -lhipblas

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

class {module.title()}HipBlas {{
private:
    hipblasHandle_t handle;
    
public:
    {module.title()}HipBlas() {{ hipblasCreate(&handle); }}
    ~{module.title()}HipBlas() {{ hipblasDestroy(handle); }}
    
    void gemm(int m, int n, int k, float alpha,
              const float* A, const float* B, float beta, float* C) {{
        hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                     m, n, k, &alpha, A, m, B, k, &beta, C, m);
    }}
    
    void gemv(int m, int n, float alpha, const float* A,
              const float* x, float beta, float* y) {{
        hipblasSgemv(handle, HIPBLAS_OP_N, m, n, &alpha, A, m, x, 1, &beta, y, 1);
    }}
    
    float dot(int n, const float* x, const float* y) {{
        float result;
        hipblasSdot(handle, n, x, 1, y, 1, &result);
        return result;
    }}
}};
'''
        self._write_file(f'{module}_hipblas.hip', code)
        return code
    
    def _emit_hipsparse_wrapper(self, module):
        """Generate hipSPARSE wrapper."""
        code = f'''// STUNIR hipSPARSE Wrapper (Generated)
// Module: {module}
// Schema: stunir.gpu.rocm.wrapper.hipsparse.v1
// Epoch: {self.epoch}
// Link with: -lhipsparse

#include <hip/hip_runtime.h>
#include <hipsparse/hipsparse.h>

class {module.title()}HipSparse {{
private:
    hipsparseHandle_t handle;
    
public:
    {module.title()}HipSparse() {{ hipsparseCreate(&handle); }}
    ~{module.title()}HipSparse() {{ hipsparseDestroy(handle); }}
    
    // SpMV: y = alpha * A * x + beta * y
    void spmv_csr(int m, int n, int nnz,
                  const float* values, const int* row_ptr, const int* col_idx,
                  const float* x, float* y, float alpha = 1.0f, float beta = 0.0f) {{
        hipsparseSpMatDescr_t A;
        hipsparseDnVecDescr_t vecX, vecY;
        
        hipsparseCreateCsr(&A, m, n, nnz, (void*)row_ptr, (void*)col_idx,
                           (void*)values, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
                           HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F);
        hipsparseCreateDnVec(&vecX, n, (void*)x, HIP_R_32F);
        hipsparseCreateDnVec(&vecY, m, (void*)y, HIP_R_32F);
        
        size_t bufferSize;
        hipsparseSpMV_bufferSize(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                  &alpha, A, vecX, &beta, vecY, HIP_R_32F,
                                  HIPSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
        
        void* buffer;
        hipMalloc(&buffer, bufferSize);
        
        hipsparseSpMV(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                      &alpha, A, vecX, &beta, vecY, HIP_R_32F,
                      HIPSPARSE_SPMV_ALG_DEFAULT, buffer);
        
        hipFree(buffer);
        hipsparseDestroySpMat(A);
        hipsparseDestroyDnVec(vecX);
        hipsparseDestroyDnVec(vecY);
    }}
}};
'''
        self._write_file(f'{module}_hipsparse.hip', code)
        return code
    
    def emit_benchmark_harness(self):
        """Generate benchmark harness for the module."""
        module = self.ir_data.get('ir_module', 'module')
        code = f'''// STUNIR Benchmark Harness (Generated)
// Module: {module}
// Schema: stunir.gpu.rocm.benchmark.v1
// Epoch: {self.epoch}

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <chrono>

class {module.title()}Benchmark {{
private:
    hipEvent_t start_event, stop_event;
    
public:
    {module.title()}Benchmark() {{
        hipEventCreate(&start_event);
        hipEventCreate(&stop_event);
    }}
    
    ~{module.title()}Benchmark() {{
        hipEventDestroy(start_event);
        hipEventDestroy(stop_event);
    }}
    
    void start() {{ hipEventRecord(start_event); }}
    
    float stop() {{
        hipEventRecord(stop_event);
        hipEventSynchronize(stop_event);
        float ms;
        hipEventElapsedTime(&ms, start_event, stop_event);
        return ms;
    }}
    
    void report(const char* name, float ms, double flops = 0, double bytes = 0) {{
        printf("%s: %.3f ms", name, ms);
        if (flops > 0) printf(", %.2f GFLOPS", flops / (ms / 1000.0) / 1e9);
        if (bytes > 0) printf(", %.2f GB/s", bytes / (ms / 1000.0) / 1e9);
        printf("\\n");
    }}
}};
'''
        self._write_file(f'{module}_benchmark.hip', code)
        return code
    
    def emit_multi_gpu_utils(self):
        """Generate multi-GPU utilities."""
        module = self.ir_data.get('ir_module', 'module')
        code = f'''// STUNIR Multi-GPU Utilities (Generated)
// Module: {module}
// Schema: stunir.gpu.rocm.multi_gpu.v1
// Epoch: {self.epoch}

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <vector>

class {module.title()}MultiGpu {{
private:
    int num_devices;
    std::vector<hipStream_t> streams;
    
public:
    {module.title()}MultiGpu() {{
        hipGetDeviceCount(&num_devices);
        streams.resize(num_devices);
        for (int i = 0; i < num_devices; i++) {{
            hipSetDevice(i);
            hipStreamCreate(&streams[i]);
        }}
    }}
    
    ~{module.title()}MultiGpu() {{
        for (int i = 0; i < num_devices; i++) {{
            hipSetDevice(i);
            hipStreamDestroy(streams[i]);
        }}
    }}
    
    int device_count() const {{ return num_devices; }}
    hipStream_t stream(int dev) {{ return streams[dev]; }}
    
    void enable_peer_access() {{
        for (int i = 0; i < num_devices; i++) {{
            for (int j = 0; j < num_devices; j++) {{
                if (i != j) {{
                    int can_access;
                    hipDeviceCanAccessPeer(&can_access, i, j);
                    if (can_access) {{
                        hipSetDevice(i);
                        hipDeviceEnablePeerAccess(j, 0);
                    }}
                }}
            }}
        }}
    }}
    
    void synchronize_all() {{
        for (int i = 0; i < num_devices; i++) {{
            hipStreamSynchronize(streams[i]);
        }}
    }}
    
    void print_info() {{
        printf("Multi-GPU System: %d devices\\n", num_devices);
        for (int i = 0; i < num_devices; i++) {{
            hipDeviceProp_t prop;
            hipGetDeviceProperties(&prop, i);
            printf("  GPU %d: %s (%.2f GB)\\n", i, prop.name,
                   prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        }}
    }}
}};
'''
        self._write_file(f'{module}_multi_gpu.hip', code)
        return code
    
    def emit_manifest(self):
        """Generate manifest for advanced emitter output."""
        return {
            'schema': 'stunir.target.gpu.rocm.advanced.manifest.v1',
            'epoch': self.epoch,
            'files': sorted(self.generated_files, key=lambda f: f['path']),
            'file_count': len(self.generated_files)
        }


def main():
    args = {'output': None, 'input': None, 'backend': 'cuda',
            'kernel': None, 'wrapper': None, 'benchmark': False, 'multi_gpu': False}
    
    for arg in sys.argv[1:]:
        if arg.startswith('--output='):
            args['output'] = arg.split('=', 1)[1]
        elif arg.startswith('--backend='):
            args['backend'] = arg.split('=', 1)[1]
        elif arg.startswith('--kernel='):
            args['kernel'] = arg.split('=', 1)[1]
        elif arg.startswith('--wrapper='):
            args['wrapper'] = arg.split('=', 1)[1]
        elif arg == '--benchmark':
            args['benchmark'] = True
        elif arg == '--multi-gpu':
            args['multi_gpu'] = True
        elif arg == '--help':
            print(__doc__)
            sys.exit(0)
        elif not arg.startswith('--'):
            args['input'] = arg
    
    if not args['input']:
        print(f"Usage: {sys.argv[0]} <ir.json> --output=<dir>", file=sys.stderr)
        sys.exit(1)
    
    out_dir = args['output'] or 'gpu_output'
    
    try:
        with open(args['input'], 'r') as f:
            ir_data = json.load(f)
        
        # Standard GPU emitter
        emitter = GpuEmitter(ir_data, out_dir, {'backend': args['backend']})
        emitter.emit()
        
        # Advanced ROCm features
        if args['backend'] == 'rocm':
            advanced = RocmAdvancedEmitter(ir_data, out_dir)
            
            if args['kernel']:
                advanced.emit_kernel_pattern(args['kernel'])
                print(f"Generated {args['kernel']} kernel pattern", file=sys.stderr)
            
            if args['wrapper']:
                advanced.emit_wrapper(args['wrapper'])
                print(f"Generated {args['wrapper']} wrapper", file=sys.stderr)
            
            if args['benchmark']:
                advanced.emit_benchmark_harness()
                print("Generated benchmark harness", file=sys.stderr)
            
            if args['multi_gpu']:
                advanced.emit_multi_gpu_utils()
                print("Generated multi-GPU utilities", file=sys.stderr)
        
        manifest = emitter.emit_manifest()
        manifest_path = Path(out_dir) / 'manifest.json'
        manifest_path.write_text(canonical_json(manifest), encoding='utf-8')
        
        print(f"GPU kernels emitted to {out_dir}/", file=sys.stderr)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
