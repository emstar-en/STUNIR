# STUNIR ROCm/HIP GPU Target

Comprehensive ROCm support for AMD GPU compute workloads.

## Overview

ROCm (Radeon Open Compute) is AMD's open-source platform for GPU computing. HIP (Heterogeneous-compute Interface for Portability) provides portable code for both AMD and NVIDIA GPUs.

## Features

### Core Capabilities
- **Portable Kernels**: HIP code compiles for AMD and NVIDIA GPUs
- **CUDA-Like Syntax**: Familiar programming model
- **Full Hardware Access**: Direct access to AMD GPU features
- **Deterministic Output**: SHA256-verified kernel generation

### Advanced Features (New)
- **Kernel Patterns**: Conv2D, FFT, Sparse MatVec, Transpose
- **Library Wrappers**: hipBLAS, hipSPARSE with simplified API
- **Memory Management**: Pool allocator with arena support
- **Benchmarking**: Performance framework with JSON reporting
- **Multi-GPU**: Device management and data distribution

## Directory Structure

```
targets/gpu/
├── emitter.py              # GPU emitter with ROCm support
├── ROCm_README.md          # This file
├── README.md               # General GPU target readme
├── examples/rocm/
│   ├── vector_add.hip      # Basic example
│   ├── matmul.hip          # Matrix multiplication
│   ├── reduction.hip       # Parallel reduction
│   └── kernels/            # Advanced kernel patterns
│       ├── conv2d.hip      # 2D convolution
│       ├── fft.hip         # Fast Fourier Transform
│       ├── sparse_matvec.hip # Sparse matrix-vector
│       └── transpose.hip   # Optimized transpose
└── rocm/
    ├── wrappers/           # Library wrappers
    │   ├── hipblas_wrapper.hip
    │   ├── hipsparse_wrapper.hip
    │   └── wrapper_utils.hip
    ├── memory/             # Memory management
    │   ├── memory_pool.hip
    │   └── memory_utils.hip
    ├── benchmarks/         # Performance tools
    │   ├── benchmark_harness.hip
    │   └── kernel_benchmarks.hip
    ├── multi_gpu/          # Multi-GPU support
    │   ├── multi_gpu_utils.hip
    │   └── multi_gpu_example.hip
    ├── PERFORMANCE.md      # Performance tuning guide
    └── MULTI_GPU.md        # Multi-GPU programming guide
```

## Installation

### AMD GPU (ROCm)

```bash
# Ubuntu 22.04+
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_6.0.60000-1_all.deb
sudo dpkg -i amdgpu-install_*.deb
sudo amdgpu-install --usecase=rocm,hip

# Verify
hipcc --version
rocm-smi
```

### NVIDIA GPU (HIP on CUDA)

```bash
# Install CUDA Toolkit first
sudo apt install hip-nvcc
export HIP_PLATFORM=nvidia
```

## Usage

### Generate ROCm Kernels

```bash
# Basic kernel generation
python3 targets/gpu/emitter.py ir.json --output=rocm_out --backend=rocm

# With specific kernel pattern
python3 targets/gpu/emitter.py ir.json --output=rocm_out --backend=rocm --kernel=conv2d

# With library wrapper
python3 targets/gpu/emitter.py ir.json --output=rocm_out --backend=rocm --wrapper=hipblas

# With benchmark harness
python3 targets/gpu/emitter.py ir.json --output=rocm_out --backend=rocm --benchmark

# With multi-GPU support
python3 targets/gpu/emitter.py ir.json --output=rocm_out --backend=rocm --multi-gpu
```

### Build Generated Code

```bash
cd rocm_out
./build.sh           # AMD GPU (default)
./build.sh --nvidia  # NVIDIA GPU
./build.sh --debug   # Debug build
```

## Kernel Patterns

### 2D Convolution

```cpp
// Tiled convolution with shared memory
#include "kernels/conv2d.hip"

dim3 block(16, 16);
dim3 grid((width + 15) / 16, (height + 15) / 16);
conv2d_tiled<<<grid, block>>>(input, filter, output, h, w, fh, fw, ...);
```

### FFT

```cpp
// Radix-2 Cooley-Tukey FFT
#include "kernels/fft.hip"

fft_radix2<<<batches, N, N * sizeof(Complex)>>>(data, N, log2N, -1);
```

### Sparse Matrix-Vector

```cpp
// CSR format SpMV with warp-level reduction
#include "kernels/sparse_matvec.hip"

int warps = (rows + WARP_SIZE - 1) / WARP_SIZE;
spmv_csr_vector<<<warps, 256>>>(values, cols, row_ptr, x, y, rows);
```

### Transpose

```cpp
// Bank-conflict-free transpose
#include "kernels/transpose.hip"

dim3 block(32, 8);
dim3 grid((cols + 31) / 32, (rows + 31) / 32);
transpose_optimized<<<grid, block>>>(input, output, rows, cols);
```

## Library Wrappers

### hipBLAS

```cpp
#include "wrappers/hipblas_wrapper.hip"
using namespace stunir::rocm;

HipBlas blas;
blas.matmul(m, n, k, d_A, d_B, d_C);  // C = A * B
blas.matvec(m, n, d_A, d_x, d_y);     // y = A * x
float norm = blas.norm(n, d_x);       // ||x||_2
```

### hipSPARSE

```cpp
#include "wrappers/hipsparse_wrapper.hip"
using namespace stunir::rocm;

CsrMatrix A(rows, cols, nnz);
A.copyFromHost(values, row_ptr, col_indices);

HipSparse sparse;
sparse.matvec(A, x, y);  // y = A * x
```

## Memory Management

```cpp
#include "memory/memory_pool.hip"
using namespace stunir::rocm::memory;

// Global pool for frequent allocations
float* data = GlobalPool::get().allocate_typed<float>(1024);
GlobalPool::get().deallocate(data);

// RAII buffer
{
    PooledBuffer<float> buf(1024);
    // Automatically returned to pool at scope end
}

// Statistics
GlobalPool::get().print_stats();
```

## Benchmarking

```cpp
#include "benchmarks/benchmark_harness.hip"
using namespace stunir::rocm::benchmark;

BenchmarkRunner runner(5, 100);  // 5 warmup, 100 iterations

runner.run("MyKernel", [&]() {
    my_kernel<<<grid, block>>>(...);
}, flops, bytes);

runner.print_summary();
runner.export_json("results.json");
```

## Multi-GPU

```cpp
#include "multi_gpu/multi_gpu_utils.hip"
using namespace stunir::rocm::multigpu;

DeviceManager mgr;
mgr.enable_all_peer_access();
mgr.print_info();

MultiGpuExecutor executor(mgr);
executor.parallel_for_devices([&](int dev, hipStream_t stream) {
    hipSetDevice(dev);
    kernel<<<grid, block, 0, stream>>>(...);
});
executor.synchronize_all();
```

## Type Mapping

| STUNIR IR | HIP Type |
|-----------|----------|
| i32 | int |
| i64 | long long |
| f32 | float |
| f64 | double |
| f16/half | __half |
| bool | bool |
| byte | unsigned char |

## Performance Tips

1. **Memory Coalescing**: Consecutive threads access consecutive memory
2. **Shared Memory**: Use for data reuse (100x faster than global)
3. **Occupancy**: Target > 50% for compute-bound kernels
4. **Wavefront Size**: 64 on AMD compute GPUs, 32 on RDNA gaming
5. **Memory Pool**: Use for frequent small allocations
6. **hipBLAS/hipSPARSE**: Use for standard operations
7. **Multi-GPU**: Scale large workloads across GPUs

See `rocm/PERFORMANCE.md` for detailed tuning guide.

## Documentation

- `rocm/PERFORMANCE.md` - Performance tuning guide
- `rocm/MULTI_GPU.md` - Multi-GPU programming guide
- `examples/rocm/kernels/README.md` - Kernel patterns documentation
- `rocm/wrappers/README.md` - Library wrappers documentation
- `rocm/memory/README.md` - Memory management documentation
- `rocm/benchmarks/README.md` - Benchmarking documentation

## Schema Versions

| Component | Schema |
|-----------|--------|
| Kernel output | `stunir.gpu.rocm.v1` |
| Manifest | `stunir.target.gpu.rocm.manifest.v1` |
| Receipt | `stunir.target.gpu.rocm.receipt.v1` |
| Conv2D | `stunir.gpu.rocm.kernel.conv2d.v1` |
| FFT | `stunir.gpu.rocm.kernel.fft.v1` |
| SpMV | `stunir.gpu.rocm.kernel.sparse.v1` |
| Transpose | `stunir.gpu.rocm.kernel.transpose.v1` |
| hipBLAS | `stunir.gpu.rocm.wrapper.hipblas.v1` |
| hipSPARSE | `stunir.gpu.rocm.wrapper.hipsparse.v1` |
| Memory Pool | `stunir.gpu.rocm.memory.pool.v1` |
| Benchmark | `stunir.gpu.rocm.benchmark.harness.v1` |
| Multi-GPU | `stunir.gpu.rocm.multi_gpu.utils.v1` |

## References

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/)
- [hipBLAS Documentation](https://rocm.docs.amd.com/projects/hipBLAS/)
- [hipSPARSE Documentation](https://rocm.docs.amd.com/projects/hipSPARSE/)
- [AMD GPU Architecture](https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch.html)
