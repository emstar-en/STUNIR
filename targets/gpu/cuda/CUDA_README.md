# STUNIR CUDA/cuBLAS GPU Target

Comprehensive CUDA support for NVIDIA GPU compute workloads.

## Overview

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform and programming model. This target provides feature-complete CUDA code generation matching the ROCm target capabilities.

## Features

### Core Capabilities
- **Native CUDA Kernels**: Optimized kernels for NVIDIA GPUs
- **cuBLAS Integration**: High-performance linear algebra
- **cuSPARSE Integration**: Sparse matrix operations
- **Deterministic Output**: SHA256-verified kernel generation
- **Multi-GPU Support**: Device management and data distribution

### Advanced Features
- **Kernel Patterns**: Conv2D, FFT, Sparse MatVec, Transpose
- **Library Wrappers**: cuBLAS, cuSPARSE with simplified API
- **Memory Management**: Pool allocator with arena support
- **Benchmarking**: Performance framework with JSON reporting
- **Unified Memory**: Automatic memory management support

## Directory Structure

```
targets/gpu/cuda/
├── README.md               # This file
├── examples/
│   ├── vector_add.cu       # Basic example
│   ├── matmul.cu          # Matrix multiplication
│   └── reduction.cu       # Parallel reduction
├── kernels/                # Advanced kernel patterns
│   ├── conv2d.cu          # 2D convolution
│   ├── fft.cu             # Fast Fourier Transform
│   ├── sparse_matvec.cu   # Sparse matrix-vector
│   └── transpose.cu       # Optimized transpose
├── wrappers/              # Library wrappers
│   ├── cublas_wrapper.cu
│   ├── cusparse_wrapper.cu
│   └── wrapper_utils.cu
├── memory/                # Memory management
│   ├── memory_pool.cu
│   └── memory_utils.cu
├── benchmarks/            # Performance tools
│   ├── benchmark_harness.cu
│   └── kernel_benchmarks.cu
├── multi_gpu/             # Multi-GPU support
│   ├── multi_gpu_utils.cu
│   └── multi_gpu_example.cu
├── PERFORMANCE.md         # Performance tuning guide
└── MULTI_GPU.md           # Multi-GPU programming guide
```

## Installation

### Ubuntu/Debian

```bash
# Install CUDA Toolkit (12.x or later recommended)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-3

# Verify installation
nvcc --version
nvidia-smi
```

### Environment Setup

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## Usage

### Generate CUDA Kernels

```bash
# Basic kernel generation
python3 targets/gpu/emitter.py ir.json --output=cuda_out --backend=cuda

# With specific kernel pattern
python3 targets/gpu/emitter.py ir.json --output=cuda_out --backend=cuda --kernel=conv2d

# With library wrapper
python3 targets/gpu/emitter.py ir.json --output=cuda_out --backend=cuda --wrapper=cublas

# With benchmark harness
python3 targets/gpu/emitter.py ir.json --output=cuda_out --backend=cuda --benchmark

# With multi-GPU support
python3 targets/gpu/emitter.py ir.json --output=cuda_out --backend=cuda --multi-gpu
```

### Build Generated Code

```bash
cd cuda_out
./build.sh         # Release build
./build.sh --debug # Debug build
./build.sh --ptx   # Generate PTX assembly
```

## Kernel Patterns

### 2D Convolution

```cuda
// Tiled convolution with shared memory
#include "kernels/conv2d.cu"

dim3 block(16, 16);
dim3 grid((width + 15) / 16, (height + 15) / 16);
conv2d_tiled<<<grid, block>>>(input, filter, output, h, w, fh, fw, ...);
```

### FFT

```cuda
// Radix-2 Cooley-Tukey FFT using shared memory
#include "kernels/fft.cu"

fft_radix2<<<batches, N, N * sizeof(cuComplex)>>>(data, N, log2N, -1);
```

### Sparse Matrix-Vector

```cuda
// CSR format SpMV with warp-level reduction
#include "kernels/sparse_matvec.cu"

int warps = (rows + 31) / 32;
spmv_csr_vector<<<warps, 256>>>(values, cols, row_ptr, x, y, rows);
```

### Transpose

```cuda
// Bank-conflict-free transpose using shared memory
#include "kernels/transpose.cu"

dim3 block(32, 8);
dim3 grid((cols + 31) / 32, (rows + 31) / 32);
transpose_optimized<<<grid, block>>>(input, output, rows, cols);
```

## Library Wrappers

### cuBLAS

```cuda
#include "wrappers/cublas_wrapper.cu"
using namespace stunir::cuda;

CuBlas blas;
blas.matmul(m, n, k, d_A, d_B, d_C);  // C = A * B
blas.matvec(m, n, d_A, d_x, d_y);     // y = A * x
float norm = blas.norm(n, d_x);       // ||x||_2
```

### cuSPARSE

```cuda
#include "wrappers/cusparse_wrapper.cu"
using namespace stunir::cuda;

CsrMatrix A(rows, cols, nnz);
A.copyFromHost(values, row_ptr, col_indices);

CuSparse sparse;
sparse.matvec(A, x, y);  // y = A * x
```

## Memory Management

```cuda
#include "memory/memory_pool.cu"
using namespace stunir::cuda::memory;

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

```cuda
#include "benchmarks/benchmark_harness.cu"
using namespace stunir::cuda::benchmark;

BenchmarkRunner runner(5, 100);  // 5 warmup, 100 iterations

runner.run("MyKernel", [&]() {
    my_kernel<<<grid, block>>>(...);
}, flops, bytes);

runner.print_summary();
runner.export_json("results.json");
```

## Multi-GPU

```cuda
#include "multi_gpu/multi_gpu_utils.cu"
using namespace stunir::cuda::multigpu;

DeviceManager mgr;
mgr.enable_all_peer_access();
mgr.print_info();

MultiGpuExecutor executor(mgr);
executor.parallel_for_devices([&](int dev, cudaStream_t stream) {
    cudaSetDevice(dev);
    kernel<<<grid, block, 0, stream>>>(...);
});
executor.synchronize_all();
```

## Type Mapping

| STUNIR IR | CUDA Type |
|-----------|-----------|
| i32 | int |
| i64 | long long |
| f32 | float |
| f64 | double |
| f16/half | __half |
| bool | bool |
| byte | unsigned char |

## Performance Tips

1. **Memory Coalescing**: Consecutive threads access consecutive memory (128-byte transactions)
2. **Shared Memory**: Use for data reuse (100x faster than global memory)
3. **Occupancy**: Target 50-75% for compute-bound kernels
4. **Warp Size**: Always 32 threads on NVIDIA GPUs
5. **Bank Conflicts**: Avoid in shared memory (32 banks, 4-byte width)
6. **cuBLAS/cuSPARSE**: Use for standard operations (highly optimized)
7. **Tensor Cores**: Available on Volta+ for mixed-precision operations
8. **Multi-GPU**: Scale large workloads with NVLINK or PCIe

See `PERFORMANCE.md` for detailed tuning guide.

## Compute Capability

| GPU Architecture | Compute Capability | Key Features |
|-----------------|-------------------|--------------|
| Turing (RTX 20xx) | 7.5 | Tensor Cores, RT Cores |
| Ampere (RTX 30xx, A100) | 8.0/8.6 | 2nd Gen Tensor Cores |
| Hopper (H100) | 9.0 | 4th Gen Tensor Cores, FP8 |
| Ada Lovelace (RTX 40xx) | 8.9 | 3rd Gen RT Cores, DLSS 3 |

## Documentation

- `PERFORMANCE.md` - Performance tuning guide
- `MULTI_GPU.md` - Multi-GPU programming guide
- `kernels/README.md` - Kernel patterns documentation
- `wrappers/README.md` - Library wrappers documentation
- `memory/README.md` - Memory management documentation
- `benchmarks/README.md` - Benchmarking documentation

## Schema Versions

| Component | Schema |
|-----------|--------|
| Kernel output | `stunir.gpu.cuda.v1` |
| Manifest | `stunir.target.gpu.cuda.manifest.v1` |
| Receipt | `stunir.target.gpu.cuda.receipt.v1` |
| Conv2D | `stunir.gpu.cuda.kernel.conv2d.v1` |
| FFT | `stunir.gpu.cuda.kernel.fft.v1` |
| SpMV | `stunir.gpu.cuda.kernel.sparse.v1` |
| Transpose | `stunir.gpu.cuda.kernel.transpose.v1` |
| cuBLAS | `stunir.gpu.cuda.wrapper.cublas.v1` |
| cuSPARSE | `stunir.gpu.cuda.wrapper.cusparse.v1` |
| Memory Pool | `stunir.gpu.cuda.memory.pool.v1` |
| Benchmark | `stunir.gpu.cuda.benchmark.harness.v1` |
| Multi-GPU | `stunir.gpu.cuda.multi_gpu.utils.v1` |

## Parity with ROCm

This CUDA target provides complete feature parity with the ROCm target:
- ✅ Advanced kernel patterns (Conv2D, FFT, Sparse, Transpose)
- ✅ Library wrappers (cuBLAS/cuSPARSE match hipBLAS/hipSPARSE)
- ✅ Memory pool management
- ✅ Benchmarking framework
- ✅ Multi-GPU support
- ✅ Performance optimization guides
- ✅ Comprehensive documentation

## References

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [cuSPARSE Documentation](https://docs.nvidia.com/cuda/cusparse/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/)
