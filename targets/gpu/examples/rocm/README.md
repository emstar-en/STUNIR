# STUNIR ROCm/HIP Examples

This directory contains production-ready ROCm/HIP examples demonstrating GPU compute patterns.

## Examples

### 1. Vector Addition (`vector_add.hip`)

**Difficulty:** Basic

Simple parallel vector addition demonstrating:
- Basic HIP kernel structure
- Memory allocation and transfer
- Thread indexing
- Performance measurement

```bash
./build/vector_add
```

### 2. Matrix Multiplication (`matmul.hip`)

**Difficulty:** Intermediate

Optimized matrix multiplication using:
- Shared memory tiling
- Memory coalescing
- Loop unrolling
- Naive vs optimized comparison

```bash
./build/matmul
```

### 3. Parallel Reduction (`reduction.hip`)

**Difficulty:** Advanced

Parallel sum reduction demonstrating:
- Warp-level primitives (`__shfl_down`)
- Shared memory reduction
- Multi-stage reduction
- Atomic operations
- Grid-stride loops

```bash
./build/reduction
```

## Building

### Prerequisites

- ROCm 5.0+ for AMD GPUs
- OR CUDA 11.0+ with HIP for NVIDIA GPUs

### Build Commands

```bash
# Build all examples for AMD GPU (default)
./build.sh

# Build for NVIDIA GPU
./build.sh --nvidia

# Build with debug symbols
./build.sh --debug

# Build single example
./build.sh vector_add

# Clean build artifacts
./build.sh --clean
```

## Output

Each example outputs:
- Device information
- Kernel execution time
- Verification results
- Performance metrics (bandwidth, GFLOPS)

## Hardware Differences

| Feature | AMD GPU | NVIDIA GPU |
|---------|---------|------------|
| Warp/Wavefront Size | 64 | 32 |
| Shared Memory | 64 KB | 48-164 KB |
| L2 Cache | 2-8 MB | 4-60 MB |
| Architecture | CDNA/RDNA | Ampere/Hopper |

## Porting from CUDA

HIP provides easy porting from CUDA:

| CUDA | HIP |
|------|-----|
| `cudaMalloc` | `hipMalloc` |
| `cudaMemcpy` | `hipMemcpy` |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` |
| `threadIdx.x` | `hipThreadIdx_x` |
| `blockIdx.x` | `hipBlockIdx_x` |
| `__syncthreads()` | `__syncthreads()` |

## Adding New Examples

1. Create `new_example.hip` in this directory
2. Follow the existing pattern:
   - Include `hip/hip_runtime.h`
   - Use `CHECK_HIP` macro for error handling
   - Add timing with `hipEvent`
   - Include verification
3. Add to `build.sh` if needed
4. Update this README

## References

- [HIP API Documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [ROCm Programming Guide](https://rocm.docs.amd.com/en/latest/programming_guide.html)
- [AMD GPU Architecture](https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch.html)
