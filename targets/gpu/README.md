# STUNIR GPU Target

Comprehensive GPU compute kernel emitter for STUNIR supporting multiple backends.

## Overview

This emitter converts STUNIR IR to GPU compute kernels with full support for:
- **CUDA** (NVIDIA GPUs)
- **ROCm/HIP** (AMD GPUs, portable to NVIDIA)
- **OpenCL** (Cross-vendor)
- **Metal** (Apple Silicon)

All backends provide feature parity with advanced patterns, library wrappers, and optimization support.

## Quick Start

```bash
# CUDA (NVIDIA)
python emitter.py <ir.json> --output=<dir> --backend=cuda

# ROCm/HIP (AMD, portable)
python emitter.py <ir.json> --output=<dir> --backend=rocm

# OpenCL (cross-vendor)
python emitter.py <ir.json> --output=<dir> --backend=opencl

# Metal (Apple Silicon)
python emitter.py <ir.json> --output=<dir> --backend=metal
```

## Backend Comparison

| Feature | CUDA | ROCm/HIP | OpenCL | Metal |
|---------|------|----------|---------|--------|
| Vendor | NVIDIA | AMD (+ NVIDIA) | Any | Apple |
| Advanced Patterns | ✅ | ✅ | 🔄 | 🔄 |
| Library Wrappers | ✅ cuBLAS/cuSPARSE | ✅ hipBLAS/hipSPARSE | ⚠️ Limited | ⚠️ Limited |
| Memory Pool | ✅ | ✅ | 🔄 | 🔄 |
| Benchmarking | ✅ | ✅ | 🔄 | 🔄 |
| Multi-GPU | ✅ | ✅ | ⚠️ Partial | ⚠️ Limited |
| Portability | NVIDIA only | AMD + NVIDIA | High | Apple only |

**Legend**: ✅ Full support | 🔄 In progress | ⚠️ Basic support

## Backend-Specific Documentation

- **CUDA**: See `cuda/CUDA_README.md` - Full feature set for NVIDIA GPUs
- **ROCm/HIP**: See `ROCm_README.md` - Full feature set for AMD GPUs (portable to NVIDIA)
- **OpenCL/Metal**: See `emitter.py` docstring - Basic kernel generation

## Advanced Features

### Kernel Patterns (CUDA & ROCm)
- 2D Convolution (tiled, separable)
- Fast Fourier Transform (radix-2, radix-4)
- Sparse Matrix-Vector (CSR, ELL)
- Optimized Transpose (bank-conflict-free)

### Library Wrappers (CUDA & ROCm)
- **CUDA**: cuBLAS, cuSPARSE
- **ROCm**: hipBLAS, hipSPARSE

### Infrastructure (CUDA & ROCm)
- Memory pool management
- Performance benchmarking framework
- Multi-GPU support with peer access
- Unified memory (CUDA)

## Usage Examples

### Basic Kernel
```bash
python emitter.py ir.json --output=gpu_out --backend=cuda
```

### Advanced Pattern
```bash
python emitter.py ir.json --output=gpu_out --backend=cuda --kernel=conv2d
```

### With Library Wrapper
```bash
python emitter.py ir.json --output=gpu_out --backend=rocm --wrapper=hipblas
```

### With Benchmarking
```bash
python emitter.py ir.json --output=gpu_out --backend=cuda --benchmark
```

### Multi-GPU
```bash
python emitter.py ir.json --output=gpu_out --backend=rocm --multi-gpu
```

## Output Files

All backends generate:
- Kernel source files (.cu / .hip / .cl / .metal)
- Host wrapper code (.cpp / .hpp)
- Build scripts (build.sh)
- Deterministic manifest (manifest.json)
- Generated documentation (README.md)
- Receipt with SHA-256 hashes (receipt.json)

## Dependencies

### CUDA
- CUDA Toolkit 11.0+ (12.x recommended)
- nvcc compiler
- NVIDIA driver 470+

### ROCm
- ROCm 5.0+ (6.x recommended)
- hipcc compiler
- AMD GPU with GCN 3.0+ or RDNA

### OpenCL
- OpenCL SDK (vendor-specific)
- ICD loader

### Metal
- Xcode with Metal support
- macOS 10.13+ or iOS 11+

## Schema Versions

| Backend | Schema |
|---------|--------|
| CUDA | `stunir.gpu.cuda.v1` |
| ROCm | `stunir.gpu.rocm.v1` |
| OpenCL | `stunir.gpu.opencl.v1` |
| Metal | `stunir.gpu.metal.v1` |

## Implementation Status

- **Ada SPARK**: GPU emitter at `targets/spark/gpu/gpu_emitter.ads/adb`
- **Python**: Unified emitter at `targets/gpu/emitter.py`
- **CUDA Examples**: `targets/gpu/cuda/` (NEW - establishes parity with ROCm)
- **ROCm Examples**: `targets/gpu/rocm/` (comprehensive)

## Performance

All backends support:
- Memory coalescing optimization
- Shared memory usage
- Occupancy calculation
- Kernel fusion opportunities
- Multi-stream execution

See backend-specific READMEs for detailed performance tuning guides.
