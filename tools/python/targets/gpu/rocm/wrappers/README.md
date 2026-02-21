# STUNIR ROCm Library Wrappers

High-level C++ wrappers for hipBLAS and hipSPARSE libraries.

## Features

- RAII handle management
- Automatic error handling
- Simplified API for common operations
- Performance timing utilities
- Device memory helpers

## Files

- `wrapper_utils.hip` - Common utilities (error handling, timers, memory)
- `hipblas_wrapper.hip` - hipBLAS wrapper (GEMM, GEMV, BLAS L1/L2/L3)
- `hipsparse_wrapper.hip` - hipSPARSE wrapper (SpMV, SpMM, CSR operations)

## Usage

### hipBLAS Example

```cpp
#include "hipblas_wrapper.hip"

using namespace stunir::rocm;

int main() {
    // Allocate device memory
    DeviceBuffer<float> d_A(m * k), d_B(k * n), d_C(m * n);
    d_A.copyFromHost(h_A);
    d_B.copyFromHost(h_B);
    
    // Create hipBLAS wrapper
    HipBlas blas;
    
    // Matrix multiplication: C = A * B
    blas.matmul(m, n, k, d_A, d_B, d_C);
    
    // Copy result back
    d_C.copyToHost(h_C);
}
```

### hipSPARSE Example

```cpp
#include "hipsparse_wrapper.hip"

using namespace stunir::rocm;

int main() {
    // Create CSR matrix
    CsrMatrix A(rows, cols, nnz);
    A.copyFromHost(values, row_ptr, col_indices);
    
    // Create vectors
    DenseVector x(cols), y(rows);
    x.copyFromHost(h_x);
    
    // Sparse matrix-vector: y = A * x
    HipSparse sparse;
    sparse.matvec(A, x, y);
}
```

## Building

### Build hipBLAS Test

```bash
hipcc -DSTUNIR_HIPBLAS_TEST -lhipblas -o test_hipblas hipblas_wrapper.hip
./test_hipblas
```

### Build hipSPARSE Test

```bash
hipcc -DSTUNIR_HIPSPARSE_TEST -lhipsparse -o test_hipsparse hipsparse_wrapper.hip
./test_hipsparse
```

## API Reference

### DeviceBuffer<T>

```cpp
DeviceBuffer<float> buf(1024);  // Allocate
buf.copyFromHost(host_ptr);     // Copy H2D
buf.copyToHost(host_ptr);       // Copy D2H
buf.zero();                     // Zero memory
float* ptr = buf.data();        // Get pointer
```

### HipBlas

| Method | Operation |
|--------|----------|
| `matmul(m,n,k,A,B,C)` | C = A * B |
| `matvec(m,n,A,x,y)` | y = A * x |
| `dot(n,x,y)` | x · y |
| `norm(n,x)` | ||x||₂ |
| `scale(n,alpha,x)` | x = α * x |
| `axpy(n,alpha,x,y)` | y = α*x + y |

### HipSparse

| Method | Operation |
|--------|----------|
| `matvec(A,x,y)` | y = A * x (SpMV) |
| `matmul(A,B,C)` | C = A * B (SpMM) |
| `transpose(A,AT)` | AT = A^T |

## Performance Tips

1. **Reuse handles**: Create `HipBlas`/`HipSparse` once, use many times
2. **Use streams**: Set stream with `handle.setStream(stream)` for overlap
3. **Pin host memory**: Use `hipHostMalloc` for async transfers
4. **Batch operations**: Use batched GEMM for many small matrices

## Dependencies

- ROCm 5.0+
- hipBLAS
- hipSPARSE

## Schema

- `stunir.gpu.rocm.wrapper.utils.v1`
- `stunir.gpu.rocm.wrapper.hipblas.v1`
- `stunir.gpu.rocm.wrapper.hipsparse.v1`
