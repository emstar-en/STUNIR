/**
 * STUNIR hipBLAS Wrapper
 *
 * High-level wrapper for common hipBLAS operations.
 * Provides simplified interface with automatic handle management.
 *
 * Schema: stunir.gpu.rocm.wrapper.hipblas.v1
 */

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include "wrapper_utils.hip"

namespace stunir {
namespace rocm {

/**
 * hipBLAS Handle Manager (RAII)
 */
class HipBlasHandle {
private:
    hipblasHandle_t handle;
    
public:
    HipBlasHandle() {
        STUNIR_HIPBLAS_CHECK(hipblasCreate(&handle));
    }
    
    ~HipBlasHandle() {
        hipblasDestroy(handle);
    }
    
    hipblasHandle_t get() { return handle; }
    operator hipblasHandle_t() { return handle; }
    
    void setStream(hipStream_t stream) {
        STUNIR_HIPBLAS_CHECK(hipblasSetStream(handle, stream));
    }
};

/**
 * BLAS Level 1 Wrappers
 */
class BlasL1 {
public:
    // Vector scaling: x = alpha * x
    static void scal(HipBlasHandle& handle, int n, float alpha, float* x, int incx = 1) {
        STUNIR_HIPBLAS_CHECK(hipblasSscal(handle, n, &alpha, x, incx));
    }
    
    static void scal(HipBlasHandle& handle, int n, double alpha, double* x, int incx = 1) {
        STUNIR_HIPBLAS_CHECK(hipblasDscal(handle, n, &alpha, x, incx));
    }
    
    // Vector copy: y = x
    static void copy(HipBlasHandle& handle, int n, const float* x, float* y, 
                     int incx = 1, int incy = 1) {
        STUNIR_HIPBLAS_CHECK(hipblasScopy(handle, n, x, incx, y, incy));
    }
    
    // AXPY: y = alpha * x + y
    static void axpy(HipBlasHandle& handle, int n, float alpha, 
                     const float* x, float* y, int incx = 1, int incy = 1) {
        STUNIR_HIPBLAS_CHECK(hipblasSaxpy(handle, n, &alpha, x, incx, y, incy));
    }
    
    static void axpy(HipBlasHandle& handle, int n, double alpha,
                     const double* x, double* y, int incx = 1, int incy = 1) {
        STUNIR_HIPBLAS_CHECK(hipblasDaxpy(handle, n, &alpha, x, incx, y, incy));
    }
    
    // Dot product: result = x . y
    static float dot(HipBlasHandle& handle, int n, const float* x, const float* y,
                     int incx = 1, int incy = 1) {
        float result;
        STUNIR_HIPBLAS_CHECK(hipblasSdot(handle, n, x, incx, y, incy, &result));
        return result;
    }
    
    // Vector norm: ||x||_2
    static float nrm2(HipBlasHandle& handle, int n, const float* x, int incx = 1) {
        float result;
        STUNIR_HIPBLAS_CHECK(hipblasSnrm2(handle, n, x, incx, &result));
        return result;
    }
    
    // Index of max absolute value
    static int iamax(HipBlasHandle& handle, int n, const float* x, int incx = 1) {
        int result;
        STUNIR_HIPBLAS_CHECK(hipblasIsamax(handle, n, x, incx, &result));
        return result - 1;  // hipBLAS returns 1-based index
    }
};

/**
 * BLAS Level 2 Wrappers
 */
class BlasL2 {
public:
    // GEMV: y = alpha * A * x + beta * y (or A^T)
    static void gemv(HipBlasHandle& handle, hipblasOperation_t trans,
                     int m, int n, float alpha, const float* A, int lda,
                     const float* x, int incx, float beta, float* y, int incy) {
        STUNIR_HIPBLAS_CHECK(hipblasSgemv(handle, trans, m, n, &alpha, 
                                          A, lda, x, incx, &beta, y, incy));
    }
    
    // Simplified GEMV with default parameters
    static void gemv(HipBlasHandle& handle, int m, int n, 
                     const float* A, const float* x, float* y,
                     float alpha = 1.0f, float beta = 0.0f,
                     bool transpose = false) {
        hipblasOperation_t op = transpose ? HIPBLAS_OP_T : HIPBLAS_OP_N;
        int lda = transpose ? n : m;
        gemv(handle, op, m, n, alpha, A, lda, x, 1, beta, y, 1);
    }
    
    // Symmetric matrix-vector: y = alpha * A * x + beta * y
    static void symv(HipBlasHandle& handle, hipblasFillMode_t uplo,
                     int n, float alpha, const float* A, int lda,
                     const float* x, float beta, float* y) {
        STUNIR_HIPBLAS_CHECK(hipblasSsymv(handle, uplo, n, &alpha,
                                          A, lda, x, 1, &beta, y, 1));
    }
    
    // GER: A = alpha * x * y^T + A (rank-1 update)
    static void ger(HipBlasHandle& handle, int m, int n, float alpha,
                    const float* x, const float* y, float* A, int lda) {
        STUNIR_HIPBLAS_CHECK(hipblasSger(handle, m, n, &alpha, 
                                         x, 1, y, 1, A, lda));
    }
};

/**
 * BLAS Level 3 Wrappers
 */
class BlasL3 {
public:
    // GEMM: C = alpha * A * B + beta * C
    static void gemm(HipBlasHandle& handle,
                     hipblasOperation_t transA, hipblasOperation_t transB,
                     int m, int n, int k,
                     float alpha, const float* A, int lda,
                     const float* B, int ldb,
                     float beta, float* C, int ldc) {
        STUNIR_HIPBLAS_CHECK(hipblasSgemm(handle, transA, transB,
                                          m, n, k, &alpha, A, lda,
                                          B, ldb, &beta, C, ldc));
    }
    
    static void gemm(HipBlasHandle& handle,
                     hipblasOperation_t transA, hipblasOperation_t transB,
                     int m, int n, int k,
                     double alpha, const double* A, int lda,
                     const double* B, int ldb,
                     double beta, double* C, int ldc) {
        STUNIR_HIPBLAS_CHECK(hipblasDgemm(handle, transA, transB,
                                          m, n, k, &alpha, A, lda,
                                          B, ldb, &beta, C, ldc));
    }
    
    // Simplified GEMM: C = A * B
    static void gemm(HipBlasHandle& handle, int m, int n, int k,
                     const float* A, const float* B, float* C,
                     float alpha = 1.0f, float beta = 0.0f) {
        gemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
             m, n, k, alpha, A, m, B, k, beta, C, m);
    }
    
    // Batched GEMM
    static void gemmBatched(HipBlasHandle& handle,
                            hipblasOperation_t transA, hipblasOperation_t transB,
                            int m, int n, int k,
                            float alpha, const float* const A[], int lda,
                            const float* const B[], int ldb,
                            float beta, float* const C[], int ldc,
                            int batchCount) {
        STUNIR_HIPBLAS_CHECK(hipblasSgemmBatched(handle, transA, transB,
                                                  m, n, k, &alpha, A, lda,
                                                  B, ldb, &beta, C, ldc, batchCount));
    }
    
    // Strided Batched GEMM (more efficient for contiguous batches)
    static void gemmStridedBatched(HipBlasHandle& handle,
                                   hipblasOperation_t transA, hipblasOperation_t transB,
                                   int m, int n, int k,
                                   float alpha, const float* A, int lda, long long strideA,
                                   const float* B, int ldb, long long strideB,
                                   float beta, float* C, int ldc, long long strideC,
                                   int batchCount) {
        STUNIR_HIPBLAS_CHECK(hipblasSgemmStridedBatched(handle, transA, transB,
                                                        m, n, k, &alpha, A, lda, strideA,
                                                        B, ldb, strideB, &beta, C, ldc, strideC,
                                                        batchCount));
    }
    
    // SYRK: C = alpha * A * A^T + beta * C (symmetric rank-k update)
    static void syrk(HipBlasHandle& handle, hipblasFillMode_t uplo,
                     hipblasOperation_t trans, int n, int k,
                     float alpha, const float* A, int lda,
                     float beta, float* C, int ldc) {
        STUNIR_HIPBLAS_CHECK(hipblasSsyrk(handle, uplo, trans, n, k,
                                          &alpha, A, lda, &beta, C, ldc));
    }
    
    // TRSM: Solve triangular system X = A^-1 * B or X = B * A^-1
    static void trsm(HipBlasHandle& handle, hipblasSideMode_t side,
                     hipblasFillMode_t uplo, hipblasOperation_t trans,
                     hipblasDiagType_t diag, int m, int n,
                     float alpha, const float* A, int lda,
                     float* B, int ldb) {
        STUNIR_HIPBLAS_CHECK(hipblasStrsm(handle, side, uplo, trans, diag,
                                          m, n, &alpha, A, lda, B, ldb));
    }
};

/**
 * High-level convenience functions
 */
class HipBlas {
private:
    HipBlasHandle handle;
    
public:
    HipBlas() {}
    
    // Matrix multiplication: C = A * B
    void matmul(int m, int n, int k, 
                const float* d_A, const float* d_B, float* d_C,
                float alpha = 1.0f, float beta = 0.0f) {
        BlasL3::gemm(handle, m, n, k, d_A, d_B, d_C, alpha, beta);
    }
    
    // Matrix-vector multiplication: y = A * x
    void matvec(int m, int n, const float* d_A, const float* d_x, float* d_y,
                float alpha = 1.0f, float beta = 0.0f) {
        BlasL2::gemv(handle, m, n, d_A, d_x, d_y, alpha, beta);
    }
    
    // Dot product
    float dot(int n, const float* d_x, const float* d_y) {
        return BlasL1::dot(handle, n, d_x, d_y);
    }
    
    // Vector norm
    float norm(int n, const float* d_x) {
        return BlasL1::nrm2(handle, n, d_x);
    }
    
    // Scale vector
    void scale(int n, float alpha, float* d_x) {
        BlasL1::scal(handle, n, alpha, d_x);
    }
    
    // AXPY: y = alpha * x + y
    void axpy(int n, float alpha, const float* d_x, float* d_y) {
        BlasL1::axpy(handle, n, alpha, d_x, d_y);
    }
    
    HipBlasHandle& getHandle() { return handle; }
};

} // namespace rocm
} // namespace stunir

// Example usage and verification
#ifdef STUNIR_HIPBLAS_TEST

using namespace stunir::rocm;

void verify_gemm(const float* A, const float* B, const float* C, 
                 int m, int n, int k, float tolerance = 1e-4f) {
    bool passed = true;
    for (int i = 0; i < m && passed; i++) {
        for (int j = 0; j < n && passed; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += A[i + l * m] * B[l + j * k];  // Column-major
            }
            if (fabsf(C[i + j * m] - sum) > tolerance) {
                printf("Mismatch at (%d,%d): %.6f vs %.6f\n", i, j, sum, C[i + j * m]);
                passed = false;
            }
        }
    }
    printf("GEMM verification: %s\n", passed ? "PASSED" : "FAILED");
}

int main() {
    printf("=== STUNIR hipBLAS Wrapper Test ===\n\n");
    
    // Print device info
    DeviceInfo info = DeviceInfo::get();
    info.print();
    printf("\n");
    
    // Test GEMM
    int m = 1024, n = 1024, k = 1024;
    printf("Testing SGEMM: (%d x %d) * (%d x %d)\n", m, k, k, n);
    
    // Allocate host memory
    std::vector<float> h_A(m * k), h_B(k * n), h_C(m * n);
    
    // Initialize
    srand(42);
    for (int i = 0; i < m * k; i++) h_A[i] = (float)(rand() % 10) / 10.0f;
    for (int i = 0; i < k * n; i++) h_B[i] = (float)(rand() % 10) / 10.0f;
    
    // Allocate device memory
    DeviceBuffer<float> d_A(m * k), d_B(k * n), d_C(m * n);
    d_A.copyFromHost(h_A.data());
    d_B.copyFromHost(h_B.data());
    d_C.zero();
    
    // Create hipBLAS wrapper and run GEMM
    HipBlas blas;
    GpuTimer timer;
    
    // Warmup
    blas.matmul(m, n, k, d_A, d_B, d_C);
    STUNIR_HIP_CHECK(hipDeviceSynchronize());
    
    // Benchmark
    int iterations = 10;
    timer.start();
    for (int i = 0; i < iterations; i++) {
        blas.matmul(m, n, k, d_A, d_B, d_C);
    }
    timer.stop();
    
    float avg_time = timer.elapsed_ms() / iterations;
    auto metrics = PerfMetrics::compute_gemm(m, n, k, avg_time);
    metrics.print("SGEMM");
    
    // Verify
    d_C.copyToHost(h_C.data());
    verify_gemm(h_A.data(), h_B.data(), h_C.data(), m, n, k);
    
    // Test Level 1 BLAS
    printf("\nTesting BLAS Level 1...\n");
    DeviceBuffer<float> d_x(1024), d_y(1024);
    std::vector<float> h_x(1024, 1.0f), h_y(1024, 2.0f);
    d_x.copyFromHost(h_x.data());
    d_y.copyFromHost(h_y.data());
    
    float dot_result = blas.dot(1024, d_x, d_y);
    printf("Dot product: %.2f (expected: 2048.00)\n", dot_result);
    
    float norm_result = blas.norm(1024, d_x);
    printf("L2 norm: %.2f (expected: 32.00)\n", norm_result);
    
    printf("\nhipBLAS wrapper test complete!\n");
    return 0;
}

#endif // STUNIR_HIPBLAS_TEST
