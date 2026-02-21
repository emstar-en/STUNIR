/**
 * STUNIR hipSPARSE Wrapper
 *
 * High-level wrapper for sparse matrix operations.
 * Supports CSR, CSC, COO, and ELL formats.
 *
 * Schema: stunir.gpu.rocm.wrapper.hipsparse.v1
 */

#include <hip/hip_runtime.h>
#include <hipsparse/hipsparse.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include "wrapper_utils.hip"

namespace stunir {
namespace rocm {

/**
 * hipSPARSE Handle Manager (RAII)
 */
class HipSparseHandle {
private:
    hipsparseHandle_t handle;
    
public:
    HipSparseHandle() {
        STUNIR_HIPSPARSE_CHECK(hipsparseCreate(&handle));
    }
    
    ~HipSparseHandle() {
        hipsparseDestroy(handle);
    }
    
    hipsparseHandle_t get() { return handle; }
    operator hipsparseHandle_t() { return handle; }
    
    void setStream(hipStream_t stream) {
        STUNIR_HIPSPARSE_CHECK(hipsparseSetStream(handle, stream));
    }
};

/**
 * CSR Matrix Descriptor
 */
class CsrMatrix {
public:
    int rows, cols, nnz;
    DeviceBuffer<float> values;
    DeviceBuffer<int> row_ptr;
    DeviceBuffer<int> col_indices;
    hipsparseSpMatDescr_t descr;
    
    CsrMatrix() : rows(0), cols(0), nnz(0), descr(nullptr) {}
    
    CsrMatrix(int m, int n, int nz) : rows(m), cols(n), nnz(nz), descr(nullptr) {
        values.allocate(nnz);
        row_ptr.allocate(rows + 1);
        col_indices.allocate(nnz);
    }
    
    ~CsrMatrix() {
        if (descr) hipsparseDestroySpMat(descr);
    }
    
    void createDescriptor() {
        if (descr) hipsparseDestroySpMat(descr);
        STUNIR_HIPSPARSE_CHECK(hipsparseCreateCsr(&descr, rows, cols, nnz,
                                                   row_ptr.data(), col_indices.data(),
                                                   values.data(),
                                                   HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
                                                   HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F));
    }
    
    void copyFromHost(const float* h_values, const int* h_row_ptr, const int* h_col_indices) {
        values.copyFromHost(h_values, nnz);
        row_ptr.copyFromHost(h_row_ptr, rows + 1);
        col_indices.copyFromHost(h_col_indices, nnz);
    }
    
    // Create from dense matrix (host)
    static CsrMatrix fromDense(const float* dense, int rows, int cols, float threshold = 0.0f) {
        std::vector<float> vals;
        std::vector<int> col_idx;
        std::vector<int> row_p(rows + 1);
        
        for (int i = 0; i < rows; i++) {
            row_p[i] = vals.size();
            for (int j = 0; j < cols; j++) {
                float val = dense[i * cols + j];
                if (fabsf(val) > threshold) {
                    vals.push_back(val);
                    col_idx.push_back(j);
                }
            }
        }
        row_p[rows] = vals.size();
        
        CsrMatrix mat(rows, cols, vals.size());
        mat.copyFromHost(vals.data(), row_p.data(), col_idx.data());
        mat.createDescriptor();
        return mat;
    }
};

/**
 * Dense Vector Descriptor
 */
class DenseVector {
public:
    int size;
    DeviceBuffer<float> values;
    hipsparseDnVecDescr_t descr;
    
    DenseVector() : size(0), descr(nullptr) {}
    
    explicit DenseVector(int n) : size(n), descr(nullptr) {
        values.allocate(n);
    }
    
    ~DenseVector() {
        if (descr) hipsparseDestroyDnVec(descr);
    }
    
    void createDescriptor() {
        if (descr) hipsparseDestroyDnVec(descr);
        STUNIR_HIPSPARSE_CHECK(hipsparseCreateDnVec(&descr, size, values.data(), HIP_R_32F));
    }
    
    void copyFromHost(const float* h_values) {
        values.copyFromHost(h_values, size);
    }
    
    void copyToHost(float* h_values) const {
        values.copyToHost(h_values, size);
    }
    
    void zero() {
        values.zero();
    }
};

/**
 * Sparse Matrix-Vector Multiplication (SpMV)
 */
class SpMV {
private:
    HipSparseHandle handle;
    void* buffer;
    size_t buffer_size;
    
public:
    SpMV() : buffer(nullptr), buffer_size(0) {}
    
    ~SpMV() {
        if (buffer) hipFree(buffer);
    }
    
    // y = alpha * A * x + beta * y
    void compute(CsrMatrix& A, DenseVector& x, DenseVector& y,
                 float alpha = 1.0f, float beta = 0.0f) {
        // Ensure descriptors exist
        if (!A.descr) A.createDescriptor();
        if (!x.descr) x.createDescriptor();
        if (!y.descr) y.createDescriptor();
        
        // Get buffer size
        size_t new_buffer_size;
        STUNIR_HIPSPARSE_CHECK(hipsparseSpMV_bufferSize(
            handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, A.descr, x.descr, &beta, y.descr, HIP_R_32F,
            HIPSPARSE_SPMV_ALG_DEFAULT, &new_buffer_size));
        
        // Allocate buffer if needed
        if (new_buffer_size > buffer_size) {
            if (buffer) hipFree(buffer);
            STUNIR_HIP_CHECK(hipMalloc(&buffer, new_buffer_size));
            buffer_size = new_buffer_size;
        }
        
        // Execute SpMV
        STUNIR_HIPSPARSE_CHECK(hipsparseSpMV(
            handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, A.descr, x.descr, &beta, y.descr, HIP_R_32F,
            HIPSPARSE_SPMV_ALG_DEFAULT, buffer));
    }
    
    HipSparseHandle& getHandle() { return handle; }
};

/**
 * Sparse Matrix-Matrix Multiplication (SpMM)
 */
class SpMM {
private:
    HipSparseHandle handle;
    void* buffer;
    size_t buffer_size;
    
public:
    SpMM() : buffer(nullptr), buffer_size(0) {}
    
    ~SpMM() {
        if (buffer) hipFree(buffer);
    }
    
    // C = alpha * A * B + beta * C (A sparse, B and C dense)
    void compute(CsrMatrix& A, int B_rows, int B_cols, 
                 float* d_B, float* d_C, int ldb, int ldc,
                 float alpha = 1.0f, float beta = 0.0f) {
        if (!A.descr) A.createDescriptor();
        
        // Create dense matrix descriptors
        hipsparseDnMatDescr_t B_descr, C_descr;
        STUNIR_HIPSPARSE_CHECK(hipsparseCreateDnMat(&B_descr, B_rows, B_cols, ldb,
                                                     d_B, HIP_R_32F, HIPSPARSE_ORDER_COL));
        STUNIR_HIPSPARSE_CHECK(hipsparseCreateDnMat(&C_descr, A.rows, B_cols, ldc,
                                                     d_C, HIP_R_32F, HIPSPARSE_ORDER_COL));
        
        // Get buffer size
        size_t new_buffer_size;
        STUNIR_HIPSPARSE_CHECK(hipsparseSpMM_bufferSize(
            handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, A.descr, B_descr, &beta, C_descr, HIP_R_32F,
            HIPSPARSE_SPMM_ALG_DEFAULT, &new_buffer_size));
        
        if (new_buffer_size > buffer_size) {
            if (buffer) hipFree(buffer);
            STUNIR_HIP_CHECK(hipMalloc(&buffer, new_buffer_size));
            buffer_size = new_buffer_size;
        }
        
        // Execute SpMM
        STUNIR_HIPSPARSE_CHECK(hipsparseSpMM(
            handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, A.descr, B_descr, &beta, C_descr, HIP_R_32F,
            HIPSPARSE_SPMM_ALG_DEFAULT, buffer));
        
        hipsparseDestroyDnMat(B_descr);
        hipsparseDestroyDnMat(C_descr);
    }
};

/**
 * CSR Matrix Transpose
 */
class CsrTranspose {
private:
    HipSparseHandle handle;
    void* buffer;
    size_t buffer_size;
    
public:
    CsrTranspose() : buffer(nullptr), buffer_size(0) {}
    
    ~CsrTranspose() {
        if (buffer) hipFree(buffer);
    }
    
    void compute(const CsrMatrix& A, CsrMatrix& AT) {
        AT = CsrMatrix(A.cols, A.rows, A.nnz);
        
        // Get buffer size
        size_t new_buffer_size;
        STUNIR_HIPSPARSE_CHECK(hipsparseCsr2cscEx2_bufferSize(
            handle, A.rows, A.cols, A.nnz,
            A.values.data(), A.row_ptr.data(), A.col_indices.data(),
            AT.values.data(), AT.row_ptr.data(), AT.col_indices.data(),
            HIP_R_32F, HIPSPARSE_ACTION_NUMERIC, HIPSPARSE_INDEX_BASE_ZERO,
            HIPSPARSE_CSR2CSC_ALG1, &new_buffer_size));
        
        if (new_buffer_size > buffer_size) {
            if (buffer) hipFree(buffer);
            STUNIR_HIP_CHECK(hipMalloc(&buffer, new_buffer_size));
            buffer_size = new_buffer_size;
        }
        
        STUNIR_HIPSPARSE_CHECK(hipsparseCsr2cscEx2(
            handle, A.rows, A.cols, A.nnz,
            A.values.data(), A.row_ptr.data(), A.col_indices.data(),
            AT.values.data(), AT.row_ptr.data(), AT.col_indices.data(),
            HIP_R_32F, HIPSPARSE_ACTION_NUMERIC, HIPSPARSE_INDEX_BASE_ZERO,
            HIPSPARSE_CSR2CSC_ALG1, buffer));
        
        AT.createDescriptor();
    }
};

/**
 * High-level sparse operations interface
 */
class HipSparse {
private:
    SpMV spmv;
    SpMM spmm;
    CsrTranspose transpose_op;
    
public:
    // SpMV: y = A * x
    void matvec(CsrMatrix& A, DenseVector& x, DenseVector& y,
                float alpha = 1.0f, float beta = 0.0f) {
        spmv.compute(A, x, y, alpha, beta);
    }
    
    // SpMM: C = A * B (A sparse, B dense)
    void matmul(CsrMatrix& A, int B_rows, int B_cols,
                float* d_B, float* d_C, int ldb, int ldc,
                float alpha = 1.0f, float beta = 0.0f) {
        spmm.compute(A, B_rows, B_cols, d_B, d_C, ldb, ldc, alpha, beta);
    }
    
    // Transpose
    void transpose(const CsrMatrix& A, CsrMatrix& AT) {
        transpose_op.compute(A, AT);
    }
};

} // namespace rocm
} // namespace stunir

// Example and verification
#ifdef STUNIR_HIPSPARSE_TEST

using namespace stunir::rocm;

void create_test_csr(int rows, int cols, float density,
                     std::vector<float>& values,
                     std::vector<int>& row_ptr,
                     std::vector<int>& col_indices) {
    row_ptr.resize(rows + 1);
    values.clear();
    col_indices.clear();
    
    srand(42);
    for (int i = 0; i < rows; i++) {
        row_ptr[i] = values.size();
        for (int j = 0; j < cols; j++) {
            if ((float)rand() / RAND_MAX < density) {
                values.push_back((float)(rand() % 10 + 1) / 10.0f);
                col_indices.push_back(j);
            }
        }
    }
    row_ptr[rows] = values.size();
}

void spmv_reference(const std::vector<float>& values,
                    const std::vector<int>& row_ptr,
                    const std::vector<int>& col_indices,
                    const std::vector<float>& x,
                    std::vector<float>& y, int rows) {
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            sum += values[j] * x[col_indices[j]];
        }
        y[i] = sum;
    }
}

int main() {
    printf("=== STUNIR hipSPARSE Wrapper Test ===\n\n");
    
    DeviceInfo info = DeviceInfo::get();
    info.print();
    printf("\n");
    
    // Create test matrix
    int rows = 10000, cols = 10000;
    float density = 0.01f;
    
    std::vector<float> h_values;
    std::vector<int> h_row_ptr, h_col_indices;
    create_test_csr(rows, cols, density, h_values, h_row_ptr, h_col_indices);
    int nnz = h_values.size();
    
    printf("Test matrix: %d x %d, nnz: %d (%.2f%%)\n", 
           rows, cols, nnz, 100.0f * nnz / (rows * cols));
    
    // Create device CSR matrix
    CsrMatrix d_A(rows, cols, nnz);
    d_A.copyFromHost(h_values.data(), h_row_ptr.data(), h_col_indices.data());
    
    // Create vectors
    std::vector<float> h_x(cols, 1.0f), h_y(rows, 0.0f), h_y_ref(rows);
    DenseVector d_x(cols), d_y(rows);
    d_x.copyFromHost(h_x.data());
    d_y.zero();
    
    // Compute reference
    spmv_reference(h_values, h_row_ptr, h_col_indices, h_x, h_y_ref, rows);
    
    // Test SpMV
    HipSparse sparse;
    GpuTimer timer;
    
    // Warmup
    sparse.matvec(d_A, d_x, d_y);
    STUNIR_HIP_CHECK(hipDeviceSynchronize());
    
    // Benchmark
    int iterations = 100;
    timer.start();
    for (int i = 0; i < iterations; i++) {
        sparse.matvec(d_A, d_x, d_y);
    }
    timer.stop();
    
    float avg_time = timer.elapsed_ms() / iterations;
    auto metrics = PerfMetrics::compute_spmv(nnz, rows, cols, avg_time);
    metrics.print("SpMV");
    
    // Verify
    d_y.copyToHost(h_y.data());
    bool passed = true;
    for (int i = 0; i < rows && passed; i++) {
        if (fabsf(h_y[i] - h_y_ref[i]) > 1e-4f) {
            printf("Mismatch at %d: %.6f vs %.6f\n", i, h_y[i], h_y_ref[i]);
            passed = false;
        }
    }
    printf("SpMV verification: %s\n", passed ? "PASSED" : "FAILED");
    
    printf("\nhipSPARSE wrapper test complete!\n");
    return 0;
}

#endif // STUNIR_HIPSPARSE_TEST
