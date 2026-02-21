/**
 * STUNIR ROCm Kernel: Sparse Matrix-Vector Multiplication (SpMV)
 *
 * CSR (Compressed Sparse Row) format implementation.
 * Supports scalar and vector-based approaches.
 *
 * Schema: stunir.gpu.rocm.kernel.sparse_matvec.v1
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WARP_SIZE 64  // AMD wavefront size
#define BLOCK_SIZE 256

#define CHECK_HIP(call) { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, \
                hipGetErrorString(err)); \
        exit(1); \
    } \
}

/**
 * CSR (Compressed Sparse Row) format
 * - values[nnz]: Non-zero values
 * - col_indices[nnz]: Column indices for each non-zero
 * - row_ptr[num_rows+1]: Start index of each row in values/col_indices
 */
struct CSRMatrix {
    float* values;
    int* col_indices;
    int* row_ptr;
    int num_rows;
    int num_cols;
    int nnz;
};

/**
 * Scalar SpMV kernel - one thread per row
 * Simple but can have load imbalance
 */
__global__ void spmv_csr_scalar(const float* __restrict__ values,
                                 const int* __restrict__ col_indices,
                                 const int* __restrict__ row_ptr,
                                 const float* __restrict__ x,
                                 float* __restrict__ y,
                                 int num_rows) {
    int row = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    
    if (row < num_rows) {
        float sum = 0.0f;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        
        for (int j = row_start; j < row_end; j++) {
            sum += values[j] * x[col_indices[j]];
        }
        
        y[row] = sum;
    }
}

/**
 * Vector SpMV kernel - one warp per row
 * Better load balancing for rows with many non-zeros
 */
__global__ void spmv_csr_vector(const float* __restrict__ values,
                                 const int* __restrict__ col_indices,
                                 const int* __restrict__ row_ptr,
                                 const float* __restrict__ x,
                                 float* __restrict__ y,
                                 int num_rows) {
    int lane = hipThreadIdx_x & (WARP_SIZE - 1);  // Lane within warp
    int row = hipBlockIdx_x * (hipBlockDim_x / WARP_SIZE) + (hipThreadIdx_x / WARP_SIZE);
    
    if (row < num_rows) {
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        
        float sum = 0.0f;
        
        // Each lane processes elements stride apart
        for (int j = row_start + lane; j < row_end; j += WARP_SIZE) {
            sum += values[j] * x[col_indices[j]];
        }
        
        // Warp reduction using shuffle
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down(sum, offset, WARP_SIZE);
        }
        
        // Lane 0 writes result
        if (lane == 0) {
            y[row] = sum;
        }
    }
}

/**
 * Adaptive SpMV - choose strategy based on row length
 * Uses segmented scan for irregular rows
 */
__global__ void spmv_csr_adaptive(const float* __restrict__ values,
                                   const int* __restrict__ col_indices,
                                   const int* __restrict__ row_ptr,
                                   const float* __restrict__ x,
                                   float* __restrict__ y,
                                   int num_rows,
                                   int threshold) {
    __shared__ float shared_vals[BLOCK_SIZE];
    __shared__ int shared_rows[BLOCK_SIZE + 1];
    
    int tid = hipThreadIdx_x;
    int gid = hipBlockIdx_x * hipBlockDim_x + tid;
    
    // Load row boundaries into shared memory
    if (gid <= num_rows) {
        shared_rows[tid] = (gid < num_rows) ? row_ptr[gid] : row_ptr[num_rows];
    }
    if (tid == BLOCK_SIZE - 1 || gid == num_rows - 1) {
        shared_rows[tid + 1] = row_ptr[min(gid + 1, num_rows)];
    }
    __syncthreads();
    
    int row = gid;
    if (row < num_rows) {
        int row_start = shared_rows[tid];
        int row_end = shared_rows[tid + 1];
        int row_len = row_end - row_start;
        
        float sum = 0.0f;
        
        if (row_len <= threshold) {
            // Scalar approach for short rows
            for (int j = row_start; j < row_end; j++) {
                sum += values[j] * x[col_indices[j]];
            }
        } else {
            // Use shared memory for long rows
            for (int j = row_start; j < row_end; j += BLOCK_SIZE) {
                int idx = j + tid;
                if (idx < row_end && tid < BLOCK_SIZE) {
                    shared_vals[tid] = values[idx] * x[col_indices[idx]];
                } else {
                    shared_vals[tid] = 0.0f;
                }
                __syncthreads();
                
                // Partial reduction in shared memory
                if (tid == 0) {
                    int limit = min(BLOCK_SIZE, row_end - j);
                    for (int k = 0; k < limit; k++) {
                        sum += shared_vals[k];
                    }
                }
                __syncthreads();
            }
        }
        
        if (row_len <= threshold || tid == 0) {
            y[row] = sum;
        }
    }
}

/**
 * ELL format SpMV for better memory coalescing
 * Requires padding to max row length
 */
__global__ void spmv_ell(const float* __restrict__ values,
                          const int* __restrict__ col_indices,
                          float* __restrict__ y,
                          const float* __restrict__ x,
                          int num_rows,
                          int max_nnz_per_row) {
    int row = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    
    if (row < num_rows) {
        float sum = 0.0f;
        
        for (int j = 0; j < max_nnz_per_row; j++) {
            int idx = j * num_rows + row;  // Column-major for coalescing
            int col = col_indices[idx];
            if (col >= 0) {  // -1 indicates padding
                sum += values[idx] * x[col];
            }
        }
        
        y[row] = sum;
    }
}

// Host functions for CSR creation and manipulation
CSRMatrix* create_random_csr(int rows, int cols, float density) {
    CSRMatrix* mat = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    mat->num_rows = rows;
    mat->num_cols = cols;
    
    // Estimate nnz
    int est_nnz = (int)(rows * cols * density);
    float* temp_values = (float*)malloc(est_nnz * sizeof(float));
    int* temp_cols = (int*)malloc(est_nnz * sizeof(int));
    mat->row_ptr = (int*)malloc((rows + 1) * sizeof(int));
    
    int nnz = 0;
    mat->row_ptr[0] = 0;
    
    srand(42);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if ((float)rand() / RAND_MAX < density) {
                temp_values[nnz] = (float)(rand() % 10 + 1) / 10.0f;
                temp_cols[nnz] = j;
                nnz++;
            }
        }
        mat->row_ptr[i + 1] = nnz;
    }
    
    mat->nnz = nnz;
    mat->values = (float*)malloc(nnz * sizeof(float));
    mat->col_indices = (int*)malloc(nnz * sizeof(int));
    memcpy(mat->values, temp_values, nnz * sizeof(float));
    memcpy(mat->col_indices, temp_cols, nnz * sizeof(int));
    
    free(temp_values);
    free(temp_cols);
    
    return mat;
}

void free_csr(CSRMatrix* mat) {
    free(mat->values);
    free(mat->col_indices);
    free(mat->row_ptr);
    free(mat);
}

void spmv_reference(const CSRMatrix* mat, const float* x, float* y) {
    for (int i = 0; i < mat->num_rows; i++) {
        float sum = 0.0f;
        for (int j = mat->row_ptr[i]; j < mat->row_ptr[i + 1]; j++) {
            sum += mat->values[j] * x[mat->col_indices[j]];
        }
        y[i] = sum;
    }
}

bool verify_spmv(const float* result, const float* reference, int n) {
    int errors = 0;
    for (int i = 0; i < n && errors < 5; i++) {
        if (fabsf(result[i] - reference[i]) > 1e-4f) {
            printf("Mismatch at %d: %.6f vs %.6f\n", i, result[i], reference[i]);
            errors++;
        }
    }
    return errors == 0;
}

int main(int argc, char** argv) {
    int rows = 10000;
    int cols = 10000;
    float density = 0.01f;  // 1% non-zeros
    
    printf("=== STUNIR ROCm: Sparse Matrix-Vector Multiplication ===\n");
    printf("Matrix: %d x %d, Density: %.1f%%\n", rows, cols, density * 100);
    
    // Create sparse matrix
    CSRMatrix* mat = create_random_csr(rows, cols, density);
    printf("Non-zeros: %d (actual density: %.2f%%)\n", 
           mat->nnz, 100.0f * mat->nnz / (rows * cols));
    
    // Allocate vectors
    float* h_x = (float*)malloc(cols * sizeof(float));
    float* h_y = (float*)malloc(rows * sizeof(float));
    float* h_ref = (float*)malloc(rows * sizeof(float));
    
    for (int i = 0; i < cols; i++) h_x[i] = 1.0f;
    
    // Compute reference
    spmv_reference(mat, h_x, h_ref);
    
    // Device memory
    float *d_values, *d_x, *d_y;
    int *d_col_indices, *d_row_ptr;
    
    CHECK_HIP(hipMalloc(&d_values, mat->nnz * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_col_indices, mat->nnz * sizeof(int)));
    CHECK_HIP(hipMalloc(&d_row_ptr, (rows + 1) * sizeof(int)));
    CHECK_HIP(hipMalloc(&d_x, cols * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_y, rows * sizeof(float)));
    
    CHECK_HIP(hipMemcpy(d_values, mat->values, mat->nnz * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_col_indices, mat->col_indices, mat->nnz * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_row_ptr, mat->row_ptr, (rows + 1) * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_x, h_x, cols * sizeof(float), hipMemcpyHostToDevice));
    
    // Timing
    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));
    
    int numBlocks = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Scalar kernel
    CHECK_HIP(hipEventRecord(start));
    hipLaunchKernelGGL(spmv_csr_scalar, dim3(numBlocks), dim3(BLOCK_SIZE), 0, 0,
                       d_values, d_col_indices, d_row_ptr, d_x, d_y, rows);
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));
    
    float scalar_time;
    CHECK_HIP(hipEventElapsedTime(&scalar_time, start, stop));
    printf("Scalar kernel: %.3f ms\n", scalar_time);
    
    CHECK_HIP(hipMemcpy(h_y, d_y, rows * sizeof(float), hipMemcpyDeviceToHost));
    printf("Scalar verification: %s\n", verify_spmv(h_y, h_ref, rows) ? "PASSED" : "FAILED");
    
    // Vector kernel (warps per row)
    int warpsPerBlock = BLOCK_SIZE / WARP_SIZE;
    int numBlocksVector = (rows + warpsPerBlock - 1) / warpsPerBlock;
    
    CHECK_HIP(hipEventRecord(start));
    hipLaunchKernelGGL(spmv_csr_vector, dim3(numBlocksVector), dim3(BLOCK_SIZE), 0, 0,
                       d_values, d_col_indices, d_row_ptr, d_x, d_y, rows);
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));
    
    float vector_time;
    CHECK_HIP(hipEventElapsedTime(&vector_time, start, stop));
    printf("Vector kernel: %.3f ms\n", vector_time);
    
    CHECK_HIP(hipMemcpy(h_y, d_y, rows * sizeof(float), hipMemcpyDeviceToHost));
    printf("Vector verification: %s\n", verify_spmv(h_y, h_ref, rows) ? "PASSED" : "FAILED");
    
    // Performance metrics
    double bytes = (mat->nnz * (sizeof(float) + sizeof(int)) + 
                   (rows + 1) * sizeof(int) + 
                   (cols + rows) * sizeof(float));
    double gb_per_sec = bytes / (scalar_time / 1000.0) / 1e9;
    printf("\nPerformance (scalar): %.2f GB/s\n", gb_per_sec);
    
    double gflops = 2.0 * mat->nnz / (scalar_time / 1000.0) / 1e9;
    printf("Performance: %.2f GFLOPS\n", gflops);
    
    printf("Speedup (vector vs scalar): %.2fx\n", scalar_time / vector_time);
    
    // Cleanup
    CHECK_HIP(hipFree(d_values));
    CHECK_HIP(hipFree(d_col_indices));
    CHECK_HIP(hipFree(d_row_ptr));
    CHECK_HIP(hipFree(d_x));
    CHECK_HIP(hipFree(d_y));
    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));
    
    free(h_x);
    free(h_y);
    free(h_ref);
    free_csr(mat);
    
    return 0;
}
