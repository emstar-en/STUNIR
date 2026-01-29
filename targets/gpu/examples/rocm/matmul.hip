/**
 * STUNIR ROCm Example: Matrix Multiplication
 * 
 * Optimized matrix multiplication using shared memory tiling.
 * Computes: C = A * B (where A is MxK, B is KxN, C is MxN)
 * 
 * This example demonstrates:
 * - Shared memory usage for data reuse
 * - Tiled computation pattern
 * - Memory coalescing optimization
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_SIZE 16

#define CHECK_HIP(call) { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, \
                hipGetErrorString(err)); \
        exit(1); \
    } \
}

// Naive matrix multiplication (for comparison)
__global__ void matmul_naive(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    int row = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int col = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Tiled matrix multiplication with shared memory
__global__ void matmul_tiled(const float* __restrict__ A, 
                             const float* __restrict__ B, 
                             float* __restrict__ C,
                             int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int tx = hipThreadIdx_x;
    int ty = hipThreadIdx_y;
    int row = hipBlockIdx_y * TILE_SIZE + ty;
    int col = hipBlockIdx_x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Collaborative loading of tiles into shared memory
        int a_col = t * TILE_SIZE + tx;
        int b_row = t * TILE_SIZE + ty;
        
        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (b_row < K && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void init_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)(rand() % 10) / 10.0f;
    }
}

bool verify_matmul(const float* A, const float* B, const float* C,
                   int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            if (fabsf(C[i * N + j] - sum) > 1e-3f) {
                printf("Mismatch at (%d,%d): expected %f, got %f\n",
                       i, j, sum, C[i * N + j]);
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char** argv) {
    int M = 1024, N = 1024, K = 1024;
    
    printf("=== STUNIR ROCm Example: Matrix Multiplication ===\n");
    printf("Matrix sizes: A(%dx%d) * B(%dx%d) = C(%dx%d)\n", M, K, K, N, M, N);
    printf("Tile size: %d\n", TILE_SIZE);
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    // Allocate host memory
    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C = (float*)malloc(size_C);
    
    // Initialize matrices
    srand(42);
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_HIP(hipMalloc(&d_A, size_A));
    CHECK_HIP(hipMalloc(&d_B, size_B));
    CHECK_HIP(hipMalloc(&d_C, size_C));
    
    // Copy data to device
    CHECK_HIP(hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_B, h_B, size_B, hipMemcpyHostToDevice));
    
    // Setup execution configuration
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE,
                       (M + TILE_SIZE - 1) / TILE_SIZE);
    
    printf("Grid: (%d, %d), Block: (%d, %d)\n",
           blocksPerGrid.x, blocksPerGrid.y, 
           threadsPerBlock.x, threadsPerBlock.y);
    
    // Timing
    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));
    
    // Run naive kernel
    CHECK_HIP(hipEventRecord(start));
    hipLaunchKernelGGL(matmul_naive, blocksPerGrid, threadsPerBlock, 
                       0, 0, d_A, d_B, d_C, M, N, K);
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));
    
    float naive_time;
    CHECK_HIP(hipEventElapsedTime(&naive_time, start, stop));
    printf("Naive kernel time: %.3f ms\n", naive_time);
    
    // Run tiled kernel
    CHECK_HIP(hipEventRecord(start));
    hipLaunchKernelGGL(matmul_tiled, blocksPerGrid, threadsPerBlock, 
                       0, 0, d_A, d_B, d_C, M, N, K);
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));
    
    float tiled_time;
    CHECK_HIP(hipEventElapsedTime(&tiled_time, start, stop));
    printf("Tiled kernel time: %.3f ms\n", tiled_time);
    printf("Speedup: %.2fx\n", naive_time / tiled_time);
    
    // Copy result back
    CHECK_HIP(hipMemcpy(h_C, d_C, size_C, hipMemcpyDeviceToHost));
    
    // Verify (on smaller matrix for speed)
    printf("Verifying results...\n");
    if (M <= 512 && N <= 512) {
        if (verify_matmul(h_A, h_B, h_C, M, N, K)) {
            printf("Verification: PASSED\n");
        } else {
            printf("Verification: FAILED\n");
        }
    } else {
        printf("Verification: SKIPPED (matrix too large)\n");
    }
    
    // Calculate GFLOPS
    double ops = 2.0 * M * N * K;
    double gflops = ops / (tiled_time / 1000.0) / 1e9;
    printf("Performance: %.2f GFLOPS\n", gflops);
    
    // Cleanup
    CHECK_HIP(hipFree(d_A));
    CHECK_HIP(hipFree(d_B));
    CHECK_HIP(hipFree(d_C));
    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
