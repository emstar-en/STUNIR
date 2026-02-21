/**
 * STUNIR CUDA Example: Matrix Multiplication
 * 
 * Tiled matrix multiplication using shared memory.
 * Computes: C = A * B
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_SIZE 16

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// Tiled matrix multiplication kernel
__global__ void matmul_tiled(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile from A
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile from B
        if (col < N && t * TILE_SIZE + threadIdx.y < K) {
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Initialize matrix with random values
void init_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

// CPU matrix multiplication for verification
void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Verify GPU result against CPU
bool verify_results(const float* gpu_result, const float* cpu_result, int size) {
    for (int i = 0; i < size; i++) {
        if (fabsf(gpu_result[i] - cpu_result[i]) > 1e-3f) {
            printf("Verification failed at index %d: GPU=%f, CPU=%f\n",
                   i, gpu_result[i], cpu_result[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    // Matrix dimensions: C(M x N) = A(M x K) * B(K x N)
    int M = 1024, N = 1024, K = 1024;
    
    printf("=== STUNIR CUDA Example: Matrix Multiplication ===\n");
    printf("Matrix dimensions: C(%d x %d) = A(%d x %d) * B(%d x %d)\n",
           M, N, M, K, K, N);
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    // Allocate host memory
    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C_gpu = (float*)malloc(size_C);
    float* h_C_cpu = (float*)malloc(size_C);
    
    // Initialize matrices
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    
    // Launch kernel
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE,
                       (M + TILE_SIZE - 1) / TILE_SIZE);
    
    printf("Launching kernel: (%d, %d) blocks, (%d, %d) threads/block\n",
           blocksPerGrid.x, blocksPerGrid.y,
           threadsPerBlock.x, threadsPerBlock.y);
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start));
    matmul_tiled<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %.3f ms\n", milliseconds);
    
    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost));
    
    // Verify with CPU computation
    printf("Running CPU verification...\n");
    matmul_cpu(h_A, h_B, h_C_cpu, M, N, K);
    
    if (verify_results(h_C_gpu, h_C_cpu, M * N)) {
        printf("Verification: PASSED\n");
    } else {
        printf("Verification: FAILED\n");
    }
    
    // Calculate GFLOPS
    double gflops = (2.0 * M * N * K) / (milliseconds / 1000.0) / 1e9;
    printf("Performance: %.2f GFLOPS\n", gflops);
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);
    
    return 0;
}
