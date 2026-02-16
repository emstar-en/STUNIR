/**
 * STUNIR CUDA Example: Vector Addition
 * 
 * Simple parallel vector addition demonstrating basic CUDA kernel structure.
 * Computes: C[i] = A[i] + B[i]
 * 
 * This is the CUDA version of the HIP vector_add example.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// Vector addition kernel
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Host function to initialize vectors
void init_vectors(float* A, float* B, int N) {
    for (int i = 0; i < N; i++) {
        A[i] = (float)i;
        B[i] = (float)(N - i);
    }
}

// Host function to verify results
bool verify_results(const float* A, const float* B, const float* C, int N) {
    for (int i = 0; i < N; i++) {
        float expected = A[i] + B[i];
        if (fabsf(C[i] - expected) > 1e-5f) {
            printf("Verification failed at index %d: expected %f, got %f\n",
                   i, expected, C[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    int N = 1024 * 1024;  // 1M elements
    size_t size = N * sizeof(float);
    
    printf("=== STUNIR CUDA Example: Vector Addition ===\n");
    printf("Vector size: %d elements (%.2f MB)\n", N, size / (1024.0f * 1024.0f));
    
    // Allocate host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);
    
    // Initialize vectors
    init_vectors(h_A, h_B, N);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size));
    CHECK_CUDA(cudaMalloc(&d_B, size));
    CHECK_CUDA(cudaMalloc(&d_C, size));
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Launching kernel: %d blocks, %d threads/block\n", 
           blocksPerGrid, threadsPerBlock);
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start));
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %.3f ms\n", milliseconds);
    
    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    
    // Verify results
    if (verify_results(h_A, h_B, h_C, N)) {
        printf("Verification: PASSED\n");
    } else {
        printf("Verification: FAILED\n");
    }
    
    // Calculate bandwidth
    float bandwidth = (3 * size) / (milliseconds / 1000.0f) / 1e9;
    printf("Effective bandwidth: %.2f GB/s\n", bandwidth);
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
