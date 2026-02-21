/**
 * STUNIR ROCm Example: Vector Addition
 * 
 * Simple parallel vector addition demonstrating basic HIP kernel structure.
 * Computes: C[i] = A[i] + B[i]
 * 
 * This example is portable and compiles for both AMD and NVIDIA GPUs.
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_HIP(call) { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, \
                hipGetErrorString(err)); \
        exit(1); \
    } \
}

// Vector addition kernel
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    
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
    
    printf("=== STUNIR ROCm Example: Vector Addition ===\n");
    printf("Vector size: %d elements (%.2f MB)\n", N, size / (1024.0f * 1024.0f));
    
    // Allocate host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);
    
    // Initialize vectors
    init_vectors(h_A, h_B, N);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_HIP(hipMalloc(&d_A, size));
    CHECK_HIP(hipMalloc(&d_B, size));
    CHECK_HIP(hipMalloc(&d_C, size));
    
    // Copy data to device
    CHECK_HIP(hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice));
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Launching kernel: %d blocks, %d threads/block\n", 
           blocksPerGrid, threadsPerBlock);
    
    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));
    
    CHECK_HIP(hipEventRecord(start));
    hipLaunchKernelGGL(vector_add, dim3(blocksPerGrid), dim3(threadsPerBlock), 
                       0, 0, d_A, d_B, d_C, N);
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK_HIP(hipEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %.3f ms\n", milliseconds);
    
    // Copy result back to host
    CHECK_HIP(hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost));
    
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
