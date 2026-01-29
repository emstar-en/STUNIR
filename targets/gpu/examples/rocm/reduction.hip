/**
 * STUNIR ROCm Example: Parallel Reduction
 * 
 * Parallel sum reduction demonstrating advanced GPU patterns:
 * - Warp-level primitives (__shfl_down)
 * - Shared memory reduction
 * - Multi-stage reduction for large arrays
 * - Atomic operations
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WARP_SIZE 64  // AMD wavefront size (32 for NVIDIA)
#define BLOCK_SIZE 256

#define CHECK_HIP(call) { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, \
                hipGetErrorString(err)); \
        exit(1); \
    } \
}

// Warp-level reduction using shuffle
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Block-level reduction using shared memory
__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];
    
    int lane = hipThreadIdx_x % WARP_SIZE;
    int wid = hipThreadIdx_x / WARP_SIZE;
    
    // Each warp performs partial reduction
    val = warp_reduce_sum(val);
    
    // Write reduced value of each warp to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    
    __syncthreads();
    
    // First warp reduces all partial sums
    val = (hipThreadIdx_x < hipBlockDim_x / WARP_SIZE) ? shared[lane] : 0.0f;
    
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }
    
    return val;
}

// Reduction kernel - reduces array to partial sums (one per block)
__global__ void reduce_sum(const float* __restrict__ input, 
                           float* __restrict__ output, int N) {
    float sum = 0.0f;
    
    // Grid-stride loop for handling large arrays
    for (int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
         i < N;
         i += hipBlockDim_x * hipGridDim_x) {
        sum += input[i];
    }
    
    // Block-level reduction
    sum = block_reduce_sum(sum);
    
    // First thread of each block writes its partial sum
    if (hipThreadIdx_x == 0) {
        output[hipBlockIdx_x] = sum;
    }
}

// Final reduction using atomics
__global__ void reduce_sum_atomic(const float* __restrict__ input,
                                   float* __restrict__ output, int N) {
    float sum = 0.0f;
    
    for (int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
         i < N;
         i += hipBlockDim_x * hipGridDim_x) {
        sum += input[i];
    }
    
    sum = block_reduce_sum(sum);
    
    if (hipThreadIdx_x == 0) {
        atomicAdd(output, sum);
    }
}

float cpu_reduce(const float* data, int N) {
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum += data[i];
    }
    return (float)sum;
}

int main(int argc, char** argv) {
    int N = 16 * 1024 * 1024;  // 16M elements
    size_t size = N * sizeof(float);
    
    printf("=== STUNIR ROCm Example: Parallel Reduction ===\n");
    printf("Array size: %d elements (%.2f MB)\n", N, size / (1024.0f * 1024.0f));
    printf("Block size: %d, Warp size: %d\n", BLOCK_SIZE, WARP_SIZE);
    
    // Allocate host memory
    float* h_input = (float*)malloc(size);
    
    // Initialize with values
    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 100) / 100.0f;
    }
    
    // CPU reference
    float cpu_sum = cpu_reduce(h_input, N);
    printf("CPU reference sum: %f\n", cpu_sum);
    
    // Allocate device memory
    float *d_input, *d_output, *d_result;
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    numBlocks = min(numBlocks, 1024);  // Limit blocks
    
    CHECK_HIP(hipMalloc(&d_input, size));
    CHECK_HIP(hipMalloc(&d_output, numBlocks * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_result, sizeof(float)));
    
    // Copy data to device
    CHECK_HIP(hipMemcpy(d_input, h_input, size, hipMemcpyHostToDevice));
    
    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));
    
    // ============ Method 1: Two-stage reduction ============
    printf("\n--- Two-stage reduction ---\n");
    
    CHECK_HIP(hipEventRecord(start));
    
    // Stage 1: Reduce to partial sums
    hipLaunchKernelGGL(reduce_sum, dim3(numBlocks), dim3(BLOCK_SIZE),
                       0, 0, d_input, d_output, N);
    
    // Stage 2: Reduce partial sums
    hipLaunchKernelGGL(reduce_sum, dim3(1), dim3(BLOCK_SIZE),
                       0, 0, d_output, d_result, numBlocks);
    
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));
    
    float two_stage_time;
    CHECK_HIP(hipEventElapsedTime(&two_stage_time, start, stop));
    
    float gpu_sum_two_stage;
    CHECK_HIP(hipMemcpy(&gpu_sum_two_stage, d_result, sizeof(float),
                        hipMemcpyDeviceToHost));
    
    printf("GPU sum (two-stage): %f\n", gpu_sum_two_stage);
    printf("Time: %.3f ms\n", two_stage_time);
    printf("Error: %.6f%%\n", fabsf(gpu_sum_two_stage - cpu_sum) / cpu_sum * 100);
    
    // ============ Method 2: Atomic reduction ============
    printf("\n--- Atomic reduction ---\n");
    
    // Reset result
    float zero = 0.0f;
    CHECK_HIP(hipMemcpy(d_result, &zero, sizeof(float), hipMemcpyHostToDevice));
    
    CHECK_HIP(hipEventRecord(start));
    hipLaunchKernelGGL(reduce_sum_atomic, dim3(numBlocks), dim3(BLOCK_SIZE),
                       0, 0, d_input, d_result, N);
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));
    
    float atomic_time;
    CHECK_HIP(hipEventElapsedTime(&atomic_time, start, stop));
    
    float gpu_sum_atomic;
    CHECK_HIP(hipMemcpy(&gpu_sum_atomic, d_result, sizeof(float),
                        hipMemcpyDeviceToHost));
    
    printf("GPU sum (atomic): %f\n", gpu_sum_atomic);
    printf("Time: %.3f ms\n", atomic_time);
    printf("Error: %.6f%%\n", fabsf(gpu_sum_atomic - cpu_sum) / cpu_sum * 100);
    
    // Performance metrics
    printf("\n--- Performance ---\n");
    float bandwidth = size / (two_stage_time / 1000.0f) / 1e9;
    printf("Effective bandwidth (two-stage): %.2f GB/s\n", bandwidth);
    printf("Speedup (two-stage vs atomic): %.2fx\n", atomic_time / two_stage_time);
    
    // Verification
    bool pass = fabsf(gpu_sum_two_stage - cpu_sum) / cpu_sum < 0.001f &&
                fabsf(gpu_sum_atomic - cpu_sum) / cpu_sum < 0.001f;
    printf("\nVerification: %s\n", pass ? "PASSED" : "FAILED");
    
    // Cleanup
    CHECK_HIP(hipFree(d_input));
    CHECK_HIP(hipFree(d_output));
    CHECK_HIP(hipFree(d_result));
    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));
    free(h_input);
    
    return pass ? 0 : 1;
}
