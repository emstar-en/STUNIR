/**
 * STUNIR ROCm Multi-GPU Example
 *
 * Demonstrates data-parallel vector addition across multiple GPUs.
 *
 * Schema: stunir.gpu.rocm.multi_gpu.example.v1
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <chrono>
#include "multi_gpu_utils.hip"

using namespace stunir::rocm::multigpu;

// Simple vector add kernel
__global__ void vector_add(const float* a, const float* b, float* c, size_t n) {
    size_t idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void single_gpu_vector_add(const float* h_a, const float* h_b, float* h_c, size_t n) {
    float *d_a, *d_b, *d_c;
    
    hipSetDevice(0);
    hipMalloc(&d_a, n * sizeof(float));
    hipMalloc(&d_b, n * sizeof(float));
    hipMalloc(&d_c, n * sizeof(float));
    
    hipMemcpy(d_a, h_a, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, n * sizeof(float), hipMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    hipLaunchKernelGGL(vector_add, dim3(gridSize), dim3(blockSize), 0, 0,
                       d_a, d_b, d_c, n);
    
    hipMemcpy(h_c, d_c, n * sizeof(float), hipMemcpyDeviceToHost);
    
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
}

void multi_gpu_vector_add(DeviceManager& mgr, const float* h_a, const float* h_b, 
                          float* h_c, size_t n) {
    int num_gpus = mgr.device_count();
    
    // Create executor
    MultiGpuExecutor executor(mgr);
    
    // Create distributor
    DataDistributor<float> dist(num_gpus, DistributionStrategy::BLOCK);
    auto sizes = dist.partition(n);
    
    // Allocate per-GPU buffers
    MultiGpuBuffer<float> d_a, d_b, d_c;
    d_a.allocate_varying(sizes);
    d_b.allocate_varying(sizes);
    d_c.allocate_varying(sizes);
    
    // Scatter data to GPUs
    size_t offset = 0;
    for (int i = 0; i < num_gpus; i++) {
        hipSetDevice(i);
        hipMemcpyAsync(d_a[i], h_a + offset, sizes[i] * sizeof(float),
                       hipMemcpyHostToDevice, executor.stream(i));
        hipMemcpyAsync(d_b[i], h_b + offset, sizes[i] * sizeof(float),
                       hipMemcpyHostToDevice, executor.stream(i));
        offset += sizes[i];
    }
    
    // Launch kernels on all GPUs
    executor.parallel_for_devices([&](int device, hipStream_t stream) {
        int blockSize = 256;
        int gridSize = (sizes[device] + blockSize - 1) / blockSize;
        
        hipLaunchKernelGGL(vector_add, dim3(gridSize), dim3(blockSize), 0, stream,
                           d_a[device], d_b[device], d_c[device], sizes[device]);
    });
    
    // Gather results
    offset = 0;
    for (int i = 0; i < num_gpus; i++) {
        hipSetDevice(i);
        hipMemcpyAsync(h_c + offset, d_c[i], sizes[i] * sizeof(float),
                       hipMemcpyDeviceToHost, executor.stream(i));
        offset += sizes[i];
    }
    
    executor.synchronize_all();
}

bool verify(const float* a, const float* b, const float* c, size_t n) {
    int errors = 0;
    for (size_t i = 0; i < n && errors < 5; i++) {
        float expected = a[i] + b[i];
        if (fabsf(c[i] - expected) > 1e-5f) {
            printf("Mismatch at %zu: %.6f + %.6f = %.6f (got %.6f)\n",
                   i, a[i], b[i], expected, c[i]);
            errors++;
        }
    }
    return errors == 0;
}

int main(int argc, char** argv) {
    printf("=== STUNIR Multi-GPU Example: Vector Addition ===\n\n");
    
    // Initialize device manager
    DeviceManager mgr;
    mgr.print_info();
    printf("\n");
    
    if (mgr.device_count() < 2) {
        printf("Note: Only %d GPU available. Multi-GPU demonstration limited.\n",
               mgr.device_count());
    }
    
    // Enable peer access if available
    mgr.enable_all_peer_access();
    
    // Test parameters
    size_t n = 100 * 1024 * 1024;  // 100M elements
    printf("Vector size: %zu elements (%.2f GB total data)\n", 
           n, 3.0 * n * sizeof(float) / (1024.0 * 1024.0 * 1024.0));
    
    // Allocate host memory (pinned for better transfer speed)
    float *h_a, *h_b, *h_c_single, *h_c_multi;
    hipHostMalloc(&h_a, n * sizeof(float));
    hipHostMalloc(&h_b, n * sizeof(float));
    hipHostMalloc(&h_c_single, n * sizeof(float));
    hipHostMalloc(&h_c_multi, n * sizeof(float));
    
    // Initialize data
    printf("Initializing data...\n");
    srand(42);
    for (size_t i = 0; i < n; i++) {
        h_a[i] = (float)(rand() % 100) / 100.0f;
        h_b[i] = (float)(rand() % 100) / 100.0f;
    }
    
    // Warm up
    printf("Warming up...\n");
    single_gpu_vector_add(h_a, h_b, h_c_single, n / 10);
    if (mgr.device_count() > 0) {
        multi_gpu_vector_add(mgr, h_a, h_b, h_c_multi, n / 10);
    }
    
    // Benchmark single GPU
    printf("\nRunning single-GPU benchmark...\n");
    auto start = std::chrono::high_resolution_clock::now();
    single_gpu_vector_add(h_a, h_b, h_c_single, n);
    auto end = std::chrono::high_resolution_clock::now();
    double single_ms = std::chrono::duration<double, std::milli>(end - start).count();
    printf("Single GPU time: %.2f ms\n", single_ms);
    
    // Benchmark multi-GPU
    printf("\nRunning multi-GPU benchmark...\n");
    start = std::chrono::high_resolution_clock::now();
    multi_gpu_vector_add(mgr, h_a, h_b, h_c_multi, n);
    end = std::chrono::high_resolution_clock::now();
    double multi_ms = std::chrono::duration<double, std::milli>(end - start).count();
    printf("Multi GPU time: %.2f ms (using %d GPUs)\n", multi_ms, mgr.device_count());
    
    // Performance metrics
    double bytes = 3.0 * n * sizeof(float);  // Read A, B, write C
    printf("\nPerformance:\n");
    printf("  Single GPU: %.2f GB/s\n", bytes / (single_ms / 1000.0) / 1e9);
    printf("  Multi GPU:  %.2f GB/s\n", bytes / (multi_ms / 1000.0) / 1e9);
    if (mgr.device_count() > 1) {
        printf("  Speedup: %.2fx\n", single_ms / multi_ms);
    }
    
    // Verify results
    printf("\nVerifying single-GPU results...\n");
    printf("Single GPU: %s\n", verify(h_a, h_b, h_c_single, n) ? "PASSED" : "FAILED");
    
    printf("Verifying multi-GPU results...\n");
    printf("Multi GPU: %s\n", verify(h_a, h_b, h_c_multi, n) ? "PASSED" : "FAILED");
    
    // Cleanup
    hipHostFree(h_a);
    hipHostFree(h_b);
    hipHostFree(h_c_single);
    hipHostFree(h_c_multi);
    
    printf("\nDone!\n");
    return 0;
}
