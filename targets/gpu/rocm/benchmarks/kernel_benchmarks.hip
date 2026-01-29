/**
 * STUNIR ROCm Kernel Benchmark Suite
 *
 * Benchmarks for all kernel patterns: matmul, conv2d, fft, sparse_matvec, transpose.
 *
 * Schema: stunir.gpu.rocm.benchmark.kernels.v1
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "benchmark_harness.hip"

using namespace stunir::rocm::benchmark;

// ============ Matrix Multiplication Benchmark ============

#define TILE_SIZE 16

__global__ void benchmark_matmul_naive(const float* A, const float* B, float* C,
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

__global__ void benchmark_matmul_tiled(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int tx = hipThreadIdx_x, ty = hipThreadIdx_y;
    int row = hipBlockIdx_y * TILE_SIZE + ty;
    int col = hipBlockIdx_x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < M && t * TILE_SIZE + tx < K)
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;
        
        if (t * TILE_SIZE + ty < K && col < N)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void run_matmul_benchmarks(BenchmarkRunner& runner) {
    std::vector<int> sizes = {512, 1024, 2048, 4096};
    
    for (int size : sizes) {
        int M = size, N = size, K = size;
        size_t bytes = M * K + K * N + M * N;
        bytes *= sizeof(float);
        double flops = 2.0 * M * N * K;
        
        float *d_A, *d_B, *d_C;
        hipMalloc(&d_A, M * K * sizeof(float));
        hipMalloc(&d_B, K * N * sizeof(float));
        hipMalloc(&d_C, M * N * sizeof(float));
        
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        
        // Naive
        runner.run("MatMul_Naive_" + std::to_string(size), [&]() {
            hipLaunchKernelGGL(benchmark_matmul_naive, grid, block, 0, 0,
                               d_A, d_B, d_C, M, N, K);
        }, flops, bytes);
        
        // Tiled
        runner.run("MatMul_Tiled_" + std::to_string(size), [&]() {
            hipLaunchKernelGGL(benchmark_matmul_tiled, grid, block, 0, 0,
                               d_A, d_B, d_C, M, N, K);
        }, flops, bytes);
        
        hipFree(d_A);
        hipFree(d_B);
        hipFree(d_C);
    }
}

// ============ Transpose Benchmark ============

#define TRANS_TILE 32
#define TRANS_ROWS 8

__global__ void benchmark_transpose_naive(const float* input, float* output,
                                           int rows, int cols) {
    int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    
    if (x < cols && y < rows) {
        output[x * rows + y] = input[y * cols + x];
    }
}

__global__ void benchmark_transpose_shared(const float* input, float* output,
                                            int rows, int cols) {
    __shared__ float tile[TRANS_TILE][TRANS_TILE + 1];
    
    int x = hipBlockIdx_x * TRANS_TILE + hipThreadIdx_x;
    int y = hipBlockIdx_y * TRANS_TILE + hipThreadIdx_y;
    
    for (int j = 0; j < TRANS_TILE; j += TRANS_ROWS) {
        if (x < cols && (y + j) < rows) {
            tile[hipThreadIdx_y + j][hipThreadIdx_x] = input[(y + j) * cols + x];
        }
    }
    
    __syncthreads();
    
    x = hipBlockIdx_y * TRANS_TILE + hipThreadIdx_x;
    y = hipBlockIdx_x * TRANS_TILE + hipThreadIdx_y;
    
    for (int j = 0; j < TRANS_TILE; j += TRANS_ROWS) {
        if (x < rows && (y + j) < cols) {
            output[(y + j) * rows + x] = tile[hipThreadIdx_x][hipThreadIdx_y + j];
        }
    }
}

void run_transpose_benchmarks(BenchmarkRunner& runner) {
    std::vector<int> sizes = {1024, 2048, 4096, 8192};
    
    for (int size : sizes) {
        size_t bytes = 2 * size * size * sizeof(float);  // Read + write
        
        float *d_input, *d_output;
        hipMalloc(&d_input, size * size * sizeof(float));
        hipMalloc(&d_output, size * size * sizeof(float));
        
        dim3 block(TRANS_TILE, TRANS_ROWS);
        dim3 grid((size + TRANS_TILE - 1) / TRANS_TILE,
                  (size + TRANS_TILE - 1) / TRANS_TILE);
        
        runner.run("Transpose_Naive_" + std::to_string(size), [&]() {
            hipLaunchKernelGGL(benchmark_transpose_naive, grid, block, 0, 0,
                               d_input, d_output, size, size);
        }, 0, bytes);
        
        runner.run("Transpose_Shared_" + std::to_string(size), [&]() {
            hipLaunchKernelGGL(benchmark_transpose_shared, grid, block, 0, 0,
                               d_input, d_output, size, size);
        }, 0, bytes);
        
        hipFree(d_input);
        hipFree(d_output);
    }
}

// ============ Reduction Benchmark ============

__global__ void benchmark_reduce_naive(const float* input, float* output, int n) {
    __shared__ float sdata[256];
    
    int tid = hipThreadIdx_x;
    int idx = hipBlockIdx_x * hipBlockDim_x + tid;
    
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    for (int s = hipBlockDim_x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

__global__ void benchmark_reduce_warp(const float* input, float* output, int n) {
    __shared__ float sdata[256];
    
    int tid = hipThreadIdx_x;
    int idx = hipBlockIdx_x * hipBlockDim_x * 2 + tid;
    
    float sum = 0.0f;
    if (idx < n) sum += input[idx];
    if (idx + hipBlockDim_x < n) sum += input[idx + hipBlockDim_x];
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduce within block
    for (int s = hipBlockDim_x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    // Warp-level reduction (no sync needed)
    if (tid < 32) {
        volatile float* vdata = sdata;
        vdata[tid] += vdata[tid + 32];
        vdata[tid] += vdata[tid + 16];
        vdata[tid] += vdata[tid + 8];
        vdata[tid] += vdata[tid + 4];
        vdata[tid] += vdata[tid + 2];
        vdata[tid] += vdata[tid + 1];
    }
    
    if (tid == 0) atomicAdd(output, sdata[0]);
}

void run_reduction_benchmarks(BenchmarkRunner& runner) {
    std::vector<int> sizes = {1024*1024, 16*1024*1024, 64*1024*1024};
    
    for (int size : sizes) {
        size_t bytes = size * sizeof(float);
        
        float *d_input, *d_output;
        hipMalloc(&d_input, bytes);
        hipMalloc(&d_output, sizeof(float));
        
        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        
        std::string label = std::to_string(size / (1024 * 1024)) + "M";
        
        runner.run("Reduce_Naive_" + label, [&]() {
            hipMemset(d_output, 0, sizeof(float));
            hipLaunchKernelGGL(benchmark_reduce_naive, dim3(gridSize), dim3(blockSize), 0, 0,
                               d_input, d_output, size);
        }, size, bytes);
        
        gridSize = (size + blockSize * 2 - 1) / (blockSize * 2);
        runner.run("Reduce_Warp_" + label, [&]() {
            hipMemset(d_output, 0, sizeof(float));
            hipLaunchKernelGGL(benchmark_reduce_warp, dim3(gridSize), dim3(blockSize), 0, 0,
                               d_input, d_output, size);
        }, size, bytes);
        
        hipFree(d_input);
        hipFree(d_output);
    }
}

// ============ Memory Copy Benchmark ============

void run_memory_benchmarks(BenchmarkRunner& runner) {
    std::vector<size_t> sizes = {1*1024*1024, 16*1024*1024, 256*1024*1024};
    
    for (size_t size : sizes) {
        void *h_data, *d_data1, *d_data2;
        hipHostMalloc(&h_data, size);
        hipMalloc(&d_data1, size);
        hipMalloc(&d_data2, size);
        
        std::string label = std::to_string(size / (1024 * 1024)) + "MB";
        
        runner.run("MemCopy_H2D_" + label, [&]() {
            hipMemcpy(d_data1, h_data, size, hipMemcpyHostToDevice);
        }, 0, size);
        
        runner.run("MemCopy_D2H_" + label, [&]() {
            hipMemcpy(h_data, d_data1, size, hipMemcpyDeviceToHost);
        }, 0, size);
        
        runner.run("MemCopy_D2D_" + label, [&]() {
            hipMemcpy(d_data2, d_data1, size, hipMemcpyDeviceToDevice);
        }, 0, size);
        
        hipHostFree(h_data);
        hipFree(d_data1);
        hipFree(d_data2);
    }
}

// ============ Main ============

int main(int argc, char** argv) {
    printf("=== STUNIR ROCm Kernel Benchmark Suite ===\n\n");
    
    BenchmarkRunner runner(5, 50, true);
    
    printf("\n--- Memory Benchmarks ---\n");
    run_memory_benchmarks(runner);
    
    printf("\n--- Matrix Multiplication Benchmarks ---\n");
    run_matmul_benchmarks(runner);
    
    printf("\n--- Transpose Benchmarks ---\n");
    run_transpose_benchmarks(runner);
    
    printf("\n--- Reduction Benchmarks ---\n");
    run_reduction_benchmarks(runner);
    
    // Print summary and export
    runner.print_summary();
    runner.export_json("benchmark_results.json");
    
    return 0;
}
