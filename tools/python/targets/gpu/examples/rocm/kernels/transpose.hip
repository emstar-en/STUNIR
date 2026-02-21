/**
 * STUNIR ROCm Kernel: Optimized Matrix Transpose
 *
 * Multiple transpose implementations with shared memory optimization.
 * Avoids bank conflicts using padding.
 *
 * Schema: stunir.gpu.rocm.kernel.transpose.v1
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8
#define PADDING 1  // Avoid bank conflicts

#define CHECK_HIP(call) { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, \
                hipGetErrorString(err)); \
        exit(1); \
    } \
}

/**
 * Naive transpose - simple but slow (non-coalesced writes)
 */
__global__ void transpose_naive(const float* __restrict__ input,
                                float* __restrict__ output,
                                int rows, int cols) {
    int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    
    if (x < cols && y < rows) {
        output[x * rows + y] = input[y * cols + x];
    }
}

/**
 * Coalesced transpose using shared memory
 * Reads and writes are coalesced, but may have bank conflicts
 */
__global__ void transpose_coalesced(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int rows, int cols) {
    __shared__ float tile[TILE_DIM][TILE_DIM];
    
    int x = hipBlockIdx_x * TILE_DIM + hipThreadIdx_x;
    int y = hipBlockIdx_y * TILE_DIM + hipThreadIdx_y;
    
    // Coalesced read from global memory
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < cols && (y + j) < rows) {
            tile[hipThreadIdx_y + j][hipThreadIdx_x] = input[(y + j) * cols + x];
        }
    }
    
    __syncthreads();
    
    // Transpose block indices for coalesced write
    x = hipBlockIdx_y * TILE_DIM + hipThreadIdx_x;
    y = hipBlockIdx_x * TILE_DIM + hipThreadIdx_y;
    
    // Coalesced write to global memory
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < rows && (y + j) < cols) {
            output[(y + j) * rows + x] = tile[hipThreadIdx_x][hipThreadIdx_y + j];
        }
    }
}

/**
 * Optimized transpose - no bank conflicts using padding
 * This is the fastest version
 */
__global__ void transpose_optimized(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int rows, int cols) {
    // Padded shared memory to avoid bank conflicts
    __shared__ float tile[TILE_DIM][TILE_DIM + PADDING];
    
    int x = hipBlockIdx_x * TILE_DIM + hipThreadIdx_x;
    int y = hipBlockIdx_y * TILE_DIM + hipThreadIdx_y;
    
    // Coalesced read from global memory
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < cols && (y + j) < rows) {
            tile[hipThreadIdx_y + j][hipThreadIdx_x] = input[(y + j) * cols + x];
        }
    }
    
    __syncthreads();
    
    // Transpose block indices
    x = hipBlockIdx_y * TILE_DIM + hipThreadIdx_x;
    y = hipBlockIdx_x * TILE_DIM + hipThreadIdx_y;
    
    // Coalesced write with transposed access pattern
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < rows && (y + j) < cols) {
            output[(y + j) * rows + x] = tile[hipThreadIdx_x][hipThreadIdx_y + j];
        }
    }
}

/**
 * In-place square matrix transpose
 * Only works for square matrices
 */
__global__ void transpose_inplace(float* data, int n) {
    __shared__ float tile1[TILE_DIM][TILE_DIM + PADDING];
    __shared__ float tile2[TILE_DIM][TILE_DIM + PADDING];
    
    int bx = hipBlockIdx_x;
    int by = hipBlockIdx_y;
    int tx = hipThreadIdx_x;
    int ty = hipThreadIdx_y;
    
    // Only process upper triangle blocks (including diagonal)
    if (bx > by) return;
    
    int x1 = bx * TILE_DIM + tx;
    int y1 = by * TILE_DIM + ty;
    int x2 = by * TILE_DIM + tx;
    int y2 = bx * TILE_DIM + ty;
    
    // Load tiles
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x1 < n && (y1 + j) < n) {
            tile1[ty + j][tx] = data[(y1 + j) * n + x1];
        }
        if (bx != by && x2 < n && (y2 + j) < n) {
            tile2[ty + j][tx] = data[(y2 + j) * n + x2];
        }
    }
    
    __syncthreads();
    
    // Write transposed tiles
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        // Write tile1 to tile2's position (transposed)
        if (x2 < n && (y2 + j) < n) {
            data[(y2 + j) * n + x2] = tile1[tx][ty + j];
        }
        // Write tile2 to tile1's position (transposed)
        if (bx != by && x1 < n && (y1 + j) < n) {
            data[(y1 + j) * n + x1] = tile2[tx][ty + j];
        }
    }
}

/**
 * Batched transpose - process multiple matrices
 */
__global__ void transpose_batched(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int rows, int cols, int batch_size) {
    __shared__ float tile[TILE_DIM][TILE_DIM + PADDING];
    
    int batch = hipBlockIdx_z;
    int x = hipBlockIdx_x * TILE_DIM + hipThreadIdx_x;
    int y = hipBlockIdx_y * TILE_DIM + hipThreadIdx_y;
    
    if (batch >= batch_size) return;
    
    const float* in = input + batch * rows * cols;
    float* out = output + batch * cols * rows;
    
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < cols && (y + j) < rows) {
            tile[hipThreadIdx_y + j][hipThreadIdx_x] = in[(y + j) * cols + x];
        }
    }
    
    __syncthreads();
    
    x = hipBlockIdx_y * TILE_DIM + hipThreadIdx_x;
    y = hipBlockIdx_x * TILE_DIM + hipThreadIdx_y;
    
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < rows && (y + j) < cols) {
            out[(y + j) * rows + x] = tile[hipThreadIdx_x][hipThreadIdx_y + j];
        }
    }
}

// Host verification
bool verify_transpose(const float* original, const float* transposed,
                     int rows, int cols) {
    int errors = 0;
    for (int i = 0; i < rows && errors < 5; i++) {
        for (int j = 0; j < cols && errors < 5; j++) {
            if (fabsf(original[i * cols + j] - transposed[j * rows + i]) > 1e-6f) {
                printf("Mismatch at (%d,%d): %.6f vs %.6f\n",
                       i, j, original[i * cols + j], transposed[j * rows + i]);
                errors++;
            }
        }
    }
    return errors == 0;
}

int main(int argc, char** argv) {
    int rows = 4096;
    int cols = 4096;
    
    printf("=== STUNIR ROCm: Matrix Transpose ===\n");
    printf("Matrix: %d x %d\n", rows, cols);
    printf("Tile size: %d, Block rows: %d\n", TILE_DIM, BLOCK_ROWS);
    
    size_t size = rows * cols * sizeof(float);
    
    // Allocate host memory
    float* h_input = (float*)malloc(size);
    float* h_output = (float*)malloc(size);
    
    // Initialize
    srand(42);
    for (int i = 0; i < rows * cols; i++) {
        h_input[i] = (float)(rand() % 100) / 10.0f;
    }
    
    // Device memory
    float *d_input, *d_output;
    CHECK_HIP(hipMalloc(&d_input, size));
    CHECK_HIP(hipMalloc(&d_output, size));
    CHECK_HIP(hipMemcpy(d_input, h_input, size, hipMemcpyHostToDevice));
    
    // Grid configuration
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((cols + TILE_DIM - 1) / TILE_DIM,
              (rows + TILE_DIM - 1) / TILE_DIM);
    
    // Timing
    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));
    
    // Warm up
    hipLaunchKernelGGL(transpose_optimized, grid, block, 0, 0,
                       d_input, d_output, rows, cols);
    CHECK_HIP(hipDeviceSynchronize());
    
    // Benchmark naive
    CHECK_HIP(hipEventRecord(start));
    for (int i = 0; i < 10; i++) {
        hipLaunchKernelGGL(transpose_naive, grid, block, 0, 0,
                           d_input, d_output, rows, cols);
    }
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));
    
    float naive_time;
    CHECK_HIP(hipEventElapsedTime(&naive_time, start, stop));
    naive_time /= 10;
    printf("Naive: %.3f ms\n", naive_time);
    
    // Benchmark coalesced
    CHECK_HIP(hipEventRecord(start));
    for (int i = 0; i < 10; i++) {
        hipLaunchKernelGGL(transpose_coalesced, grid, block, 0, 0,
                           d_input, d_output, rows, cols);
    }
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));
    
    float coalesced_time;
    CHECK_HIP(hipEventElapsedTime(&coalesced_time, start, stop));
    coalesced_time /= 10;
    printf("Coalesced: %.3f ms\n", coalesced_time);
    
    // Benchmark optimized
    CHECK_HIP(hipEventRecord(start));
    for (int i = 0; i < 10; i++) {
        hipLaunchKernelGGL(transpose_optimized, grid, block, 0, 0,
                           d_input, d_output, rows, cols);
    }
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));
    
    float optimized_time;
    CHECK_HIP(hipEventElapsedTime(&optimized_time, start, stop));
    optimized_time /= 10;
    printf("Optimized (no bank conflicts): %.3f ms\n", optimized_time);
    
    // Copy result and verify
    CHECK_HIP(hipMemcpy(h_output, d_output, size, hipMemcpyDeviceToHost));
    printf("Verification: %s\n", verify_transpose(h_input, h_output, rows, cols) ? "PASSED" : "FAILED");
    
    // Performance metrics
    double bytes = 2.0 * size;  // Read + write
    double bandwidth_naive = bytes / (naive_time / 1000.0) / 1e9;
    double bandwidth_coalesced = bytes / (coalesced_time / 1000.0) / 1e9;
    double bandwidth_optimized = bytes / (optimized_time / 1000.0) / 1e9;
    
    printf("\nEffective Bandwidth:\n");
    printf("  Naive: %.2f GB/s\n", bandwidth_naive);
    printf("  Coalesced: %.2f GB/s\n", bandwidth_coalesced);
    printf("  Optimized: %.2f GB/s\n", bandwidth_optimized);
    printf("\nSpeedup:\n");
    printf("  Coalesced vs Naive: %.2fx\n", naive_time / coalesced_time);
    printf("  Optimized vs Naive: %.2fx\n", naive_time / optimized_time);
    printf("  Optimized vs Coalesced: %.2fx\n", coalesced_time / optimized_time);
    
    // Cleanup
    CHECK_HIP(hipFree(d_input));
    CHECK_HIP(hipFree(d_output));
    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));
    free(h_input);
    free(h_output);
    
    return 0;
}
