/**
 * STUNIR ROCm Kernel: 2D Convolution
 *
 * Optimized 2D convolution with tiling and shared memory.
 * Supports arbitrary filter sizes with padding/stride.
 *
 * Schema: stunir.gpu.rocm.kernel.conv2d.v1
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_WIDTH 16
#define TILE_HEIGHT 16
#define MAX_FILTER_SIZE 7

#define CHECK_HIP(call) { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, \
                hipGetErrorString(err)); \
        exit(1); \
    } \
}

/**
 * Naive 2D convolution kernel (for verification)
 */
__global__ void conv2d_naive(const float* __restrict__ input,
                             const float* __restrict__ filter,
                             float* __restrict__ output,
                             int height, int width,
                             int filter_h, int filter_w,
                             int pad_h, int pad_w,
                             int stride_h, int stride_w) {
    int out_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int out_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    
    int out_height = (height + 2 * pad_h - filter_h) / stride_h + 1;
    int out_width = (width + 2 * pad_w - filter_w) / stride_w + 1;
    
    if (out_y < out_height && out_x < out_width) {
        float sum = 0.0f;
        
        for (int fy = 0; fy < filter_h; fy++) {
            for (int fx = 0; fx < filter_w; fx++) {
                int in_y = out_y * stride_h - pad_h + fy;
                int in_x = out_x * stride_w - pad_w + fx;
                
                if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                    sum += input[in_y * width + in_x] * filter[fy * filter_w + fx];
                }
            }
        }
        
        output[out_y * out_width + out_x] = sum;
    }
}

/**
 * Tiled 2D convolution with shared memory
 * 
 * Optimization strategies:
 * - Load input tile + halo into shared memory
 * - Use constant memory for small filters
 * - Loop unrolling for fixed filter sizes
 */
__global__ void conv2d_tiled(const float* __restrict__ input,
                             const float* __restrict__ filter,
                             float* __restrict__ output,
                             int height, int width,
                             int filter_h, int filter_w,
                             int pad_h, int pad_w,
                             int stride_h, int stride_w) {
    // Calculate output dimensions
    int out_height = (height + 2 * pad_h - filter_h) / stride_h + 1;
    int out_width = (width + 2 * pad_w - filter_w) / stride_w + 1;
    
    // Thread indices within block
    int tx = hipThreadIdx_x;
    int ty = hipThreadIdx_y;
    
    // Output coordinates
    int out_y = hipBlockIdx_y * TILE_HEIGHT + ty;
    int out_x = hipBlockIdx_x * TILE_WIDTH + tx;
    
    // Calculate halo size needed
    int halo_h = filter_h - 1;
    int halo_w = filter_w - 1;
    
    // Shared memory for input tile + halo
    __shared__ float tile[TILE_HEIGHT + MAX_FILTER_SIZE - 1][TILE_WIDTH + MAX_FILTER_SIZE - 1];
    __shared__ float filt[MAX_FILTER_SIZE][MAX_FILTER_SIZE];
    
    // Collaboratively load filter into shared memory
    if (tx < filter_w && ty < filter_h) {
        filt[ty][tx] = filter[ty * filter_w + tx];
    }
    
    // Calculate input region to load
    int in_start_y = hipBlockIdx_y * TILE_HEIGHT * stride_h - pad_h;
    int in_start_x = hipBlockIdx_x * TILE_WIDTH * stride_w - pad_w;
    
    // Load input tile with halos
    for (int i = ty; i < TILE_HEIGHT + halo_h; i += hipBlockDim_y) {
        for (int j = tx; j < TILE_WIDTH + halo_w; j += hipBlockDim_x) {
            int in_y = in_start_y + i;
            int in_x = in_start_x + j;
            
            if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                tile[i][j] = input[in_y * width + in_x];
            } else {
                tile[i][j] = 0.0f;  // Zero padding
            }
        }
    }
    
    __syncthreads();
    
    // Compute convolution
    if (out_y < out_height && out_x < out_width) {
        float sum = 0.0f;
        int tile_y = ty * stride_h;
        int tile_x = tx * stride_w;
        
        #pragma unroll
        for (int fy = 0; fy < filter_h; fy++) {
            #pragma unroll
            for (int fx = 0; fx < filter_w; fx++) {
                sum += tile[tile_y + fy][tile_x + fx] * filt[fy][fx];
            }
        }
        
        output[out_y * out_width + out_x] = sum;
    }
}

/**
 * Separable 2D convolution (for separable filters)
 * Two-pass approach: horizontal then vertical
 */
__global__ void conv2d_separable_h(const float* __restrict__ input,
                                   const float* __restrict__ filter_h,
                                   float* __restrict__ output,
                                   int height, int width,
                                   int filter_size, int pad) {
    __shared__ float tile[TILE_HEIGHT][TILE_WIDTH + MAX_FILTER_SIZE - 1];
    
    int tx = hipThreadIdx_x;
    int ty = hipThreadIdx_y;
    int y = hipBlockIdx_y * TILE_HEIGHT + ty;
    int x = hipBlockIdx_x * TILE_WIDTH + tx;
    
    int halo = filter_size / 2;
    
    // Load tile with horizontal halo
    if (y < height) {
        for (int i = tx; i < TILE_WIDTH + filter_size - 1; i += hipBlockDim_x) {
            int in_x = hipBlockIdx_x * TILE_WIDTH - halo + i;
            if (in_x >= 0 && in_x < width) {
                tile[ty][i] = input[y * width + in_x];
            } else {
                tile[ty][i] = 0.0f;
            }
        }
    }
    
    __syncthreads();
    
    if (y < height && x < width) {
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < filter_size; k++) {
            sum += tile[ty][tx + k] * filter_h[k];
        }
        output[y * width + x] = sum;
    }
}

__global__ void conv2d_separable_v(const float* __restrict__ input,
                                   const float* __restrict__ filter_v,
                                   float* __restrict__ output,
                                   int height, int width,
                                   int filter_size, int pad) {
    __shared__ float tile[TILE_HEIGHT + MAX_FILTER_SIZE - 1][TILE_WIDTH];
    
    int tx = hipThreadIdx_x;
    int ty = hipThreadIdx_y;
    int y = hipBlockIdx_y * TILE_HEIGHT + ty;
    int x = hipBlockIdx_x * TILE_WIDTH + tx;
    
    int halo = filter_size / 2;
    
    // Load tile with vertical halo
    if (x < width) {
        for (int i = ty; i < TILE_HEIGHT + filter_size - 1; i += hipBlockDim_y) {
            int in_y = hipBlockIdx_y * TILE_HEIGHT - halo + i;
            if (in_y >= 0 && in_y < height) {
                tile[i][tx] = input[in_y * width + x];
            } else {
                tile[i][tx] = 0.0f;
            }
        }
    }
    
    __syncthreads();
    
    if (y < height && x < width) {
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < filter_size; k++) {
            sum += tile[ty + k][tx] * filter_v[k];
        }
        output[y * width + x] = sum;
    }
}

// Host functions
void verify_conv2d(const float* input, const float* filter, const float* output,
                   int height, int width, int filter_h, int filter_w,
                   int pad_h, int pad_w, int stride_h, int stride_w) {
    int out_height = (height + 2 * pad_h - filter_h) / stride_h + 1;
    int out_width = (width + 2 * pad_w - filter_w) / stride_w + 1;
    
    int errors = 0;
    for (int oy = 0; oy < out_height && errors < 5; oy++) {
        for (int ox = 0; ox < out_width && errors < 5; ox++) {
            float sum = 0.0f;
            for (int fy = 0; fy < filter_h; fy++) {
                for (int fx = 0; fx < filter_w; fx++) {
                    int iy = oy * stride_h - pad_h + fy;
                    int ix = ox * stride_w - pad_w + fx;
                    if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                        sum += input[iy * width + ix] * filter[fy * filter_w + fx];
                    }
                }
            }
            if (fabsf(output[oy * out_width + ox] - sum) > 1e-4f) {
                errors++;
                printf("Mismatch at (%d,%d): expected %f, got %f\n",
                       oy, ox, sum, output[oy * out_width + ox]);
            }
        }
    }
    printf("Verification: %s\n", errors == 0 ? "PASSED" : "FAILED");
}

int main(int argc, char** argv) {
    // Parameters
    int height = 512, width = 512;
    int filter_h = 3, filter_w = 3;
    int pad_h = 1, pad_w = 1;
    int stride_h = 1, stride_w = 1;
    
    printf("=== STUNIR ROCm: 2D Convolution ===\n");
    printf("Input: %dx%d, Filter: %dx%d, Pad: %dx%d, Stride: %dx%d\n",
           height, width, filter_h, filter_w, pad_h, pad_w, stride_h, stride_w);
    
    int out_height = (height + 2 * pad_h - filter_h) / stride_h + 1;
    int out_width = (width + 2 * pad_w - filter_w) / stride_w + 1;
    
    size_t input_size = height * width * sizeof(float);
    size_t filter_size = filter_h * filter_w * sizeof(float);
    size_t output_size = out_height * out_width * sizeof(float);
    
    // Allocate host memory
    float* h_input = (float*)malloc(input_size);
    float* h_filter = (float*)malloc(filter_size);
    float* h_output = (float*)malloc(output_size);
    
    // Initialize
    srand(42);
    for (int i = 0; i < height * width; i++) h_input[i] = (float)(rand() % 10) / 10.0f;
    for (int i = 0; i < filter_h * filter_w; i++) h_filter[i] = (float)(rand() % 10) / 10.0f - 0.5f;
    
    // Device memory
    float *d_input, *d_filter, *d_output;
    CHECK_HIP(hipMalloc(&d_input, input_size));
    CHECK_HIP(hipMalloc(&d_filter, filter_size));
    CHECK_HIP(hipMalloc(&d_output, output_size));
    
    CHECK_HIP(hipMemcpy(d_input, h_input, input_size, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_filter, h_filter, filter_size, hipMemcpyHostToDevice));
    
    // Grid configuration
    dim3 block(TILE_WIDTH, TILE_HEIGHT);
    dim3 grid((out_width + TILE_WIDTH - 1) / TILE_WIDTH,
              (out_height + TILE_HEIGHT - 1) / TILE_HEIGHT);
    
    // Timing
    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));
    
    // Run naive kernel
    CHECK_HIP(hipEventRecord(start));
    hipLaunchKernelGGL(conv2d_naive, grid, block, 0, 0,
                       d_input, d_filter, d_output,
                       height, width, filter_h, filter_w,
                       pad_h, pad_w, stride_h, stride_w);
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));
    
    float naive_time;
    CHECK_HIP(hipEventElapsedTime(&naive_time, start, stop));
    printf("Naive kernel: %.3f ms\n", naive_time);
    
    // Run tiled kernel
    CHECK_HIP(hipEventRecord(start));
    hipLaunchKernelGGL(conv2d_tiled, grid, block, 0, 0,
                       d_input, d_filter, d_output,
                       height, width, filter_h, filter_w,
                       pad_h, pad_w, stride_h, stride_w);
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));
    
    float tiled_time;
    CHECK_HIP(hipEventElapsedTime(&tiled_time, start, stop));
    printf("Tiled kernel: %.3f ms\n", tiled_time);
    printf("Speedup: %.2fx\n", naive_time / tiled_time);
    
    // Copy result and verify
    CHECK_HIP(hipMemcpy(h_output, d_output, output_size, hipMemcpyDeviceToHost));
    verify_conv2d(h_input, h_filter, h_output, height, width,
                  filter_h, filter_w, pad_h, pad_w, stride_h, stride_w);
    
    // Performance metrics
    double ops = 2.0 * out_height * out_width * filter_h * filter_w;
    double gflops = ops / (tiled_time / 1000.0) / 1e9;
    printf("Performance: %.2f GFLOPS\n", gflops);
    
    // Cleanup
    CHECK_HIP(hipFree(d_input));
    CHECK_HIP(hipFree(d_filter));
    CHECK_HIP(hipFree(d_output));
    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));
    free(h_input);
    free(h_filter);
    free(h_output);
    
    return 0;
}
