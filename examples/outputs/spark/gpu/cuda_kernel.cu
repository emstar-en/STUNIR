/* STUNIR Generated GPU Code */
/* Platform: CUDA */

#include <cuda_runtime.h>

typedef struct {
    float x;
    float y;
    float z;
} float3_data;

__global__ void vector_add(float3_data* a, float3_data* b, float3_data* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    /* WASM Function Body */
}

__global__ void matrix_mul(float* a, float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    /* WASM Function Body */
}

