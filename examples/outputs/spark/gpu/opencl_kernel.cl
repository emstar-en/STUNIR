/* STUNIR Generated GPU Code */
/* Platform: OpenCL */

#include <CL/cl.h>

typedef struct {
    float x;
    float y;
    float z;
} float3_data;

__kernel void vector_add(__global float3_data* a, __global float3_data* b, __global float3_data* result) {
    int idx = get_global_id(0);
    /* WASM Function Body */
}

