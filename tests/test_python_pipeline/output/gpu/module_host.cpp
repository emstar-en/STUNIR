// STUNIR CUDA Host Wrapper
// Module: module

#include <stdio.h>
#include <cuda_runtime.h>

extern void parse_heartbeat();

int main(int argc, char** argv) {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("CUDA devices: %d\n", deviceCount);
    
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("Device: %s\n", prop.name);
    }
    
    return 0;
}
