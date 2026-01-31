# GPU Code Examples

## CUDA Kernel Example

**Platform:** NVIDIA CUDA  
**Compute Capability:** sm_75 (Turing)  
**Max Threads:** 1024

### Compilation

```bash
nvcc -arch=sm_75 -O2 cuda_kernel.cu -o kernel
./kernel
```

## OpenCL Kernel Example

**Platform:** OpenCL 1.2+  
**Devices:** CPU, GPU, FPGA

### Compilation

```bash
gcc -O2 -I/usr/local/cuda/include \
  -L/usr/local/cuda/lib64 \
  -lOpenCL \
  host.c -o opencl_host
```

### Features

- Thread-level parallelism
- Shared memory utilization
- Coalesced memory access
- Formally verified safety
