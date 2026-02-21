# STUNIR ROCm Multi-GPU Support

Foundation for multi-GPU programming with ROCm.

## Features

- Device enumeration and management
- Peer-to-peer (P2P) memory access
- Data distribution patterns
- Parallel execution utilities
- RAII stream management

## Components

### Device Manager

Enumerate and query GPU devices:

```cpp
#include "multi_gpu_utils.hip"
using namespace stunir::rocm::multigpu;

DeviceManager mgr;
mgr.print_info();

printf("GPUs: %d\n", mgr.device_count());
printf("Can GPU0 access GPU1? %s\n", 
       mgr.can_peer_access(0, 1) ? "Yes" : "No");
```

### Multi-GPU Buffer

Allocate memory across multiple GPUs:

```cpp
MultiGpuBuffer<float> buffer;
buffer.allocate(4, 1024);  // 4 GPUs, 1024 elements each

// Or varying sizes
std::vector<size_t> sizes = {1000, 2000, 1500, 2500};
buffer.allocate_varying(sizes);

float* gpu0_ptr = buffer[0];
float* gpu1_ptr = buffer[1];
```

### Data Distributor

Distribute data across GPUs:

```cpp
DataDistributor<float> dist(4, DistributionStrategy::BLOCK);
auto sizes = dist.partition(total_elements);

// Scatter host data to GPUs
dist.scatter(host_data, total, device_buffers, streams);

// Gather results back
dist.gather(device_buffers, host_result, streams);
```

### Parallel Executor

Run operations on all GPUs:

```cpp
MultiGpuExecutor executor(mgr);

executor.parallel_for_devices([&](int device, hipStream_t stream) {
    hipLaunchKernelGGL(my_kernel, grid, block, 0, stream, ...);
});

executor.synchronize_all();
```

## Distribution Strategies

| Strategy | Description |
|----------|-------------|
| BLOCK | Contiguous partitions per GPU |
| CYCLIC | Round-robin element distribution |
| BLOCK_CYCLIC | Blocked cyclic distribution |

## Example: Multi-GPU Vector Addition

```cpp
// See multi_gpu_example.hip for full implementation

DeviceManager mgr;
MultiGpuExecutor executor(mgr);
DataDistributor<float> dist(mgr.device_count());

// Allocate and scatter
MultiGpuBuffer<float> d_a, d_b, d_c;
dist.scatter(h_a, n, d_a, executor.all_streams());
dist.scatter(h_b, n, d_b, executor.all_streams());
d_c.allocate_varying(dist.partition(n));

// Launch on all GPUs
executor.parallel_for_devices([&](int dev, hipStream_t stream) {
    hipLaunchKernelGGL(vector_add, grid, block, 0, stream,
                       d_a[dev], d_b[dev], d_c[dev], sizes[dev]);
});

// Gather results
dist.gather(d_c, h_c, executor.all_streams());
```

## Building

```bash
# Build example
hipcc -O3 -o multi_gpu_example multi_gpu_example.hip

# Run
./multi_gpu_example
```

## Performance Considerations

1. **Enable P2P access** for direct GPU-GPU transfers
2. **Use pinned host memory** for faster H2D/D2H transfers
3. **Overlap transfers with computation** using streams
4. **Balance workloads** based on GPU capabilities
5. **Minimize synchronization** points

## Expansion Path

Future enhancements:
- NCCL integration for collective operations
- GPU-aware MPI support
- Automatic load balancing
- Multi-node support
- Memory-managed (unified) multi-GPU

## Schema

- `stunir.gpu.rocm.multi_gpu.utils.v1`
- `stunir.gpu.rocm.multi_gpu.example.v1`
