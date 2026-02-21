# STUNIR ROCm Memory Management

Advanced memory management utilities for ROCm GPU applications.

## Components

### Memory Pool (`memory_pool.hip`)

Arena-based allocator with pooling for efficient GPU memory management.

**Features:**
- Size-class based allocation (256B to 16MB)
- Block reuse to reduce allocation overhead
- Statistics tracking (hit rate, usage)
- Thread-safe with mutex protection
- RAII wrappers for automatic cleanup

### Memory Utilities (`memory_utils.hip`)

Comprehensive memory operation utilities.

**Features:**
- Synchronous and async memory copies
- 2D/3D memory operations
- Managed (unified) memory support
- Pinned host memory
- Peer-to-peer transfers
- Bandwidth testing

## Usage

### Memory Pool

```cpp
#include "memory_pool.hip"
using namespace stunir::rocm::memory;

// Using global pool
float* data = GlobalPool::get().allocate_typed<float>(1024);
// ... use data ...
GlobalPool::get().deallocate(data);

// RAII wrapper
{
    PooledBuffer<float> buffer(1024);
    // buffer automatically freed at scope end
}

// Statistics
GlobalPool::get().print_stats();
```

### Arena Allocator

```cpp
#include "memory_pool.hip"
using namespace stunir::rocm::memory;

// For temporary allocations within a kernel/operation
Arena arena;
arena.initialize(64 * 1024 * 1024);  // 64MB

void* temp1 = arena.allocate(1024);
void* temp2 = arena.allocate(2048);
// ... use memory ...

arena.reset();  // Free all at once (very fast)
```

### Memory Utilities

```cpp
#include "memory_utils.hip"
using namespace stunir::rocm::memory;

// Basic copies
MemCopy::hostToDevice(d_data, h_data, size);
MemCopy::deviceToHost(h_result, d_result, size);

// Async copies
MemCopy::hostToDeviceAsync(d_data, h_data, size, stream);

// Memory info
MemInfo::printDeviceMemory();
auto mem = MemInfo::getDeviceMemory();
printf("Free: %.2f GB\n", mem.free / 1e9);

// Bandwidth test
auto bw = BandwidthTest::run();
BandwidthTest::print(bw);

// Pinned memory
void* pinned;
PinnedMemory::allocate(&pinned, size);
PinnedMemory::free(pinned);

// Managed memory
void* managed;
ManagedMemory::allocate(&managed, size);
ManagedMemory::prefetch(managed, size, 0);  // Prefetch to GPU 0
```

## Performance Tips

1. **Use the pool for frequent allocations**: Pool allocation is ~10-100x faster than `hipMalloc`
2. **Use arenas for temporary data**: Arena reset is O(1)
3. **Pin host memory for async transfers**: Improves H2D/D2H throughput
4. **Prefetch managed memory**: Reduces page faults
5. **Align allocations**: Pool uses 256-byte alignment automatically

## Building

```bash
hipcc -o memory_test memory_pool.hip memory_utils.hip -DSTUNIR_MEMORY_TEST
./memory_test
```

## Schema

- `stunir.gpu.rocm.memory.pool.v1`
- `stunir.gpu.rocm.memory.utils.v1`
