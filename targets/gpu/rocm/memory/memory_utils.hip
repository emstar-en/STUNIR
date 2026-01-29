/**
 * STUNIR ROCm Memory Utilities
 *
 * Memory copy, set, prefetch, and diagnostic utilities.
 *
 * Schema: stunir.gpu.rocm.memory.utils.v1
 */

#ifndef STUNIR_MEMORY_UTILS_HIP
#define STUNIR_MEMORY_UTILS_HIP

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <vector>

namespace stunir {
namespace rocm {
namespace memory {

/**
 * Memory copy utilities
 */
class MemCopy {
public:
    // Synchronous copies
    static hipError_t hostToDevice(void* dst, const void* src, size_t size) {
        return hipMemcpy(dst, src, size, hipMemcpyHostToDevice);
    }
    
    static hipError_t deviceToHost(void* dst, const void* src, size_t size) {
        return hipMemcpy(dst, src, size, hipMemcpyDeviceToHost);
    }
    
    static hipError_t deviceToDevice(void* dst, const void* src, size_t size) {
        return hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice);
    }
    
    // Asynchronous copies
    static hipError_t hostToDeviceAsync(void* dst, const void* src, size_t size, 
                                         hipStream_t stream) {
        return hipMemcpyAsync(dst, src, size, hipMemcpyHostToDevice, stream);
    }
    
    static hipError_t deviceToHostAsync(void* dst, const void* src, size_t size,
                                         hipStream_t stream) {
        return hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToHost, stream);
    }
    
    static hipError_t deviceToDeviceAsync(void* dst, const void* src, size_t size,
                                           hipStream_t stream) {
        return hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToDevice, stream);
    }
    
    // 2D copies
    static hipError_t copy2D(void* dst, size_t dpitch,
                              const void* src, size_t spitch,
                              size_t width, size_t height,
                              hipMemcpyKind kind) {
        return hipMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
    }
    
    // 3D copies
    static hipError_t copy3D(const hipMemcpy3DParms* parms) {
        return hipMemcpy3D(parms);
    }
    
    // Peer-to-peer copy
    static hipError_t peerToPeer(void* dst, int dstDevice,
                                  const void* src, int srcDevice,
                                  size_t size) {
        return hipMemcpyPeer(dst, dstDevice, src, srcDevice, size);
    }
};

/**
 * Memory set utilities
 */
class MemSet {
public:
    static hipError_t zero(void* ptr, size_t size) {
        return hipMemset(ptr, 0, size);
    }
    
    static hipError_t set(void* ptr, int value, size_t size) {
        return hipMemset(ptr, value, size);
    }
    
    static hipError_t zeroAsync(void* ptr, size_t size, hipStream_t stream) {
        return hipMemsetAsync(ptr, 0, size, stream);
    }
    
    static hipError_t setAsync(void* ptr, int value, size_t size, hipStream_t stream) {
        return hipMemsetAsync(ptr, value, size, stream);
    }
    
    // 2D memset
    static hipError_t zero2D(void* ptr, size_t pitch, size_t width, size_t height) {
        return hipMemset2D(ptr, pitch, 0, width, height);
    }
};

/**
 * Managed/Unified memory utilities
 */
class ManagedMemory {
public:
    static hipError_t allocate(void** ptr, size_t size) {
        return hipMallocManaged(ptr, size);
    }
    
    // Memory advice hints
    static hipError_t prefetch(const void* ptr, size_t size, int device,
                                hipStream_t stream = 0) {
        return hipMemPrefetchAsync(ptr, size, device, stream);
    }
    
    static hipError_t adviseReadMostly(const void* ptr, size_t size, int device) {
        return hipMemAdvise(ptr, size, hipMemAdviseSetReadMostly, device);
    }
    
    static hipError_t advisePreferredLocation(const void* ptr, size_t size, int device) {
        return hipMemAdvise(ptr, size, hipMemAdviseSetPreferredLocation, device);
    }
    
    static hipError_t adviseAccessedBy(const void* ptr, size_t size, int device) {
        return hipMemAdvise(ptr, size, hipMemAdviseSetAccessedBy, device);
    }
    
    // Query attributes
    static hipError_t getAttributes(hipPointerAttribute_t* attr, const void* ptr) {
        return hipPointerGetAttributes(attr, ptr);
    }
};

/**
 * Pinned (page-locked) host memory
 */
class PinnedMemory {
public:
    static hipError_t allocate(void** ptr, size_t size) {
        return hipHostMalloc(ptr, size, hipHostMallocDefault);
    }
    
    static hipError_t allocateMapped(void** ptr, size_t size) {
        return hipHostMalloc(ptr, size, hipHostMallocMapped);
    }
    
    static hipError_t allocateWriteCombined(void** ptr, size_t size) {
        return hipHostMalloc(ptr, size, hipHostMallocWriteCombined);
    }
    
    static hipError_t free(void* ptr) {
        return hipHostFree(ptr);
    }
    
    static hipError_t getDevicePointer(void** devPtr, void* hostPtr) {
        return hipHostGetDevicePointer(devPtr, hostPtr, 0);
    }
    
    static hipError_t register_(void* ptr, size_t size) {
        return hipHostRegister(ptr, size, hipHostRegisterDefault);
    }
    
    static hipError_t unregister(void* ptr) {
        return hipHostUnregister(ptr);
    }
};

/**
 * Memory information and diagnostics
 */
class MemInfo {
public:
    struct DeviceMemory {
        size_t free;
        size_t total;
        double usage_percent;
    };
    
    static DeviceMemory getDeviceMemory(int device = -1) {
        DeviceMemory mem;
        if (device >= 0) hipSetDevice(device);
        hipMemGetInfo(&mem.free, &mem.total);
        mem.usage_percent = 100.0 * (mem.total - mem.free) / mem.total;
        return mem;
    }
    
    static void printDeviceMemory(int device = -1) {
        DeviceMemory mem = getDeviceMemory(device);
        printf("Device Memory: %.2f / %.2f GB (%.1f%% used)\n",
               (mem.total - mem.free) / (1024.0 * 1024.0 * 1024.0),
               mem.total / (1024.0 * 1024.0 * 1024.0),
               mem.usage_percent);
    }
    
    enum class MemoryType {
        Unknown,
        Host,
        Device,
        Managed
    };
    
    static MemoryType getPointerType(const void* ptr) {
        hipPointerAttribute_t attr;
        if (hipPointerGetAttributes(&attr, ptr) != hipSuccess) {
            return MemoryType::Unknown;
        }
        
        if (attr.isManaged) return MemoryType::Managed;
        if (attr.type == hipMemoryTypeDevice) return MemoryType::Device;
        if (attr.type == hipMemoryTypeHost) return MemoryType::Host;
        return MemoryType::Unknown;
    }
    
    static const char* memoryTypeString(MemoryType type) {
        switch (type) {
            case MemoryType::Host: return "Host";
            case MemoryType::Device: return "Device";
            case MemoryType::Managed: return "Managed";
            default: return "Unknown";
        }
    }
};

/**
 * Memory bandwidth test
 */
class BandwidthTest {
public:
    struct Result {
        double h2d_bandwidth_gb_s;
        double d2h_bandwidth_gb_s;
        double d2d_bandwidth_gb_s;
        size_t test_size;
    };
    
    static Result run(size_t size = 256 * 1024 * 1024, int iterations = 10) {
        Result result;
        result.test_size = size;
        
        // Allocate memory
        void *h_data, *d_data1, *d_data2;
        hipHostMalloc(&h_data, size);
        hipMalloc(&d_data1, size);
        hipMalloc(&d_data2, size);
        
        hipEvent_t start, stop;
        hipEventCreate(&start);
        hipEventCreate(&stop);
        float elapsed;
        
        // H2D bandwidth
        hipEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            hipMemcpy(d_data1, h_data, size, hipMemcpyHostToDevice);
        }
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&elapsed, start, stop);
        result.h2d_bandwidth_gb_s = (double)size * iterations / (elapsed / 1000.0) / 1e9;
        
        // D2H bandwidth
        hipEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            hipMemcpy(h_data, d_data1, size, hipMemcpyDeviceToHost);
        }
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&elapsed, start, stop);
        result.d2h_bandwidth_gb_s = (double)size * iterations / (elapsed / 1000.0) / 1e9;
        
        // D2D bandwidth
        hipEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            hipMemcpy(d_data2, d_data1, size, hipMemcpyDeviceToDevice);
        }
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&elapsed, start, stop);
        result.d2d_bandwidth_gb_s = (double)size * iterations / (elapsed / 1000.0) / 1e9;
        
        // Cleanup
        hipHostFree(h_data);
        hipFree(d_data1);
        hipFree(d_data2);
        hipEventDestroy(start);
        hipEventDestroy(stop);
        
        return result;
    }
    
    static void print(const Result& r) {
        printf("=== Memory Bandwidth Test (%.0f MB) ===\n", r.test_size / (1024.0 * 1024.0));
        printf("H2D: %.2f GB/s\n", r.h2d_bandwidth_gb_s);
        printf("D2H: %.2f GB/s\n", r.d2h_bandwidth_gb_s);
        printf("D2D: %.2f GB/s\n", r.d2d_bandwidth_gb_s);
    }
};

} // namespace memory
} // namespace rocm
} // namespace stunir

#endif // STUNIR_MEMORY_UTILS_HIP
