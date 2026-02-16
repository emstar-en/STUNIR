/**
 * STUNIR ROCm Multi-GPU Utilities
 *
 * Foundation for multi-GPU operations including device enumeration,
 * peer-to-peer access, and data distribution patterns.
 *
 * Schema: stunir.gpu.rocm.multi_gpu.utils.v1
 */

#ifndef STUNIR_MULTI_GPU_UTILS_HIP
#define STUNIR_MULTI_GPU_UTILS_HIP

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <map>
#include <functional>

namespace stunir {
namespace rocm {
namespace multigpu {

#define MGPU_CHECK(call) { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "MGPU Error at %s:%d: %s\n", \
                __FILE__, __LINE__, hipGetErrorString(err)); \
        return false; \
    } \
}

#define MGPU_CHECK_VOID(call) { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "MGPU Error at %s:%d: %s\n", \
                __FILE__, __LINE__, hipGetErrorString(err)); \
    } \
}

/**
 * GPU Device Information
 */
struct GpuDevice {
    int id;
    std::string name;
    size_t total_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    bool is_integrated;
    bool can_map_host_memory;
    
    static GpuDevice from_id(int device_id) {
        GpuDevice dev;
        dev.id = device_id;
        
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, device_id);
        
        dev.name = prop.name;
        dev.total_memory = prop.totalGlobalMem;
        dev.compute_capability_major = prop.major;
        dev.compute_capability_minor = prop.minor;
        dev.multiprocessor_count = prop.multiProcessorCount;
        dev.is_integrated = prop.integrated;
        dev.can_map_host_memory = prop.canMapHostMemory;
        
        return dev;
    }
    
    void print() const {
        printf("GPU %d: %s\n", id, name.c_str());
        printf("  Memory: %.2f GB\n", total_memory / (1024.0 * 1024.0 * 1024.0));
        printf("  Compute: %d.%d\n", compute_capability_major, compute_capability_minor);
        printf("  SMs: %d\n", multiprocessor_count);
    }
};

/**
 * Device Manager - Enumerate and manage GPUs
 */
class DeviceManager {
private:
    std::vector<GpuDevice> devices;
    std::vector<std::vector<bool>> p2p_matrix;
    int current_device;
    
public:
    DeviceManager() : current_device(0) {
        enumerate();
    }
    
    bool enumerate() {
        int count = 0;
        hipError_t err = hipGetDeviceCount(&count);
        if (err != hipSuccess || count == 0) {
            fprintf(stderr, "No HIP devices found\n");
            return false;
        }
        
        devices.clear();
        for (int i = 0; i < count; i++) {
            devices.push_back(GpuDevice::from_id(i));
        }
        
        // Build P2P capability matrix
        p2p_matrix.resize(count, std::vector<bool>(count, false));
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < count; j++) {
                if (i == j) {
                    p2p_matrix[i][j] = true;
                } else {
                    int can_access;
                    hipDeviceCanAccessPeer(&can_access, i, j);
                    p2p_matrix[i][j] = (can_access == 1);
                }
            }
        }
        
        return true;
    }
    
    int device_count() const { return devices.size(); }
    const GpuDevice& device(int idx) const { return devices[idx]; }
    const std::vector<GpuDevice>& all_devices() const { return devices; }
    
    bool can_peer_access(int src, int dst) const {
        if (src < 0 || src >= (int)devices.size() || 
            dst < 0 || dst >= (int)devices.size()) {
            return false;
        }
        return p2p_matrix[src][dst];
    }
    
    bool set_device(int device_id) {
        if (device_id < 0 || device_id >= (int)devices.size()) {
            return false;
        }
        MGPU_CHECK(hipSetDevice(device_id));
        current_device = device_id;
        return true;
    }
    
    int get_current_device() const { return current_device; }
    
    bool enable_peer_access(int src, int dst) {
        if (!can_peer_access(src, dst)) return false;
        
        hipSetDevice(src);
        hipError_t err = hipDeviceEnablePeerAccess(dst, 0);
        return (err == hipSuccess || err == hipErrorPeerAccessAlreadyEnabled);
    }
    
    bool enable_all_peer_access() {
        bool all_success = true;
        for (size_t i = 0; i < devices.size(); i++) {
            for (size_t j = 0; j < devices.size(); j++) {
                if (i != j && can_peer_access(i, j)) {
                    if (!enable_peer_access(i, j)) {
                        all_success = false;
                    }
                }
            }
        }
        return all_success;
    }
    
    void print_info() const {
        printf("=== Multi-GPU System Info ===\n");
        printf("Devices: %zu\n\n", devices.size());
        
        for (const auto& dev : devices) {
            dev.print();
            printf("\n");
        }
        
        printf("P2P Access Matrix:\n");
        printf("     ");
        for (size_t j = 0; j < devices.size(); j++) {
            printf("GPU%zu ", j);
        }
        printf("\n");
        for (size_t i = 0; i < devices.size(); i++) {
            printf("GPU%zu ", i);
            for (size_t j = 0; j < devices.size(); j++) {
                printf("  %s  ", p2p_matrix[i][j] ? "Y" : "N");
            }
            printf("\n");
        }
    }
};

/**
 * Per-device stream for overlapped execution
 */
class DeviceStream {
private:
    int device_id;
    hipStream_t stream;
    bool owns_stream;
    
public:
    DeviceStream(int dev, hipStream_t s = nullptr) : device_id(dev), owns_stream(false) {
        if (s == nullptr) {
            hipSetDevice(device_id);
            hipStreamCreate(&stream);
            owns_stream = true;
        } else {
            stream = s;
        }
    }
    
    ~DeviceStream() {
        if (owns_stream) {
            hipSetDevice(device_id);
            hipStreamDestroy(stream);
        }
    }
    
    void synchronize() {
        hipStreamSynchronize(stream);
    }
    
    hipStream_t get() const { return stream; }
    int get_device() const { return device_id; }
    operator hipStream_t() const { return stream; }
};

/**
 * Multi-GPU memory allocation helper
 */
template<typename T>
class MultiGpuBuffer {
private:
    std::vector<T*> device_ptrs;
    std::vector<size_t> device_sizes;
    int num_devices;
    
public:
    MultiGpuBuffer() : num_devices(0) {}
    
    MultiGpuBuffer(int devices, size_t elements_per_device) {
        allocate(devices, elements_per_device);
    }
    
    ~MultiGpuBuffer() {
        free();
    }
    
    bool allocate(int devices, size_t elements_per_device) {
        free();
        
        num_devices = devices;
        device_ptrs.resize(devices);
        device_sizes.resize(devices, elements_per_device);
        
        for (int i = 0; i < devices; i++) {
            hipSetDevice(i);
            hipError_t err = hipMalloc(&device_ptrs[i], elements_per_device * sizeof(T));
            if (err != hipSuccess) {
                free();
                return false;
            }
        }
        return true;
    }
    
    bool allocate_varying(const std::vector<size_t>& sizes) {
        free();
        
        num_devices = sizes.size();
        device_ptrs.resize(num_devices);
        device_sizes = sizes;
        
        for (int i = 0; i < num_devices; i++) {
            hipSetDevice(i);
            hipError_t err = hipMalloc(&device_ptrs[i], sizes[i] * sizeof(T));
            if (err != hipSuccess) {
                free();
                return false;
            }
        }
        return true;
    }
    
    void free() {
        for (int i = 0; i < num_devices; i++) {
            if (device_ptrs[i]) {
                hipSetDevice(i);
                hipFree(device_ptrs[i]);
            }
        }
        device_ptrs.clear();
        device_sizes.clear();
        num_devices = 0;
    }
    
    T* operator[](int device) { return device_ptrs[device]; }
    const T* operator[](int device) const { return device_ptrs[device]; }
    
    size_t size(int device) const { return device_sizes[device]; }
    int devices() const { return num_devices; }
};

/**
 * Data distribution strategies
 */
enum class DistributionStrategy {
    BLOCK,      // Contiguous blocks per GPU
    CYCLIC,     // Round-robin distribution
    BLOCK_CYCLIC,  // Blocked cyclic
    CUSTOM      // User-defined
};

/**
 * Data distributor for multi-GPU workloads
 */
template<typename T>
class DataDistributor {
private:
    int num_devices;
    DistributionStrategy strategy;
    size_t block_size;
    
public:
    DataDistributor(int devices, DistributionStrategy strat = DistributionStrategy::BLOCK)
        : num_devices(devices), strategy(strat), block_size(1) {}
    
    void set_block_size(size_t bs) { block_size = bs; }
    
    // Calculate partition sizes for each device
    std::vector<size_t> partition(size_t total_elements) const {
        std::vector<size_t> sizes(num_devices);
        
        switch (strategy) {
            case DistributionStrategy::BLOCK: {
                size_t base = total_elements / num_devices;
                size_t remainder = total_elements % num_devices;
                for (int i = 0; i < num_devices; i++) {
                    sizes[i] = base + (i < (int)remainder ? 1 : 0);
                }
                break;
            }
            case DistributionStrategy::CYCLIC:
            case DistributionStrategy::BLOCK_CYCLIC: {
                // For cyclic, each GPU gets approximately equal elements
                size_t per_device = total_elements / num_devices;
                size_t remainder = total_elements % num_devices;
                for (int i = 0; i < num_devices; i++) {
                    sizes[i] = per_device + (i < (int)remainder ? 1 : 0);
                }
                break;
            }
            default:
                break;
        }
        
        return sizes;
    }
    
    // Distribute data from host to multiple GPUs
    bool scatter(const T* host_data, size_t total_elements,
                 MultiGpuBuffer<T>& device_buffers,
                 std::vector<hipStream_t>& streams) {
        auto sizes = partition(total_elements);
        
        if (!device_buffers.allocate_varying(sizes)) {
            return false;
        }
        
        size_t offset = 0;
        for (int i = 0; i < num_devices; i++) {
            hipSetDevice(i);
            hipMemcpyAsync(device_buffers[i], host_data + offset,
                          sizes[i] * sizeof(T), hipMemcpyHostToDevice,
                          streams[i]);
            offset += sizes[i];
        }
        
        return true;
    }
    
    // Gather results from GPUs to host
    bool gather(MultiGpuBuffer<T>& device_buffers, T* host_data,
               std::vector<hipStream_t>& streams) {
        size_t offset = 0;
        for (int i = 0; i < num_devices; i++) {
            hipSetDevice(i);
            hipMemcpyAsync(host_data + offset, device_buffers[i],
                          device_buffers.size(i) * sizeof(T),
                          hipMemcpyDeviceToHost, streams[i]);
            offset += device_buffers.size(i);
        }
        
        // Synchronize all streams
        for (int i = 0; i < num_devices; i++) {
            hipStreamSynchronize(streams[i]);
        }
        
        return true;
    }
};

/**
 * Simple multi-GPU parallel executor
 */
class MultiGpuExecutor {
private:
    DeviceManager& manager;
    std::vector<hipStream_t> streams;
    int num_devices;
    
public:
    MultiGpuExecutor(DeviceManager& mgr) : manager(mgr) {
        num_devices = manager.device_count();
        streams.resize(num_devices);
        
        for (int i = 0; i < num_devices; i++) {
            hipSetDevice(i);
            hipStreamCreate(&streams[i]);
        }
    }
    
    ~MultiGpuExecutor() {
        for (int i = 0; i < num_devices; i++) {
            hipSetDevice(i);
            hipStreamDestroy(streams[i]);
        }
    }
    
    // Execute a function on all GPUs in parallel
    template<typename Func>
    void parallel_for_devices(Func func) {
        for (int i = 0; i < num_devices; i++) {
            hipSetDevice(i);
            func(i, streams[i]);
        }
    }
    
    void synchronize_all() {
        for (int i = 0; i < num_devices; i++) {
            hipStreamSynchronize(streams[i]);
        }
    }
    
    hipStream_t stream(int device) { return streams[device]; }
    std::vector<hipStream_t>& all_streams() { return streams; }
};

} // namespace multigpu
} // namespace rocm
} // namespace stunir

#endif // STUNIR_MULTI_GPU_UTILS_HIP
