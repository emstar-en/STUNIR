/**
 * STUNIR ROCm Wrapper Utilities
 *
 * Common utilities for hipBLAS and hipSPARSE wrappers.
 * Error handling, memory management, and performance utilities.
 *
 * Schema: stunir.gpu.rocm.wrapper.utils.v1
 */

#ifndef STUNIR_WRAPPER_UTILS_HIP
#define STUNIR_WRAPPER_UTILS_HIP

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <string>

namespace stunir {
namespace rocm {

/**
 * Error handling utilities
 */
#define STUNIR_HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "STUNIR HIP Error at %s:%d: %s\n", \
                    __FILE__, __LINE__, hipGetErrorString(err)); \
            throw std::runtime_error(hipGetErrorString(err)); \
        } \
    } while (0)

#define STUNIR_HIPBLAS_CHECK(call) \
    do { \
        hipblasStatus_t status = call; \
        if (status != HIPBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "STUNIR hipBLAS Error at %s:%d: %d\n", \
                    __FILE__, __LINE__, status); \
            throw std::runtime_error("hipBLAS error"); \
        } \
    } while (0)

#define STUNIR_HIPSPARSE_CHECK(call) \
    do { \
        hipsparseStatus_t status = call; \
        if (status != HIPSPARSE_STATUS_SUCCESS) { \
            fprintf(stderr, "STUNIR hipSPARSE Error at %s:%d: %d\n", \
                    __FILE__, __LINE__, status); \
            throw std::runtime_error("hipSPARSE error"); \
        } \
    } while (0)

/**
 * Timer class for performance measurement
 */
class GpuTimer {
private:
    hipEvent_t start_event, stop_event;
    bool running;
    
public:
    GpuTimer() : running(false) {
        STUNIR_HIP_CHECK(hipEventCreate(&start_event));
        STUNIR_HIP_CHECK(hipEventCreate(&stop_event));
    }
    
    ~GpuTimer() {
        hipEventDestroy(start_event);
        hipEventDestroy(stop_event);
    }
    
    void start(hipStream_t stream = 0) {
        STUNIR_HIP_CHECK(hipEventRecord(start_event, stream));
        running = true;
    }
    
    void stop(hipStream_t stream = 0) {
        STUNIR_HIP_CHECK(hipEventRecord(stop_event, stream));
        STUNIR_HIP_CHECK(hipEventSynchronize(stop_event));
        running = false;
    }
    
    float elapsed_ms() {
        float ms = 0.0f;
        STUNIR_HIP_CHECK(hipEventElapsedTime(&ms, start_event, stop_event));
        return ms;
    }
    
    double elapsed_seconds() {
        return elapsed_ms() / 1000.0;
    }
};

/**
 * Device memory wrapper with RAII
 */
template<typename T>
class DeviceBuffer {
private:
    T* ptr;
    size_t count;
    
public:
    DeviceBuffer() : ptr(nullptr), count(0) {}
    
    explicit DeviceBuffer(size_t n) : count(n) {
        STUNIR_HIP_CHECK(hipMalloc(&ptr, n * sizeof(T)));
    }
    
    ~DeviceBuffer() {
        if (ptr) hipFree(ptr);
    }
    
    // Disable copy
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    
    // Enable move
    DeviceBuffer(DeviceBuffer&& other) : ptr(other.ptr), count(other.count) {
        other.ptr = nullptr;
        other.count = 0;
    }
    
    DeviceBuffer& operator=(DeviceBuffer&& other) {
        if (this != &other) {
            if (ptr) hipFree(ptr);
            ptr = other.ptr;
            count = other.count;
            other.ptr = nullptr;
            other.count = 0;
        }
        return *this;
    }
    
    void allocate(size_t n) {
        if (ptr) hipFree(ptr);
        count = n;
        STUNIR_HIP_CHECK(hipMalloc(&ptr, n * sizeof(T)));
    }
    
    void copyFromHost(const T* host_data, size_t n = 0) {
        if (n == 0) n = count;
        STUNIR_HIP_CHECK(hipMemcpy(ptr, host_data, n * sizeof(T), hipMemcpyHostToDevice));
    }
    
    void copyToHost(T* host_data, size_t n = 0) const {
        if (n == 0) n = count;
        STUNIR_HIP_CHECK(hipMemcpy(host_data, ptr, n * sizeof(T), hipMemcpyDeviceToHost));
    }
    
    void zero() {
        STUNIR_HIP_CHECK(hipMemset(ptr, 0, count * sizeof(T)));
    }
    
    T* data() { return ptr; }
    const T* data() const { return ptr; }
    size_t size() const { return count; }
    operator T*() { return ptr; }
    operator const T*() const { return ptr; }
};

/**
 * Device information utilities
 */
struct DeviceInfo {
    int device_id;
    std::string name;
    size_t total_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int warp_size;
    int max_threads_per_block;
    
    static DeviceInfo get(int device_id = 0) {
        DeviceInfo info;
        info.device_id = device_id;
        
        hipDeviceProp_t prop;
        STUNIR_HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
        
        info.name = prop.name;
        info.total_memory = prop.totalGlobalMem;
        info.compute_capability_major = prop.major;
        info.compute_capability_minor = prop.minor;
        info.multiprocessor_count = prop.multiProcessorCount;
        info.warp_size = prop.warpSize;
        info.max_threads_per_block = prop.maxThreadsPerBlock;
        
        return info;
    }
    
    void print() const {
        printf("Device %d: %s\n", device_id, name.c_str());
        printf("  Memory: %.2f GB\n", total_memory / (1024.0 * 1024.0 * 1024.0));
        printf("  Compute: %d.%d\n", compute_capability_major, compute_capability_minor);
        printf("  SMs: %d\n", multiprocessor_count);
        printf("  Warp size: %d\n", warp_size);
    }
};

/**
 * Performance metrics calculation
 */
struct PerfMetrics {
    double elapsed_ms;
    double gflops;
    double bandwidth_gb_s;
    
    static PerfMetrics compute_gemm(int M, int N, int K, double elapsed_ms) {
        PerfMetrics pm;
        pm.elapsed_ms = elapsed_ms;
        double ops = 2.0 * M * N * K;  // multiply-add
        pm.gflops = ops / (elapsed_ms / 1000.0) / 1e9;
        double bytes = (M * K + K * N + M * N) * sizeof(float);
        pm.bandwidth_gb_s = bytes / (elapsed_ms / 1000.0) / 1e9;
        return pm;
    }
    
    static PerfMetrics compute_spmv(int nnz, int rows, int cols, double elapsed_ms) {
        PerfMetrics pm;
        pm.elapsed_ms = elapsed_ms;
        double ops = 2.0 * nnz;  // multiply-add per non-zero
        pm.gflops = ops / (elapsed_ms / 1000.0) / 1e9;
        // Bytes: values, col_indices, row_ptr, x, y
        double bytes = nnz * (sizeof(float) + sizeof(int)) + 
                      (rows + 1) * sizeof(int) + 
                      (cols + rows) * sizeof(float);
        pm.bandwidth_gb_s = bytes / (elapsed_ms / 1000.0) / 1e9;
        return pm;
    }
    
    void print(const char* label = "") const {
        printf("%s: %.3f ms, %.2f GFLOPS, %.2f GB/s\n", 
               label, elapsed_ms, gflops, bandwidth_gb_s);
    }
};

} // namespace rocm
} // namespace stunir

#endif // STUNIR_WRAPPER_UTILS_HIP
