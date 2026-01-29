/**
 * STUNIR ROCm Memory Pool
 *
 * Arena-based GPU memory allocator with pooling, reuse, and defragmentation.
 * Optimized for workloads with frequent allocations of similar sizes.
 *
 * Schema: stunir.gpu.rocm.memory.pool.v1
 */

#ifndef STUNIR_MEMORY_POOL_HIP
#define STUNIR_MEMORY_POOL_HIP

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <map>
#include <mutex>
#include <algorithm>
#include <cstring>

namespace stunir {
namespace rocm {
namespace memory {

#define STUNIR_MEM_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "STUNIR Memory Error at %s:%d: %s\n", \
                    __FILE__, __LINE__, hipGetErrorString(err)); \
            return nullptr; \
        } \
    } while (0)

#define STUNIR_MEM_CHECK_VOID(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "STUNIR Memory Error at %s:%d: %s\n", \
                    __FILE__, __LINE__, hipGetErrorString(err)); \
            return; \
        } \
    } while (0)

// Size classes for small allocations (power of 2)
constexpr size_t MIN_BLOCK_SIZE = 256;           // 256 bytes
constexpr size_t MAX_POOL_BLOCK = 16 * 1024 * 1024;  // 16 MB
constexpr size_t ALIGNMENT = 256;                 // 256-byte alignment
constexpr size_t NUM_SIZE_CLASSES = 18;          // 256B to 16MB

/**
 * Memory block metadata
 */
struct MemBlock {
    void* ptr;
    size_t size;
    size_t allocated_size;  // Actual allocation size (may be larger)
    bool in_use;
    int device_id;
    hipStream_t stream;
    
    MemBlock() : ptr(nullptr), size(0), allocated_size(0), 
                 in_use(false), device_id(0), stream(0) {}
};

/**
 * Size class for pooled allocations
 */
class SizeClass {
public:
    size_t block_size;
    std::vector<MemBlock> free_blocks;
    std::vector<MemBlock> used_blocks;
    size_t total_allocated;
    size_t high_water_mark;
    
    SizeClass(size_t size) : block_size(size), total_allocated(0), high_water_mark(0) {}
    
    void* allocate(int device_id) {
        // Try to reuse existing free block
        if (!free_blocks.empty()) {
            MemBlock block = free_blocks.back();
            free_blocks.pop_back();
            block.in_use = true;
            used_blocks.push_back(block);
            return block.ptr;
        }
        
        // Allocate new block
        void* ptr = nullptr;
        if (hipMalloc(&ptr, block_size) != hipSuccess) {
            return nullptr;
        }
        
        MemBlock block;
        block.ptr = ptr;
        block.size = block_size;
        block.allocated_size = block_size;
        block.in_use = true;
        block.device_id = device_id;
        used_blocks.push_back(block);
        
        total_allocated += block_size;
        high_water_mark = std::max(high_water_mark, total_allocated);
        
        return ptr;
    }
    
    bool deallocate(void* ptr) {
        for (auto it = used_blocks.begin(); it != used_blocks.end(); ++it) {
            if (it->ptr == ptr) {
                it->in_use = false;
                free_blocks.push_back(*it);
                used_blocks.erase(it);
                return true;
            }
        }
        return false;
    }
    
    void trim(size_t keep_count = 2) {
        while (free_blocks.size() > keep_count) {
            MemBlock& block = free_blocks.back();
            hipFree(block.ptr);
            total_allocated -= block.allocated_size;
            free_blocks.pop_back();
        }
    }
    
    void release_all() {
        for (auto& block : free_blocks) {
            hipFree(block.ptr);
        }
        for (auto& block : used_blocks) {
            hipFree(block.ptr);
        }
        free_blocks.clear();
        used_blocks.clear();
        total_allocated = 0;
    }
};

/**
 * Arena allocator for contiguous allocations
 */
class Arena {
private:
    void* base_ptr;
    size_t total_size;
    size_t used_size;
    int device_id;
    std::vector<std::pair<size_t, size_t>> allocations;  // offset, size
    
public:
    Arena() : base_ptr(nullptr), total_size(0), used_size(0), device_id(0) {}
    
    bool initialize(size_t size, int dev = 0) {
        device_id = dev;
        hipSetDevice(device_id);
        
        if (hipMalloc(&base_ptr, size) != hipSuccess) {
            return false;
        }
        total_size = size;
        used_size = 0;
        return true;
    }
    
    ~Arena() {
        if (base_ptr) {
            hipSetDevice(device_id);
            hipFree(base_ptr);
        }
    }
    
    void* allocate(size_t size) {
        // Align size
        size_t aligned_size = (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
        
        if (used_size + aligned_size > total_size) {
            return nullptr;  // Arena full
        }
        
        void* ptr = static_cast<char*>(base_ptr) + used_size;
        allocations.push_back({used_size, aligned_size});
        used_size += aligned_size;
        
        return ptr;
    }
    
    void reset() {
        used_size = 0;
        allocations.clear();
    }
    
    size_t available() const { return total_size - used_size; }
    size_t used() const { return used_size; }
    size_t capacity() const { return total_size; }
    void* base() const { return base_ptr; }
};

/**
 * Main Memory Pool Manager
 */
class MemoryPool {
private:
    std::map<size_t, SizeClass*> size_classes;
    std::vector<MemBlock> large_allocations;
    std::mutex pool_mutex;
    int device_id;
    bool initialized;
    
    // Statistics
    size_t total_allocated;
    size_t total_freed;
    size_t allocation_count;
    size_t cache_hits;
    size_t cache_misses;
    
    size_t get_size_class(size_t size) {
        // Round up to next power of 2 >= MIN_BLOCK_SIZE
        if (size < MIN_BLOCK_SIZE) return MIN_BLOCK_SIZE;
        
        size_t block_size = MIN_BLOCK_SIZE;
        while (block_size < size && block_size < MAX_POOL_BLOCK) {
            block_size *= 2;
        }
        return block_size;
    }
    
public:
    MemoryPool() : device_id(0), initialized(false),
                   total_allocated(0), total_freed(0),
                   allocation_count(0), cache_hits(0), cache_misses(0) {}
    
    ~MemoryPool() {
        release();
    }
    
    bool initialize(int dev = 0) {
        std::lock_guard<std::mutex> lock(pool_mutex);
        
        if (initialized) return true;
        
        device_id = dev;
        hipSetDevice(device_id);
        
        // Pre-create size classes
        size_t size = MIN_BLOCK_SIZE;
        for (size_t i = 0; i < NUM_SIZE_CLASSES && size <= MAX_POOL_BLOCK; i++) {
            size_classes[size] = new SizeClass(size);
            size *= 2;
        }
        
        initialized = true;
        return true;
    }
    
    void* allocate(size_t size) {
        std::lock_guard<std::mutex> lock(pool_mutex);
        
        if (!initialized) {
            if (!initialize()) return nullptr;
        }
        
        hipSetDevice(device_id);
        allocation_count++;
        
        // Use pooled allocation for small sizes
        if (size <= MAX_POOL_BLOCK) {
            size_t class_size = get_size_class(size);
            auto it = size_classes.find(class_size);
            
            if (it != size_classes.end()) {
                void* ptr = it->second->allocate(device_id);
                if (ptr) {
                    if (!it->second->free_blocks.empty()) {
                        cache_hits++;
                    } else {
                        cache_misses++;
                    }
                    total_allocated += class_size;
                    return ptr;
                }
            }
        }
        
        // Large allocation - direct hipMalloc
        void* ptr = nullptr;
        STUNIR_MEM_CHECK(hipMalloc(&ptr, size));
        
        MemBlock block;
        block.ptr = ptr;
        block.size = size;
        block.allocated_size = size;
        block.in_use = true;
        block.device_id = device_id;
        large_allocations.push_back(block);
        
        total_allocated += size;
        cache_misses++;
        
        return ptr;
    }
    
    void deallocate(void* ptr) {
        if (!ptr) return;
        
        std::lock_guard<std::mutex> lock(pool_mutex);
        hipSetDevice(device_id);
        
        // Check pooled allocations
        for (auto& pair : size_classes) {
            if (pair.second->deallocate(ptr)) {
                total_freed += pair.first;
                return;
            }
        }
        
        // Check large allocations
        for (auto it = large_allocations.begin(); it != large_allocations.end(); ++it) {
            if (it->ptr == ptr) {
                hipFree(ptr);
                total_freed += it->size;
                large_allocations.erase(it);
                return;
            }
        }
        
        // Unknown pointer - try direct free
        hipFree(ptr);
    }
    
    void trim() {
        std::lock_guard<std::mutex> lock(pool_mutex);
        
        for (auto& pair : size_classes) {
            pair.second->trim();
        }
    }
    
    void release() {
        std::lock_guard<std::mutex> lock(pool_mutex);
        
        if (!initialized) return;
        
        hipSetDevice(device_id);
        
        for (auto& pair : size_classes) {
            pair.second->release_all();
            delete pair.second;
        }
        size_classes.clear();
        
        for (auto& block : large_allocations) {
            hipFree(block.ptr);
        }
        large_allocations.clear();
        
        initialized = false;
    }
    
    // Convenience template methods
    template<typename T>
    T* allocate_typed(size_t count) {
        return static_cast<T*>(allocate(count * sizeof(T)));
    }
    
    // Statistics
    struct Stats {
        size_t total_allocated;
        size_t total_freed;
        size_t current_usage;
        size_t allocation_count;
        size_t cache_hits;
        size_t cache_misses;
        double cache_hit_rate;
    };
    
    Stats get_stats() const {
        Stats s;
        s.total_allocated = total_allocated;
        s.total_freed = total_freed;
        s.current_usage = total_allocated - total_freed;
        s.allocation_count = allocation_count;
        s.cache_hits = cache_hits;
        s.cache_misses = cache_misses;
        s.cache_hit_rate = (cache_hits + cache_misses > 0) ?
            (double)cache_hits / (cache_hits + cache_misses) : 0.0;
        return s;
    }
    
    void print_stats() const {
        Stats s = get_stats();
        printf("=== Memory Pool Statistics ===\n");
        printf("Total allocated: %.2f MB\n", s.total_allocated / (1024.0 * 1024.0));
        printf("Total freed: %.2f MB\n", s.total_freed / (1024.0 * 1024.0));
        printf("Current usage: %.2f MB\n", s.current_usage / (1024.0 * 1024.0));
        printf("Allocations: %zu\n", s.allocation_count);
        printf("Cache hits: %zu, misses: %zu\n", s.cache_hits, s.cache_misses);
        printf("Cache hit rate: %.2f%%\n", s.cache_hit_rate * 100.0);
    }
};

/**
 * Global memory pool instance
 */
class GlobalPool {
private:
    static MemoryPool* instance;
    static std::mutex init_mutex;
    
public:
    static MemoryPool& get() {
        if (!instance) {
            std::lock_guard<std::mutex> lock(init_mutex);
            if (!instance) {
                instance = new MemoryPool();
                instance->initialize();
            }
        }
        return *instance;
    }
    
    static void destroy() {
        std::lock_guard<std::mutex> lock(init_mutex);
        if (instance) {
            delete instance;
            instance = nullptr;
        }
    }
};

MemoryPool* GlobalPool::instance = nullptr;
std::mutex GlobalPool::init_mutex;

/**
 * RAII wrapper for pool allocations
 */
template<typename T>
class PooledBuffer {
private:
    T* ptr;
    size_t count;
    MemoryPool* pool;
    
public:
    PooledBuffer() : ptr(nullptr), count(0), pool(nullptr) {}
    
    PooledBuffer(size_t n, MemoryPool& p) : count(n), pool(&p) {
        ptr = pool->allocate_typed<T>(n);
    }
    
    PooledBuffer(size_t n) : count(n), pool(&GlobalPool::get()) {
        ptr = pool->allocate_typed<T>(n);
    }
    
    ~PooledBuffer() {
        if (ptr && pool) {
            pool->deallocate(ptr);
        }
    }
    
    // Move semantics
    PooledBuffer(PooledBuffer&& other) : ptr(other.ptr), count(other.count), pool(other.pool) {
        other.ptr = nullptr;
        other.count = 0;
    }
    
    PooledBuffer& operator=(PooledBuffer&& other) {
        if (this != &other) {
            if (ptr && pool) pool->deallocate(ptr);
            ptr = other.ptr;
            count = other.count;
            pool = other.pool;
            other.ptr = nullptr;
            other.count = 0;
        }
        return *this;
    }
    
    // Disable copy
    PooledBuffer(const PooledBuffer&) = delete;
    PooledBuffer& operator=(const PooledBuffer&) = delete;
    
    T* data() { return ptr; }
    const T* data() const { return ptr; }
    size_t size() const { return count; }
    operator T*() { return ptr; }
    operator const T*() const { return ptr; }
};

} // namespace memory
} // namespace rocm
} // namespace stunir

#endif // STUNIR_MEMORY_POOL_HIP
