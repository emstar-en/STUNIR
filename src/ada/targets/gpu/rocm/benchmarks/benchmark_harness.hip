/**
 * STUNIR ROCm Benchmark Harness
 *
 * Core benchmarking infrastructure with timing utilities,
 * throughput/latency measurement, and JSON reporting.
 *
 * Schema: stunir.gpu.rocm.benchmark.harness.v1
 */

#ifndef STUNIR_BENCHMARK_HARNESS_HIP
#define STUNIR_BENCHMARK_HARNESS_HIP

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
#include <chrono>
#include <functional>

namespace stunir {
namespace rocm {
namespace benchmark {

#define BENCH_HIP_CHECK(call) { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP Error: %s\n", hipGetErrorString(err)); \
        exit(1); \
    } \
}

/**
 * High-precision GPU timer
 */
class GpuTimer {
private:
    hipEvent_t start_event, stop_event;
    std::vector<float> samples;
    bool recording;
    
public:
    GpuTimer() : recording(false) {
        BENCH_HIP_CHECK(hipEventCreate(&start_event));
        BENCH_HIP_CHECK(hipEventCreate(&stop_event));
    }
    
    ~GpuTimer() {
        hipEventDestroy(start_event);
        hipEventDestroy(stop_event);
    }
    
    void start(hipStream_t stream = 0) {
        BENCH_HIP_CHECK(hipEventRecord(start_event, stream));
        recording = true;
    }
    
    float stop(hipStream_t stream = 0) {
        BENCH_HIP_CHECK(hipEventRecord(stop_event, stream));
        BENCH_HIP_CHECK(hipEventSynchronize(stop_event));
        recording = false;
        
        float ms;
        BENCH_HIP_CHECK(hipEventElapsedTime(&ms, start_event, stop_event));
        samples.push_back(ms);
        return ms;
    }
    
    void reset() { samples.clear(); }
    const std::vector<float>& get_samples() const { return samples; }
};

/**
 * Statistical analysis of benchmark results
 */
struct BenchmarkStats {
    double min_ms;
    double max_ms;
    double mean_ms;
    double median_ms;
    double stddev_ms;
    double p95_ms;       // 95th percentile
    double p99_ms;       // 99th percentile
    int sample_count;
    
    static BenchmarkStats compute(std::vector<float> samples) {
        BenchmarkStats stats;
        stats.sample_count = samples.size();
        
        if (samples.empty()) {
            stats.min_ms = stats.max_ms = stats.mean_ms = 0;
            stats.median_ms = stats.stddev_ms = stats.p95_ms = stats.p99_ms = 0;
            return stats;
        }
        
        std::sort(samples.begin(), samples.end());
        
        stats.min_ms = samples.front();
        stats.max_ms = samples.back();
        stats.mean_ms = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
        
        // Median
        int mid = samples.size() / 2;
        stats.median_ms = (samples.size() % 2 == 0) ?
            (samples[mid - 1] + samples[mid]) / 2.0 : samples[mid];
        
        // Standard deviation
        double sq_sum = 0;
        for (float s : samples) {
            sq_sum += (s - stats.mean_ms) * (s - stats.mean_ms);
        }
        stats.stddev_ms = std::sqrt(sq_sum / samples.size());
        
        // Percentiles
        stats.p95_ms = samples[(int)(samples.size() * 0.95)];
        stats.p99_ms = samples[(int)(samples.size() * 0.99)];
        
        return stats;
    }
    
    void print(const char* name = "") const {
        printf("=== %s Benchmark Results ===\n", name);
        printf("Samples: %d\n", sample_count);
        printf("Min: %.3f ms, Max: %.3f ms\n", min_ms, max_ms);
        printf("Mean: %.3f ms, Median: %.3f ms\n", mean_ms, median_ms);
        printf("Stddev: %.3f ms\n", stddev_ms);
        printf("P95: %.3f ms, P99: %.3f ms\n", p95_ms, p99_ms);
    }
};

/**
 * Performance metrics
 */
struct PerfMetrics {
    std::string name;
    double elapsed_ms;
    double gflops;
    double bandwidth_gb_s;
    double throughput;      // Operations per second
    std::string unit;       // Unit for throughput
    
    void print() const {
        printf("%s: %.3f ms", name.c_str(), elapsed_ms);
        if (gflops > 0) printf(", %.2f GFLOPS", gflops);
        if (bandwidth_gb_s > 0) printf(", %.2f GB/s", bandwidth_gb_s);
        if (throughput > 0) printf(", %.2f %s", throughput, unit.c_str());
        printf("\n");
    }
};

/**
 * Benchmark result for JSON export
 */
struct BenchmarkResult {
    std::string name;
    std::string category;
    std::string description;
    BenchmarkStats stats;
    PerfMetrics metrics;
    std::map<std::string, std::string> params;
    std::map<std::string, double> extra_metrics;
    std::string timestamp;
    std::string gpu_name;
    std::string driver_version;
    
    std::string to_json() const {
        std::string json = "{\n";
        json += "  \"name\": \"" + name + "\",\n";
        json += "  \"category\": \"" + category + "\",\n";
        json += "  \"description\": \"" + description + "\",\n";
        json += "  \"timestamp\": \"" + timestamp + "\",\n";
        json += "  \"gpu\": \"" + gpu_name + "\",\n";
        json += "  \"driver\": \"" + driver_version + "\",\n";
        
        // Stats
        json += "  \"stats\": {\n";
        json += "    \"min_ms\": " + std::to_string(stats.min_ms) + ",\n";
        json += "    \"max_ms\": " + std::to_string(stats.max_ms) + ",\n";
        json += "    \"mean_ms\": " + std::to_string(stats.mean_ms) + ",\n";
        json += "    \"median_ms\": " + std::to_string(stats.median_ms) + ",\n";
        json += "    \"stddev_ms\": " + std::to_string(stats.stddev_ms) + ",\n";
        json += "    \"p95_ms\": " + std::to_string(stats.p95_ms) + ",\n";
        json += "    \"p99_ms\": " + std::to_string(stats.p99_ms) + ",\n";
        json += "    \"samples\": " + std::to_string(stats.sample_count) + "\n";
        json += "  },\n";
        
        // Metrics
        json += "  \"metrics\": {\n";
        json += "    \"gflops\": " + std::to_string(metrics.gflops) + ",\n";
        json += "    \"bandwidth_gb_s\": " + std::to_string(metrics.bandwidth_gb_s) + ",\n";
        json += "    \"throughput\": " + std::to_string(metrics.throughput) + ",\n";
        json += "    \"throughput_unit\": \"" + metrics.unit + "\"\n";
        json += "  },\n";
        
        // Params
        json += "  \"params\": {\n";
        bool first = true;
        for (const auto& p : params) {
            if (!first) json += ",\n";
            json += "    \"" + p.first + "\": \"" + p.second + "\"";
            first = false;
        }
        json += "\n  }\n";
        
        json += "}";
        return json;
    }
};

/**
 * Device information for benchmarks
 */
struct DeviceInfo {
    int device_id;
    std::string name;
    std::string architecture;
    size_t total_memory;
    int compute_units;
    int clock_mhz;
    int memory_clock_mhz;
    int memory_bus_width;
    double peak_memory_bandwidth;
    int warp_size;
    
    static DeviceInfo get(int device = 0) {
        DeviceInfo info;
        info.device_id = device;
        
        hipDeviceProp_t prop;
        BENCH_HIP_CHECK(hipGetDeviceProperties(&prop, device));
        
        info.name = prop.name;
        info.architecture = "GCN/CDNA";  // AMD architecture
        info.total_memory = prop.totalGlobalMem;
        info.compute_units = prop.multiProcessorCount;
        info.clock_mhz = prop.clockRate / 1000;
        info.memory_clock_mhz = prop.memoryClockRate / 1000;
        info.memory_bus_width = prop.memoryBusWidth;
        info.peak_memory_bandwidth = 2.0 * prop.memoryClockRate * 1000 * 
                                     (prop.memoryBusWidth / 8) / 1e9;
        info.warp_size = prop.warpSize;
        
        return info;
    }
    
    void print() const {
        printf("=== GPU Information ===\n");
        printf("Device: %s\n", name.c_str());
        printf("Compute Units: %d\n", compute_units);
        printf("Memory: %.2f GB\n", total_memory / (1024.0 * 1024.0 * 1024.0));
        printf("Clock: %d MHz\n", clock_mhz);
        printf("Memory Bandwidth: %.2f GB/s (theoretical)\n", peak_memory_bandwidth);
        printf("Wavefront Size: %d\n", warp_size);
    }
};

/**
 * Main Benchmark Runner
 */
class BenchmarkRunner {
private:
    std::vector<BenchmarkResult> results;
    DeviceInfo device_info;
    int warmup_iterations;
    int benchmark_iterations;
    bool verbose;
    
public:
    BenchmarkRunner(int warmup = 5, int iterations = 100, bool verb = true)
        : warmup_iterations(warmup), benchmark_iterations(iterations), verbose(verb) {
        device_info = DeviceInfo::get();
        if (verbose) {
            device_info.print();
            printf("\n");
        }
    }
    
    template<typename Func>
    BenchmarkResult run(const std::string& name, Func kernel_func,
                        double flops = 0, double bytes = 0) {
        if (verbose) printf("Running: %s\n", name.c_str());
        
        GpuTimer timer;
        
        // Warmup
        for (int i = 0; i < warmup_iterations; i++) {
            kernel_func();
        }
        BENCH_HIP_CHECK(hipDeviceSynchronize());
        
        // Benchmark
        timer.reset();
        for (int i = 0; i < benchmark_iterations; i++) {
            timer.start();
            kernel_func();
            timer.stop();
        }
        
        // Compute statistics
        BenchmarkResult result;
        result.name = name;
        result.stats = BenchmarkStats::compute(timer.get_samples());
        result.gpu_name = device_info.name;
        
        // Compute performance metrics
        result.metrics.name = name;
        result.metrics.elapsed_ms = result.stats.median_ms;
        if (flops > 0) {
            result.metrics.gflops = flops / (result.stats.median_ms / 1000.0) / 1e9;
        }
        if (bytes > 0) {
            result.metrics.bandwidth_gb_s = bytes / (result.stats.median_ms / 1000.0) / 1e9;
        }
        
        // Timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        result.timestamp = std::ctime(&time_t);
        result.timestamp.pop_back();  // Remove newline
        
        if (verbose) {
            result.stats.print(name.c_str());
            result.metrics.print();
            printf("\n");
        }
        
        results.push_back(result);
        return result;
    }
    
    void add_result(const BenchmarkResult& result) {
        results.push_back(result);
    }
    
    void export_json(const std::string& filename) {
        std::ofstream file(filename);
        file << "{\n";
        file << "  \"schema\": \"stunir.gpu.rocm.benchmark.results.v1\",\n";
        file << "  \"device\": \"" << device_info.name << "\",\n";
        file << "  \"benchmarks\": [\n";
        
        for (size_t i = 0; i < results.size(); i++) {
            if (i > 0) file << ",\n";
            file << "    " << results[i].to_json();
        }
        
        file << "\n  ]\n";
        file << "}\n";
        file.close();
        
        if (verbose) printf("Results exported to: %s\n", filename.c_str());
    }
    
    void print_summary() {
        printf("\n=== Benchmark Summary ===\n");
        printf("%-30s %12s %12s %12s\n", "Benchmark", "Median (ms)", "GFLOPS", "GB/s");
        printf("%-30s %12s %12s %12s\n", "----------", "----------", "------", "----");
        
        for (const auto& r : results) {
            printf("%-30s %12.3f", r.name.c_str(), r.stats.median_ms);
            if (r.metrics.gflops > 0) printf(" %12.2f", r.metrics.gflops);
            else printf(" %12s", "-");
            if (r.metrics.bandwidth_gb_s > 0) printf(" %12.2f", r.metrics.bandwidth_gb_s);
            else printf(" %12s", "-");
            printf("\n");
        }
    }
    
    const std::vector<BenchmarkResult>& get_results() const { return results; }
    void clear_results() { results.clear(); }
};

/**
 * Roofline model analysis
 */
class RooflineModel {
private:
    double peak_gflops;
    double peak_bandwidth;
    
public:
    RooflineModel(double gflops, double bandwidth_gb_s)
        : peak_gflops(gflops), peak_bandwidth(bandwidth_gb_s) {}
    
    // From device properties
    static RooflineModel from_device(int device = 0) {
        DeviceInfo info = DeviceInfo::get(device);
        // Estimate peak FLOPS: CUs * ops_per_cycle * clock
        // AMD CDNA2: 128 FP32 ops per CU per cycle
        double peak_gflops = info.compute_units * 128 * info.clock_mhz / 1000.0;
        return RooflineModel(peak_gflops, info.peak_memory_bandwidth);
    }
    
    double get_attainable_gflops(double arithmetic_intensity) const {
        // arithmetic_intensity = FLOPS / Bytes
        return std::min(peak_gflops, arithmetic_intensity * peak_bandwidth);
    }
    
    double get_ridge_point() const {
        // Point where compute and memory bound intersect
        return peak_gflops / peak_bandwidth;
    }
    
    bool is_memory_bound(double arithmetic_intensity) const {
        return arithmetic_intensity < get_ridge_point();
    }
    
    void analyze(const std::string& name, double flops, double bytes, double achieved_gflops) {
        double ai = flops / bytes;
        double attainable = get_attainable_gflops(ai);
        double efficiency = achieved_gflops / attainable * 100.0;
        
        printf("=== Roofline Analysis: %s ===\n", name.c_str());
        printf("Arithmetic Intensity: %.2f FLOP/Byte\n", ai);
        printf("Ridge Point: %.2f FLOP/Byte\n", get_ridge_point());
        printf("Bound: %s\n", is_memory_bound(ai) ? "Memory" : "Compute");
        printf("Attainable: %.2f GFLOPS\n", attainable);
        printf("Achieved: %.2f GFLOPS\n", achieved_gflops);
        printf("Efficiency: %.1f%%\n", efficiency);
    }
};

} // namespace benchmark
} // namespace rocm
} // namespace stunir

#endif // STUNIR_BENCHMARK_HARNESS_HIP
