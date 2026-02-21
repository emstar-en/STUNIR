# STUNIR ROCm Benchmark Framework

Comprehensive performance benchmarking for ROCm GPU applications.

## Features

- High-precision GPU timing with HIP events
- Statistical analysis (mean, median, stddev, percentiles)
- JSON export for automated analysis
- Roofline model analysis
- Comparison across GPU architectures

## Components

### Benchmark Harness (`benchmark_harness.hip`)

Core infrastructure:
- `GpuTimer`: High-precision timing
- `BenchmarkStats`: Statistical analysis
- `BenchmarkRunner`: Test orchestration
- `RooflineModel`: Performance analysis

### Kernel Benchmarks (`kernel_benchmarks.hip`)

Benchmark suite for:
- Matrix multiplication (naive vs tiled)
- Matrix transpose (naive vs shared)
- Parallel reduction (naive vs warp-optimized)
- Memory operations (H2D, D2H, D2D)

## Usage

### Quick Start

```bash
# Build
hipcc -O3 -o kernel_bench kernel_benchmarks.hip

# Run
./kernel_bench
```

### Custom Benchmarks

```cpp
#include "benchmark_harness.hip"
using namespace stunir::rocm::benchmark;

int main() {
    BenchmarkRunner runner(5, 100);  // 5 warmup, 100 iterations
    
    // Benchmark a kernel
    double flops = 2.0 * M * N * K;  // GEMM flops
    double bytes = (M*K + K*N + M*N) * sizeof(float);
    
    runner.run("MyKernel", [&]() {
        hipLaunchKernelGGL(my_kernel, grid, block, 0, 0, args...);
    }, flops, bytes);
    
    runner.print_summary();
    runner.export_json("results.json");
}
```

### Roofline Analysis

```cpp
RooflineModel roofline = RooflineModel::from_device();

double flops = 2.0 * M * N * K;
double bytes = 3 * M * N * sizeof(float);
double achieved = flops / (elapsed_ms / 1000.0) / 1e9;

roofline.analyze("GEMM", flops, bytes, achieved);
```

## JSON Output Format

```json
{
  "schema": "stunir.gpu.rocm.benchmark.results.v1",
  "device": "AMD Radeon RX 7900 XTX",
  "benchmarks": [
    {
      "name": "MatMul_Tiled_1024",
      "stats": {
        "min_ms": 1.234,
        "max_ms": 1.456,
        "mean_ms": 1.345,
        "median_ms": 1.340,
        "stddev_ms": 0.045,
        "p95_ms": 1.420,
        "p99_ms": 1.450
      },
      "metrics": {
        "gflops": 1598.51,
        "bandwidth_gb_s": 450.23
      }
    }
  ]
}
```

## Performance Metrics

| Metric | Description |
|--------|-------------|
| GFLOPS | Billions of floating-point operations per second |
| GB/s | Memory bandwidth in gigabytes per second |
| Latency | Time per operation (ms) |
| Throughput | Operations per second |
| Efficiency | Achieved vs theoretical peak (%) |

## Building

```bash
# Standard build
hipcc -O3 -o benchmarks kernel_benchmarks.hip

# With profiling
hipcc -O3 -DPROFILE -o benchmarks kernel_benchmarks.hip

# Debug build
hipcc -g -O0 -o benchmarks kernel_benchmarks.hip
```

## Comparing Architectures

Run benchmarks on different GPUs and compare JSON outputs:

```python
import json

def compare(file1, file2):
    r1 = json.load(open(file1))['benchmarks']
    r2 = json.load(open(file2))['benchmarks']
    
    for b1 in r1:
        for b2 in r2:
            if b1['name'] == b2['name']:
                speedup = b1['stats']['median_ms'] / b2['stats']['median_ms']
                print(f"{b1['name']}: {speedup:.2f}x")
```

## Schema

- `stunir.gpu.rocm.benchmark.harness.v1`
- `stunir.gpu.rocm.benchmark.kernels.v1`
- `stunir.gpu.rocm.benchmark.results.v1`
