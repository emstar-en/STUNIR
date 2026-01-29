#!/usr/bin/env python3
"""
STUNIR Load Test Runner
=======================

Runs load tests and generates reports.

Usage:
    python tests/load/run_load_tests.py [--users N] [--duration S] [--report]
"""

import subprocess
import sys
import os
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

# Performance baselines (95th percentile in ms)
PERFORMANCE_BASELINES = {
    "compute_sha256": 10,      # 10ms
    "canonical_json": 50,      # 50ms
    "emit_ir": 200,            # 200ms
    "large_manifest_processing": 500,  # 500ms
    "file_hash": 100,          # 100ms
}

# Minimum requests per second targets
RPS_TARGETS = {
    "compute_sha256": 100,
    "canonical_json": 50,
    "emit_ir": 10,
}


def check_locust_installed() -> bool:
    """Check if locust is installed."""
    try:
        subprocess.run(["locust", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def run_load_test(users: int, spawn_rate: int, duration: str, output_dir: Path) -> dict:
    """Run locust load test and return results."""
    locustfile = Path(__file__).parent / "locustfile.py"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_prefix = output_dir / f"load_test_{timestamp}"
    html_report = output_dir / f"load_test_{timestamp}.html"
    
    cmd = [
        "locust",
        "-f", str(locustfile),
        "--headless",
        "-u", str(users),
        "-r", str(spawn_rate),
        "-t", duration,
        "--csv", str(csv_prefix),
        "--html", str(html_report),
        "--only-summary",
    ]
    
    print(f"Running load test: {users} users, {duration}")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse results from CSV
    stats_file = Path(f"{csv_prefix}_stats.csv")
    results = {
        "timestamp": timestamp,
        "users": users,
        "duration": duration,
        "operations": {},
        "passed": True,
        "failures": []
    }
    
    if stats_file.exists():
        import csv
        with open(stats_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("Name", "")
                if name and name != "Aggregated":
                    results["operations"][name] = {
                        "requests": int(row.get("Request Count", 0)),
                        "failures": int(row.get("Failure Count", 0)),
                        "median_response_time": float(row.get("Median Response Time", 0)),
                        "p95_response_time": float(row.get("95%", 0)),
                        "p99_response_time": float(row.get("99%", 0)),
                        "rps": float(row.get("Requests/s", 0)),
                    }
    
    # Check against baselines
    for op_name, op_stats in results["operations"].items():
        baseline = PERFORMANCE_BASELINES.get(op_name)
        if baseline and op_stats["p95_response_time"] > baseline:
            results["passed"] = False
            results["failures"].append(
                f"{op_name}: p95 ({op_stats['p95_response_time']:.1f}ms) > baseline ({baseline}ms)"
            )
    
    return results


def print_results(results: dict):
    """Print formatted load test results."""
    print("\n" + "="*60)
    print("Load Test Results")
    print("="*60)
    print(f"Timestamp: {results['timestamp']}")
    print(f"Users: {results['users']}")
    print(f"Duration: {results['duration']}")
    print()
    
    print(f"{'Operation':<30} {'Requests':<10} {'p95 (ms)':<12} {'RPS':<10} {'Status'}")
    print("-"*70)
    
    for name, stats in results["operations"].items():
        baseline = PERFORMANCE_BASELINES.get(name, float('inf'))
        status = "✅" if stats["p95_response_time"] <= baseline else "❌"
        print(f"{name:<30} {stats['requests']:<10} {stats['p95_response_time']:<12.1f} {stats['rps']:<10.1f} {status}")
    
    print()
    if results["passed"]:
        print("✅ All performance baselines met")
    else:
        print("❌ Performance issues:")
        for failure in results["failures"]:
            print(f"   - {failure}")


def main():
    parser = argparse.ArgumentParser(description="Run STUNIR load tests")
    parser.add_argument("--users", "-u", type=int, default=10, help="Number of concurrent users")
    parser.add_argument("--spawn-rate", "-r", type=int, default=2, help="Users spawned per second")
    parser.add_argument("--duration", "-t", default="60s", help="Test duration")
    parser.add_argument("--output-dir", default="tests/load/results", help="Output directory")
    parser.add_argument("--json", action="store_true", help="Output JSON results")
    args = parser.parse_args()
    
    print("="*60)
    print("STUNIR Load Testing")
    print("="*60)
    
    if not check_locust_installed():
        print("\n⚠️  locust not installed. Install with: pip install locust")
        print("    Running manual performance tests instead...")
        
        # Fallback to simple performance tests
        return run_simple_perf_tests()
    
    results = run_load_test(
        users=args.users,
        spawn_rate=args.spawn_rate,
        duration=args.duration,
        output_dir=Path(args.output_dir)
    )
    
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_results(results)
    
    return 0 if results["passed"] else 1


def run_simple_perf_tests() -> int:
    """Run simple performance tests without locust."""
    import time
    
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    try:
        from manifests.base import canonical_json, compute_sha256
    except ImportError:
        print("Cannot import required modules")
        return 1
    
    # Simple performance measurements
    iterations = 1000
    
    print(f"\nRunning {iterations} iterations of each operation...")
    
    # Test compute_sha256
    data = "test data " * 100
    start = time.perf_counter()
    for _ in range(iterations):
        compute_sha256(data)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"compute_sha256: {elapsed/iterations:.3f}ms per call")
    
    # Test canonical_json
    test_dict = {"key": "value", "nested": {"a": 1, "b": 2}, "list": [1, 2, 3]}
    start = time.perf_counter()
    for _ in range(iterations):
        canonical_json(test_dict)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"canonical_json: {elapsed/iterations:.3f}ms per call")
    
    print("\n✅ Performance tests completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
