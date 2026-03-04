#!/usr/bin/env python3
"""
STUNIR Extractor Test Harness
Tests Rust, Python, and SPARK extractors with golden test files.

Usage:
    python test_extractors.py [--extractor rust|python|spark|all] [--golden]
"""

import subprocess
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Base paths
SCRIPT_DIR = Path(__file__).parent
STUNIR_ROOT = SCRIPT_DIR.parent.parent
TOOLS_SPARK = STUNIR_ROOT / "tools" / "spark"
GOLDEN_DIR = TOOLS_SPARK / "test_data" / "golden"

# Extractor configurations
EXTRACTORS = {
    "rust": {
        "exe": TOOLS_SPARK / "bin" / "rust_extract_main.exe",
        "test_file": TOOLS_SPARK / "test_data" / "test_rust.rs",
        "expected_functions": ["add", "multiply", "main"],
        "golden_file": GOLDEN_DIR / "test_rust.expected.json",
    },
    "python": {
        "exe": TOOLS_SPARK / "bin" / "python_extract_main.exe",
        "test_file": TOOLS_SPARK / "test_data" / "test_python.py",
        "expected_functions": ["simple_func", "with_type_hints", "with_defaults"],
        "golden_file": GOLDEN_DIR / "test_python.expected.json",
    },
    "spark": {
        "exe": TOOLS_SPARK / "bin" / "spark_extract_main.exe",
        "test_file": TOOLS_SPARK / "test_data" / "golden_test.ads",
        "expected_functions": [],  # Will be populated if file exists
        "golden_file": GOLDEN_DIR / "test_spark.expected.json",
    }
}


def log(msg: str, level: str = "INFO"):
    """Print log message with level"""
    print(f"[{level}] {msg}")


def normalize_json(data: Dict) -> Dict:
    """Normalize JSON for deterministic comparison"""
    # Sort keys recursively
    def sort_keys(obj):
        if isinstance(obj, dict):
            return {k: sort_keys(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, list):
            return [sort_keys(item) for item in obj]
        return obj
    return sort_keys(data)


def compare_with_golden(output_data: Dict, golden_file: Path) -> Tuple[bool, str]:
    """Compare output with golden fixture"""
    if not golden_file.exists():
        return True, "No golden file (skip comparison)"
    
    try:
        with open(golden_file, 'r', encoding='utf-8') as f:
            golden_data = json.load(f)
    except json.JSONDecodeError as e:
        return False, f"Golden file JSON error: {e}"
    
    # Normalize both for comparison
    output_norm = normalize_json(output_data)
    golden_norm = normalize_json(golden_data)
    
    if output_norm == golden_norm:
        return True, "Matches golden fixture"
    
    # Find differences
    diffs = []
    
    # Check schema_version
    if output_norm.get("schema_version") != golden_norm.get("schema_version"):
        diffs.append(f"schema_version: got {output_norm.get('schema_version')}, expected {golden_norm.get('schema_version')}")
    
    # Check module_name
    if output_norm.get("module_name") != golden_norm.get("module_name"):
        diffs.append(f"module_name: got {output_norm.get('module_name')}, expected {golden_norm.get('module_name')}")
    
    # Check functions
    out_funcs = output_norm.get("functions", [])
    gold_funcs = golden_norm.get("functions", [])
    
    if len(out_funcs) != len(gold_funcs):
        diffs.append(f"function count: got {len(out_funcs)}, expected {len(gold_funcs)}")
    else:
        for i, (out_f, gold_f) in enumerate(zip(out_funcs, gold_funcs)):
            if out_f != gold_f:
                diffs.append(f"function[{i}]: got {out_f}, expected {gold_f}")
    
    if diffs:
        return False, "Golden mismatch: " + "; ".join(diffs[:3])
    return False, "Golden mismatch (structural difference)"


def run_extractor(extractor_name: str, config: Dict, use_golden: bool = False) -> Tuple[bool, str, Optional[Dict]]:
    """Run a single extractor and validate output"""
    exe = config["exe"]
    test_file = config["test_file"]
    expected_funcs = config["expected_functions"]
    
    # Check if extractor exists
    if not exe.exists():
        return False, f"Extractor not found: {exe}", None
    
    # Check if test file exists
    if not test_file.exists():
        return False, f"Test file not found: {test_file}", None
    
    # Create output file path
    output_file = SCRIPT_DIR / f"test_output_{extractor_name}.json"
    
    # Run extractor
    cmd = [
        str(exe),
        "-i", str(test_file),
        "-o", str(output_file),
        "-m", f"test_{extractor_name}",
        "--lang", extractor_name
    ]
    
    log(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(TOOLS_SPARK)
        )
        
        if result.returncode != 0:
            return False, f"Exit code {result.returncode}: {result.stderr}", None
        
        # Validate output
        if not output_file.exists():
            return False, f"Output file not created: {output_file}", None
        
        # Parse JSON
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate schema
        schema_version = data.get("schema_version", "")
        if schema_version != "extraction.v2":
            return False, f"Invalid schema_version: {schema_version} (expected 'extraction.v2')", data
        
        # Validate functions
        functions = data.get("functions", [])
        func_names = [f.get("name") for f in functions]
        
        # Check expected functions
        missing = []
        for expected in expected_funcs:
            if expected not in func_names:
                missing.append(expected)
        
        if missing:
            return False, f"Missing expected functions: {missing}", data
        
        # Validate function structure
        for func in functions:
            if "name" not in func:
                return False, f"Function missing 'name' field: {func}", data
            if "return_type" not in func:
                return False, f"Function missing 'return_type' field: {func}", data
            if "parameters" not in func:
                return False, f"Function missing 'parameters' field: {func}", data
        
        # Golden comparison if requested
        if use_golden:
            golden_ok, golden_msg = compare_with_golden(data, config.get("golden_file", Path()))
            if not golden_ok:
                return False, golden_msg, data
            return True, f"OK - {len(functions)} functions, {golden_msg}", data
        
        return True, f"OK - {len(functions)} functions extracted", data
        
    except subprocess.TimeoutExpired:
        return False, "Timeout (30s)", None
    except json.JSONDecodeError as e:
        return False, f"JSON parse error: {e}", None
    except Exception as e:
        return False, f"Error: {e}", None


def test_all_extractors(extractors_to_test: List[str], use_golden: bool = False) -> Dict[str, Tuple[bool, str]]:
    """Test multiple extractors"""
    results = {}
    
    for name in extractors_to_test:
        if name not in EXTRACTORS:
            results[name] = (False, f"Unknown extractor: {name}")
            continue
        
        log(f"\n{'='*50}")
        log(f"Testing {name} extractor")
        log(f"{'='*50}")
        
        success, msg, data = run_extractor(name, EXTRACTORS[name], use_golden)
        results[name] = (success, msg)
        
        if success:
            log(f"PASSED: {msg}", "PASS")
            if data:
                log(f"  Functions: {[f['name'] for f in data.get('functions', [])]}")
        else:
            log(f"FAILED: {msg}", "FAIL")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test STUNIR extractors")
    parser.add_argument(
        "--extractor", "-e",
        choices=["rust", "python", "spark", "all"],
        default="all",
        help="Which extractor to test (default: all)"
    )
    parser.add_argument(
        "--golden", "-g",
        action="store_true",
        help="Compare output against golden fixtures"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Determine which extractors to test
    if args.extractor == "all":
        extractors_to_test = ["rust", "python", "spark"]
    else:
        extractors_to_test = [args.extractor]
    
    log(f"STUNIR Extractor Test Harness")
    log(f"Testing: {extractors_to_test}")
    log(f"STUNIR root: {STUNIR_ROOT}")
    if args.golden:
        log(f"Golden comparison: enabled")
    
    # Run tests
    results = test_all_extractors(extractors_to_test, args.golden)
    
    # Summary
    log(f"\n{'='*50}")
    log("SUMMARY")
    log(f"{'='*50}")
    
    passed = sum(1 for s, m in results.values() if s)
    failed = sum(1 for s, m in results.values() if not s)
    
    for name, (success, msg) in results.items():
        status = "PASS" if success else "FAIL"
        log(f"  {name}: {status} - {msg}")
    
    log(f"\nTotal: {passed} passed, {failed} failed")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
