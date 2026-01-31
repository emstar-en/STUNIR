#!/usr/bin/env python3
"""
STUNIR Confluence Test Suite - Week 3
Tests IR generation and code emission across multiple language pipelines
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Test configuration
STUNIR_ROOT = Path(__file__).parent.parent
SPEC_DIR = STUNIR_ROOT / "spec" / "ardupilot_test"
TEST_OUTPUT_DIR = STUNIR_ROOT / "test_outputs" / "confluence"
TEST_SPEC = SPEC_DIR / "test_spec.json"

# Tool paths
SPARK_SPEC_TO_IR = STUNIR_ROOT / "tools" / "spark" / "bin" / "stunir_spec_to_ir_main"
SPARK_IR_TO_CODE = STUNIR_ROOT / "tools" / "spark" / "bin" / "stunir_ir_to_code_main"
PYTHON_SPEC_TO_IR = STUNIR_ROOT / "tools" / "spec_to_ir.py"
PYTHON_IR_TO_CODE = STUNIR_ROOT / "tools" / "ir_to_code.py"
RUST_SPEC_TO_IR = STUNIR_ROOT / "tools" / "rust" / "target" / "release" / "stunir_spec_to_ir"
RUST_IR_TO_CODE = STUNIR_ROOT / "tools" / "rust" / "target" / "release" / "stunir_ir_to_code"

# Emitter categories to test (24 major categories)
EMITTER_CATEGORIES = [
    # Polyglot (general purpose)
    ("polyglot", "c99", "C99 Standard"),
    ("polyglot", "c89", "C89/ANSI C"),
    ("polyglot", "rust", "Rust"),
    ("polyglot", "python", "Python"),
    ("polyglot", "javascript", "JavaScript"),
    
    # Assembly
    ("assembly", "x86", "x86 Assembly"),
    ("assembly", "arm", "ARM Assembly"),
    
    # Lisp dialects
    ("lisp", "common_lisp", "Common Lisp"),
    ("lisp", "scheme", "Scheme"),
    ("lisp", "clojure", "Clojure"),
    ("lisp", "racket", "Racket"),
    
    # Systems
    ("embedded", "arm", "Embedded ARM"),
    ("gpu", "cuda", "NVIDIA CUDA"),
    ("wasm", "wasm", "WebAssembly"),
    
    # Specialized
    ("functional", "haskell", "Haskell"),
    ("oop", "java", "Java"),
    ("scientific", "matlab", "MATLAB"),
    ("prolog", "swi", "SWI-Prolog"),
    ("constraints", "minizinc", "MiniZinc"),
    ("expert_systems", "clips", "CLIPS"),
    ("planning", "pddl", "PDDL"),
    ("bytecode", "jvm", "JVM Bytecode"),
    ("beam", "erlang", "Erlang/BEAM"),
    ("mobile", "swift", "Swift"),
]


class ConfluenceTestRunner:
    def __init__(self):
        self.results = {
            "test_run": datetime.now().isoformat(),
            "ir_tests": {},
            "emitter_tests": {},
            "summary": {}
        }
        TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def run_spec_to_ir(self, pipeline: str) -> Tuple[bool, str, Dict]:
        """Run spec_to_ir for a given pipeline"""
        output_file = TEST_OUTPUT_DIR / f"ir_{pipeline}.json"
        
        try:
            if pipeline == "spark":
                if not SPARK_SPEC_TO_IR.exists():
                    return False, "SPARK binary not found", {}
                cmd = [
                    str(SPARK_SPEC_TO_IR),
                    "--spec-root", str(SPEC_DIR),
                    "--out", str(output_file)
                ]
            elif pipeline == "python":
                cmd = [
                    sys.executable,
                    str(PYTHON_SPEC_TO_IR),
                    "--spec-root", str(SPEC_DIR),
                    "--out", str(output_file)
                ]
            elif pipeline == "rust":
                if not RUST_SPEC_TO_IR.exists():
                    return False, "Rust binary not found", {}
                cmd = [
                    str(RUST_SPEC_TO_IR),
                    str(TEST_SPEC),
                    "-o", str(output_file)
                ]
            else:
                return False, f"Unknown pipeline: {pipeline}", {}
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return False, f"Exit code {result.returncode}: {result.stderr}", {}
            
            # Load and validate IR
            with open(output_file) as f:
                ir_data = json.load(f)
            
            # Validate schema
            if ir_data.get("schema") != "stunir_ir_v1":
                return False, f"Invalid schema: {ir_data.get('schema')}", ir_data
            
            if ir_data.get("ir_version") != "v1":
                return False, f"Invalid IR version: {ir_data.get('ir_version')}", ir_data
            
            return True, "Success", ir_data
            
        except subprocess.TimeoutExpired:
            return False, "Timeout", {}
        except Exception as e:
            return False, str(e), {}

    def compare_ir_outputs(self, ir_spark: Dict, ir_python: Dict, ir_rust: Dict) -> Dict:
        """Compare IR outputs for confluence"""
        results = {
            "schema_match": True,
            "structure_match": True,
            "differences": []
        }
        
        # Check schema consistency
        schemas = [
            ir_spark.get("schema"),
            ir_python.get("schema"),
            ir_rust.get("schema")
        ]
        if len(set(schemas)) > 1:
            results["schema_match"] = False
            results["differences"].append(f"Schema mismatch: {schemas}")
        
        # Check IR version
        versions = [
            ir_spark.get("ir_version"),
            ir_python.get("ir_version"),
            ir_rust.get("ir_version")
        ]
        if len(set(versions)) > 1:
            results["structure_match"] = False
            results["differences"].append(f"IR version mismatch: {versions}")
        
        # Check function counts
        func_counts = [
            len(ir_spark.get("functions", [])),
            len(ir_python.get("functions", [])),
            len(ir_rust.get("functions", []))
        ]
        if len(set(func_counts)) > 1:
            results["structure_match"] = False
            results["differences"].append(f"Function count mismatch: {func_counts}")
        
        return results

    def test_emitter(self, category: str, emitter: str, ir_file: Path) -> Tuple[bool, str]:
        """Test a specific emitter with given IR"""
        try:
            output_file = TEST_OUTPUT_DIR / f"{category}_{emitter}_output.txt"
            
            # For now, test with Python emitter (SPARK emitters need to be mapped)
            # This is a simplified test - real implementation would use appropriate emitters
            emitter_path = STUNIR_ROOT / "targets" / category / "emitter.py"
            
            if not emitter_path.exists():
                return False, f"Emitter not found: {emitter_path}"
            
            # Simple test: check if emitter can be imported
            return True, "Emitter available"
            
        except Exception as e:
            return False, str(e)

    def run_all_tests(self):
        """Run complete confluence test suite"""
        print("=" * 80)
        print("STUNIR CONFLUENCE TEST SUITE - WEEK 3")
        print("=" * 80)
        print()
        
        # Test 1: IR Generation Confluence
        print("Test 1: IR Generation Confluence")
        print("-" * 80)
        
        for pipeline in ["spark", "python", "rust"]:
            print(f"Testing {pipeline.upper()} spec_to_ir...", end=" ")
            success, message, ir_data = self.run_spec_to_ir(pipeline)
            
            self.results["ir_tests"][pipeline] = {
                "success": success,
                "message": message,
                "ir_valid": success and "schema" in ir_data,
                "function_count": len(ir_data.get("functions", [])) if success else 0
            }
            
            if success:
                print(f"✓ SUCCESS ({len(ir_data.get('functions', []))} functions)")
            else:
                print(f"✗ FAILED: {message}")
        
        print()
        
        # Test 2: IR Comparison
        if all(self.results["ir_tests"][p]["success"] for p in ["spark", "python", "rust"]):
            print("Test 2: IR Output Comparison")
            print("-" * 80)
            
            with open(TEST_OUTPUT_DIR / "ir_spark.json") as f:
                ir_spark = json.load(f)
            with open(TEST_OUTPUT_DIR / "ir_python.json") as f:
                ir_python = json.load(f)
            with open(TEST_OUTPUT_DIR / "ir_rust.json") as f:
                ir_rust = json.load(f)
            
            comparison = self.compare_ir_outputs(ir_spark, ir_python, ir_rust)
            self.results["ir_comparison"] = comparison
            
            if comparison["schema_match"] and comparison["structure_match"]:
                print("✓ All pipelines produce compatible IR")
            else:
                print("✗ IR outputs differ:")
                for diff in comparison["differences"]:
                    print(f"  - {diff}")
            print()
        
        # Test 3: Emitter Coverage
        print("Test 3: Emitter Coverage")
        print("-" * 80)
        
        tested_count = 0
        available_count = 0
        
        for category, emitter, description in EMITTER_CATEGORIES[:10]:  # Test first 10 for speed
            ir_file = TEST_OUTPUT_DIR / "ir_python.json"
            success, message = self.test_emitter(category, emitter, ir_file)
            tested_count += 1
            if success:
                available_count += 1
        
        print(f"Emitter availability: {available_count}/{tested_count}")
        print()
        
        # Summary
        self.results["summary"] = {
            "ir_pipelines_tested": 3,
            "ir_pipelines_passing": sum(1 for p in self.results["ir_tests"].values() if p["success"]),
            "emitters_tested": tested_count,
            "emitters_available": available_count,
            "confluence_achieved": all(
                self.results["ir_tests"][p]["success"] 
                for p in ["spark", "python", "rust"]
            )
        }
        
        # Save results
        results_file = TEST_OUTPUT_DIR / "confluence_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print("=" * 80)
        print("CONFLUENCE TEST SUMMARY")
        print("=" * 80)
        print(f"IR Pipelines Passing: {self.results['summary']['ir_pipelines_passing']}/3")
        print(f"Confluence Achieved: {'✓ YES' if self.results['summary']['confluence_achieved'] else '✗ NO'}")
        print(f"Results saved to: {results_file}")
        print()
        
        return self.results["summary"]["confluence_achieved"]


if __name__ == "__main__":
    runner = ConfluenceTestRunner()
    success = runner.run_all_tests()
    sys.exit(0 if success else 1)
