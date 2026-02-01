#!/usr/bin/env python3
"""
STUNIR SPARK Pipeline Integration Tests - v0.8.6
End-to-end tests for the complete SPARK pipeline
"""
import json
import sys
import os
import subprocess
import tempfile
from pathlib import Path

TESTS_PASSED = 0
TESTS_FAILED = 0
TESTS_TOTAL = 0

def run_test(name, test_func):
    global TESTS_PASSED, TESTS_FAILED, TESTS_TOTAL
    TESTS_TOTAL += 1
    try:
        result = test_func()
        if result:
            TESTS_PASSED += 1
            print(f"  ✓ {name}")
            return True
        else:
            TESTS_FAILED += 1
            print(f"  ✗ {name}")
            return False
    except Exception as e:
        TESTS_FAILED += 1
        print(f"  ✗ {name}: {e}")
        return False


class TestPipeline:
    def __init__(self, project_root):
        self.project_root = project_root
        self.spark_dir = os.path.join(project_root, "tools", "spark")
        self.spec_to_ir = os.path.join(self.spark_dir, "bin", "stunir_spec_to_ir_main")
        self.ir_to_code = os.path.join(self.spark_dir, "bin", "stunir_ir_to_code_main")
        
        if not os.path.exists(self.spec_to_ir):
            self.spec_to_ir = None
            self.ir_to_code = None
    
    def run_pipeline(self, spec, output_dir):
        """Run the full spec -> IR -> code pipeline."""
        spec_dir = os.path.join(output_dir, "spec")
        os.makedirs(spec_dir, exist_ok=True)
        spec_file = os.path.join(spec_dir, f"{spec['module']}.json")
        with open(spec_file, 'w') as f:
            json.dump(spec, f)
        
        subprocess.run([self.spec_to_ir, spec_dir, output_dir], capture_output=True, timeout=30)
        
        ir_file = os.path.join(output_dir, "ir.json")
        if not os.path.exists(ir_file):
            return None, None
        
        subprocess.run([self.ir_to_code, ir_file, output_dir], capture_output=True, timeout=30)
        
        with open(ir_file) as f:
            ir = json.load(f)
        
        code = None
        for f in os.listdir(output_dir):
            if f.endswith('.c') and 'spec' not in f:
                with open(os.path.join(output_dir, f)) as fp:
                    code = fp.read()
                break
        
        return ir, code
    
    # ==================== v0.8.4 Feature Integration Tests ====================
    
    def test_break_while(self):
        spec = {
            "module": "break_while_test",
            "functions": [{
                "name": "find_first",
                "returns": "i32",
                "params": [{"name": "max", "type": "i32"}, {"name": "target", "type": "i32"}],
                "body": [
                    {"type": "assign", "target": "i", "value": "0"},
                    {"type": "assign", "target": "result", "value": "-1"},
                    {"type": "while", "condition": "i < max", "body": [
                        {"type": "if", "condition": "i == target", "then": [
                            {"type": "assign", "target": "result", "value": "i"},
                            {"type": "break"}
                        ]},
                        {"type": "assign", "target": "i", "value": "i + 1"}
                    ]},
                    {"type": "return", "value": "result"}
                ]
            }]
        }
        with tempfile.TemporaryDirectory() as td:
            ir, code = self.run_pipeline(spec, td)
            return ir is not None and code is not None and "break" in code
    
    def test_break_nested(self):
        spec = {
            "module": "break_nested_test",
            "functions": [{
                "name": "find_match",
                "returns": "i32",
                "params": [],
                "body": [
                    {"type": "assign", "target": "found", "value": "0"},
                    {"type": "while", "condition": "found == 0", "body": [
                        {"type": "assign", "target": "i", "value": "0"},
                        {"type": "while", "condition": "i < 10", "body": [
                            {"type": "if", "condition": "i == 5", "then": [
                                {"type": "assign", "target": "found", "value": "1"},
                                {"type": "break"}
                            ]},
                            {"type": "assign", "target": "i", "value": "i + 1"}
                        ]},
                        {"type": "if", "condition": "found == 1", "then": [{"type": "break"}]}
                    ]},
                    {"type": "return", "value": "found"}
                ]
            }]
        }
        with tempfile.TemporaryDirectory() as td:
            ir, code = self.run_pipeline(spec, td)
            return ir is not None and code is not None
    
    def test_continue_for(self):
        spec = {
            "module": "continue_for_test",
            "functions": [{
                "name": "sum_odd",
                "returns": "i32",
                "params": [{"name": "n", "type": "i32"}],
                "body": [
                    {"type": "assign", "target": "sum", "value": "0"},
                    {"type": "for", "init": "i = 0", "condition": "i < n", "update": "i = i + 1", "body": [
                        {"type": "if", "condition": "i % 2 == 0", "then": [{"type": "continue"}]},
                        {"type": "assign", "target": "sum", "value": "sum + i"}
                    ]},
                    {"type": "return", "value": "sum"}
                ]
            }]
        }
        with tempfile.TemporaryDirectory() as td:
            ir, code = self.run_pipeline(spec, td)
            return ir is not None and code is not None and "continue" in code
    
    def test_switch_simple(self):
        spec = {
            "module": "switch_simple_test",
            "functions": [{
                "name": "get_day_type",
                "returns": "i32",
                "params": [{"name": "day", "type": "i32"}],
                "body": [
                    {"type": "assign", "target": "result", "value": "0"},
                    {"type": "switch", "expr": "day", "cases": [
                        {"value": 1, "body": [{"type": "assign", "target": "result", "value": "1"}, {"type": "break"}]},
                        {"value": 2, "body": [{"type": "assign", "target": "result", "value": "1"}, {"type": "break"}]},
                        {"value": 6, "body": [{"type": "assign", "target": "result", "value": "2"}, {"type": "break"}]},
                        {"value": 7, "body": [{"type": "assign", "target": "result", "value": "2"}, {"type": "break"}]}
                    ], "default": [{"type": "assign", "target": "result", "value": "1"}]},
                    {"type": "return", "value": "result"}
                ]
            }]
        }
        with tempfile.TemporaryDirectory() as td:
            ir, code = self.run_pipeline(spec, td)
            return ir is not None and code is not None and "switch" in code
    
    def test_switch_fallthrough(self):
        spec = {
            "module": "switch_fall_test",
            "functions": [{
                "name": "is_weekend",
                "returns": "i32",
                "params": [{"name": "day", "type": "i32"}],
                "body": [
                    {"type": "switch", "expr": "day", "cases": [
                        {"value": 6, "body": []},
                        {"value": 7, "body": [{"type": "return", "value": "1"}]}
                    ], "default": [{"type": "return", "value": "0"}]}
                ]
            }]
        }
        with tempfile.TemporaryDirectory() as td:
            ir, code = self.run_pipeline(spec, td)
            return ir is not None and code is not None
    
    def test_combined_features(self):
        spec = {
            "module": "combined_test",
            "functions": [{
                "name": "process",
                "returns": "i32",
                "params": [{"name": "mode", "type": "i32"}, {"name": "limit", "type": "i32"}],
                "body": [
                    {"type": "assign", "target": "result", "value": "0"},
                    {"type": "switch", "expr": "mode", "cases": [
                        {"value": 1, "body": [
                            {"type": "for", "init": "i = 0", "condition": "i < limit", "update": "i = i + 1", "body": [
                                {"type": "if", "condition": "i % 2 == 0", "then": [{"type": "continue"}]},
                                {"type": "assign", "target": "result", "value": "result + i"},
                                {"type": "if", "condition": "result > 100", "then": [{"type": "break"}]}
                            ]},
                            {"type": "break"}
                        ]},
                        {"value": 2, "body": [
                            {"type": "assign", "target": "result", "value": "limit * 2"},
                            {"type": "break"}
                        ]}
                    ], "default": [{"type": "assign", "target": "result", "value": "-1"}]},
                    {"type": "return", "value": "result"}
                ]
            }]
        }
        with tempfile.TemporaryDirectory() as td:
            ir, code = self.run_pipeline(spec, td)
            return ir is not None and code is not None
    
    # ==================== v0.7.x Nested Control Flow Tests ====================
    
    def test_nested_if_while(self):
        spec = {
            "module": "nested_if_while",
            "functions": [{
                "name": "test",
                "returns": "i32",
                "params": [{"name": "n", "type": "i32"}],
                "body": [
                    {"type": "assign", "target": "sum", "value": "0"},
                    {"type": "assign", "target": "i", "value": "0"},
                    {"type": "while", "condition": "i < n", "body": [
                        {"type": "if", "condition": "i % 2 == 0", "then": [
                            {"type": "assign", "target": "sum", "value": "sum + i"}
                        ]},
                        {"type": "assign", "target": "i", "value": "i + 1"}
                    ]},
                    {"type": "return", "value": "sum"}
                ]
            }]
        }
        with tempfile.TemporaryDirectory() as td:
            ir, code = self.run_pipeline(spec, td)
            return ir is not None and code is not None
    
    def test_nested_while_if(self):
        spec = {
            "module": "nested_while_if",
            "functions": [{
                "name": "test",
                "returns": "i32",
                "params": [{"name": "run", "type": "i32"}, {"name": "n", "type": "i32"}],
                "body": [
                    {"type": "assign", "target": "result", "value": "0"},
                    {"type": "if", "condition": "run != 0", "then": [
                        {"type": "assign", "target": "i", "value": "0"},
                        {"type": "while", "condition": "i < n", "body": [
                            {"type": "assign", "target": "result", "value": "result + 1"},
                            {"type": "assign", "target": "i", "value": "i + 1"}
                        ]}
                    ]},
                    {"type": "return", "value": "result"}
                ]
            }]
        }
        with tempfile.TemporaryDirectory() as td:
            ir, code = self.run_pipeline(spec, td)
            return ir is not None and code is not None
    
    def test_triple_nested(self):
        spec = {
            "module": "triple_nested",
            "functions": [{
                "name": "matrix_sum",
                "returns": "i32",
                "params": [],
                "body": [
                    {"type": "assign", "target": "sum", "value": "0"},
                    {"type": "for", "init": "i = 0", "condition": "i < 3", "update": "i = i + 1", "body": [
                        {"type": "for", "init": "j = 0", "condition": "j < 3", "update": "j = j + 1", "body": [
                            {"type": "for", "init": "k = 0", "condition": "k < 3", "update": "k = k + 1", "body": [
                                {"type": "assign", "target": "sum", "value": "sum + 1"}
                            ]}
                        ]}
                    ]},
                    {"type": "return", "value": "sum"}
                ]
            }]
        }
        with tempfile.TemporaryDirectory() as td:
            ir, code = self.run_pipeline(spec, td)
            return ir is not None and code is not None
    
    # ==================== Determinism Tests ====================
    
    def test_deterministic_ir(self):
        spec = {
            "module": "determinism_test",
            "functions": [{
                "name": "test",
                "returns": "i32",
                "params": [],
                "body": [{"type": "return", "value": "42"}]
            }]
        }
        
        ir_hashes = []
        for _ in range(3):
            with tempfile.TemporaryDirectory() as td:
                ir, _ = self.run_pipeline(spec, td)
                if ir:
                    ir_hashes.append(json.dumps(ir, sort_keys=True))
        
        return len(set(ir_hashes)) == 1
    
    def test_deterministic_code(self):
        spec = {
            "module": "code_determinism",
            "functions": [{
                "name": "test",
                "returns": "i32",
                "params": [],
                "body": [{"type": "return", "value": "42"}]
            }]
        }
        
        code_outputs = []
        for _ in range(3):
            with tempfile.TemporaryDirectory() as td:
                _, code = self.run_pipeline(spec, td)
                if code:
                    code_outputs.append(code)
        
        return len(set(code_outputs)) == 1
    
    # ==================== Compliance Tests ====================
    
    def test_c_compilation(self):
        spec = {
            "module": "compile_test",
            "functions": [{
                "name": "add",
                "returns": "i32",
                "params": [{"name": "a", "type": "i32"}, {"name": "b", "type": "i32"}],
                "body": [{"type": "return", "value": "a + b"}]
            }]
        }
        
        with tempfile.TemporaryDirectory() as td:
            ir, code = self.run_pipeline(spec, td)
            if code is None:
                return False
            
            code_file = os.path.join(td, "test.c")
            with open(code_file, 'w') as f:
                f.write(code)
            
            result = subprocess.run(["gcc", "-c", "-fsyntax-only", code_file], capture_output=True)
            return result.returncode == 0


def main():
    if len(sys.argv) < 2:
        print("Usage: test_pipeline.py <project_root>")
        sys.exit(1)
    
    project_root = sys.argv[1]
    tests = TestPipeline(project_root)
    
    if tests.spec_to_ir is None:
        print("SPARK binaries not found. Build with: gprbuild -P stunir_tools.gpr")
        print("0 passed, 0 failed, 0 total")
        sys.exit(0)
    
    print("Running SPARK pipeline integration tests...")
    print("")
    
    test_methods = [m for m in dir(tests) if m.startswith('test_')]
    
    for method_name in sorted(test_methods):
        method = getattr(tests, method_name)
        run_test(method_name, method)
    
    print("")
    print(f"{TESTS_PASSED} passed, {TESTS_FAILED} failed, {TESTS_TOTAL} total")
    
    sys.exit(0 if TESTS_FAILED == 0 else 1)


if __name__ == "__main__":
    main()
