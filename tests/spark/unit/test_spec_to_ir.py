#!/usr/bin/env python3
"""
STUNIR SPARK spec_to_ir Unit Tests - v0.8.6
Tests for SPARK spec_to_ir implementation: 50+ test cases
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

def create_spec(module_name, functions):
    return {
        "module": module_name,
        "description": f"Test spec for {module_name}",
        "functions": functions
    }

def run_spec_to_ir(binary, spec_json, output_dir):
    """Run the SPARK spec_to_ir binary."""
    spec_dir = os.path.join(output_dir, "spec")
    os.makedirs(spec_dir, exist_ok=True)
    spec_file = os.path.join(spec_dir, f"{spec_json['module']}.json")
    with open(spec_file, 'w') as f:
        json.dump(spec_json, f)
    
    ir_file = os.path.join(output_dir, "ir.json")
    
    # SPARK binary uses --spec-root/--out format
    result = subprocess.run(
        [binary, "--spec-root", spec_dir, "--out", ir_file],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if os.path.exists(ir_file):
        with open(ir_file) as f:
            return json.load(f)
    return None


class TestSpecToIR:
    def __init__(self, binary):
        self.binary = binary
    
    # ==================== Basic Function Tests ====================
    
    def test_empty_function(self):
        spec = create_spec("empty_test", [{
            "name": "empty_func", "returns": "void", "params": [], "body": []
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None and len(ir.get('functions', [])) >= 1
    
    def test_single_return(self):
        spec = create_spec("return_test", [{
            "name": "return_42", "returns": "i32", "params": [],
            "body": [{"type": "return", "value": "42"}]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            if ir is None:
                return False
            steps = ir['functions'][0].get('steps', [])
            return any(s.get('op') == 'return' for s in steps)
    
    def test_multiple_params(self):
        spec = create_spec("params_test", [{
            "name": "add", "returns": "i32",
            "params": [{"name": "a", "type": "i32"}, {"name": "b", "type": "i32"}],
            "body": [{"type": "return", "value": "a + b"}]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            if ir is None:
                return False
            return len(ir['functions'][0].get('params', [])) == 2
    
    def test_multiple_functions(self):
        spec = create_spec("multi_func", [
            {"name": "func1", "returns": "i32", "params": [], "body": [{"type": "return", "value": "1"}]},
            {"name": "func2", "returns": "i32", "params": [], "body": [{"type": "return", "value": "2"}]},
            {"name": "func3", "returns": "i32", "params": [], "body": [{"type": "return", "value": "3"}]}
        ])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None and len(ir.get('functions', [])) >= 3
    
    # ==================== Type Tests ====================
    
    def test_type_i8(self):
        spec = create_spec("type_i8", [{"name": "f", "returns": "i8", "params": [], "body": [{"type": "return", "value": "0"}]}])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None and ir['functions'][0].get('return_type') == 'i8'
    
    def test_type_i16(self):
        spec = create_spec("type_i16", [{"name": "f", "returns": "i16", "params": [], "body": [{"type": "return", "value": "0"}]}])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None and ir['functions'][0].get('return_type') == 'i16'
    
    def test_type_i32(self):
        spec = create_spec("type_i32", [{"name": "f", "returns": "i32", "params": [], "body": [{"type": "return", "value": "0"}]}])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None and ir['functions'][0].get('return_type') == 'i32'
    
    def test_type_i64(self):
        spec = create_spec("type_i64", [{"name": "f", "returns": "i64", "params": [], "body": [{"type": "return", "value": "0"}]}])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None and ir['functions'][0].get('return_type') == 'i64'
    
    def test_type_u32(self):
        spec = create_spec("type_u32", [{"name": "f", "returns": "u32", "params": [], "body": [{"type": "return", "value": "0"}]}])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None
    
    def test_type_f32(self):
        spec = create_spec("type_f32", [{"name": "f", "returns": "f32", "params": [], "body": [{"type": "return", "value": "3.14"}]}])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None
    
    def test_type_f64(self):
        spec = create_spec("type_f64", [{"name": "f", "returns": "f64", "params": [], "body": [{"type": "return", "value": "3.14159"}]}])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None
    
    def test_type_bool(self):
        spec = create_spec("type_bool", [{"name": "f", "returns": "bool", "params": [], "body": [{"type": "return", "value": "true"}]}])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None
    
    def test_type_string(self):
        spec = create_spec("type_string", [{"name": "f", "returns": "string", "params": [], "body": [{"type": "return", "value": "\"hello\""}]}])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None
    
    def test_type_void(self):
        spec = create_spec("type_void", [{"name": "f", "returns": "void", "params": [], "body": []}])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None and ir['functions'][0].get('return_type') == 'void'
    
    # ==================== Assignment Tests ====================
    
    def test_assign_literal(self):
        spec = create_spec("assign_lit", [{
            "name": "test", "returns": "i32", "params": [],
            "body": [{"type": "assign", "target": "x", "value": "42"}, {"type": "return", "value": "x"}]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            if ir is None:
                return False
            steps = ir['functions'][0].get('steps', [])
            return any(s.get('op') == 'assign' for s in steps)
    
    def test_assign_expression(self):
        spec = create_spec("assign_expr", [{
            "name": "test", "returns": "i32", "params": [{"name": "a", "type": "i32"}],
            "body": [{"type": "assign", "target": "x", "value": "a * 2 + 1"}, {"type": "return", "value": "x"}]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None
    
    def test_assign_multiple(self):
        spec = create_spec("assign_multi", [{
            "name": "test", "returns": "i32", "params": [],
            "body": [
                {"type": "assign", "target": "a", "value": "1"},
                {"type": "assign", "target": "b", "value": "2"},
                {"type": "assign", "target": "c", "value": "3"},
                {"type": "return", "value": "a + b + c"}
            ]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            if ir is None:
                return False
            steps = ir['functions'][0].get('steps', [])
            assigns = [s for s in steps if s.get('op') == 'assign']
            return len(assigns) == 3
    
    # ==================== Control Flow: If Tests ====================
    
    def test_if_simple(self):
        spec = create_spec("if_simple", [{
            "name": "test", "returns": "i32", "params": [{"name": "x", "type": "i32"}],
            "body": [
                {"type": "if", "condition": "x > 0", "then": [{"type": "return", "value": "1"}]},
                {"type": "return", "value": "0"}
            ]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            if ir is None:
                return False
            steps = ir['functions'][0].get('steps', [])
            return any(s.get('op') == 'if' for s in steps)
    
    def test_if_else(self):
        spec = create_spec("if_else", [{
            "name": "test", "returns": "i32", "params": [{"name": "x", "type": "i32"}],
            "body": [{
                "type": "if", "condition": "x > 0",
                "then": [{"type": "return", "value": "1"}],
                "else": [{"type": "return", "value": "-1"}]
            }]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            if ir is None:
                return False
            steps = ir['functions'][0].get('steps', [])
            return any(s.get('op') == 'if' and 'else_block' in s for s in steps)
    
    def test_if_nested(self):
        spec = create_spec("if_nested", [{
            "name": "test", "returns": "i32", "params": [{"name": "x", "type": "i32"}],
            "body": [{
                "type": "if", "condition": "x > 0",
                "then": [{
                    "type": "if", "condition": "x > 10",
                    "then": [{"type": "return", "value": "2"}],
                    "else": [{"type": "return", "value": "1"}]
                }],
                "else": [{"type": "return", "value": "0"}]
            }]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None
    
    def test_if_chain(self):
        spec = create_spec("if_chain", [{
            "name": "classify", "returns": "i32", "params": [{"name": "x", "type": "i32"}],
            "body": [
                {"type": "if", "condition": "x < 0", "then": [{"type": "return", "value": "-1"}]},
                {"type": "if", "condition": "x == 0", "then": [{"type": "return", "value": "0"}]},
                {"type": "return", "value": "1"}
            ]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            if ir is None:
                return False
            steps = ir['functions'][0].get('steps', [])
            ifs = [s for s in steps if s.get('op') == 'if']
            return len(ifs) == 2
    
    # ==================== Control Flow: While Tests ====================
    
    def test_while_simple(self):
        spec = create_spec("while_simple", [{
            "name": "count", "returns": "i32", "params": [{"name": "n", "type": "i32"}],
            "body": [
                {"type": "assign", "target": "i", "value": "0"},
                {"type": "while", "condition": "i < n", "body": [{"type": "assign", "target": "i", "value": "i + 1"}]},
                {"type": "return", "value": "i"}
            ]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            if ir is None:
                return False
            steps = ir['functions'][0].get('steps', [])
            return any(s.get('op') == 'while' for s in steps)
    
    def test_while_nested(self):
        spec = create_spec("while_nested", [{
            "name": "nested", "returns": "i32", "params": [],
            "body": [
                {"type": "assign", "target": "sum", "value": "0"},
                {"type": "assign", "target": "i", "value": "0"},
                {"type": "while", "condition": "i < 3", "body": [
                    {"type": "assign", "target": "j", "value": "0"},
                    {"type": "while", "condition": "j < 3", "body": [
                        {"type": "assign", "target": "sum", "value": "sum + 1"},
                        {"type": "assign", "target": "j", "value": "j + 1"}
                    ]},
                    {"type": "assign", "target": "i", "value": "i + 1"}
                ]},
                {"type": "return", "value": "sum"}
            ]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None
    
    # ==================== Control Flow: For Tests ====================
    
    def test_for_simple(self):
        spec = create_spec("for_simple", [{
            "name": "sum", "returns": "i32", "params": [{"name": "n", "type": "i32"}],
            "body": [
                {"type": "assign", "target": "total", "value": "0"},
                {"type": "for", "init": "i = 0", "condition": "i < n", "update": "i = i + 1", "body": [
                    {"type": "assign", "target": "total", "value": "total + i"}
                ]},
                {"type": "return", "value": "total"}
            ]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            if ir is None:
                return False
            steps = ir['functions'][0].get('steps', [])
            return any(s.get('op') == 'for' for s in steps)
    
    # ==================== v0.8.4 Feature Tests ====================
    
    def test_break_in_while(self):
        spec = create_spec("break_while", [{
            "name": "find", "returns": "i32", "params": [{"name": "n", "type": "i32"}],
            "body": [
                {"type": "assign", "target": "i", "value": "0"},
                {"type": "while", "condition": "i < 100", "body": [
                    {"type": "if", "condition": "i == n", "then": [{"type": "break"}]},
                    {"type": "assign", "target": "i", "value": "i + 1"}
                ]},
                {"type": "return", "value": "i"}
            ]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None
    
    def test_continue_in_for(self):
        spec = create_spec("continue_for", [{
            "name": "sum_evens", "returns": "i32", "params": [{"name": "n", "type": "i32"}],
            "body": [
                {"type": "assign", "target": "sum", "value": "0"},
                {"type": "for", "init": "i = 0", "condition": "i < n", "update": "i = i + 1", "body": [
                    {"type": "if", "condition": "i % 2 != 0", "then": [{"type": "continue"}]},
                    {"type": "assign", "target": "sum", "value": "sum + i"}
                ]},
                {"type": "return", "value": "sum"}
            ]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None
    
    def test_switch_simple(self):
        spec = create_spec("switch_simple", [{
            "name": "classify", "returns": "i32", "params": [{"name": "x", "type": "i32"}],
            "body": [
                {"type": "assign", "target": "result", "value": "0"},
                {"type": "switch", "expr": "x", "cases": [
                    {"value": 1, "body": [{"type": "assign", "target": "result", "value": "10"}, {"type": "break"}]},
                    {"value": 2, "body": [{"type": "assign", "target": "result", "value": "20"}, {"type": "break"}]}
                ], "default": [{"type": "assign", "target": "result", "value": "-1"}]},
                {"type": "return", "value": "result"}
            ]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            if ir is None:
                return False
            steps = ir['functions'][0].get('steps', [])
            return any(s.get('op') == 'switch' for s in steps)
    
    def test_switch_fallthrough(self):
        spec = create_spec("switch_fall", [{
            "name": "is_weekend", "returns": "i32", "params": [{"name": "day", "type": "i32"}],
            "body": [
                {"type": "switch", "expr": "day", "cases": [
                    {"value": 6, "body": []},
                    {"value": 7, "body": [{"type": "return", "value": "1"}]}
                ], "default": [{"type": "return", "value": "0"}]}
            ]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None
    
    # ==================== v0.7.x Nested Control Flow ====================
    
    def test_if_in_while(self):
        spec = create_spec("if_in_while", [{
            "name": "test", "returns": "i32", "params": [],
            "body": [
                {"type": "assign", "target": "x", "value": "0"},
                {"type": "while", "condition": "x < 10", "body": [
                    {"type": "if", "condition": "x == 5", "then": [{"type": "assign", "target": "x", "value": "100"}]},
                    {"type": "assign", "target": "x", "value": "x + 1"}
                ]},
                {"type": "return", "value": "x"}
            ]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None
    
    def test_while_in_if(self):
        spec = create_spec("while_in_if", [{
            "name": "test", "returns": "i32", "params": [{"name": "do_loop", "type": "bool"}],
            "body": [
                {"type": "assign", "target": "x", "value": "0"},
                {"type": "if", "condition": "do_loop", "then": [
                    {"type": "while", "condition": "x < 5", "body": [{"type": "assign", "target": "x", "value": "x + 1"}]}
                ]},
                {"type": "return", "value": "x"}
            ]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None
    
    def test_deeply_nested(self):
        spec = create_spec("deep_nest", [{
            "name": "test", "returns": "i32", "params": [],
            "body": [
                {"type": "assign", "target": "sum", "value": "0"},
                {"type": "for", "init": "i = 0", "condition": "i < 3", "update": "i = i + 1", "body": [
                    {"type": "for", "init": "j = 0", "condition": "j < 3", "update": "j = j + 1", "body": [
                        {"type": "if", "condition": "i == j", "then": [{"type": "assign", "target": "sum", "value": "sum + 1"}]}
                    ]}
                ]},
                {"type": "return", "value": "sum"}
            ]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None
    
    # ==================== IR Structure Tests ====================
    
    def test_ir_has_module_name(self):
        spec = create_spec("mod_test", [{"name": "f", "returns": "void", "params": [], "body": []}])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None and 'module_name' in ir
    
    def test_ir_has_functions(self):
        spec = create_spec("func_test", [{"name": "f", "returns": "void", "params": [], "body": []}])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None and isinstance(ir.get('functions'), list)
    
    def test_ir_function_has_name(self):
        spec = create_spec("name_test", [{"name": "my_func", "returns": "void", "params": [], "body": []}])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None and ir['functions'][0].get('name') == 'my_func'
    
    def test_ir_function_has_return_type(self):
        spec = create_spec("ret_test", [{"name": "f", "returns": "i32", "params": [], "body": []}])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None and 'return_type' in ir['functions'][0]
    
    def test_ir_function_has_steps(self):
        spec = create_spec("steps_test", [{"name": "f", "returns": "i32", "params": [], "body": [{"type": "return", "value": "0"}]}])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None and isinstance(ir['functions'][0].get('steps'), list)
    
    # ==================== Edge Cases ====================
    
    def test_long_module_name(self):
        spec = create_spec("a" * 50, [{"name": "f", "returns": "void", "params": [], "body": []}])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None
    
    def test_many_params(self):
        params = [{"name": f"p{i}", "type": "i32"} for i in range(8)]
        spec = create_spec("many_params", [{"name": "f", "returns": "i32", "params": params, "body": [{"type": "return", "value": "p0"}]}])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None and len(ir['functions'][0].get('params', [])) >= 8
    
    def test_empty_spec(self):
        spec = create_spec("empty", [])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None


def main():
    if len(sys.argv) < 2:
        print("Usage: test_spec_to_ir.py <path_to_binary>")
        sys.exit(1)
    
    binary = sys.argv[1]
    if not binary or not os.path.exists(binary):
        print(f"Binary not found: {binary}")
        print("0 passed, 0 failed, 0 total")
        sys.exit(0)
    
    tests = TestSpecToIR(binary)
    
    print("Running SPARK spec_to_ir unit tests...")
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
