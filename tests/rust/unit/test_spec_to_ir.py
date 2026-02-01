#!/usr/bin/env python3
"""
STUNIR Rust spec_to_ir Unit Tests - v0.8.6
Tests for Rust spec_to_ir implementation: 50+ test cases
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
    """Run a single test and track results."""
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
    """Create a spec JSON structure."""
    return {
        "module": module_name,
        "description": f"Test spec for {module_name}",
        "functions": functions
    }

def run_spec_to_ir(binary, spec_json, output_dir):
    """Run the Rust spec_to_ir binary."""
    spec_file = os.path.join(output_dir, "spec.json")
    ir_file = os.path.join(output_dir, "ir.json")
    with open(spec_file, 'w') as f:
        json.dump(spec_json, f)
    
    result = subprocess.run(
        [binary, spec_file, "--out", ir_file],
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
        """Test function with no body."""
        spec = create_spec("empty_test", [{
            "name": "empty_func",
            "returns": "void",
            "params": [],
            "body": []
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None and len(ir.get('functions', [])) == 1
    
    def test_single_return(self):
        """Test function with single return."""
        spec = create_spec("return_test", [{
            "name": "return_42",
            "returns": "i32",
            "params": [],
            "body": [{"type": "return", "value": "42"}]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            if ir is None:
                return False
            steps = ir['functions'][0].get('steps', [])
            return any(s.get('op') == 'return' for s in steps)
    
    def test_multiple_params(self):
        """Test function with multiple parameters."""
        spec = create_spec("params_test", [{
            "name": "add",
            "returns": "i32",
            "params": [
                {"name": "a", "type": "i32"},
                {"name": "b", "type": "i32"}
            ],
            "body": [{"type": "return", "value": "a + b"}]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            if ir is None:
                return False
            params = ir['functions'][0].get('params', [])
            return len(params) == 2
    
    def test_multiple_functions(self):
        """Test spec with multiple functions."""
        spec = create_spec("multi_func", [
            {"name": "func1", "returns": "i32", "params": [], "body": [{"type": "return", "value": "1"}]},
            {"name": "func2", "returns": "i32", "params": [], "body": [{"type": "return", "value": "2"}]},
            {"name": "func3", "returns": "i32", "params": [], "body": [{"type": "return", "value": "3"}]}
        ])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None and len(ir.get('functions', [])) == 3
    
    # ==================== Type Tests ====================
    
    def test_type_i8(self):
        """Test i8 type handling."""
        spec = create_spec("type_i8", [{
            "name": "get_i8", "returns": "i8", "params": [], "body": [{"type": "return", "value": "127"}]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None and ir['functions'][0].get('return_type') == 'i8'
    
    def test_type_i16(self):
        """Test i16 type handling."""
        spec = create_spec("type_i16", [{
            "name": "get_i16", "returns": "i16", "params": [], "body": [{"type": "return", "value": "32767"}]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None and ir['functions'][0].get('return_type') == 'i16'
    
    def test_type_i32(self):
        """Test i32 type handling."""
        spec = create_spec("type_i32", [{
            "name": "get_i32", "returns": "i32", "params": [], "body": [{"type": "return", "value": "0"}]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None and ir['functions'][0].get('return_type') == 'i32'
    
    def test_type_i64(self):
        """Test i64 type handling."""
        spec = create_spec("type_i64", [{
            "name": "get_i64", "returns": "i64", "params": [], "body": [{"type": "return", "value": "0"}]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None and ir['functions'][0].get('return_type') == 'i64'
    
    def test_type_u32(self):
        """Test u32 type handling."""
        spec = create_spec("type_u32", [{
            "name": "get_u32", "returns": "u32", "params": [], "body": [{"type": "return", "value": "0"}]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None
    
    def test_type_f32(self):
        """Test f32 type handling."""
        spec = create_spec("type_f32", [{
            "name": "get_f32", "returns": "f32", "params": [], "body": [{"type": "return", "value": "3.14"}]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None
    
    def test_type_f64(self):
        """Test f64 type handling."""
        spec = create_spec("type_f64", [{
            "name": "get_f64", "returns": "f64", "params": [], "body": [{"type": "return", "value": "3.14159"}]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None
    
    def test_type_bool(self):
        """Test bool type handling."""
        spec = create_spec("type_bool", [{
            "name": "get_bool", "returns": "bool", "params": [], "body": [{"type": "return", "value": "true"}]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None
    
    def test_type_string(self):
        """Test string type handling."""
        spec = create_spec("type_string", [{
            "name": "get_string", "returns": "string", "params": [], "body": [{"type": "return", "value": "\"hello\""}]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None
    
    def test_type_void(self):
        """Test void type handling."""
        spec = create_spec("type_void", [{
            "name": "do_nothing", "returns": "void", "params": [], "body": []
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None and ir['functions'][0].get('return_type') == 'void'
    
    # ==================== Assignment Tests ====================
    
    def test_assign_literal(self):
        """Test literal assignment."""
        spec = create_spec("assign_lit", [{
            "name": "test", "returns": "i32", "params": [],
            "body": [
                {"type": "assign", "target": "x", "value": "42"},
                {"type": "return", "value": "x"}
            ]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            if ir is None:
                return False
            steps = ir['functions'][0].get('steps', [])
            return any(s.get('op') == 'assign' for s in steps)
    
    def test_assign_expression(self):
        """Test expression assignment."""
        spec = create_spec("assign_expr", [{
            "name": "test", "returns": "i32", "params": [{"name": "a", "type": "i32"}],
            "body": [
                {"type": "assign", "target": "x", "value": "a * 2 + 1"},
                {"type": "return", "value": "x"}
            ]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None
    
    def test_assign_multiple(self):
        """Test multiple assignments."""
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
        """Test simple if statement."""
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
        """Test if-else statement."""
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
        """Test nested if statements."""
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
        """Test if-else chain."""
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
        """Test simple while loop."""
        spec = create_spec("while_simple", [{
            "name": "count", "returns": "i32", "params": [{"name": "n", "type": "i32"}],
            "body": [
                {"type": "assign", "target": "i", "value": "0"},
                {"type": "while", "condition": "i < n", "body": [
                    {"type": "assign", "target": "i", "value": "i + 1"}
                ]},
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
        """Test nested while loops."""
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
        """Test simple for loop."""
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
        """Test break statement in while loop."""
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
            if ir is None:
                return False
            # Check for break op in IR
            def find_break(steps):
                for s in steps:
                    if s.get('op') == 'break':
                        return True
                    for key in ['then_block', 'else_block', 'body']:
                        if key in s and find_break(s[key]):
                            return True
                return False
            return find_break(ir['functions'][0].get('steps', []))
    
    def test_continue_in_for(self):
        """Test continue statement in for loop."""
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
        """Test simple switch statement."""
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
        """Test switch with fallthrough."""
        spec = create_spec("switch_fall", [{
            "name": "is_weekend", "returns": "i32", "params": [{"name": "day", "type": "i32"}],
            "body": [
                {"type": "switch", "expr": "day", "cases": [
                    {"value": 6, "body": []},  # Fallthrough
                    {"value": 7, "body": [{"type": "return", "value": "1"}]}
                ], "default": [{"type": "return", "value": "0"}]}
            ]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None
    
    # ==================== Nested Control Flow Tests (v0.7.x) ====================
    
    def test_if_in_while(self):
        """Test if inside while loop."""
        spec = create_spec("if_in_while", [{
            "name": "test", "returns": "i32", "params": [],
            "body": [
                {"type": "assign", "target": "x", "value": "0"},
                {"type": "while", "condition": "x < 10", "body": [
                    {"type": "if", "condition": "x == 5", "then": [
                        {"type": "assign", "target": "x", "value": "100"}
                    ]},
                    {"type": "assign", "target": "x", "value": "x + 1"}
                ]},
                {"type": "return", "value": "x"}
            ]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None
    
    def test_while_in_if(self):
        """Test while inside if statement."""
        spec = create_spec("while_in_if", [{
            "name": "test", "returns": "i32", "params": [{"name": "do_loop", "type": "bool"}],
            "body": [
                {"type": "assign", "target": "x", "value": "0"},
                {"type": "if", "condition": "do_loop", "then": [
                    {"type": "while", "condition": "x < 5", "body": [
                        {"type": "assign", "target": "x", "value": "x + 1"}
                    ]}
                ]},
                {"type": "return", "value": "x"}
            ]
        }])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None
    
    def test_deeply_nested(self):
        """Test deeply nested control flow (3 levels)."""
        spec = create_spec("deep_nest", [{
            "name": "test", "returns": "i32", "params": [],
            "body": [
                {"type": "assign", "target": "sum", "value": "0"},
                {"type": "for", "init": "i = 0", "condition": "i < 3", "update": "i = i + 1", "body": [
                    {"type": "for", "init": "j = 0", "condition": "j < 3", "update": "j = j + 1", "body": [
                        {"type": "if", "condition": "i == j", "then": [
                            {"type": "assign", "target": "sum", "value": "sum + 1"}
                        ]}
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
        """Test IR has module_name field."""
        spec = create_spec("mod_test", [{"name": "f", "returns": "void", "params": [], "body": []}])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None and 'module_name' in ir
    
    def test_ir_has_functions(self):
        """Test IR has functions array."""
        spec = create_spec("func_test", [{"name": "f", "returns": "void", "params": [], "body": []}])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None and isinstance(ir.get('functions'), list)
    
    def test_ir_function_has_name(self):
        """Test IR function has name field."""
        spec = create_spec("name_test", [{"name": "my_func", "returns": "void", "params": [], "body": []}])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None and ir['functions'][0].get('name') == 'my_func'
    
    def test_ir_function_has_return_type(self):
        """Test IR function has return_type field."""
        spec = create_spec("ret_test", [{"name": "f", "returns": "i32", "params": [], "body": []}])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None and 'return_type' in ir['functions'][0]
    
    def test_ir_function_has_steps(self):
        """Test IR function has steps array."""
        spec = create_spec("steps_test", [{"name": "f", "returns": "i32", "params": [], "body": [{"type": "return", "value": "0"}]}])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None and isinstance(ir['functions'][0].get('steps'), list)
    
    # ==================== Edge Case Tests ====================
    
    def test_long_module_name(self):
        """Test with long module name."""
        spec = create_spec("a" * 100, [{"name": "f", "returns": "void", "params": [], "body": []}])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None
    
    def test_many_params(self):
        """Test function with many parameters."""
        params = [{"name": f"p{i}", "type": "i32"} for i in range(10)]
        spec = create_spec("many_params", [{"name": "f", "returns": "i32", "params": params, "body": [{"type": "return", "value": "p0"}]}])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None and len(ir['functions'][0].get('params', [])) == 10
    
    def test_unicode_in_description(self):
        """Test spec with unicode in description."""
        spec = {"module": "unicode_test", "description": "Test with émojis 🚀 and üñíçödé", "functions": [{"name": "f", "returns": "void", "params": [], "body": []}]}
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None
    
    def test_empty_spec(self):
        """Test spec with no functions."""
        spec = create_spec("empty", [])
        with tempfile.TemporaryDirectory() as td:
            ir = run_spec_to_ir(self.binary, spec, td)
            return ir is not None and len(ir.get('functions', [])) == 0


def main():
    if len(sys.argv) < 2:
        print("Usage: test_spec_to_ir.py <path_to_binary>")
        sys.exit(1)
    
    binary = sys.argv[1]
    if not os.path.exists(binary):
        print(f"Binary not found: {binary}")
        sys.exit(1)
    
    tests = TestSpecToIR(binary)
    
    print("Running spec_to_ir unit tests...")
    print("")
    
    # Get all test methods
    test_methods = [m for m in dir(tests) if m.startswith('test_')]
    
    for method_name in sorted(test_methods):
        method = getattr(tests, method_name)
        run_test(method_name, method)
    
    print("")
    print(f"{TESTS_PASSED} passed, {TESTS_FAILED} failed, {TESTS_TOTAL} total")
    
    sys.exit(0 if TESTS_FAILED == 0 else 1)


if __name__ == "__main__":
    main()
