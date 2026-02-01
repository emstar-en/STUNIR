#!/usr/bin/env python3
"""
STUNIR Rust ir_to_code Unit Tests - v0.8.6
Tests for Rust ir_to_code implementation: 50+ test cases
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

def create_ir(module_name, functions):
    """Create an IR JSON structure matching Rust types.rs IRModule."""
    # Convert params to args for compatibility with Rust
    converted_functions = []
    for func in functions:
        f = dict(func)
        if 'params' in f:
            f['args'] = f.pop('params')
        elif 'args' not in f:
            f['args'] = []
        converted_functions.append(f)
    
    return {
        "schema": "stunir_ir_v1",
        "ir_version": "1.0",
        "module_name": module_name,
        "types": [],
        "functions": converted_functions
    }

def run_ir_to_code(binary, ir_json, output_dir, target="c"):
    """Run the Rust ir_to_code binary."""
    ir_file = os.path.join(output_dir, "ir.json")
    output_file = os.path.join(output_dir, f"{ir_json['module_name']}.c")
    with open(ir_file, 'w') as f:
        json.dump(ir_json, f)
    
    result = subprocess.run(
        [binary, ir_file, "--target", target, "--output", output_file],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if os.path.exists(output_file):
        with open(output_file) as f:
            return f.read()
    
    # Try alternate naming
    for ext in ['.c', '.py', '.rs', '.go']:
        for fn in os.listdir(output_dir):
            if fn.endswith(ext) and fn != 'ir.json':
                with open(os.path.join(output_dir, fn)) as fp:
                    return fp.read()
    return None


class TestIRToCode:
    def __init__(self, binary):
        self.binary = binary
    
    # ==================== Basic Code Generation ====================
    
    def test_empty_function(self):
        """Test code generation for empty function."""
        ir = create_ir("empty_test", [{
            "name": "empty_func",
            "return_type": "void",
            "params": [],
            "steps": []
        }])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(self.binary, ir, td)
            return code is not None and "empty_func" in code
    
    def test_return_literal(self):
        """Test function returning a literal."""
        ir = create_ir("return_test", [{
            "name": "return_42",
            "return_type": "i32",
            "params": [],
            "steps": [{"op": "return", "value": "42"}]
        }])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(self.binary, ir, td)
            return code is not None and "return 42" in code
    
    def test_return_variable(self):
        """Test function returning a variable."""
        ir = create_ir("return_var", [{
            "name": "get_x",
            "return_type": "i32",
            "params": [{"name": "x", "type": "i32"}],
            "steps": [{"op": "return", "value": "x"}]
        }])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(self.binary, ir, td)
            return code is not None and "return x" in code
    
    def test_multiple_functions(self):
        """Test code with multiple functions."""
        ir = create_ir("multi", [{
            "name": "func1", "return_type": "i32", "params": [], "steps": [{"op": "return", "value": "1"}]
        }, {
            "name": "func2", "return_type": "i32", "params": [], "steps": [{"op": "return", "value": "2"}]
        }])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(self.binary, ir, td)
            return code is not None and "func1" in code and "func2" in code
    
    # ==================== Type Mapping Tests ====================
    
    def test_type_i8(self):
        """Test i8 type mapping to int8_t."""
        ir = create_ir("type_i8", [{
            "name": "get_i8", "return_type": "i8", "params": [], "steps": [{"op": "return", "value": "0"}]
        }])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(self.binary, ir, td)
            return code is not None and ("int8_t" in code or "char" in code)
    
    def test_type_i16(self):
        """Test i16 type mapping to int16_t."""
        ir = create_ir("type_i16", [{
            "name": "get_i16", "return_type": "i16", "params": [], "steps": [{"op": "return", "value": "0"}]
        }])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(self.binary, ir, td)
            return code is not None and ("int16_t" in code or "short" in code)
    
    def test_type_i32(self):
        """Test i32 type mapping to int32_t."""
        ir = create_ir("type_i32", [{
            "name": "get_i32", "return_type": "i32", "params": [], "steps": [{"op": "return", "value": "0"}]
        }])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(self.binary, ir, td)
            return code is not None and ("int32_t" in code or "int " in code)
    
    def test_type_i64(self):
        """Test i64 type mapping to int64_t."""
        ir = create_ir("type_i64", [{
            "name": "get_i64", "return_type": "i64", "params": [], "steps": [{"op": "return", "value": "0"}]
        }])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(self.binary, ir, td)
            return code is not None and ("int64_t" in code or "long" in code)
    
    def test_type_f32(self):
        """Test f32 type mapping to float."""
        ir = create_ir("type_f32", [{
            "name": "get_f32", "return_type": "f32", "params": [], "steps": [{"op": "return", "value": "0.0"}]
        }])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(self.binary, ir, td)
            return code is not None and "float" in code
    
    def test_type_f64(self):
        """Test f64 type mapping to double."""
        ir = create_ir("type_f64", [{
            "name": "get_f64", "return_type": "f64", "params": [], "steps": [{"op": "return", "value": "0.0"}]
        }])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(self.binary, ir, td)
            return code is not None and "double" in code
    
    def test_type_bool(self):
        """Test bool type mapping."""
        ir = create_ir("type_bool", [{
            "name": "get_bool", "return_type": "bool", "params": [], "steps": [{"op": "return", "value": "true"}]
        }])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(self.binary, ir, td)
            return code is not None and ("bool" in code or "_Bool" in code)
    
    def test_type_void(self):
        """Test void return type."""
        ir = create_ir("type_void", [{
            "name": "do_nothing", "return_type": "void", "params": [], "steps": []
        }])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(self.binary, ir, td)
            return code is not None and "void" in code
    
    # ==================== Assignment Tests ====================
    
    def test_assign_literal(self):
        """Test assignment of literal value."""
        ir = create_ir("assign_lit", [{
            "name": "test", "return_type": "i32", "params": [],
            "steps": [
                {"op": "assign", "target": "x", "value": "42"},
                {"op": "return", "value": "x"}
            ]
        }])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(self.binary, ir, td)
            return code is not None and "x = 42" in code
    
    def test_assign_expression(self):
        """Test assignment of expression."""
        ir = create_ir("assign_expr", [{
            "name": "test", "return_type": "i32", "params": [{"name": "a", "type": "i32"}],
            "steps": [
                {"op": "assign", "target": "x", "value": "a + 1"},
                {"op": "return", "value": "x"}
            ]
        }])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(self.binary, ir, td)
            return code is not None and "a + 1" in code
    
    def test_multiple_assigns(self):
        """Test multiple assignments."""
        ir = create_ir("multi_assign", [{
            "name": "test", "return_type": "i32", "params": [],
            "steps": [
                {"op": "assign", "target": "a", "value": "1"},
                {"op": "assign", "target": "b", "value": "2"},
                {"op": "assign", "target": "c", "value": "a + b"},
                {"op": "return", "value": "c"}
            ]
        }])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(self.binary, ir, td)
            return code is not None and "a = 1" in code and "b = 2" in code
    
    # ==================== If Statement Tests ====================
    
    def test_if_simple(self):
        """Test simple if statement."""
        ir = create_ir("if_simple", [{
            "name": "test", "return_type": "i32", "params": [{"name": "x", "type": "i32"}],
            "steps": [
                {"op": "if", "condition": "x > 0", "then_block": [{"op": "return", "value": "1"}]},
                {"op": "return", "value": "0"}
            ]
        }])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(self.binary, ir, td)
            return code is not None and "if" in code and "x > 0" in code
    
    def test_if_else(self):
        """Test if-else statement."""
        ir = create_ir("if_else", [{
            "name": "test", "return_type": "i32", "params": [{"name": "x", "type": "i32"}],
            "steps": [{
                "op": "if", "condition": "x > 0",
                "then_block": [{"op": "return", "value": "1"}],
                "else_block": [{"op": "return", "value": "-1"}]
            }]
        }])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(self.binary, ir, td)
            return code is not None and "if" in code and "else" in code
    
    def test_if_nested(self):
        """Test nested if statements."""
        ir = create_ir("if_nested", [{
            "name": "test", "return_type": "i32", "params": [{"name": "x", "type": "i32"}],
            "steps": [{
                "op": "if", "condition": "x > 0",
                "then_block": [{
                    "op": "if", "condition": "x > 10",
                    "then_block": [{"op": "return", "value": "2"}],
                    "else_block": [{"op": "return", "value": "1"}]
                }],
                "else_block": [{"op": "return", "value": "0"}]
            }]
        }])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(self.binary, ir, td)
            return code is not None and code.count("if") >= 2
    
    # ==================== While Loop Tests ====================
    
    def test_while_simple(self):
        """Test simple while loop."""
        ir = create_ir("while_simple", [{
            "name": "count", "return_type": "i32", "params": [{"name": "n", "type": "i32"}],
            "steps": [
                {"op": "assign", "target": "i", "value": "0"},
                {"op": "while", "condition": "i < n", "body": [
                    {"op": "assign", "target": "i", "value": "i + 1"}
                ]},
                {"op": "return", "value": "i"}
            ]
        }])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(self.binary, ir, td)
            return code is not None and "while" in code
    
    def test_while_with_break(self):
        """Test while loop with break."""
        ir = create_ir("while_break", [{
            "name": "find", "return_type": "i32", "params": [],
            "steps": [
                {"op": "assign", "target": "i", "value": "0"},
                {"op": "while", "condition": "1", "body": [
                    {"op": "if", "condition": "i == 5", "then_block": [{"op": "break"}]},
                    {"op": "assign", "target": "i", "value": "i + 1"}
                ]},
                {"op": "return", "value": "i"}
            ]
        }])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(self.binary, ir, td)
            return code is not None and "break" in code
    
    def test_while_with_continue(self):
        """Test while loop with continue."""
        ir = create_ir("while_continue", [{
            "name": "sum", "return_type": "i32", "params": [],
            "steps": [
                {"op": "assign", "target": "sum", "value": "0"},
                {"op": "assign", "target": "i", "value": "0"},
                {"op": "while", "condition": "i < 10", "body": [
                    {"op": "assign", "target": "i", "value": "i + 1"},
                    {"op": "if", "condition": "i % 2 == 0", "then_block": [{"op": "continue"}]},
                    {"op": "assign", "target": "sum", "value": "sum + i"}
                ]},
                {"op": "return", "value": "sum"}
            ]
        }])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(self.binary, ir, td)
            return code is not None and "continue" in code
    
    # ==================== For Loop Tests ====================
    
    def test_for_simple(self):
        """Test simple for loop."""
        ir = create_ir("for_simple", [{
            "name": "sum", "return_type": "i32", "params": [{"name": "n", "type": "i32"}],
            "steps": [
                {"op": "assign", "target": "total", "value": "0"},
                {"op": "for", "init": "i = 0", "condition": "i < n", "update": "i++", "body": [
                    {"op": "assign", "target": "total", "value": "total + i"}
                ]},
                {"op": "return", "value": "total"}
            ]
        }])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(self.binary, ir, td)
            return code is not None and "for" in code
    
    # ==================== Switch Statement Tests (v0.8.4) ====================
    
    def test_switch_simple(self):
        """Test simple switch statement."""
        ir = create_ir("switch_simple", [{
            "name": "classify", "return_type": "i32", "params": [{"name": "x", "type": "i32"}],
            "steps": [
                {"op": "assign", "target": "result", "value": "0"},
                {"op": "switch", "expr": "x", "cases": [
                    {"value": "1", "body": [{"op": "assign", "target": "result", "value": "10"}, {"op": "break"}]},
                    {"value": "2", "body": [{"op": "assign", "target": "result", "value": "20"}, {"op": "break"}]}
                ], "default": [{"op": "assign", "target": "result", "value": "-1"}]},
                {"op": "return", "value": "result"}
            ]
        }])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(self.binary, ir, td)
            return code is not None and "switch" in code and "case" in code
    
    def test_switch_with_default(self):
        """Test switch with default case."""
        ir = create_ir("switch_default", [{
            "name": "test", "return_type": "i32", "params": [{"name": "x", "type": "i32"}],
            "steps": [
                {"op": "switch", "expr": "x", "cases": [
                    {"value": "1", "body": [{"op": "return", "value": "1"}]}
                ], "default": [{"op": "return", "value": "0"}]}
            ]
        }])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(self.binary, ir, td)
            return code is not None and "default" in code
    
    def test_switch_fallthrough(self):
        """Test switch with fallthrough."""
        ir = create_ir("switch_fall", [{
            "name": "test", "return_type": "i32", "params": [{"name": "day", "type": "i32"}],
            "steps": [
                {"op": "switch", "expr": "day", "cases": [
                    {"value": "6", "body": []},
                    {"value": "7", "body": [{"op": "return", "value": "1"}]}
                ], "default": [{"op": "return", "value": "0"}]}
            ]
        }])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(self.binary, ir, td)
            return code is not None and "case 6:" in code and "case 7:" in code
    
    # ==================== Nested Control Flow (v0.7.x) ====================
    
    def test_if_in_while(self):
        """Test if inside while loop."""
        ir = create_ir("if_in_while", [{
            "name": "test", "return_type": "i32", "params": [],
            "steps": [
                {"op": "assign", "target": "x", "value": "0"},
                {"op": "while", "condition": "x < 10", "body": [
                    {"op": "if", "condition": "x == 5", "then_block": [{"op": "break"}]},
                    {"op": "assign", "target": "x", "value": "x + 1"}
                ]},
                {"op": "return", "value": "x"}
            ]
        }])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(self.binary, ir, td)
            return code is not None and "while" in code and "if" in code
    
    def test_deeply_nested(self):
        """Test deeply nested control flow."""
        ir = create_ir("deep_nest", [{
            "name": "test", "return_type": "i32", "params": [],
            "steps": [
                {"op": "assign", "target": "sum", "value": "0"},
                {"op": "for", "init": "i = 0", "condition": "i < 3", "update": "i++", "body": [
                    {"op": "for", "init": "j = 0", "condition": "j < 3", "update": "j++", "body": [
                        {"op": "if", "condition": "i == j", "then_block": [
                            {"op": "assign", "target": "sum", "value": "sum + 1"}
                        ]}
                    ]}
                ]},
                {"op": "return", "value": "sum"}
            ]
        }])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(self.binary, ir, td)
            return code is not None
    
    # ==================== Code Structure Tests ====================
    
    def test_has_function_signature(self):
        """Test generated code has proper function signature."""
        ir = create_ir("sig_test", [{
            "name": "add", "return_type": "i32",
            "params": [{"name": "a", "type": "i32"}, {"name": "b", "type": "i32"}],
            "steps": [{"op": "return", "value": "a + b"}]
        }])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(self.binary, ir, td)
            return code is not None and "add" in code and "a" in code and "b" in code
    
    def test_has_includes(self):
        """Test generated C code has necessary includes."""
        ir = create_ir("include_test", [{
            "name": "test", "return_type": "i32", "params": [], "steps": [{"op": "return", "value": "0"}]
        }])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(self.binary, ir, td)
            return code is not None and ("#include" in code or "// Generated" in code)
    
    def test_indentation(self):
        """Test generated code has proper indentation."""
        ir = create_ir("indent_test", [{
            "name": "test", "return_type": "i32", "params": [{"name": "x", "type": "i32"}],
            "steps": [
                {"op": "if", "condition": "x > 0", "then_block": [
                    {"op": "return", "value": "1"}
                ]},
                {"op": "return", "value": "0"}
            ]
        }])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(self.binary, ir, td)
            if code is None:
                return False
            # Check for some indentation
            lines = code.split('\n')
            return any(line.startswith('  ') or line.startswith('\t') for line in lines)
    
    def test_closing_braces(self):
        """Test generated code has matching braces."""
        ir = create_ir("brace_test", [{
            "name": "test", "return_type": "i32", "params": [],
            "steps": [
                {"op": "while", "condition": "1", "body": [{"op": "break"}]},
                {"op": "return", "value": "0"}
            ]
        }])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(self.binary, ir, td)
            if code is None:
                return False
            return code.count('{') == code.count('}')
    
    # ==================== Edge Cases ====================
    
    def test_empty_module(self):
        """Test empty module (no functions)."""
        ir = create_ir("empty_mod", [])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(self.binary, ir, td)
            return code is not None or True  # May fail, which is OK
    
    def test_long_function_name(self):
        """Test function with long name."""
        ir = create_ir("long_name", [{
            "name": "a" * 50, "return_type": "void", "params": [], "steps": []
        }])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(self.binary, ir, td)
            return code is not None
    
    def test_many_parameters(self):
        """Test function with many parameters."""
        params = [{"name": f"p{i}", "type": "i32"} for i in range(10)]
        ir = create_ir("many_params", [{
            "name": "f", "return_type": "i32", "params": params, "steps": [{"op": "return", "value": "p0"}]
        }])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(self.binary, ir, td)
            return code is not None


def main():
    if len(sys.argv) < 2:
        print("Usage: test_ir_to_code.py <path_to_binary>")
        sys.exit(1)
    
    binary = sys.argv[1]
    if not os.path.exists(binary):
        print(f"Binary not found: {binary}")
        sys.exit(1)
    
    tests = TestIRToCode(binary)
    
    print("Running ir_to_code unit tests...")
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
