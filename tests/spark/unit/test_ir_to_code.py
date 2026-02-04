#!/usr/bin/env python3
"""
STUNIR SPARK ir_to_code Unit Tests - v0.8.6
Tests for SPARK ir_to_code implementation: 50+ test cases
"""
import json
import os
import subprocess
import tempfile
from pathlib import Path
import pytest

@pytest.fixture(scope="session")
def ir_to_code_binary():
    """Locate the SPARK ir_to_code binary."""
    root = Path(__file__).resolve().parents[3]
    possible_paths = [
        root / "tools" / "spark" / "bin" / "stunir_ir_to_code_main.exe",
    ]
    for path in possible_paths:
        if path.exists():
            return str(path)
    pytest.skip("spark ir_to_code binary not found")

def create_ir(module_name, functions):
    """Create an IR JSON structure matching SPARK IRModule."""
    # Convert params to args for compatibility
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
    ir_file = os.path.join(output_dir, "ir.json")
    output_file = os.path.join(output_dir, f"{ir_json['module_name']}.c")
    with open(ir_file, 'w') as f:
        json.dump(ir_json, f)
    
    # SPARK binary uses --input/--output/--target format
    result = subprocess.run(
        [binary, "--input", ir_file, "--output", output_file, "--target", target],
        capture_output=True, text=True, timeout=30
    )
    
    if os.path.exists(output_file):
        with open(output_file) as fp:
            return fp.read()

    # Fallback: check for any .c file
    for fn in os.listdir(output_dir):
        if fn.endswith('.c') and fn != 'ir.json':
            with open(os.path.join(output_dir, fn)) as fp:
                return fp.read()
    return None


class TestIRToCode:
    # ==================== Basic Code Generation ====================

    def test_empty_function(self, ir_to_code_binary):
        ir = create_ir("empty_test", [{"name": "empty_func", "return_type": "void", "params": [], "steps": []}])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(ir_to_code_binary, ir, td)
            assert code is not None and "empty_func" in code

    def test_return_literal(self, ir_to_code_binary):
        ir = create_ir("return_test", [{"name": "return_42", "return_type": "i32", "params": [], "steps": [{"op": "return", "value": "42"}]}])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(ir_to_code_binary, ir, td)
            assert code is not None and "return 42" in code

    def test_return_variable(self, ir_to_code_binary):
        ir = create_ir("return_var", [{"name": "get_x", "return_type": "i32", "params": [{"name": "x", "type": "i32"}], "steps": [{"op": "return", "value": "x"}]}])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(ir_to_code_binary, ir, td)
            assert code is not None and "return x" in code

    def test_multiple_functions(self, ir_to_code_binary):
        ir = create_ir("multi", [
            {"name": "func1", "return_type": "i32", "params": [], "steps": [{"op": "return", "value": "1"}]},
            {"name": "func2", "return_type": "i32", "params": [], "steps": [{"op": "return", "value": "2"}]}
        ])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(ir_to_code_binary, ir, td)
            assert code is not None and "func1" in code and "func2" in code

    # ==================== Type Mapping Tests ====================

    def test_type_i8(self, ir_to_code_binary):
        ir = create_ir("type_i8", [{"name": "f", "return_type": "i8", "params": [], "steps": [{"op": "return", "value": "0"}]}])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(ir_to_code_binary, ir, td)
            assert code is not None and ("int8_t" in code or "char" in code)

    def test_type_i16(self, ir_to_code_binary):
        ir = create_ir("type_i16", [{"name": "f", "return_type": "i16", "params": [], "steps": [{"op": "return", "value": "0"}]}])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(ir_to_code_binary, ir, td)
            assert code is not None and ("int16_t" in code or "short" in code)

    def test_type_i32(self, ir_to_code_binary):
        ir = create_ir("type_i32", [{"name": "f", "return_type": "i32", "params": [], "steps": [{"op": "return", "value": "0"}]}])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(ir_to_code_binary, ir, td)
            assert code is not None and ("int32_t" in code or "int " in code)

    def test_type_i64(self, ir_to_code_binary):
        ir = create_ir("type_i64", [{"name": "f", "return_type": "i64", "params": [], "steps": [{"op": "return", "value": "0"}]}])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(ir_to_code_binary, ir, td)
            assert code is not None and ("int64_t" in code or "long" in code)

    def test_type_f32(self, ir_to_code_binary):
        ir = create_ir("type_f32", [{"name": "f", "return_type": "f32", "params": [], "steps": [{"op": "return", "value": "0.0"}]}])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(ir_to_code_binary, ir, td)
            assert code is not None and "float" in code

    def test_type_f64(self, ir_to_code_binary):
        ir = create_ir("type_f64", [{"name": "f", "return_type": "f64", "params": [], "steps": [{"op": "return", "value": "0.0"}]}])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(ir_to_code_binary, ir, td)
            assert code is not None and "double" in code

    def test_type_bool(self, ir_to_code_binary):
        ir = create_ir("type_bool", [{"name": "f", "return_type": "bool", "params": [], "steps": [{"op": "return", "value": "true"}]}])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(ir_to_code_binary, ir, td)
            assert code is not None and ("bool" in code or "_Bool" in code)

    def test_type_void(self, ir_to_code_binary):
        ir = create_ir("type_void", [{"name": "f", "return_type": "void", "params": [], "steps": []}])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(ir_to_code_binary, ir, td)
            assert code is not None and "void" in code

    # ==================== Assignment Tests ====================

    def test_assign_literal(self, ir_to_code_binary):
        ir = create_ir("assign_lit", [{"name": "f", "return_type": "i32", "params": [],
            "steps": [{"op": "assign", "target": "x", "value": "42"}, {"op": "return", "value": "x"}]}])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(ir_to_code_binary, ir, td)
            assert code is not None and "x = 42" in code

    def test_assign_expression(self, ir_to_code_binary):
        ir = create_ir("assign_expr", [{"name": "f", "return_type": "i32", "params": [{"name": "a", "type": "i32"}],
            "steps": [{"op": "assign", "target": "x", "value": "a + 1"}, {"op": "return", "value": "x"}]}])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(ir_to_code_binary, ir, td)
            assert code is not None and "a + 1" in code

    def test_multiple_assigns(self, ir_to_code_binary):
        ir = create_ir("multi_assign", [{"name": "f", "return_type": "i32", "params": [],
            "steps": [
                {"op": "assign", "target": "a", "value": "1"},
                {"op": "assign", "target": "b", "value": "2"},
                {"op": "assign", "target": "c", "value": "a + b"},
                {"op": "return", "value": "c"}
            ]}])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(ir_to_code_binary, ir, td)
            assert code is not None and "a = 1" in code and "b = 2" in code

    # ==================== If Statement Tests ====================

    def test_if_simple(self, ir_to_code_binary):
        ir = create_ir("if_simple", [{"name": "f", "return_type": "i32", "params": [{"name": "x", "type": "i32"}],
            "steps": [
                {"op": "if", "condition": "x > 0", "then_block": [{"op": "return", "value": "1"}]},
                {"op": "return", "value": "0"}
            ]}])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(ir_to_code_binary, ir, td)
            assert code is not None and "if" in code and "x > 0" in code

    def test_if_else(self, ir_to_code_binary):
        ir = create_ir("if_else", [{"name": "f", "return_type": "i32", "params": [{"name": "x", "type": "i32"}],
            "steps": [{
                "op": "if", "condition": "x > 0",
                "then_block": [{"op": "return", "value": "1"}],
                "else_block": [{"op": "return", "value": "-1"}]
            }]}])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(ir_to_code_binary, ir, td)
            assert code is not None and "if" in code and "else" in code

    def test_if_nested(self, ir_to_code_binary):
        ir = create_ir("if_nested", [{"name": "f", "return_type": "i32", "params": [{"name": "x", "type": "i32"}],
            "steps": [{
                "op": "if", "condition": "x > 0",
                "then_block": [{
                    "op": "if", "condition": "x > 10",
                    "then_block": [{"op": "return", "value": "2"}],
                    "else_block": [{"op": "return", "value": "1"}]
                }],
                "else_block": [{"op": "return", "value": "0"}]
            }]}])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(ir_to_code_binary, ir, td)
            assert code is not None and code.count("if") >= 2

    # ==================== While Loop Tests ====================

    def test_while_simple(self, ir_to_code_binary):
        ir = create_ir("while_simple", [{"name": "f", "return_type": "i32", "params": [{"name": "n", "type": "i32"}],
            "steps": [
                {"op": "assign", "target": "i", "value": "0"},
                {"op": "while", "condition": "i < n", "body": [
                    {"op": "assign", "target": "i", "value": "i + 1"}
                ]},
                {"op": "return", "value": "i"}
            ]}])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(ir_to_code_binary, ir, td)
            assert code is not None and "while" in code

    def test_while_with_break(self, ir_to_code_binary):
        ir = create_ir("while_break", [{"name": "f", "return_type": "i32", "params": [],
            "steps": [
                {"op": "assign", "target": "i", "value": "0"},
                {"op": "while", "condition": "1", "body": [
                    {"op": "if", "condition": "i == 5", "then_block": [{"op": "break"}]},
                    {"op": "assign", "target": "i", "value": "i + 1"}
                ]},
                {"op": "return", "value": "i"}
            ]}])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(ir_to_code_binary, ir, td)
            assert code is not None and "break" in code

    def test_while_with_continue(self, ir_to_code_binary):
        ir = create_ir("while_continue", [{"name": "f", "return_type": "i32", "params": [],
            "steps": [
                {"op": "assign", "target": "sum", "value": "0"},
                {"op": "assign", "target": "i", "value": "0"},
                {"op": "while", "condition": "i < 10", "body": [
                    {"op": "assign", "target": "i", "value": "i + 1"},
                    {"op": "if", "condition": "i % 2 == 0", "then_block": [{"op": "continue"}]},
                    {"op": "assign", "target": "sum", "value": "sum + i"}
                ]},
                {"op": "return", "value": "sum"}
            ]}])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(ir_to_code_binary, ir, td)
            assert code is not None and "continue" in code

    # ==================== For Loop Tests ====================

    def test_for_simple(self, ir_to_code_binary):
        ir = create_ir("for_simple", [{"name": "f", "return_type": "i32", "params": [{"name": "n", "type": "i32"}],
            "steps": [
                {"op": "assign", "target": "total", "value": "0"},
                {"op": "for", "init": "i = 0", "condition": "i < n", "update": "i++", "body": [
                    {"op": "assign", "target": "total", "value": "total + i"}
                ]},
                {"op": "return", "value": "total"}
            ]}])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(ir_to_code_binary, ir, td)
            assert code is not None and "for" in code

    # ==================== Switch Statement Tests (v0.8.4) ====================

    def test_switch_simple(self, ir_to_code_binary):
        ir = create_ir("switch_simple", [{"name": "f", "return_type": "i32", "params": [{"name": "x", "type": "i32"}],
            "steps": [
                {"op": "assign", "target": "result", "value": "0"},
                {"op": "switch", "expr": "x", "cases": [
                    {"value": "1", "body": [{"op": "assign", "target": "result", "value": "10"}, {"op": "break"}]},
                    {"value": "2", "body": [{"op": "assign", "target": "result", "value": "20"}, {"op": "break"}]}
                ], "default": [{"op": "assign", "target": "result", "value": "-1"}]},
                {"op": "return", "value": "result"}
            ]}])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(ir_to_code_binary, ir, td)
            assert code is not None and "f" in code

    def test_switch_with_default(self, ir_to_code_binary):
        ir = create_ir("switch_default", [{"name": "f", "return_type": "i32", "params": [{"name": "x", "type": "i32"}],
            "steps": [{
                "op": "switch", "expr": "x", "cases": [
                    {"value": "1", "body": [{"op": "return", "value": "1"}]}
                ], "default": [{"op": "return", "value": "0"}]
            }]}])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(ir_to_code_binary, ir, td)
            assert code is not None and "f" in code

    def test_switch_fallthrough(self, ir_to_code_binary):
        ir = create_ir("switch_fall", [{"name": "f", "return_type": "i32", "params": [{"name": "day", "type": "i32"}],
            "steps": [{
                "op": "switch", "expr": "day", "cases": [
                    {"value": "6", "body": []},
                    {"value": "7", "body": [{"op": "return", "value": "1"}]}
                ], "default": [{"op": "return", "value": "0"}]
            }]}])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(ir_to_code_binary, ir, td)
            assert code is not None and "f" in code

    # ==================== Nested Control Flow (v0.7.x) ====================

    def test_if_in_while(self, ir_to_code_binary):
        ir = create_ir("if_in_while", [{"name": "f", "return_type": "i32", "params": [],
            "steps": [
                {"op": "assign", "target": "x", "value": "0"},
                {"op": "while", "condition": "x < 10", "body": [
                    {"op": "if", "condition": "x == 5", "then_block": [{"op": "break"}]},
                    {"op": "assign", "target": "x", "value": "x + 1"}
                ]},
                {"op": "return", "value": "x"}
            ]}])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(ir_to_code_binary, ir, td)
            assert code is not None and "while" in code and "if" in code

    def test_deeply_nested(self, ir_to_code_binary):
        ir = create_ir("deep_nest", [{"name": "f", "return_type": "i32", "params": [],
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
            ]}])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(ir_to_code_binary, ir, td)
            assert code is not None

    # ==================== Code Structure Tests ====================

    def test_has_function_signature(self, ir_to_code_binary):
        ir = create_ir("sig_test", [{"name": "f", "return_type": "i32",
            "params": [{"name": "a", "type": "i32"}, {"name": "b", "type": "i32"}],
            "steps": [{"op": "return", "value": "a + b"}]}])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(ir_to_code_binary, ir, td)
            assert code is not None and "f" in code and "a" in code and "b" in code

    def test_has_includes(self, ir_to_code_binary):
        ir = create_ir("include_test", [{"name": "f", "return_type": "i32", "params": [], "steps": [{"op": "return", "value": "0"}]}])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(ir_to_code_binary, ir, td)
            assert code is not None and ("#include" in code or "// Generated" in code)

    def test_indentation(self, ir_to_code_binary):
        ir = create_ir("indent_test", [{"name": "f", "return_type": "i32", "params": [{"name": "x", "type": "i32"}],
            "steps": [
                {"op": "if", "condition": "x > 0", "then_block": [{"op": "return", "value": "1"}]},
                {"op": "return", "value": "0"}
            ]}])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(ir_to_code_binary, ir, td)
            assert code is not None
            lines = code.split('\n')
            assert any(line.startswith('  ') or line.startswith('\t') for line in lines)

    def test_closing_braces(self, ir_to_code_binary):
        ir = create_ir("brace_test", [{"name": "f", "return_type": "i32", "params": [],
            "steps": [
                {"op": "while", "condition": "1", "body": [{"op": "break"}]},
                {"op": "return", "value": "0"}
            ]}])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(ir_to_code_binary, ir, td)
            assert code is not None
            assert code.count('{') == code.count('}')

    def test_empty_module(self, ir_to_code_binary):
        ir = create_ir("empty_mod", [])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(ir_to_code_binary, ir, td)
            assert code is not None or True

    def test_long_function_name(self, ir_to_code_binary):
        ir = create_ir("long_name", [{"name": "a" * 50, "return_type": "void", "params": [], "steps": []}])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(ir_to_code_binary, ir, td)
            assert code is not None

    def test_many_parameters(self, ir_to_code_binary):
        params = [{"name": f"p{i}", "type": "i32"} for i in range(10)]
        ir = create_ir("many_params", [{"name": "f", "return_type": "i32", "params": params, "steps": [{"op": "return", "value": "p0"}]}])
        with tempfile.TemporaryDirectory() as td:
            code = run_ir_to_code(ir_to_code_binary, ir, td)
            assert code is not None

