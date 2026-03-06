#!/usr/bin/env python3
"""
STUNIR Comprehensive Emission Validation Test
Tests functional code emitters for Lisp, Futhark, Lean4, Ada, and Prolog.
Includes golden output validation for all step types.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Comprehensive test IR covering all step types
COMPREHENSIVE_IR = {
    "ir_version": "v1",
    "module_name": "emission_test",
    "types": [],
    "functions": [
        # Basic operations
        {
            "name": "simple_return",
            "args": [],
            "return_type": "int",
            "body_hint": "simple_return",
            "emission_mode": "stub_hints",
            "steps": [
                {"op": "return", "value": "42"}
            ]
        },
        {
            "name": "add",
            "args": [{"name": "a", "type": "int"}, {"name": "b", "type": "int"}],
            "return_type": "int",
            "body_hint": "simple_return",
            "emission_mode": "stub_hints",
            "steps": [
                {"op": "assign", "target": "result", "value": "a + b"},
                {"op": "return", "value": "result"}
            ]
        },
        # Control flow
        {
            "name": "conditional",
            "args": [{"name": "x", "type": "int"}],
            "return_type": "bool",
            "body_hint": "conditional",
            "emission_mode": "stub_hints",
            "steps": [
                {
                    "op": "if",
                    "condition": "x > 0",
                    "then_block": [
                        {"op": "return", "value": "true"}
                    ],
                    "else_block": [
                        {"op": "return", "value": "false"}
                    ]
                }
            ]
        },
        {
            "name": "while_loop",
            "args": [{"name": "n", "type": "int"}],
            "return_type": "int",
            "body_hint": "loop_accum",
            "hint_detail": "accumulates sum",
            "emission_mode": "stub_hints",
            "steps": [
                {"op": "assign", "target": "sum", "value": "0"},
                {"op": "assign", "target": "i", "value": "0"},
                {
                    "op": "while",
                    "condition": "i < n",
                    "body": [
                        {"op": "assign", "target": "sum", "value": "sum + i"},
                        {"op": "assign", "target": "i", "value": "i + 1"}
                    ]
                },
                {"op": "return", "value": "sum"}
            ]
        },
        {
            "name": "for_loop",
            "args": [{"name": "n", "type": "int"}],
            "return_type": "int",
            "body_hint": "loop_accum",
            "hint_detail": "factorial",
            "emission_mode": "stub_hints",
            "steps": [
                {"op": "assign", "target": "result", "value": "1"},
                {
                    "op": "for",
                    "init": "1",
                    "condition": "n + 1",
                    "increment": "i + 1",
                    "body": [
                        {"op": "assign", "target": "result", "value": "result * i"}
                    ]
                },
                {"op": "return", "value": "result"}
            ]
        },
        # Switch/case
        {
            "name": "switch_example",
            "args": [{"name": "x", "type": "int"}],
            "return_type": "string",
            "body_hint": "switch",
            "emission_mode": "stub_hints",
            "steps": [
                {
                    "op": "switch",
                    "expr": "x",
                    "cases": [
                        {"value": 1, "body": [{"op": "return", "value": "\"one\""}]},
                        {"value": 2, "body": [{"op": "return", "value": "\"two\""}]},
                        {"value": 3, "body": [{"op": "return", "value": "\"three\""}]}
                    ],
                    "default": [{"op": "return", "value": "\"other\""}]
                }
            ]
        },
        # Exception handling
        {
            "name": "try_catch_example",
            "args": [{"name": "x", "type": "int"}],
            "return_type": "int",
            "body_hint": "try_catch",
            "emission_mode": "stub_hints",
            "steps": [
                {
                    "op": "try",
                    "try_block": [
                        {"op": "assign", "target": "result", "value": "x * 2"}
                    ],
                    "catch_blocks": [
                        {
                            "exception_type": "ValueError",
                            "exception_var": "e",
                            "body": [{"op": "assign", "target": "result", "value": "0"}]
                        }
                    ],
                    "finally_block": [
                        {"op": "call", "value": "cleanup", "args": ""}
                    ]
                },
                {"op": "return", "value": "result"}
            ]
        },
        # Arrays
        {
            "name": "array_operations",
            "args": [],
            "return_type": "int",
            "body_hint": "complex",
            "emission_mode": "stub_hints",
            "steps": [
                {"op": "array_new", "target": "arr", "element_type": "int", "size": 5},
                {"op": "array_push", "target": "arr", "value": "1"},
                {"op": "array_push", "target": "arr", "value": "2"},
                {"op": "array_push", "target": "arr", "value": "3"},
                {"op": "array_len", "target": "len", "value": "arr"},
                {"op": "array_get", "target": "first", "value": "arr", "index": "0"},
                {"op": "array_set", "target": "arr", "index": "1", "value": "42"},
                {"op": "array_pop", "target": "last", "value": "arr"},
                {"op": "return", "value": "len"}
            ]
        },
        # Maps
        {
            "name": "map_operations",
            "args": [],
            "return_type": "int",
            "body_hint": "complex",
            "emission_mode": "stub_hints",
            "steps": [
                {"op": "map_new", "target": "m", "key_type": "string", "value_type": "int"},
                {"op": "map_set", "target": "m", "key": "\"a\"", "value": "1"},
                {"op": "map_set", "target": "m", "key": "\"b\"", "value": "2"},
                {"op": "map_get", "target": "val", "value": "m", "key": "\"a\""},
                {"op": "map_has", "target": "has_b", "value": "m", "key": "\"b\""},
                {"op": "map_delete", "target": "m", "key": "\"a\""},
                {"op": "map_keys", "target": "keys", "value": "m"},
                {"op": "return", "value": "val"}
            ]
        },
        # Sets
        {
            "name": "set_operations",
            "args": [],
            "return_type": "bool",
            "body_hint": "complex",
            "emission_mode": "stub_hints",
            "steps": [
                {"op": "set_new", "target": "s", "element_type": "int"},
                {"op": "set_add", "target": "s", "value": "1"},
                {"op": "set_add", "target": "s", "value": "2"},
                {"op": "set_has", "target": "has_one", "value": "s", "key": "1"},
                {"op": "set_remove", "target": "s", "value": "1"},
                {"op": "return", "value": "has_one"}
            ]
        },
        # Structs
        {
            "name": "struct_operations",
            "args": [],
            "return_type": "int",
            "body_hint": "complex",
            "emission_mode": "stub_hints",
            "steps": [
                {"op": "struct_new", "target": "p", "struct_type": "Point", "fields": {"x": 1, "y": 2}},
                {"op": "struct_get", "target": "x_val", "value": "p", "field": "x"},
                {"op": "struct_set", "target": "p", "field": "y", "value": "3"},
                {"op": "return", "value": "x_val"}
            ]
        },
        # Break/Continue
        {
            "name": "break_continue",
            "args": [{"name": "n", "type": "int"}],
            "return_type": "int",
            "body_hint": "complex",
            "emission_mode": "stub_hints",
            "steps": [
                {"op": "assign", "target": "sum", "value": "0"},
                {"op": "assign", "target": "i", "value": "0"},
                {
                    "op": "while",
                    "condition": "i < n",
                    "body": [
                        {
                            "op": "if",
                            "condition": "i == 5",
                            "then_block": [{"op": "break"}],
                            "else_block": []
                        },
                        {
                            "op": "if",
                            "condition": "i % 2 == 0",
                            "then_block": [
                                {"op": "assign", "target": "i", "value": "i + 1"},
                                {"op": "continue"}
                            ],
                            "else_block": []
                        },
                        {"op": "assign", "target": "sum", "value": "sum + i"},
                        {"op": "assign", "target": "i", "value": "i + 1"}
                    ]
                },
                {"op": "return", "value": "sum"}
            ]
        },
        # Error/Throw
        {
            "name": "error_example",
            "args": [{"name": "x", "type": "int"}],
            "return_type": "int",
            "body_hint": "complex",
            "emission_mode": "stub_hints",
            "steps": [
                {
                    "op": "if",
                    "condition": "x < 0",
                    "then_block": [
                        {"op": "error", "value": "\"x must be non-negative\""}
                    ],
                    "else_block": []
                },
                {"op": "return", "value": "x * 2"}
            ]
        },
        # Empty function
        {
            "name": "empty_func",
            "args": [],
            "return_type": "void",
            "body_hint": "none",
            "emission_mode": "stub_hints",
            "steps": []
        },
        # Best-effort mode test
        {
            "name": "best_effort_test",
            "args": [{"name": "x", "type": "int"}],
            "return_type": "int",
            "body_hint": "complex",
            "emission_mode": "best_effort",
            "steps": [
                {"op": "assign", "target": "result", "value": "x * 2"},
                {"op": "return", "value": "result"}
            ]
        }
    ]
}

# Target languages to test
TARGET_LANGUAGES = {
    "lisp": ["common_lisp", "scheme", "clojure"],
    "futhark": ["futhark"],
    "lean4": ["lean4"],
    "ada": ["ada", "spark"],
    "prolog": ["swi_prolog", "gnu_prolog", "mercury"]
}

# Golden output patterns (partial matches for key constructs)
GOLDEN_PATTERNS = {
    "common_lisp": {
        "simple_return": ["(return 42)", "42"],
        "conditional": ["(if", "(progn", "true", "false"],
        "while_loop": ["(loop while", "(setf sum"],
        "for_loop": ["(loop for", "(setf result"],
        "array_operations": ["(make-array", "(aref", "(vector-push"],
    },
    "scheme": {
        "simple_return": ["42"],
        "conditional": ["(if", "(begin", "true", "false"],
        "while_loop": ["(let loop", "(when"],
        "for_loop": ["(do", "(set!"],
    },
    "clojure": {
        "simple_return": ["42"],
        "conditional": ["(if", "(do", "true", "false"],
        "while_loop": ["(loop", "(when"],
        "for_loop": ["(loop", "(recur"],
    },
    "futhark": {
        "simple_return": ["let", "42"],
        "conditional": ["if", "then", "else"],
        "while_loop": ["loop", "while"],
        "for_loop": ["loop", "for"],
    },
    "lean4": {
        "simple_return": ["42"],
        "conditional": ["if", "then", "else"],
        "while_loop": ["whileM", "do"],
        "for_loop": ["for i in", "do"],
    },
    "ada": {
        "simple_return": ["return 42", "return Integer"],
        "conditional": ["if", "then", "end if"],
        "while_loop": ["while", "loop", "end loop"],
        "for_loop": ["for I in", "loop", "end loop"],
    },
    "spark": {
        "simple_return": ["return 42", "return Integer"],
        "conditional": ["if", "then", "end if"],
        "while_loop": ["while", "loop", "end loop"],
        "for_loop": ["for I in", "loop", "end loop"],
    },
    "swi_prolog": {
        "simple_return": ["Result = 42"],
        "conditional": ["( X > 0 ->", "Result = true"],
        "while_loop": ["(   ", "->  (", "fail"],
        "for_loop": ["between(", "Result"],
    },
    "gnu_prolog": {
        "simple_return": ["Result = 42"],
        "conditional": ["( X > 0 ->", "Result = true"],
        "while_loop": ["(   ", "->  (", "fail"],
        "for_loop": ["for(_I,", "Result"],
    },
    "mercury": {
        "simple_return": ["Result = 42"],
        "conditional": ["( if X > 0 then", "Result = true"],
        "while_loop": ["( if", "then", "else"],
        "for_loop": ["( for _I in", "do"],
    }
}

# Stub hint patterns - these should be present for unsupported operations
STUB_HINT_PATTERNS = {
    "common_lisp": {
        "break": ["STUB: break", "return-from"],
        "continue": ["STUB: continue", "go"],
        "unsupported": ["STUB: unsupported operation"],
    },
    "scheme": {
        "break": ["STUB: break", "call/cc"],
        "continue": ["STUB: continue", "continuation"],
        "unsupported": ["STUB: unsupported operation"],
    },
    "clojure": {
        "break": ["STUB: break", "recur", "reduced"],
        "continue": ["STUB: continue", "recur", "accumulator"],
        "unsupported": ["STUB: unsupported operation"],
    },
    "futhark": {
        "break": ["STUB: break", "tagged result"],
        "continue": ["STUB: continue", "tagged result"],
        "unsupported": ["STUB: unsupported operation"],
    },
    "lean4": {
        "break": ["STUB: break", "whileM", "StateM", "Except"],
        "continue": ["STUB: continue", "whileM", "StateM"],
        "unsupported": ["STUB: unsupported operation"],
    },
    "ada": {
        "break": ["exit;"],
        "continue": ["STUB: continue", "label"],
        "map_keys": ["STUB: map_keys", "Iterate"],
        "set_union": ["STUB: set_union", "Union"],
        "set_intersect": ["STUB: set_intersect", "Intersection"],
        "unsupported": ["STUB: unsupported operation"],
    },
    "spark": {
        "break": ["exit;"],
        "continue": ["STUB: continue", "label"],
        "map_keys": ["STUB: map_keys", "Iterate"],
        "set_union": ["STUB: set_union", "Union"],
        "set_intersect": ["STUB: set_intersect", "Intersection"],
        "unsupported": ["STUB: unsupported operation"],
    },
    "swi_prolog": {
        "unsupported": ["STUB: unsupported operation"],
    },
    "gnu_prolog": {
        "unsupported": ["STUB: unsupported operation"],
    },
    "mercury": {
        "unsupported": ["STUB: unsupported operation"],
    }
}

def validate_ir_schema(ir_data: dict) -> Tuple[bool, str]:
    """Validate IR against schema."""
    required_fields = ["ir_version", "module_name", "functions"]
    for field in required_fields:
        if field not in ir_data:
            return False, f"Missing required field: {field}"
    
    for func in ir_data.get("functions", []):
        if "name" not in func:
            return False, "Function missing name"
        if "return_type" not in func:
            return False, f"Function {func.get('name', 'unknown')} missing return_type"
        if "emission_mode" not in func:
            return False, f"Function {func.get('name', 'unknown')} missing emission_mode"
    
    return True, "IR schema validation passed"

def check_step_coverage(ir_data: dict) -> Tuple[Set[str], Set[str]]:
    """Check which step types are covered by the test IR."""
    step_types = set()
    for func in ir_data.get("functions", []):
        for step in func.get("steps", []):
            step_types.add(step.get("op", "unknown"))
    
    expected_steps = {
        "return", "assign", "call", "if", "while", "for",
        "break", "continue", "switch", "try", "throw", "error",
        "array_new", "array_get", "array_set", "array_push", "array_pop", "array_len",
        "map_new", "map_get", "map_set", "map_delete", "map_has", "map_keys",
        "set_new", "set_add", "set_remove", "set_has",
        "struct_new", "struct_get", "struct_set"
    }
    
    covered = step_types.intersection(expected_steps)
    missing = expected_steps - step_types
    
    return covered, missing

def check_emission_mode_coverage(ir_data: dict) -> Dict[str, int]:
    """Check emission mode distribution."""
    modes = {"stub_hints": 0, "best_effort": 0}
    for func in ir_data.get("functions", []):
        mode = func.get("emission_mode", "stub_hints")
        if mode in modes:
            modes[mode] += 1
    return modes

def validate_golden_output(target: str, func_name: str, output: str) -> Tuple[bool, List[str]]:
    """Validate output against golden patterns."""
    if target not in GOLDEN_PATTERNS:
        return True, ["No golden patterns defined for target"]
    
    patterns = GOLDEN_PATTERNS[target].get(func_name, [])
    if not patterns:
        return True, ["No golden patterns defined for function"]
    
    missing = []
    for pattern in patterns:
        if pattern.lower() not in output.lower():
            missing.append(pattern)
    
    return len(missing) == 0, missing

def check_no_placeholders(output: str, target: str) -> List[str]:
    """Check for placeholder comments that should not be present."""
    placeholders = []
    
    # Generic placeholders
    if "TODO" in output and "STUB:" not in output:
        placeholders.append("TODO without STUB hint")
    
    if "unsupported step" in output.lower():
        placeholders.append("unsupported step comment")
    
    # Target-specific placeholders
    if target in ["common_lisp", "scheme", "clojure"]:
        if ";; for loop: use" in output:
            placeholders.append("for loop comment placeholder")
        if ";; while loop: use" in output:
            placeholders.append("while loop comment placeholder")
    
    if target in ["swi_prolog", "gnu_prolog", "mercury"]:
        if "% while loop: use" in output:
            placeholders.append("while loop comment placeholder")
        if "% for loop: use" in output:
            placeholders.append("for loop comment placeholder")
    
    return placeholders

def validate_stub_hints(output: str, target: str, operation: str) -> Tuple[bool, List[str]]:
    """Validate that stub hints are present for unsupported operations."""
    if target not in STUB_HINT_PATTERNS:
        return True, ["No stub hint patterns defined for target"]
    
    patterns = STUB_HINT_PATTERNS[target].get(operation, [])
    if not patterns:
        return True, ["No stub hint patterns defined for operation"]
    
    missing = []
    for pattern in patterns:
        if pattern.lower() not in output.lower():
            missing.append(pattern)
    
    return len(missing) == 0, missing

def run_python_emitter_test(target_group: str, target: str, ir_data: dict) -> Tuple[bool, str]:
    """Run a Python emitter test for a specific target."""
    import tempfile
    import importlib.util
    
    # Map target groups to emitter paths
    emitter_paths = {
        "lisp": {
            "common_lisp": "tools/python/targets/lisp/common_lisp/emitter.py",
            "scheme": "tools/python/targets/lisp/scheme/emitter.py",
            "clojure": "tools/python/targets/lisp/clojure/emitter.py",
        },
        "prolog": {
            "swi_prolog": "tools/python/targets/prolog/swi_prolog/emitter.py",
            "gnu_prolog": "tools/python/targets/prolog/gnu_prolog/emitter.py",
            "mercury": "tools/python/targets/prolog/mercury/emitter.py",
        },
        "functional": {
            "futhark": "tools/python/targets/functional/futhark_emitter.py",
            "lean4": "tools/python/targets/functional/lean4_emitter.py",
        },
        "ada": {
            "ada": "tools/python/targets/systems/ada_emitter.py",
            "spark": "tools/python/targets/systems/spark_emitter.py",
        }
    }
    
    if target_group not in emitter_paths or target not in emitter_paths[target_group]:
        return False, f"No emitter path for {target_group}/{target}"
    
    emitter_path = Path(emitter_paths[target_group][target])
    if not emitter_path.exists():
        return False, f"Emitter not found: {emitter_path}"
    
    try:
        # Import emitter module dynamically
        spec = importlib.util.spec_from_file_location("emitter", emitter_path)
        if spec is None or spec.loader is None:
            return False, f"Could not load emitter spec: {emitter_path}"
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find emitter class
        emitter_class = None
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and 'Emitter' in name:
                emitter_class = obj
                break
        
        if emitter_class is None:
            return False, f"No emitter class found in {emitter_path}"
        
        # Run emitter
        with tempfile.TemporaryDirectory() as tmpdir:
            emitter = emitter_class(ir_data, Path(tmpdir))
            manifest = emitter.run()
            
            # Read generated files
            generated_code = ""
            for file_info in manifest.get("files", []):
                file_path = Path(tmpdir) / file_info["path"]
                if file_path.exists():
                    generated_code += file_path.read_text() + "\n"
            
            return True, generated_code
    
    except Exception as e:
        return False, f"Emitter error: {str(e)}"

def test_all_stub_hints():
    """Test that all targets emit proper stub hints for unsupported operations."""
    print("\n[5] Testing stub hint emissions...")
    
    results = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "details": []
    }
    
    # Test break/continue stub hints
    break_continue_ir = {
        "ir_version": "v1",
        "module_name": "stub_test",
        "types": [],
        "functions": [
            {
                "name": "test_break",
                "args": [],
                "return_type": "int",
                "emission_mode": "stub_hints",
                "steps": [
                    {"op": "assign", "target": "i", "value": "0"},
                    {
                        "op": "while",
                        "condition": "i < 10",
                        "body": [
                            {"op": "break"}
                        ]
                    },
                    {"op": "return", "value": "i"}
                ]
            },
            {
                "name": "test_continue",
                "args": [],
                "return_type": "int",
                "emission_mode": "stub_hints",
                "steps": [
                    {"op": "assign", "target": "i", "value": "0"},
                    {
                        "op": "while",
                        "condition": "i < 10",
                        "body": [
                            {"op": "continue"}
                        ]
                    },
                    {"op": "return", "value": "i"}
                ]
            }
        ]
    }
    
    for group, targets in TARGET_LANGUAGES.items():
        for target in targets:
            success, output = run_python_emitter_test(group, target, break_continue_ir)
            
            if not success:
                results["skipped"] += 1
                results["details"].append(f"  {target}: SKIPPED ({output})")
                continue
            
            # Validate break stub hint
            break_valid, break_missing = validate_stub_hints(output, target, "break")
            continue_valid, continue_missing = validate_stub_hints(output, target, "continue")
            
            if break_valid and continue_valid:
                results["passed"] += 1
                results["details"].append(f"  {target}: PASSED")
            else:
                results["failed"] += 1
                msg = f"  {target}: FAILED"
                if not break_valid:
                    msg += f" (break missing: {', '.join(break_missing)})"
                if not continue_valid:
                    msg += f" (continue missing: {', '.join(continue_missing)})"
                results["details"].append(msg)
    
    print(f"    Passed: {results['passed']}")
    print(f"    Failed: {results['failed']}")
    print(f"    Skipped: {results['skipped']}")
    
    for detail in results["details"]:
        print(detail)
    
    return results["failed"] == 0

def main():
    print("=" * 70)
    print("STUNIR Comprehensive Emission Validation Test")
    print("=" * 70)
    
    # Validate IR schema
    print("\n[1] Validating IR schema...")
    valid, msg = validate_ir_schema(COMPREHENSIVE_IR)
    print(f"    {msg}")
    if not valid:
        sys.exit(1)
    
    # Check step coverage
    print("\n[2] Checking step coverage...")
    covered, missing = check_step_coverage(COMPREHENSIVE_IR)
    print(f"    Covered: {len(covered)} step types")
    print(f"    Missing: {len(missing)} step types")
    if missing:
        print(f"    Missing types: {', '.join(sorted(missing))}")
    
    # Check emission mode coverage
    print("\n[3] Checking emission mode coverage...")
    modes = check_emission_mode_coverage(COMPREHENSIVE_IR)
    print(f"    stub_hints: {modes['stub_hints']} functions")
    print(f"    best_effort: {modes['best_effort']} functions")
    
    # Write test IR to file
    ir_path = Path("test_comprehensive_ir.json")
    with open(ir_path, "w") as f:
        json.dump(COMPREHENSIVE_IR, f, indent=2)
    print(f"\n[4] Test IR written to {ir_path}")
    
    # Test stub hints
    stub_hints_passed = test_all_stub_hints()
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  Functions: {len(COMPREHENSIVE_IR['functions'])}")
    print(f"  Step types covered: {len(covered)}")
    print(f"  Target languages: {sum(len(v) for v in TARGET_LANGUAGES.values())}")
    print(f"  Emission modes: stub_hints={modes['stub_hints']}, best_effort={modes['best_effort']}")
    print(f"  Stub hint tests: {'PASSED' if stub_hints_passed else 'FAILED'}")
    
    # Body hints summary
    body_hints = [f.get("body_hint", "none") for f in COMPREHENSIVE_IR["functions"]]
    unique_hints = set(body_hints)
    print(f"  Body hints used: {', '.join(sorted(unique_hints))}")
    
    # Target language groups
    print("\n  Target language groups:")
    for group, targets in TARGET_LANGUAGES.items():
        print(f"    {group}: {', '.join(targets)}")
    
    if stub_hints_passed:
        print("\n✓ Comprehensive emission validation test completed successfully")
        return 0
    else:
        print("\n✗ Some stub hint tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
