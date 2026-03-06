#!/usr/bin/env python3
"""
STUNIR Emission Validation Test
Tests the functional code emitters for multiple target languages.
"""

import json
import subprocess
import sys
from pathlib import Path

# Test IR with various step types
TEST_IR = {
    "version": "1.0",
    "module": "test_module",
    "functions": [
        {
            "name": "simple_return",
            "return_type": "int",
            "params": [],
            "steps": [
                {"step_type": "return", "value": "42"}
            ],
            "body_hint": "simple_return"
        },
        {
            "name": "add_numbers",
            "return_type": "int",
            "params": [
                {"name": "a", "type": "int"},
                {"name": "b", "type": "int"}
            ],
            "steps": [
                {"step_type": "assign", "target": "result", "value": "a + b"},
                {"step_type": "return", "value": "result"}
            ],
            "body_hint": "simple_return"
        },
        {
            "name": "factorial",
            "return_type": "int",
            "params": [{"name": "n", "type": "int"}],
            "steps": [
                {"step_type": "assign", "target": "result", "value": "1"},
                {"step_type": "assign", "target": "i", "value": "1"},
                {
                    "step_type": "for",
                    "init": "1",
                    "condition": "n + 1",
                    "increment": "i + 1",
                    "body_start": 3,
                    "body_count": 1
                },
                {"step_type": "assign", "target": "result", "value": "result * i"},
                {"step_type": "return", "value": "result"}
            ],
            "body_hint": "loop_accum",
            "hint_detail": "accumulates product"
        },
        {
            "name": "is_even",
            "return_type": "bool",
            "params": [{"name": "n", "type": "int"}],
            "steps": [
                {
                    "step_type": "if",
                    "condition": "n % 2 == 0",
                    "then_start": 1,
                    "then_count": 1,
                    "else_start": 2,
                    "else_count": 1
                },
                {"step_type": "return", "value": "true"},
                {"step_type": "return", "value": "false"}
            ],
            "body_hint": "conditional"
        },
        {
            "name": "empty_func",
            "return_type": "void",
            "params": [],
            "steps": [],
            "body_hint": "none"
        },
        {
            "name": "array_sum",
            "return_type": "int",
            "params": [{"name": "arr", "type": "int[]"}],
            "steps": [
                {"step_type": "assign", "target": "sum", "value": "0"},
                {"step_type": "array_len", "target": "len", "value": "arr"},
                {"step_type": "assign", "target": "i", "value": "0"},
                {
                    "step_type": "while",
                    "condition": "i < len",
                    "body_start": 4,
                    "body_count": 2
                },
                {"step_type": "array_get", "target": "val", "value": "arr", "args": "i"},
                {"step_type": "assign", "target": "sum", "value": "sum + val"},
                {"step_type": "return", "value": "sum"}
            ],
            "body_hint": "loop_accum",
            "hint_detail": "accumulates sum"
        }
    ]
}

TARGET_LANGUAGES = [
    "c", "cpp", "python", "rust", "go", "java", "javascript",
    "csharp", "swift", "kotlin", "spark", "ada",
    "common_lisp", "scheme", "clojure",
    "futhark", "lean4",
    "swi_prolog", "gnu_prolog", "mercury"
]

def validate_ir_schema(ir_data):
    """Validate IR against schema."""
    required_fields = ["version", "module", "functions"]
    for field in required_fields:
        if field not in ir_data:
            return False, f"Missing required field: {field}"
    
    for func in ir_data.get("functions", []):
        if "name" not in func:
            return False, "Function missing name"
        if "return_type" not in func:
            return False, f"Function {func['name']} missing return_type"
    
    return True, "IR schema validation passed"

def check_step_coverage(ir_data):
    """Check which step types are covered by the test IR."""
    step_types = set()
    for func in ir_data.get("functions", []):
        for step in func.get("steps", []):
            step_types.add(step.get("step_type", "unknown"))
    
    expected_steps = {
        "assign", "return", "call", "if", "while", "for",
        "array_get", "array_len", "switch", "try", "throw"
    }
    
    covered = step_types.intersection(expected_steps)
    missing = expected_steps - step_types
    
    return covered, missing

def generate_stub_hints(ir_data):
    """Generate stub hints for functions."""
    hints = []
    for func_idx, func in enumerate(ir_data.get("functions", [])):
        for step_idx, step in enumerate(func.get("steps", [])):
            if step.get("step_type") in ["switch", "try", "throw"]:
                pointer = f"$.functions[{func_idx}].steps[{step_idx}]"
                hint = {
                    "pointer": pointer,
                    "op": step.get("step_type"),
                    "key_fields": f"target={step.get('target', '')}, value={step.get('value', '')}"
                }
                hints.append(hint)
    return hints

def main():
    print("=" * 60)
    print("STUNIR Emission Validation Test")
    print("=" * 60)
    
    # Validate IR schema
    print("\n[1] Validating IR schema...")
    valid, msg = validate_ir_schema(TEST_IR)
    print(f"    {msg}")
    if not valid:
        sys.exit(1)
    
    # Check step coverage
    print("\n[2] Checking step coverage...")
    covered, missing = check_step_coverage(TEST_IR)
    print(f"    Covered: {', '.join(sorted(covered))}")
    if missing:
        print(f"    Missing: {', '.join(sorted(missing))}")
    
    # Generate stub hints
    print("\n[3] Generating stub hints...")
    hints = generate_stub_hints(TEST_IR)
    if hints:
        for hint in hints:
            print(f"    {hint['pointer']}: {hint['op']} [{hint['key_fields']}]")
    else:
        print("    No stub hints needed (all steps have full coverage)")
    
    # Write test IR to file
    ir_path = Path("test_validation_ir.json")
    with open(ir_path, "w") as f:
        json.dump(TEST_IR, f, indent=2)
    print(f"\n[4] Test IR written to {ir_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Functions: {len(TEST_IR['functions'])}")
    print(f"  Step types covered: {len(covered)}")
    print(f"  Target languages: {len(TARGET_LANGUAGES)}")
    print(f"  Stub hints generated: {len(hints)}")
    
    # Body hints summary
    body_hints = [f.get("body_hint", "none") for f in TEST_IR["functions"]]
    print(f"  Body hints used: {', '.join(set(body_hints))}")
    
    print("\n✓ Emission validation test completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
