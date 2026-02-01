#!/usr/bin/env python3
"""
STUNIR v0.9.0 Test Suite
Tests break/continue/switch features across all test specs.
"""
import json
import sys
import os
from pathlib import Path

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent / "tools"))

from spec_to_ir import convert_spec_to_ir
from ir_to_code import translate_steps_to_c, build_render_context

def test_spec(spec_file):
    """Test a single spec file."""
    print(f"\n{'='*80}")
    print(f"Testing: {spec_file.name}")
    print('='*80)
    
    # Load spec
    with open(spec_file, 'r') as f:
        spec = json.load(f)
    
    # Convert to IR
    try:
        ir = convert_spec_to_ir(spec)
        print(f"✓ IR generation successful")
        print(f"  Module: {ir['module_name']}")
        print(f"  Functions: {len(ir['functions'])}")
    except Exception as e:
        print(f"✗ IR generation failed: {e}")
        return False
    
    # Check for new operations
    def check_ops(steps, ops_found):
        for step in steps:
            if not isinstance(step, dict):
                continue
            op = step.get('op', '')
            if op in ['break', 'continue', 'switch']:
                ops_found.add(op)
            # Recurse into nested blocks
            if 'then_block' in step:
                check_ops(step['then_block'], ops_found)
            if 'else_block' in step:
                check_ops(step['else_block'], ops_found)
            if 'body' in step:
                check_ops(step['body'], ops_found)
            if 'cases' in step:
                for case in step['cases']:
                    if 'body' in case:
                        check_ops(case['body'], ops_found)
            if 'default' in step:
                check_ops(step['default'], ops_found)
    
    ops_found = set()
    for func in ir['functions']:
        if 'steps' in func:
            check_ops(func['steps'], ops_found)
    
    if ops_found:
        print(f"  New operations found: {', '.join(sorted(ops_found))}")
    
    # Generate C code for each function
    try:
        for func in ir['functions']:
            func_name = func['name']
            steps = func.get('steps', [])
            ret_type = func['return_type']
            
            c_code = translate_steps_to_c(steps, ret_type)
            
            # Check that C code contains expected keywords
            if 'break' in ops_found and 'break;' not in c_code:
                print(f"✗ C code for {func_name} missing 'break;'")
                return False
            if 'continue' in ops_found and 'continue;' not in c_code:
                print(f"✗ C code for {func_name} missing 'continue;'")
                return False
            if 'switch' in ops_found and 'switch' not in c_code:
                print(f"✗ C code for {func_name} missing 'switch'")
                return False
            
        print(f"✓ C code generation successful")
        
        # Print sample C code for first function
        first_func = ir['functions'][0]
        c_code = translate_steps_to_c(first_func['steps'], first_func['return_type'])
        print(f"\n  Sample C code for {first_func['name']}:")
        for line in c_code.split('\n')[:15]:  # First 15 lines
            print(f"    {line}")
        if c_code.count('\n') > 15:
            print(f"    ... ({c_code.count(chr(10)) - 15} more lines)")
        
    except Exception as e:
        print(f"✗ C code generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Run all v0.9.0 tests."""
    test_dir = Path(__file__).parent / "test_specs" / "v0.9.0"
    
    if not test_dir.exists():
        print(f"Error: Test directory not found: {test_dir}")
        return 1
    
    # Find all test specs
    test_specs = sorted(test_dir.glob("*.json"))
    
    if not test_specs:
        print(f"Error: No test specs found in {test_dir}")
        return 1
    
    print(f"STUNIR v0.9.0 Test Suite")
    print(f"Found {len(test_specs)} test spec(s)")
    
    # Run tests
    passed = 0
    failed = 0
    
    for spec_file in test_specs:
        if test_spec(spec_file):
            passed += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f"Test Summary")
    print('='*80)
    print(f"Total: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print(f"\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
