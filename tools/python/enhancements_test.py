#!/usr/bin/env python3
"""STUNIR Enhancement Test Suite.

Tests all 5 major enhancements to verify functionality.
"""

import sys
import json
import subprocess
from pathlib import Path

# Test IR data
TEST_IR = {
    "schema": "stunir.ir.v1",
    "ir_module": "test_module",
    "ir_functions": [
        {
            "name": "test_control_flow",
            "params": [{"name": "x", "type": "i32"}],
            "returns": "i32",
            "body": [
                {"type": "var_decl", "var_name": "result", "var_type": "i32", "init": 0},
                {"type": "if", "cond": {"type": "binary", "op": ">", "left": {"type": "var", "name": "x"}, "right": 0},
                 "then": [{"type": "assign", "target": "result", "value": 1}],
                 "else": [{"type": "assign", "target": "result", "value": -1}]},
                {"type": "while", "cond": {"type": "binary", "op": "<", "left": {"type": "var", "name": "result"}, "right": 10},
                 "body": [{"type": "assign", "target": "result", "value": {"type": "binary", "op": "+", "left": {"type": "var", "name": "result"}, "right": 1}}]},
                {"type": "return", "value": {"type": "var", "name": "result"}}
            ]
        },
        {
            "name": "test_expressions",
            "params": [{"name": "a", "type": "i32"}, {"name": "b", "type": "i32"}],
            "returns": "i32",
            "body": [
                {"type": "var_decl", "var_name": "c", "var_type": "i32", 
                 "init": {"type": "binary", "op": "+", 
                         "left": {"type": "binary", "op": "*", "left": 2, "right": 3},
                         "right": {"type": "binary", "op": "*", "left": 4, "right": 5}}},
                {"type": "var_decl", "var_name": "d", "var_type": "i32",
                 "init": {"type": "binary", "op": "+", "left": {"type": "var", "name": "a"}, "right": {"type": "var", "name": "b"}}},
                {"type": "var_decl", "var_name": "e", "var_type": "i32",
                 "init": {"type": "binary", "op": "+", "left": {"type": "var", "name": "a"}, "right": {"type": "var", "name": "b"}}},
                {"type": "return", "value": {"type": "binary", "op": "+", "left": {"type": "var", "name": "d"}, "right": {"type": "var", "name": "e"}}}
            ]
        },
        {
            "name": "test_memory",
            "params": [],
            "returns": "void",
            "body": [
                {"type": "var_decl", "var_name": "ptr", "var_type": "*i32",
                 "init": {"type": "call", "func": "malloc", "args": [4]}},
                {"type": "assign", "target": "ptr", "value": 42},
                {"type": "call", "func": "free", "args": ["ptr"]}
            ]
        }
    ],
    "ir_types": [
        {"name": "Point", "kind": "struct", "fields": [
            {"name": "x", "type": "i32"},
            {"name": "y", "type": "i32"}
        ]},
        {"name": "Color", "kind": "enum", "variants": [
            {"name": "Red", "value": 0},
            {"name": "Green", "value": 1},
            {"name": "Blue", "value": 2}
        ]}
    ]
}


def test_control_flow():
    """Test Enhancement 1: Control Flow Translation."""
    print("\n=== Testing Control Flow Translation ===")
    try:
        from ir.control_flow import ControlFlowAnalyzer, ControlFlowTranslator
        
        analyzer = ControlFlowAnalyzer(TEST_IR)
        cfg = analyzer.analyze()
        
        print(f"✓ CFG constructed: {len(cfg.blocks)} blocks")
        print(f"✓ Loops detected: {len(cfg.loops)}")
        print(f"✓ Branches detected: {len(cfg.branches)}")
        
        # Test translator
        for target in ['python', 'rust', 'c', 'haskell']:
            translator = ControlFlowTranslator(target)
            if_code = translator.translate_if("x > 0", ["result = 1"], ["result = -1"])
            print(f"✓ If translation for {target}: OK")
        
        print("✓ Control Flow Translation: PASSED")
        return True
    except Exception as e:
        print(f"✗ Control Flow Translation: FAILED - {e}")
        return False


def test_type_mapping():
    """Test Enhancement 2: Type Mapping for Complex Types."""
    print("\n=== Testing Type Mapping ===")
    try:
        tools_dir = str(Path(__file__).parent)
        
        # Run a separate test script to verify types module works
        test_script = f'''
import sys
sys.path.insert(0, "{tools_dir}")
from stunir_types.type_system import TypeRegistry, parse_type, IntType, PointerType, ArrayType
from stunir_types.type_mapper import create_type_mapper

# Test registry
registry = TypeRegistry()
print(f"REGISTRY_TYPES:{{len(registry.types)}}")

# Test type parsing
int_type = parse_type("i32", registry)
ptr_type = parse_type("*mut i32", registry)
arr_type = parse_type("[i32; 10]", registry)
print(f"PARSE_OK:i32,*mut_i32,[i32;10]")

# Test mapping
for target in ['python', 'rust', 'haskell', 'c99']:
    mapper = create_type_mapper(target)
    int_mapped = mapper.map_type(IntType(bits=32, signed=True))
    ptr_mapped = mapper.map_type(PointerType(IntType(bits=32, signed=True)))
    print(f"MAP_{{target.upper()}}:{{int_mapped.code}},{{ptr_mapped.code}}")

print("TYPES_TEST:PASSED")
'''
        
        result = subprocess.run(
            [sys.executable, '-c', test_script],
            capture_output=True, text=True, timeout=10
        )
        
        if result.returncode != 0:
            print(f"✗ Type mapping subprocess error: {result.stderr}")
            return False
        
        output = result.stdout
        
        # Parse output
        for line in output.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                if key == 'REGISTRY_TYPES':
                    print(f"✓ Type registry initialized with {value} built-in types")
                elif key == 'PARSE_OK':
                    print(f"✓ Type parsing: {value}")
                elif key.startswith('MAP_'):
                    target = key[4:].lower()
                    parts = value.split(',')
                    print(f"✓ Type mapping for {target}: i32->{parts[0]}, *i32->{parts[1] if len(parts) > 1 else 'N/A'}")
        
        if 'TYPES_TEST:PASSED' in output:
            print("✓ Type Mapping: PASSED")
            return True
        else:
            print("✗ Type Mapping: Test did not complete")
            return False
    except Exception as e:
        print(f"✗ Type Mapping: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_semantic_analysis():
    """Test Enhancement 3: Semantic Analysis for Expressions."""
    print("\n=== Testing Semantic Analysis ===")
    try:
        from semantic import (
            SemanticAnalyzer, ExpressionParser, ConstantFolder,
            CommonSubexpressionEliminator, SemanticChecker
        )
        
        # Test semantic analyzer
        analyzer = SemanticAnalyzer()
        issues = analyzer.analyze(TEST_IR)
        print(f"✓ Semantic analysis: {len(issues)} issues found")
        
        # Test expression parser
        parser = ExpressionParser()
        expr = parser.parse({'type': 'binary', 'op': '+', 'left': 1, 'right': 2})
        print(f"✓ Expression parsing: {expr}")
        
        # Test constant folding
        folder = ConstantFolder()
        folded = folder.fold(expr)
        print(f"✓ Constant folding: 1+2 -> {folded.value if hasattr(folded, 'value') else folded}")
        
        # Test semantic checker
        checker = SemanticChecker()
        all_issues = checker.check(TEST_IR)
        summary = checker.get_summary()
        print(f"✓ Semantic checker: errors={summary['error']}, warnings={summary['warning']}")
        
        print("✓ Semantic Analysis: PASSED")
        return True
    except Exception as e:
        print(f"✗ Semantic Analysis: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_management():
    """Test Enhancement 4: Memory Management Patterns."""
    print("\n=== Testing Memory Management ===")
    try:
        from memory import (
            create_memory_manager, MemorySafetyAnalyzer,
            ManualMemoryManager, OwnershipMemoryManager
        )
        
        # Test different memory managers
        for strategy in ['manual', 'rust', 'python', 'raii']:
            manager = create_memory_manager(strategy)
            alloc_code = manager.emit_allocation('ptr', 'int', 4)
            free_code = manager.emit_deallocation('ptr')
            print(f"✓ {strategy} manager: alloc='{alloc_code[:50]}...'")
        
        # Test safety analyzer
        safety = MemorySafetyAnalyzer()
        violations = safety.analyze_ir(TEST_IR)
        print(f"✓ Safety analysis: {len(violations)} potential violations")
        
        print("✓ Memory Management: PASSED")
        return True
    except Exception as e:
        print(f"✗ Memory Management: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optimization():
    """Test Enhancement 5: Optimization Passes."""
    print("\n=== Testing Optimization Passes ===")
    try:
        from optimize import (
            create_optimizer, OptimizationLevel,
            DeadCodeEliminationPass, ConstantFoldingPass
        )
        
        # Test optimizer at different levels
        for level in ['O0', 'O1', 'O2', 'O3']:
            optimizer = create_optimizer(level)
            optimized = optimizer.optimize(TEST_IR)
            print(f"✓ {level} optimization: completed")
        
        # Test individual passes
        dce = DeadCodeEliminationPass()
        result = dce.run(TEST_IR)
        print(f"✓ Dead code elimination: {dce.stats.statements_removed} statements removed")
        
        cf = ConstantFoldingPass()
        result = cf.run(TEST_IR)
        print(f"✓ Constant folding: {cf.stats.constants_folded} constants folded")
        
        # Get stats summary
        optimizer = create_optimizer('O2')
        optimized = optimizer.optimize(TEST_IR)
        print(f"✓ Optimization summary:\n{optimizer.get_stats_summary()}")
        
        print("✓ Optimization Passes: PASSED")
        return True
    except Exception as e:
        print(f"✗ Optimization Passes: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all enhancement tests."""
    print("=" * 60)
    print("STUNIR Enhancement Test Suite")
    print("=" * 60)
    
    # Add tools to path
    tools_dir = Path(__file__).parent
    sys.path.insert(0, str(tools_dir))
    
    results = {
        'Control Flow Translation': test_control_flow(),
        'Type Mapping': test_type_mapping(),
        'Semantic Analysis': test_semantic_analysis(),
        'Memory Management': test_memory_management(),
        'Optimization Passes': test_optimization(),
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
