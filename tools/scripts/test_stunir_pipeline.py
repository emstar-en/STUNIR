#!/usr/bin/env python3
"""
STUNIR Pipeline Test Suite (Phase 5)
Comprehensive tests for the STUNIR pipeline bridge scripts.
"""

import json
import sys
import tempfile
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from bridge_spec_assemble import assemble_spec, validate_extraction_json
from bridge_spec_to_ir import convert_spec_to_ir, validate_spec_json
from bridge_ir_to_code import generate_code, validate_ir_json, get_file_extension


def test_extraction_validation():
    """Test extraction.json validation."""
    print("Testing extraction.json validation...")
    
    valid_extraction = {
        "kind": "stunir.extraction.v1",
        "meta": {"source_index": "test.c"},
        "extractions": [
            {
                "source_file": "test.c",
                "functions": [
                    {
                        "name": "test_func",
                        "signature": {
                            "return_type": "int",
                            "args": [{"name": "x", "type": "int"}]
                        }
                    }
                ]
            }
        ]
    }
    
    assert validate_extraction_json(valid_extraction) == True
    
    # Test invalid extraction (missing kind)
    invalid_extraction = {"meta": {}, "extractions": []}
    try:
        validate_extraction_json(invalid_extraction)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    print("  ✓ Extraction validation tests passed")


def test_spec_assembly():
    """Test spec.json assembly from extraction.json."""
    print("Testing spec.json assembly...")
    
    extraction_data = {
        "kind": "stunir.extraction.v1",
        "meta": {"source_index": "test.c"},
        "extractions": [
            {
                "source_file": "test.c",
                "functions": [
                    {
                        "name": "add",
                        "signature": {
                            "return_type": "int",
                            "args": [
                                {"name": "a", "type": "int"},
                                {"name": "b", "type": "int"}
                            ]
                        }
                    },
                    {
                        "name": "greet",
                        "signature": {
                            "return_type": "void",
                            "args": [{"name": "name", "type": "char *"}]
                        }
                    }
                ]
            }
        ]
    }
    
    spec = assemble_spec(extraction_data, "test_module")
    
    assert spec["kind"] == "stunir.spec.v1"
    assert spec["meta"]["origin"] == "bridge_spec_assemble"
    assert len(spec["modules"]) == 1
    assert spec["modules"][0]["name"] == "test_module"
    assert len(spec["modules"][0]["functions"]) == 2
    
    # Check function signatures preserved
    add_func = spec["modules"][0]["functions"][0]
    assert add_func["name"] == "add"
    assert add_func["signature"]["return_type"] == "int"
    assert len(add_func["signature"]["args"]) == 2
    assert add_func["signature"]["args"][0]["name"] == "a"
    
    print("  ✓ Spec assembly tests passed")


def test_ir_conversion():
    """Test IR conversion from spec.json."""
    print("Testing IR conversion...")
    
    spec_data = {
        "kind": "stunir.spec.v1",
        "meta": {"origin": "test"},
        "modules": [
            {
                "name": "test_module",
                "functions": [
                    {
                        "name": "multiply",
                        "signature": {
                            "return_type": "int",
                            "args": [
                                {"name": "x", "type": "int"},
                                {"name": "y", "type": "int"}
                            ]
                        }
                    }
                ]
            }
        ]
    }
    
    ir = convert_spec_to_ir(spec_data, "test_module")
    
    assert ir["schema"] == "stunir_flat_ir_v1"
    assert ir["module_name"] == "test_module"
    assert len(ir["functions"]) == 1
    
    func = ir["functions"][0]
    assert func["name"] == "multiply"
    assert func["return_type"] == "int"
    assert len(func["args"]) == 2
    assert func["args"][0]["name"] == "x"
    assert func["args"][0]["type"] == "int"
    assert len(func["steps"]) == 1
    assert func["steps"][0]["op"] == "noop"
    
    print("  ✓ IR conversion tests passed")


def test_code_generation_cpp():
    """Test C++ code generation."""
    print("Testing C++ code generation...")
    
    ir_data = {
        "schema": "stunir_flat_ir_v1",
        "ir_version": "v1",
        "module_name": "math",
        "types": [],
        "functions": [
            {
                "name": "add",
                "args": [
                    {"name": "a", "type": "int"},
                    {"name": "b", "type": "int"}
                ],
                "return_type": "int",
                "steps": [{"op": "noop"}]
            },
            {
                "name": "get_pi",
                "args": [],
                "return_type": "double",
                "steps": [{"op": "noop"}]
            }
        ]
    }
    
    code = generate_code(ir_data, "cpp")
    
    # Check includes
    assert "#include <stdint.h>" in code
    assert "#include <stdbool.h>" in code
    
    # Check function signatures
    assert "int add(int a, int b)" in code
    assert "double get_pi()" in code
    
    # Check function bodies
    assert "// TODO: Implement function body" in code
    
    print("  ✓ C++ code generation tests passed")


def test_code_generation_multi_target():
    """Test code generation for multiple targets."""
    print("Testing multi-target code generation...")
    
    ir_data = {
        "schema": "stunir_flat_ir_v1",
        "ir_version": "v1",
        "module_name": "test",
        "types": [],
        "functions": [
            {
                "name": "process",
                "args": [{"name": "data", "type": "char *"}],
                "return_type": "int",
                "steps": [{"op": "noop"}]
            }
        ]
    }
    
    targets = ["cpp", "c", "python", "rust", "go"]
    
    for target in targets:
        code = generate_code(ir_data, target)
        assert len(code) > 0, f"Empty code generated for {target}"
        assert "process" in code, f"Function name missing in {target} output"
        print(f"    ✓ {target} generation OK")
    
    print("  ✓ Multi-target code generation tests passed")


def test_file_extensions():
    """Test file extension mapping."""
    print("Testing file extensions...")
    
    extensions = {
        "cpp": ".cpp",
        "c": ".c",
        "python": ".py",
        "rust": ".rs",
        "go": ".go",
        "javascript": ".js",
        "java": ".java",
        "csharp": ".cs",
        "swift": ".swift",
        "kotlin": ".kt"
    }
    
    for target, expected_ext in extensions.items():
        ext = get_file_extension(target)
        assert ext == expected_ext, f"Wrong extension for {target}: got {ext}, expected {expected_ext}"
    
    print("  ✓ File extension tests passed")


def test_end_to_end():
    """Test complete pipeline end-to-end."""
    print("Testing end-to-end pipeline...")
    
    # Create test extraction data
    extraction_data = {
        "kind": "stunir.extraction.v1",
        "meta": {"source_index": "example.c"},
        "extractions": [
            {
                "source_file": "example.c",
                "functions": [
                    {
                        "name": "calculate_sum",
                        "signature": {
                            "return_type": "int",
                            "args": [
                                {"name": "numbers", "type": "int *"},
                                {"name": "count", "type": "size_t"}
                            ]
                        }
                    }
                ]
            }
        ]
    }
    
    # Phase 1: extraction -> spec
    spec = assemble_spec(extraction_data, "example")
    assert spec["kind"] == "stunir.spec.v1"
    
    # Phase 2: spec -> ir
    ir = convert_spec_to_ir(spec, "example")
    assert ir["schema"] == "stunir_flat_ir_v1"
    
    # Phase 3: ir -> code
    cpp_code = generate_code(ir, "cpp")
    assert "int calculate_sum(int * numbers, size_t count)" in cpp_code
    
    print("  ✓ End-to-end pipeline test passed")


def test_pointer_types():
    """Test handling of pointer types."""
    print("Testing pointer type handling...")

    ir_data = {
        "schema": "stunir_flat_ir_v1",
        "ir_version": "v1",
        "module_name": "ptr_test",
        "types": [],
        "functions": [
            {
                "name": "process_buffer",
                "args": [
                    {"name": "buf", "type": "char *"},
                    {"name": "ptr", "type": "void *"},
                    {"name": "data", "type": "int **"}
                ],
                "return_type": "void",
                "steps": [{"op": "noop"}]
            }
        ]
    }
    
    # Test C++
    cpp_code = generate_code(ir_data, "cpp")
    assert "char * buf" in cpp_code
    assert "void * ptr" in cpp_code
    assert "int ** data" in cpp_code
    
    # Test Python (pointers become references)
    py_code = generate_code(ir_data, "python")
    assert "buf: str" in py_code or "buf" in py_code
    
    print("  ✓ Pointer type tests passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("STUNIR Pipeline Test Suite")
    print("=" * 70 + "\n")
    
    tests = [
        test_extraction_validation,
        test_spec_assembly,
        test_ir_conversion,
        test_code_generation_cpp,
        test_code_generation_multi_target,
        test_file_extensions,
        test_pointer_types,
        test_end_to_end,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
