#!/usr/bin/env python3
"""
STUNIR Integration Test Suite

Tests the complete pipeline from IR generation to code generation.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))

from semantic_ir.emitters.types import IRModule, IRFunction, IRParameter, IRDataType
from ir_to_code import emit_general_language, map_type, default_return


class TestIRGeneration(unittest.TestCase):
    """Test IR generation and validation."""
    
    def test_simple_add_function(self):
        """Test generating IR for a simple add function."""
        ir_module = {
            "ir_version": "0.8.9",
            "module_name": "test_add",
            "types": [],
            "functions": [{
                "name": "add",
                "return_type": "i32",
                "parameters": [
                    {"name": "a", "param_type": "i32"},
                    {"name": "b", "param_type": "i32"}
                ],
                "docstring": "Add two integers"
            }]
        }
        
        # Validate structure
        self.assertEqual(ir_module["ir_version"], "0.8.9")
        self.assertEqual(ir_module["module_name"], "test_add")
        self.assertEqual(len(ir_module["functions"]), 1)
        
        func = ir_module["functions"][0]
        self.assertEqual(func["name"], "add")
        self.assertEqual(func["return_type"], "i32")
        self.assertEqual(len(func["parameters"]), 2)


class TestTypeMapping(unittest.TestCase):
    """Test type mapping from IR to target languages."""
    
    def test_c_type_mapping(self):
        """Test C type mapping."""
        self.assertEqual(map_type("i32", "c"), "int32_t")
        self.assertEqual(map_type("i64", "c"), "int64_t")
        self.assertEqual(map_type("f32", "c"), "float")
        self.assertEqual(map_type("f64", "c"), "double")
        self.assertEqual(map_type("void", "c"), "void")
    
    def test_rust_type_mapping(self):
        """Test Rust type mapping."""
        self.assertEqual(map_type("i32", "rust"), "i32")
        self.assertEqual(map_type("i64", "rust"), "i64")
        self.assertEqual(map_type("f32", "rust"), "f32")
        self.assertEqual(map_type("f64", "rust"), "f64")
        self.assertEqual(map_type("void", "rust"), "()")
    
    def test_python_type_mapping(self):
        """Test Python type mapping."""
        self.assertEqual(map_type("i32", "python"), "int")
        self.assertEqual(map_type("i64", "python"), "int")
        self.assertEqual(map_type("f32", "python"), "float")
        self.assertEqual(map_type("f64", "python"), "float")
        self.assertEqual(map_type("void", "python"), "None")


class TestCodeGeneration(unittest.TestCase):
    """Test code generation for different languages."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_module = IRModule(
            ir_version="0.8.9",
            module_name="test_module",
            types=[],
            functions=[
                IRFunction(
                    name="add",
                    return_type=IRDataType.I32,
                    parameters=[
                        IRParameter(name="a", param_type=IRDataType.I32),
                        IRParameter(name="b", param_type=IRDataType.I32)
                    ],
                    statements=[],
                    docstring="Add two integers"
                )
            ],
            docstring=None
        )
    
    def test_c_code_generation(self):
        """Test C code generation."""
        code = emit_general_language(self.test_module, "c")
        
        # Check for expected content
        self.assertIn("// STUNIR Generated Code", code)
        self.assertIn("// Target Language: c", code)
        self.assertIn("int32_t add(int32_t a, int32_t b)", code)
        self.assertIn("// Add two integers", code)
        self.assertIn("return 0;", code)
    
    def test_rust_code_generation(self):
        """Test Rust code generation."""
        code = emit_general_language(self.test_module, "rust")
        
        # Check for expected content
        self.assertIn("// STUNIR Generated Code", code)
        self.assertIn("// Target Language: rust", code)
        self.assertIn("pub fn add(a: i32, b: i32) -> i32", code)
        self.assertIn("/// Add two integers", code)
        self.assertIn("return 0;", code)
    
    def test_python_code_generation(self):
        """Test Python code generation."""
        code = emit_general_language(self.test_module, "python")
        
        # Check for expected content
        self.assertIn("# STUNIR Generated Code", code)
        self.assertIn("# Target Language: python", code)
        self.assertIn("def add(a: int, b: int) -> int:", code)
        self.assertIn('"""Add two integers"""', code)
        self.assertIn("return 0", code)


class TestDefaultValues(unittest.TestCase):
    """Test default return values."""
    
    def test_c_defaults(self):
        """Test C default values."""
        self.assertEqual(default_return("i32", "c"), "0")
        self.assertEqual(default_return("f32", "c"), "0.0f")
        self.assertEqual(default_return("f64", "c"), "0.0")
        self.assertEqual(default_return("bool", "c"), "false")
    
    def test_rust_defaults(self):
        """Test Rust default values."""
        self.assertEqual(default_return("i32", "rust"), "0")
        self.assertEqual(default_return("f32", "rust"), "0.0")
        self.assertEqual(default_return("f64", "rust"), "0.0")
        self.assertEqual(default_return("bool", "rust"), "false")
    
    def test_python_defaults(self):
        """Test Python default values."""
        self.assertEqual(default_return("i32", "python"), "0")
        self.assertEqual(default_return("f32", "python"), "0.0")
        self.assertEqual(default_return("f64", "python"), "0.0")
        self.assertEqual(default_return("bool", "python"), "False")


class TestEndToEndPipeline(unittest.TestCase):
    """Test the complete end-to-end pipeline."""
    
    def test_full_pipeline(self):
        """Test the complete pipeline from JSON IR to generated code."""
        # Create a temporary JSON file
        ir_data = {
            "ir_version": "0.8.9",
            "module_name": "math_ops",
            "types": [],
            "functions": [
                {
                    "name": "multiply",
                    "return_type": "i32",
                    "parameters": [
                        {"name": "x", "param_type": "i32"},
                        {"name": "y", "param_type": "i32"}
                    ],
                    "docstring": "Multiply two integers"
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(ir_data, f)
            temp_path = f.name
        
        try:
            # Import and test the full pipeline
            from ir_to_code import json_to_ir_module, emit_general_language
            
            # Load IR from JSON
            with open(temp_path, 'r') as f:
                json_data = json.load(f)
            
            ir_module = json_to_ir_module(json_data)
            
            # Verify IR module
            self.assertEqual(ir_module.module_name, "math_ops")
            self.assertEqual(len(ir_module.functions), 1)
            
            func = ir_module.functions[0]
            self.assertEqual(func.name, "multiply")
            self.assertEqual(func.return_type, IRDataType.I32)
            self.assertEqual(len(func.parameters), 2)
            
            # Generate code for multiple languages
            c_code = emit_general_language(ir_module, "c")
            rust_code = emit_general_language(ir_module, "rust")
            python_code = emit_general_language(ir_module, "python")
            
            # Verify generated code
            self.assertIn("int32_t multiply(int32_t x, int32_t y)", c_code)
            self.assertIn("pub fn multiply(x: i32, y: i32) -> i32", rust_code)
            self.assertIn("def multiply(x: int, y: int) -> int:", python_code)
            
        finally:
            os.unlink(temp_path)


def run_tests():
    """Run the integration test suite."""
    print("\n" + "=" * 60)
    print("STUNIR Integration Test Suite")
    print("=" * 60 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestIRGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestTypeMapping))
    suite.addTests(loader.loadTestsFromTestCase(TestCodeGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestDefaultValues))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndPipeline))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n[OK] All tests passed!")
        return 0
    else:
        print("\n[FAIL] Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
