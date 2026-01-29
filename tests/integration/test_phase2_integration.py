#!/usr/bin/env python3
"""STUNIR Phase 2 Integration Tests.

End-to-end tests for Phase 2 code generation, testing the full pipeline from
IR → Enhancements → Code Generation for all 4 target languages.

Part of Phase 2 (Basic Code Generation) of STUNIR Enhancement Integration.
"""

import sys
import os
import json
import tempfile
import unittest

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tools.codegen import (
    PythonCodeGenerator,
    RustCodeGenerator,
    GoCodeGenerator,
    C99CodeGenerator,
    get_generator,
    get_supported_targets,
)


class TestPhase2Integration(unittest.TestCase):
    """Integration tests for Phase 2 code generation."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Sample cJSON-like IR data
        cls.cjson_ir = {
            'ir_module': 'cjson',
            'schema': 'stunir.ir.v1',
            'ir_types': [
                {
                    'name': 'cJSON',
                    'fields': [
                        {'name': 'next', 'type': 'cJSON*'},
                        {'name': 'prev', 'type': 'cJSON*'},
                        {'name': 'child', 'type': 'cJSON*'},
                        {'name': 'type', 'type': 'i32'},
                        {'name': 'valuestring', 'type': 'string'},
                        {'name': 'valueint', 'type': 'i32'},
                        {'name': 'valuedouble', 'type': 'f64'}
                    ]
                }
            ],
            'ir_functions': [
                {
                    'name': 'cJSON_CreateObject',
                    'params': [],
                    'return_type': 'cJSON*',
                    'description': 'Create a cJSON object',
                    'body': [
                        {
                            'type': 'var_decl',
                            'var_name': 'item',
                            'var_type': 'cJSON*',
                            'init': None,
                            'mutable': True
                        },
                        {
                            'type': 'assign',
                            'target': 'item',
                            'value': {
                                'type': 'call',
                                'func': 'cJSON_New_Item',
                                'args': []
                            }
                        },
                        {
                            'type': 'return',
                            'value': {'type': 'var', 'name': 'item'}
                        }
                    ]
                },
                {
                    'name': 'cJSON_CreateNumber',
                    'params': [
                        {'name': 'num', 'type': 'f64'}
                    ],
                    'return_type': 'cJSON*',
                    'description': 'Create a cJSON number',
                    'body': [
                        {
                            'type': 'var_decl',
                            'var_name': 'item',
                            'var_type': 'cJSON*',
                            'init': None,
                            'mutable': True
                        },
                        {
                            'type': 'assign',
                            'target': 'item',
                            'value': {
                                'type': 'call',
                                'func': 'cJSON_New_Item',
                                'args': []
                            }
                        },
                        {
                            'type': 'assign',
                            'target': 'item.valuedouble',
                            'value': {'type': 'var', 'name': 'num'}
                        },
                        {
                            'type': 'return',
                            'value': {'type': 'var', 'name': 'item'}
                        }
                    ]
                },
                {
                    'name': 'cJSON_GetArraySize',
                    'params': [
                        {'name': 'array', 'type': 'cJSON*'}
                    ],
                    'return_type': 'i32',
                    'description': 'Get the size of a cJSON array',
                    'body': [
                        {
                            'type': 'var_decl',
                            'var_name': 'size',
                            'var_type': 'i32',
                            'init': 0,
                            'mutable': True
                        },
                        {
                            'type': 'var_decl',
                            'var_name': 'child',
                            'var_type': 'cJSON*',
                            'init': {'type': 'member', 'base': {'type': 'var', 'name': 'array'}, 'member': 'child'},
                            'mutable': True
                        },
                        {
                            'type': 'return',
                            'value': {'type': 'var', 'name': 'size'}
                        }
                    ]
                },
                {
                    'name': 'cJSON_IsNumber',
                    'params': [
                        {'name': 'item', 'type': 'cJSON*'}
                    ],
                    'return_type': 'bool',
                    'description': 'Check if item is a number',
                    'body': [
                        {
                            'type': 'return',
                            'value': {
                                'type': 'binary',
                                'op': '==',
                                'left': {
                                    'type': 'member',
                                    'base': {'type': 'var', 'name': 'item'},
                                    'member': 'type'
                                },
                                'right': {'type': 'literal', 'value': 8, 'lit_type': 'i32'}
                            }
                        }
                    ]
                }
            ],
            'ir_exports': [
                {'name': 'cJSON_CreateObject'},
                {'name': 'cJSON_CreateNumber'},
                {'name': 'cJSON_GetArraySize'},
                {'name': 'cJSON_IsNumber'}
            ]
        }
    
    def test_all_targets_generate_code(self):
        """Test that all 4 target languages generate code."""
        for target in get_supported_targets():
            with self.subTest(target=target):
                gen = get_generator(target)
                code = gen.generate_module(self.cjson_ir)
                
                # Verify code is not empty
                self.assertTrue(len(code) > 0, f"{target} generated empty code")
                
                # Verify function names appear
                self.assertIn('cJSON_CreateObject', code)
                self.assertIn('cJSON_CreateNumber', code)
    
    def test_python_code_is_syntactically_correct(self):
        """Test that generated Python code has valid syntax."""
        gen = get_generator('python')
        code = gen.generate_module(self.cjson_ir)
        
        # Try to compile the code
        try:
            compile(code, '<generated>', 'exec')
        except SyntaxError as e:
            self.fail(f"Generated Python code has syntax error: {e}")
    
    def test_all_targets_have_function_bodies(self):
        """Test that generated code has actual function bodies, not stubs."""
        for target in get_supported_targets():
            with self.subTest(target=target):
                gen = get_generator(target)
                code = gen.generate_module(self.cjson_ir)
                
                # Count return statements
                return_count = code.lower().count('return')
                
                # Should have at least as many returns as functions
                self.assertGreaterEqual(return_count, 4, 
                    f"{target} code seems to have stub bodies (few returns)")
    
    def test_type_mappings_consistent(self):
        """Test that type mappings are consistent across operations."""
        func_ir = {
            'name': 'test_types',
            'params': [
                {'name': 'int_val', 'type': 'i32'},
                {'name': 'float_val', 'type': 'f64'},
                {'name': 'bool_val', 'type': 'bool'},
            ],
            'return_type': 'i32',
            'body': [
                {'type': 'return', 'value': {'type': 'var', 'name': 'int_val'}}
            ]
        }
        
        type_patterns = {
            'python': ['int', 'float', 'bool'],
            'rust': ['i32', 'f64', 'bool'],
            'go': ['int32', 'float64', 'bool'],
            'c99': ['int32_t', 'double', 'bool'],
        }
        
        for target in get_supported_targets():
            with self.subTest(target=target):
                gen = get_generator(target)
                code = gen.generate_function(func_ir)
                
                for pattern in type_patterns[target]:
                    self.assertIn(pattern, code, 
                        f"{target} missing expected type pattern: {pattern}")
    
    def test_binary_operations_all_languages(self):
        """Test binary operation generation across all languages."""
        func_ir = {
            'name': 'test_binary_ops',
            'params': [
                {'name': 'a', 'type': 'i32'},
                {'name': 'b', 'type': 'i32'},
            ],
            'return_type': 'i32',
            'body': [
                {
                    'type': 'var_decl',
                    'var_name': 'sum',
                    'var_type': 'i32',
                    'init': {
                        'type': 'binary',
                        'op': '+',
                        'left': {'type': 'var', 'name': 'a'},
                        'right': {'type': 'var', 'name': 'b'}
                    },
                    'mutable': True
                },
                {
                    'type': 'return',
                    'value': {'type': 'var', 'name': 'sum'}
                }
            ]
        }
        
        for target in get_supported_targets():
            with self.subTest(target=target):
                gen = get_generator(target)
                code = gen.generate_function(func_ir)
                
                # Verify arithmetic operator present
                self.assertIn('+', code, f"{target} missing + operator")
                # Verify variable names present
                self.assertIn('a', code)
                self.assertIn('b', code)
                self.assertIn('sum', code)
    
    def test_comparison_operations(self):
        """Test comparison operation generation."""
        func_ir = {
            'name': 'is_positive',
            'params': [{'name': 'x', 'type': 'i32'}],
            'return_type': 'bool',
            'body': [
                {
                    'type': 'return',
                    'value': {
                        'type': 'binary',
                        'op': '>',
                        'left': {'type': 'var', 'name': 'x'},
                        'right': {'type': 'literal', 'value': 0, 'lit_type': 'i32'}
                    }
                }
            ]
        }
        
        for target in get_supported_targets():
            with self.subTest(target=target):
                gen = get_generator(target)
                code = gen.generate_function(func_ir)
                
                self.assertIn('>', code, f"{target} missing > operator")
                self.assertIn('0', code, f"{target} missing literal 0")
    
    def test_logical_operations_python(self):
        """Test Python logical operations use Python keywords."""
        gen = get_generator('python')
        func_ir = {
            'name': 'test_logic',
            'params': [
                {'name': 'a', 'type': 'bool'},
                {'name': 'b', 'type': 'bool'},
            ],
            'return_type': 'bool',
            'body': [
                {
                    'type': 'return',
                    'value': {
                        'type': 'binary',
                        'op': 'and',
                        'left': {'type': 'var', 'name': 'a'},
                        'right': {'type': 'var', 'name': 'b'}
                    }
                }
            ]
        }
        
        code = gen.generate_function(func_ir)
        self.assertIn(' and ', code, "Python should use 'and' keyword")
    
    def test_logical_operations_c_like(self):
        """Test C-like languages use && and ||."""
        for target in ['rust', 'go', 'c99']:
            with self.subTest(target=target):
                gen = get_generator(target)
                func_ir = {
                    'name': 'test_logic',
                    'params': [
                        {'name': 'a', 'type': 'bool'},
                        {'name': 'b', 'type': 'bool'},
                    ],
                    'return_type': 'bool',
                    'body': [
                        {
                            'type': 'return',
                            'value': {
                                'type': 'binary',
                                'op': '&&',
                                'left': {'type': 'var', 'name': 'a'},
                                'right': {'type': 'var', 'name': 'b'}
                            }
                        }
                    ]
                }
                
                code = gen.generate_function(func_ir)
                self.assertIn('&&', code, f"{target} should use && operator")
    
    def test_cjson_full_module_python(self):
        """Test full cJSON module generation for Python."""
        gen = get_generator('python')
        code = gen.generate_module(self.cjson_ir)
        
        # Verify module structure
        self.assertIn('Module: cjson', code)
        self.assertIn('from typing import', code)
        
        # Verify all functions are present with bodies
        for func_name in ['cJSON_CreateObject', 'cJSON_CreateNumber', 
                          'cJSON_GetArraySize', 'cJSON_IsNumber']:
            self.assertIn(f'def {func_name}', code)
        
        # Verify there are actual statements (not just pass)
        self.assertIn('return', code)
        
        # Verify type annotations
        self.assertIn('float', code)  # f64 -> float
        self.assertIn('bool', code)
    
    def test_cjson_full_module_rust(self):
        """Test full cJSON module generation for Rust."""
        gen = get_generator('rust')
        code = gen.generate_module(self.cjson_ir)
        
        # Verify module structure
        self.assertIn('Module: cjson', code)
        
        # Verify all functions
        for func_name in ['cJSON_CreateObject', 'cJSON_CreateNumber', 
                          'cJSON_GetArraySize', 'cJSON_IsNumber']:
            self.assertIn(f'fn {func_name}', code)
        
        # Verify Rust syntax
        self.assertIn('let', code)
        self.assertIn('return', code)
    
    def test_cjson_full_module_go(self):
        """Test full cJSON module generation for Go."""
        gen = get_generator('go')
        code = gen.generate_module(self.cjson_ir)
        
        # Verify package declaration
        self.assertIn('package', code)
        
        # Verify all functions
        for func_name in ['cJSON_CreateObject', 'cJSON_CreateNumber', 
                          'cJSON_GetArraySize', 'cJSON_IsNumber']:
            self.assertIn(f'func {func_name}', code)
        
        # Verify Go syntax
        self.assertIn('return', code)
    
    def test_cjson_full_module_c99(self):
        """Test full cJSON module generation for C99."""
        gen = get_generator('c99')
        code = gen.generate_module(self.cjson_ir)
        
        # Verify includes
        self.assertIn('#include <stdint.h>', code)
        self.assertIn('#include <stdbool.h>', code)
        
        # Verify all functions
        for func_name in ['cJSON_CreateObject', 'cJSON_CreateNumber', 
                          'cJSON_GetArraySize', 'cJSON_IsNumber']:
            self.assertIn(func_name, code)
        
        # Verify C syntax
        self.assertIn('return', code)
        self.assertIn(';', code)
    
    def test_c99_header_generation(self):
        """Test C99 header file generation."""
        gen = C99CodeGenerator()
        header = gen.generate_header(self.cjson_ir)
        
        # Verify header guard
        self.assertIn('#ifndef', header)
        self.assertIn('#define', header)
        self.assertIn('#endif', header)
        
        # Verify function declarations
        self.assertIn('cJSON_CreateObject', header)
        self.assertIn(';', header)
    
    def test_output_line_counts(self):
        """Test that generated code has reasonable line counts."""
        for target in get_supported_targets():
            with self.subTest(target=target):
                gen = get_generator(target)
                code = gen.generate_module(self.cjson_ir)
                line_count = len(code.splitlines())
                
                # Should have at least 20 lines (not empty stubs)
                self.assertGreater(line_count, 20, 
                    f"{target} generated too few lines ({line_count})")
                
                # Should have less than 500 lines (sanity check)
                self.assertLess(line_count, 500, 
                    f"{target} generated too many lines ({line_count})")


class TestEnhancementContextIntegration(unittest.TestCase):
    """Test integration with EnhancementContext."""
    
    def test_generator_accepts_none_context(self):
        """Test that generators work without enhancement context."""
        for target in get_supported_targets():
            with self.subTest(target=target):
                gen = get_generator(target, enhancement_context=None)
                self.assertIsNotNone(gen)
                
                # Should still generate code
                func_ir = {
                    'name': 'test',
                    'params': [],
                    'return_type': 'void',
                    'body': []
                }
                code = gen.generate_function(func_ir)
                self.assertIn('test', code)
    
    def test_generator_with_mock_context(self):
        """Test generators with a mock enhancement context."""
        # Create a minimal mock context
        class MockContext:
            def lookup_variable(self, name):
                return None
            def lookup_function(self, name):
                return None
            def get_expression_type(self, expr_id):
                return None
        
        mock_ctx = MockContext()
        
        for target in get_supported_targets():
            with self.subTest(target=target):
                gen = get_generator(target, enhancement_context=mock_ctx)
                self.assertIsNotNone(gen)
                
                func_ir = {
                    'name': 'with_context',
                    'params': [{'name': 'x', 'type': 'i32'}],
                    'return_type': 'i32',
                    'body': [
                        {'type': 'return', 'value': {'type': 'var', 'name': 'x'}}
                    ]
                }
                code = gen.generate_function(func_ir)
                self.assertIn('with_context', code)


class TestErrorHandling(unittest.TestCase):
    """Test error handling in code generation."""
    
    def test_unknown_statement_type(self):
        """Test handling of unknown statement types."""
        gen = PythonCodeGenerator()
        func_ir = {
            'name': 'test',
            'params': [],
            'return_type': 'void',
            'body': [
                {'type': 'unknown_stmt_type', 'data': 'test'}
            ]
        }
        
        # Should not raise, should include TODO comment
        code = gen.generate_function(func_ir)
        self.assertIn('TODO', code)
    
    def test_empty_function_body(self):
        """Test handling of empty function body."""
        for target in get_supported_targets():
            with self.subTest(target=target):
                gen = get_generator(target)
                func_ir = {
                    'name': 'empty_func',
                    'params': [],
                    'return_type': 'void',
                    'body': []
                }
                
                # Should not raise
                code = gen.generate_function(func_ir)
                self.assertIn('empty_func', code)
    
    def test_missing_init_value(self):
        """Test handling of variable declaration without init value."""
        for target in get_supported_targets():
            with self.subTest(target=target):
                gen = get_generator(target)
                func_ir = {
                    'name': 'test',
                    'params': [],
                    'return_type': 'void',
                    'body': [
                        {
                            'type': 'var_decl',
                            'var_name': 'x',
                            'var_type': 'i32',
                            'mutable': True
                            # No 'init' key
                        }
                    ]
                }
                
                # Should not raise
                code = gen.generate_function(func_ir)
                self.assertIn('x', code)


if __name__ == '__main__':
    unittest.main(verbosity=2)
