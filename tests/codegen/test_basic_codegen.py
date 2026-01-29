#!/usr/bin/env python3
"""STUNIR Basic Code Generation Tests.

Tests for Phase 2 code generation including statement and expression
translation for Python, Rust, Go, and C99.

Part of Phase 2 (Basic Code Generation) of STUNIR Enhancement Integration.
"""

import sys
import os
import unittest

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tools.codegen import (
    PythonCodeGenerator,
    RustCodeGenerator,
    GoCodeGenerator,
    C99CodeGenerator,
    PythonExpressionTranslator,
    RustExpressionTranslator,
    GoExpressionTranslator,
    C99ExpressionTranslator,
    get_generator,
    get_supported_targets,
)


class TestExpressionTranslation(unittest.TestCase):
    """Test expression translation for all languages."""
    
    def setUp(self):
        """Set up translators for each language."""
        self.py_expr = PythonExpressionTranslator()
        self.rs_expr = RustExpressionTranslator()
        self.go_expr = GoExpressionTranslator()
        self.c_expr = C99ExpressionTranslator()
    
    # -------------------------------------------------------------------------
    # Literal Translation Tests
    # -------------------------------------------------------------------------
    
    def test_python_integer_literal(self):
        """Test Python integer literal translation."""
        result = self.py_expr.translate_literal(42, 'i32')
        self.assertEqual(result, '42')
    
    def test_python_float_literal(self):
        """Test Python float literal translation."""
        result = self.py_expr.translate_literal(3.14, 'f64')
        self.assertEqual(result, '3.14')
    
    def test_python_string_literal(self):
        """Test Python string literal translation."""
        result = self.py_expr.translate_literal('hello', 'string')
        self.assertEqual(result, '"hello"')
    
    def test_python_bool_literal(self):
        """Test Python boolean literal translation."""
        self.assertEqual(self.py_expr.translate_literal(True, 'bool'), 'True')
        self.assertEqual(self.py_expr.translate_literal(False, 'bool'), 'False')
    
    def test_python_none_literal(self):
        """Test Python None literal translation."""
        result = self.py_expr.translate_literal(None, 'void')
        self.assertEqual(result, 'None')
    
    def test_rust_integer_literal(self):
        """Test Rust integer literal translation."""
        result = self.rs_expr.translate_literal(42, 'i32')
        self.assertEqual(result, '42_i32')
    
    def test_rust_float_literal(self):
        """Test Rust float literal translation."""
        result = self.rs_expr.translate_literal(3.14, 'f64')
        self.assertEqual(result, '3.14_f64')
    
    def test_rust_string_literal(self):
        """Test Rust String literal translation."""
        result = self.rs_expr.translate_literal('hello', 'string')
        self.assertEqual(result, '"hello".to_string()')
    
    def test_rust_bool_literal(self):
        """Test Rust boolean literal translation."""
        self.assertEqual(self.rs_expr.translate_literal(True, 'bool'), 'true')
        self.assertEqual(self.rs_expr.translate_literal(False, 'bool'), 'false')
    
    def test_go_integer_literal(self):
        """Test Go integer literal translation."""
        result = self.go_expr.translate_literal(42, 'i32')
        self.assertEqual(result, '42')
    
    def test_go_bool_literal(self):
        """Test Go boolean literal translation."""
        self.assertEqual(self.go_expr.translate_literal(True, 'bool'), 'true')
        self.assertEqual(self.go_expr.translate_literal(False, 'bool'), 'false')
    
    def test_go_nil_literal(self):
        """Test Go nil literal translation."""
        result = self.go_expr.translate_literal(None, 'void')
        self.assertEqual(result, 'nil')
    
    def test_c99_integer_literal(self):
        """Test C99 integer literal translation."""
        result = self.c_expr.translate_literal(42, 'i32')
        self.assertEqual(result, '42')
    
    def test_c99_float_literal(self):
        """Test C99 float literal with suffix."""
        result = self.c_expr.translate_literal(3.14, 'f32')
        self.assertEqual(result, '3.14f')
    
    def test_c99_bool_literal(self):
        """Test C99 boolean literal translation."""
        self.assertEqual(self.c_expr.translate_literal(True, 'bool'), 'true')
        self.assertEqual(self.c_expr.translate_literal(False, 'bool'), 'false')
    
    def test_c99_null_literal(self):
        """Test C99 NULL literal translation."""
        result = self.c_expr.translate_literal(None, 'void')
        self.assertEqual(result, 'NULL')
    
    # -------------------------------------------------------------------------
    # Binary Operation Tests
    # -------------------------------------------------------------------------
    
    def test_python_arithmetic(self):
        """Test Python arithmetic operations."""
        self.assertEqual(self.py_expr.translate_binary_op('a', '+', 'b'), '(a + b)')
        self.assertEqual(self.py_expr.translate_binary_op('a', '-', 'b'), '(a - b)')
        self.assertEqual(self.py_expr.translate_binary_op('a', '*', 'b'), '(a * b)')
        self.assertEqual(self.py_expr.translate_binary_op('a', '/', 'b'), '(a / b)')
    
    def test_python_comparison(self):
        """Test Python comparison operations."""
        self.assertEqual(self.py_expr.translate_binary_op('a', '==', 'b'), '(a == b)')
        self.assertEqual(self.py_expr.translate_binary_op('a', '<', 'b'), '(a < b)')
        self.assertEqual(self.py_expr.translate_binary_op('a', '>', 'b'), '(a > b)')
    
    def test_python_logical(self):
        """Test Python logical operations."""
        self.assertEqual(self.py_expr.translate_binary_op('a', 'and', 'b'), '(a and b)')
        self.assertEqual(self.py_expr.translate_binary_op('a', 'or', 'b'), '(a or b)')
    
    def test_rust_logical(self):
        """Test Rust logical operations."""
        self.assertEqual(self.rs_expr.translate_binary_op('a', '&&', 'b'), '(a && b)')
        self.assertEqual(self.rs_expr.translate_binary_op('a', '||', 'b'), '(a || b)')
    
    def test_go_logical(self):
        """Test Go logical operations."""
        self.assertEqual(self.go_expr.translate_binary_op('a', '&&', 'b'), '(a && b)')
        self.assertEqual(self.go_expr.translate_binary_op('a', '||', 'b'), '(a || b)')
    
    def test_c99_logical(self):
        """Test C99 logical operations."""
        self.assertEqual(self.c_expr.translate_binary_op('a', '&&', 'b'), '(a && b)')
        self.assertEqual(self.c_expr.translate_binary_op('a', '||', 'b'), '(a || b)')
    
    # -------------------------------------------------------------------------
    # Unary Operation Tests
    # -------------------------------------------------------------------------
    
    def test_python_unary_not(self):
        """Test Python unary not operation."""
        result = self.py_expr.translate_unary_op('not ', 'x')
        self.assertEqual(result, '(not x)')
    
    def test_python_unary_negation(self):
        """Test Python unary negation."""
        result = self.py_expr.translate_unary_op('-', 'x')
        self.assertEqual(result, '(-x)')
    
    def test_rust_unary_not(self):
        """Test Rust unary not operation."""
        result = self.rs_expr.translate_unary_op('!', 'x')
        self.assertEqual(result, '(!x)')
    
    def test_c99_unary_not(self):
        """Test C99 unary not operation."""
        result = self.c_expr.translate_unary_op('!', 'x')
        self.assertEqual(result, '(!x)')
    
    # -------------------------------------------------------------------------
    # Function Call Tests
    # -------------------------------------------------------------------------
    
    def test_python_function_call(self):
        """Test Python function call translation."""
        result = self.py_expr.translate_function_call('print', ['message'])
        self.assertEqual(result, 'print(message)')
    
    def test_python_method_call(self):
        """Test Python method call translation."""
        result = self.py_expr.translate_function_call('append', ['item'], 'lst')
        self.assertEqual(result, 'lst.append(item)')
    
    def test_rust_function_call(self):
        """Test Rust function call translation."""
        result = self.rs_expr.translate_function_call('println!', ['"hello"'])
        self.assertEqual(result, 'println!("hello")')
    
    def test_go_function_call(self):
        """Test Go function call translation."""
        result = self.go_expr.translate_function_call('fmt.Println', ['message'])
        self.assertEqual(result, 'fmt.Println(message)')
    
    def test_c99_function_call(self):
        """Test C99 function call translation."""
        result = self.c_expr.translate_function_call('printf', ['"%d"', 'x'])
        self.assertEqual(result, 'printf("%d", x)')
    
    # -------------------------------------------------------------------------
    # Expression IR Translation Tests
    # -------------------------------------------------------------------------
    
    def test_translate_binary_expr_ir(self):
        """Test translating binary expression IR."""
        expr_ir = {
            'type': 'binary',
            'op': '+',
            'left': {'type': 'var', 'name': 'a'},
            'right': {'type': 'var', 'name': 'b'}
        }
        
        py_result = self.py_expr.translate_expression(expr_ir)
        self.assertIn('a', py_result)
        self.assertIn('b', py_result)
        self.assertIn('+', py_result)
    
    def test_translate_nested_expr_ir(self):
        """Test translating nested expression IR."""
        expr_ir = {
            'type': 'binary',
            'op': '*',
            'left': {
                'type': 'binary',
                'op': '+',
                'left': {'type': 'var', 'name': 'a'},
                'right': {'type': 'var', 'name': 'b'}
            },
            'right': {'type': 'literal', 'value': 2, 'lit_type': 'i32'}
        }
        
        py_result = self.py_expr.translate_expression(expr_ir)
        self.assertIn('a', py_result)
        self.assertIn('b', py_result)
        self.assertIn('+', py_result)
        self.assertIn('*', py_result)


class TestStatementTranslation(unittest.TestCase):
    """Test statement translation for all languages."""
    
    def setUp(self):
        """Set up code generators for each language."""
        self.py_gen = PythonCodeGenerator()
        self.rs_gen = RustCodeGenerator()
        self.go_gen = GoCodeGenerator()
        self.c_gen = C99CodeGenerator()
    
    # -------------------------------------------------------------------------
    # Variable Declaration Tests
    # -------------------------------------------------------------------------
    
    def test_python_var_decl(self):
        """Test Python variable declaration."""
        stmt = {
            'type': 'var_decl',
            'var_name': 'x',
            'var_type': 'i32',
            'init': 0,
            'mutable': True
        }
        result = self.py_gen.stmt_translator.translate_statement(stmt)
        self.assertIn('x', result)
        self.assertIn('int', result)
        self.assertIn('0', result)
    
    def test_rust_var_decl_mutable(self):
        """Test Rust mutable variable declaration."""
        stmt = {
            'type': 'var_decl',
            'var_name': 'x',
            'var_type': 'i32',
            'init': 0,
            'mutable': True
        }
        result = self.rs_gen.stmt_translator.translate_statement(stmt)
        self.assertIn('let mut', result)
        self.assertIn('x', result)
        self.assertIn('i32', result)
    
    def test_rust_var_decl_immutable(self):
        """Test Rust immutable variable declaration."""
        stmt = {
            'type': 'var_decl',
            'var_name': 'x',
            'var_type': 'i32',
            'init': 0,
            'mutable': False
        }
        result = self.rs_gen.stmt_translator.translate_statement(stmt)
        self.assertIn('let ', result)
        self.assertNotIn('let mut', result)
    
    def test_go_var_decl_short(self):
        """Test Go short variable declaration."""
        stmt = {
            'type': 'var_decl',
            'var_name': 'x',
            'var_type': 'i32',
            'init': 0,
            'mutable': True
        }
        result = self.go_gen.stmt_translator.translate_statement(stmt)
        self.assertIn(':=', result)
        self.assertIn('x', result)
    
    def test_c99_var_decl(self):
        """Test C99 variable declaration."""
        stmt = {
            'type': 'var_decl',
            'var_name': 'x',
            'var_type': 'i32',
            'init': 0,
            'mutable': True
        }
        result = self.c_gen.stmt_translator.translate_statement(stmt)
        self.assertIn('int32_t', result)
        self.assertIn('x', result)
        self.assertIn('0', result)
        self.assertIn(';', result)
    
    # -------------------------------------------------------------------------
    # Assignment Tests
    # -------------------------------------------------------------------------
    
    def test_python_assignment(self):
        """Test Python assignment."""
        stmt = {
            'type': 'assign',
            'target': 'x',
            'value': 42
        }
        result = self.py_gen.stmt_translator.translate_statement(stmt)
        self.assertEqual(result.strip(), 'x = 42')
    
    def test_rust_assignment(self):
        """Test Rust assignment."""
        stmt = {
            'type': 'assign',
            'target': 'x',
            'value': 42
        }
        result = self.rs_gen.stmt_translator.translate_statement(stmt)
        self.assertEqual(result.strip(), 'x = 42;')
    
    def test_go_assignment(self):
        """Test Go assignment."""
        stmt = {
            'type': 'assign',
            'target': 'x',
            'value': 42
        }
        result = self.go_gen.stmt_translator.translate_statement(stmt)
        self.assertEqual(result.strip(), 'x = 42')
    
    def test_c99_assignment(self):
        """Test C99 assignment."""
        stmt = {
            'type': 'assign',
            'target': 'x',
            'value': 42
        }
        result = self.c_gen.stmt_translator.translate_statement(stmt)
        self.assertEqual(result.strip(), 'x = 42;')
    
    # -------------------------------------------------------------------------
    # Return Statement Tests
    # -------------------------------------------------------------------------
    
    def test_python_return_value(self):
        """Test Python return with value."""
        stmt = {
            'type': 'return',
            'value': 42
        }
        result = self.py_gen.stmt_translator.translate_statement(stmt)
        self.assertEqual(result.strip(), 'return 42')
    
    def test_python_return_void(self):
        """Test Python return without value."""
        stmt = {
            'type': 'return',
            'value': None
        }
        result = self.py_gen.stmt_translator.translate_statement(stmt)
        self.assertEqual(result.strip(), 'return')
    
    def test_rust_return_value(self):
        """Test Rust return with value."""
        stmt = {
            'type': 'return',
            'value': 42
        }
        result = self.rs_gen.stmt_translator.translate_statement(stmt)
        self.assertEqual(result.strip(), 'return 42;')
    
    def test_go_return_value(self):
        """Test Go return with value."""
        stmt = {
            'type': 'return',
            'value': 42
        }
        result = self.go_gen.stmt_translator.translate_statement(stmt)
        self.assertEqual(result.strip(), 'return 42')
    
    def test_c99_return_value(self):
        """Test C99 return with value."""
        stmt = {
            'type': 'return',
            'value': 42
        }
        result = self.c_gen.stmt_translator.translate_statement(stmt)
        self.assertEqual(result.strip(), 'return 42;')


class TestFunctionGeneration(unittest.TestCase):
    """Test complete function generation for all languages."""
    
    def setUp(self):
        """Set up code generators."""
        self.py_gen = PythonCodeGenerator()
        self.rs_gen = RustCodeGenerator()
        self.go_gen = GoCodeGenerator()
        self.c_gen = C99CodeGenerator()
    
    def get_add_function_ir(self):
        """Get IR for a simple add function."""
        return {
            'name': 'add',
            'params': [
                {'name': 'a', 'type': 'i32'},
                {'name': 'b', 'type': 'i32'}
            ],
            'return_type': 'i32',
            'body': [
                {
                    'type': 'return',
                    'value': {
                        'type': 'binary',
                        'op': '+',
                        'left': {'type': 'var', 'name': 'a'},
                        'right': {'type': 'var', 'name': 'b'}
                    }
                }
            ]
        }
    
    def test_python_add_function(self):
        """Test Python add function generation."""
        func_ir = self.get_add_function_ir()
        result = self.py_gen.generate_function(func_ir)
        
        self.assertIn('def add', result)
        self.assertIn('a: int', result)
        self.assertIn('b: int', result)
        self.assertIn('-> int', result)
        self.assertIn('return', result)
        self.assertIn('+', result)
    
    def test_rust_add_function(self):
        """Test Rust add function generation."""
        func_ir = self.get_add_function_ir()
        result = self.rs_gen.generate_function(func_ir)
        
        self.assertIn('fn add', result)
        self.assertIn('a: i32', result)
        self.assertIn('b: i32', result)
        self.assertIn('-> i32', result)
        self.assertIn('return', result)
    
    def test_go_add_function(self):
        """Test Go add function generation."""
        func_ir = self.get_add_function_ir()
        result = self.go_gen.generate_function(func_ir)
        
        self.assertIn('func add', result)
        self.assertIn('a int32', result)
        self.assertIn('b int32', result)
        self.assertIn('int32 {', result)
        self.assertIn('return', result)
    
    def test_c99_add_function(self):
        """Test C99 add function generation."""
        func_ir = self.get_add_function_ir()
        result = self.c_gen.generate_function(func_ir)
        
        self.assertIn('int32_t add', result)
        self.assertIn('int32_t a', result)
        self.assertIn('int32_t b', result)
        self.assertIn('return', result)
        self.assertIn(';', result)


class TestModuleGeneration(unittest.TestCase):
    """Test complete module generation for all languages."""
    
    def setUp(self):
        """Set up code generators."""
        self.py_gen = PythonCodeGenerator()
        self.rs_gen = RustCodeGenerator()
        self.go_gen = GoCodeGenerator()
        self.c_gen = C99CodeGenerator()
    
    def get_sample_module_ir(self):
        """Get IR for a sample module."""
        return {
            'ir_module': 'math_utils',
            'ir_functions': [
                {
                    'name': 'add',
                    'params': [
                        {'name': 'a', 'type': 'i32'},
                        {'name': 'b', 'type': 'i32'}
                    ],
                    'return_type': 'i32',
                    'body': [
                        {
                            'type': 'return',
                            'value': {
                                'type': 'binary',
                                'op': '+',
                                'left': {'type': 'var', 'name': 'a'},
                                'right': {'type': 'var', 'name': 'b'}
                            }
                        }
                    ]
                },
                {
                    'name': 'multiply',
                    'params': [
                        {'name': 'x', 'type': 'i32'},
                        {'name': 'y', 'type': 'i32'}
                    ],
                    'return_type': 'i32',
                    'body': [
                        {
                            'type': 'return',
                            'value': {
                                'type': 'binary',
                                'op': '*',
                                'left': {'type': 'var', 'name': 'x'},
                                'right': {'type': 'var', 'name': 'y'}
                            }
                        }
                    ]
                }
            ],
            'ir_exports': [
                {'name': 'add'},
                {'name': 'multiply'}
            ]
        }
    
    def test_python_module(self):
        """Test Python module generation."""
        module_ir = self.get_sample_module_ir()
        result = self.py_gen.generate_module(module_ir)
        
        self.assertIn('math_utils', result)
        self.assertIn('def add', result)
        self.assertIn('def multiply', result)
        self.assertIn('__all__', result)
    
    def test_rust_module(self):
        """Test Rust module generation."""
        module_ir = self.get_sample_module_ir()
        result = self.rs_gen.generate_module(module_ir)
        
        self.assertIn('math_utils', result)
        self.assertIn('fn add', result)
        self.assertIn('fn multiply', result)
    
    def test_go_module(self):
        """Test Go module generation."""
        module_ir = self.get_sample_module_ir()
        result = self.go_gen.generate_module(module_ir)
        
        self.assertIn('package', result)
        self.assertIn('func add', result)
        self.assertIn('func multiply', result)
    
    def test_c99_module(self):
        """Test C99 module generation."""
        module_ir = self.get_sample_module_ir()
        result = self.c_gen.generate_module(module_ir)
        
        self.assertIn('math_utils', result)
        self.assertIn('int32_t add', result)
        self.assertIn('int32_t multiply', result)
        self.assertIn('#include', result)


class TestGetGenerator(unittest.TestCase):
    """Test the get_generator factory function."""
    
    def test_get_python_generator(self):
        """Test getting Python generator."""
        gen = get_generator('python')
        self.assertIsInstance(gen, PythonCodeGenerator)
    
    def test_get_rust_generator(self):
        """Test getting Rust generator."""
        gen = get_generator('rust')
        self.assertIsInstance(gen, RustCodeGenerator)
    
    def test_get_go_generator(self):
        """Test getting Go generator."""
        gen = get_generator('go')
        self.assertIsInstance(gen, GoCodeGenerator)
    
    def test_get_c99_generator(self):
        """Test getting C99 generator."""
        gen = get_generator('c99')
        self.assertIsInstance(gen, C99CodeGenerator)
    
    def test_get_c_generator(self):
        """Test getting C generator (alias for C99)."""
        gen = get_generator('c')
        self.assertIsInstance(gen, C99CodeGenerator)
    
    def test_unsupported_target(self):
        """Test error for unsupported target."""
        with self.assertRaises(ValueError):
            get_generator('unsupported_language')
    
    def test_get_supported_targets(self):
        """Test getting list of supported targets."""
        targets = get_supported_targets()
        self.assertIn('python', targets)
        self.assertIn('rust', targets)
        self.assertIn('go', targets)
        self.assertIn('c99', targets)


class TestCrossLanguageComparison(unittest.TestCase):
    """Test that the same IR produces valid code across all languages."""
    
    def setUp(self):
        """Set up generators."""
        self.generators = {
            'python': PythonCodeGenerator(),
            'rust': RustCodeGenerator(),
            'go': GoCodeGenerator(),
            'c99': C99CodeGenerator()
        }
    
    def test_simple_return(self):
        """Test simple return statement across languages."""
        func_ir = {
            'name': 'get_value',
            'params': [],
            'return_type': 'i32',
            'body': [
                {'type': 'return', 'value': 42}
            ]
        }
        
        for lang, gen in self.generators.items():
            with self.subTest(lang=lang):
                result = gen.generate_function(func_ir)
                self.assertIn('get_value', result)
                self.assertIn('42', result)
    
    def test_variable_and_return(self):
        """Test variable declaration and return across languages."""
        func_ir = {
            'name': 'compute',
            'params': [],
            'return_type': 'i32',
            'body': [
                {
                    'type': 'var_decl',
                    'var_name': 'result',
                    'var_type': 'i32',
                    'init': 10,
                    'mutable': True
                },
                {
                    'type': 'return',
                    'value': {'type': 'var', 'name': 'result'}
                }
            ]
        }
        
        for lang, gen in self.generators.items():
            with self.subTest(lang=lang):
                result = gen.generate_function(func_ir)
                self.assertIn('compute', result)
                self.assertIn('result', result)
                self.assertIn('10', result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
