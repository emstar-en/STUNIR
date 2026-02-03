#!/usr/bin/env python3
"""STUNIR Advanced Code Generation Tests.

Tests for Phase 3 advanced code generation including control flow and
complex expressions for all 8 target languages.

Part of Phase 3 (Advanced Code Generation) of STUNIR Enhancement Integration.
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
    JavaScriptCodeGenerator,
    TypeScriptCodeGenerator,
    JavaCodeGenerator,
    CppCodeGenerator,
    get_generator,
    get_supported_targets,
)


class TestControlFlowGeneration(unittest.TestCase):
    """Test control flow generation for all languages."""
    
    def setUp(self):
        """Set up generators for each language."""
        self.generators = {
            'python': PythonCodeGenerator(),
            'rust': RustCodeGenerator(),
            'go': GoCodeGenerator(),
            'c99': C99CodeGenerator(),
            'javascript': JavaScriptCodeGenerator(),
            'typescript': TypeScriptCodeGenerator(),
            'java': JavaCodeGenerator(),
            'cpp': CppCodeGenerator(),
        }
    
    # -------------------------------------------------------------------------
    # If Statement Tests
    # -------------------------------------------------------------------------
    
    def test_if_statement_python(self):
        """Test Python if statement generation."""
        func_ir = {
            'name': 'is_positive',
            'params': [{'name': 'x', 'type': 'i32'}],
            'return_type': 'bool',
            'body': [
                {
                    'type': 'if',
                    'condition': {'type': 'binary', 'op': '>', 
                                 'left': {'type': 'var', 'name': 'x'},
                                 'right': 0},
                    'then': [{'type': 'return', 'value': True}],
                    'else': [{'type': 'return', 'value': False}]
                }
            ]
        }
        code = self.generators['python'].generate_function(func_ir)
        self.assertIn('if', code)
        self.assertIn('else:', code)
        self.assertIn('return True', code)
        self.assertIn('return False', code)
    
    def test_if_statement_rust(self):
        """Test Rust if statement generation."""
        func_ir = {
            'name': 'is_positive',
            'params': [{'name': 'x', 'type': 'i32'}],
            'return_type': 'bool',
            'body': [
                {
                    'type': 'if',
                    'condition': {'type': 'binary', 'op': '>', 
                                 'left': {'type': 'var', 'name': 'x'},
                                 'right': 0},
                    'then': [{'type': 'return', 'value': True}],
                    'else': [{'type': 'return', 'value': False}]
                }
            ]
        }
        code = self.generators['rust'].generate_function(func_ir)
        self.assertIn('if', code)
        self.assertIn('} else {', code)
        self.assertIn('return true', code)
        self.assertIn('return false', code)
    
    def test_if_elif_else_all_languages(self):
        """Test if-elif-else generation in all languages."""
        func_ir = {
            'name': 'sign',
            'params': [{'name': 'x', 'type': 'i32'}],
            'return_type': 'i32',
            'body': [
                {
                    'type': 'if',
                    'condition': {'type': 'binary', 'op': '>', 
                                 'left': {'type': 'var', 'name': 'x'},
                                 'right': 0},
                    'then': [{'type': 'return', 'value': 1}],
                    'elif': [
                        {
                            'condition': {'type': 'binary', 'op': '<',
                                         'left': {'type': 'var', 'name': 'x'},
                                         'right': 0},
                            'body': [{'type': 'return', 'value': -1}]
                        }
                    ],
                    'else': [{'type': 'return', 'value': 0}]
                }
            ]
        }
        
        for lang, gen in self.generators.items():
            with self.subTest(language=lang):
                code = gen.generate_function(func_ir)
                self.assertIn('if', code.lower())
    
    # -------------------------------------------------------------------------
    # While Loop Tests
    # -------------------------------------------------------------------------
    
    def test_while_loop_all_languages(self):
        """Test while loop generation in all languages."""
        func_ir = {
            'name': 'count_down',
            'params': [{'name': 'n', 'type': 'i32'}],
            'return_type': 'void',
            'body': [
                {
                    'type': 'while',
                    'condition': {'type': 'binary', 'op': '>',
                                 'left': {'type': 'var', 'name': 'n'},
                                 'right': 0},
                    'body': [
                        {
                            'type': 'assign',
                            'target': 'n',
                            'value': {'type': 'binary', 'op': '-',
                                     'left': {'type': 'var', 'name': 'n'},
                                     'right': 1}
                        }
                    ]
                }
            ]
        }
        
        for lang, gen in self.generators.items():
            with self.subTest(language=lang):
                code = gen.generate_function(func_ir)
                # Go uses 'for' for while loops
                if lang == 'go':
                    self.assertIn('for', code)
                else:
                    self.assertIn('while', code)
    
    # -------------------------------------------------------------------------
    # For Loop Tests
    # -------------------------------------------------------------------------
    
    def test_for_range_all_languages(self):
        """Test range-based for loop generation in all languages."""
        func_ir = {
            'name': 'sum_range',
            'params': [{'name': 'n', 'type': 'i32'}],
            'return_type': 'i32',
            'body': [
                {
                    'type': 'var_decl',
                    'var_name': 'total',
                    'var_type': 'i32',
                    'init': 0
                },
                {
                    'type': 'for_range',
                    'var': 'i',
                    'start': 0,
                    'end': {'type': 'var', 'name': 'n'},
                    'body': [
                        {
                            'type': 'assign',
                            'target': 'total',
                            'op': '+=',
                            'value': {'type': 'var', 'name': 'i'}
                        }
                    ]
                },
                {'type': 'return', 'value': {'type': 'var', 'name': 'total'}}
            ]
        }
        
        for lang, gen in self.generators.items():
            with self.subTest(language=lang):
                code = gen.generate_function(func_ir)
                self.assertIn('for', code)
    
    def test_for_each_all_languages(self):
        """Test for-each loop generation in all languages."""
        func_ir = {
            'name': 'print_items',
            'params': [{'name': 'items', 'type': 'array'}],
            'return_type': 'void',
            'body': [
                {
                    'type': 'for_each',
                    'var': 'item',
                    'iterable': {'type': 'var', 'name': 'items'},
                    'body': [
                        {
                            'type': 'call',
                            'func': 'print',
                            'args': [{'type': 'var', 'name': 'item'}]
                        }
                    ]
                }
            ]
        }
        
        for lang, gen in self.generators.items():
            with self.subTest(language=lang):
                code = gen.generate_function(func_ir)
                self.assertIn('for', code)
    
    # -------------------------------------------------------------------------
    # Switch/Match Tests
    # -------------------------------------------------------------------------
    
    def test_switch_statement_all_languages(self):
        """Test switch/match statement generation in all languages."""
        func_ir = {
            'name': 'day_name',
            'params': [{'name': 'day', 'type': 'i32'}],
            'return_type': 'string',
            'body': [
                {
                    'type': 'switch',
                    'value': {'type': 'var', 'name': 'day'},
                    'cases': [
                        {'value': 1, 'body': [{'type': 'return', 'value': 'Monday'}]},
                        {'value': 2, 'body': [{'type': 'return', 'value': 'Tuesday'}]},
                        {'value': 3, 'body': [{'type': 'return', 'value': 'Wednesday'}]}
                    ],
                    'default': [{'type': 'return', 'value': 'Unknown'}]
                }
            ]
        }
        
        for lang, gen in self.generators.items():
            with self.subTest(language=lang):
                code = gen.generate_function(func_ir)
                # Python uses 'match', Rust uses 'match', others use 'switch'
                self.assertTrue('switch' in code.lower() or 'match' in code.lower())


class TestComplexExpressions(unittest.TestCase):
    """Test complex expression generation."""
    
    def setUp(self):
        """Set up generators."""
        self.generators = {
            'python': PythonCodeGenerator(),
            'javascript': JavaScriptCodeGenerator(),
            'typescript': TypeScriptCodeGenerator(),
            'java': JavaCodeGenerator(),
            'cpp': CppCodeGenerator(),
        }
    
    def test_method_chain_expression(self):
        """Test method chaining expression generation."""
        func_ir = {
            'name': 'process',
            'params': [{'name': 'data', 'type': 'string'}],
            'return_type': 'string',
            'body': [
                {
                    'type': 'return',
                    'value': {
                        'type': 'chain',
                        'base': {'type': 'var', 'name': 'data'},
                        'calls': [
                            {'method': 'trim', 'args': []},
                            {'method': 'toLowerCase', 'args': []},
                        ]
                    }
                }
            ]
        }
        
        for lang, gen in self.generators.items():
            with self.subTest(language=lang):
                code = gen.generate_function(func_ir)
                self.assertIn('.trim()', code)
    
    def test_lambda_expression(self):
        """Test lambda expression generation."""
        func_ir = {
            'name': 'get_adder',
            'params': [{'name': 'x', 'type': 'i32'}],
            'return_type': 'function',
            'body': [
                {
                    'type': 'return',
                    'value': {
                        'type': 'lambda',
                        'params': [{'name': 'y', 'type': 'i32'}],
                        'body': {
                            'type': 'binary',
                            'op': '+',
                            'left': {'type': 'var', 'name': 'x'},
                            'right': {'type': 'var', 'name': 'y'}
                        }
                    }
                }
            ]
        }
        
        for lang, gen in self.generators.items():
            with self.subTest(language=lang):
                code = gen.generate_function(func_ir)
                # All should have some form of lambda/arrow syntax
                # C++ uses [&] capture syntax
                self.assertTrue('lambda' in code or '=>' in code or '->' in code or '[&]' in code)
    
    def test_ternary_expression(self):
        """Test ternary expression generation."""
        func_ir = {
            'name': 'abs',
            'params': [{'name': 'x', 'type': 'i32'}],
            'return_type': 'i32',
            'body': [
                {
                    'type': 'return',
                    'value': {
                        'type': 'ternary',
                        'condition': {'type': 'binary', 'op': '>=',
                                     'left': {'type': 'var', 'name': 'x'},
                                     'right': 0},
                        'then': {'type': 'var', 'name': 'x'},
                        'else': {'type': 'unary', 'op': '-',
                                'operand': {'type': 'var', 'name': 'x'}}
                    }
                }
            ]
        }
        
        # JavaScript/TypeScript/Java/C++ support ternary
        for lang in ['javascript', 'typescript', 'java', 'cpp']:
            gen = self.generators[lang]
            with self.subTest(language=lang):
                code = gen.generate_function(func_ir)
                self.assertIn('?', code)


class TestGetGenerator(unittest.TestCase):
    """Test the get_generator helper function."""
    
    def test_get_all_generators(self):
        """Test that all generators can be retrieved."""
        targets = get_supported_targets()
        self.assertEqual(len(targets), 8)
        
        for target in targets:
            gen = get_generator(target)
            self.assertIsNotNone(gen)
    
    def test_generator_aliases(self):
        """Test generator aliases work correctly."""
        aliases = {
            'py': 'python',
            'rs': 'rust',
            'golang': 'go',
            'c': 'c99',
            'js': 'javascript',
            'ts': 'typescript',
            'c++': 'cpp',
            'cxx': 'cpp',
        }
        
        for alias, target in aliases.items():
            with self.subTest(alias=alias):
                gen = get_generator(alias)
                self.assertIsNotNone(gen)
    
    def test_unsupported_target(self):
        """Test error for unsupported target."""
        with self.assertRaises(ValueError):
            get_generator('cobol')


class TestAllLanguagesFactorial(unittest.TestCase):
    """Test factorial function generation in all languages."""
    
    def get_factorial_ir(self):
        """Get IR for recursive factorial function."""
        return {
            'name': 'factorial',
            'params': [{'name': 'n', 'type': 'i32'}],
            'return_type': 'i32',
            'docstring': 'Compute factorial of n',
            'body': [
                {
                    'type': 'if',
                    'condition': {'type': 'binary', 'op': '<=',
                                 'left': {'type': 'var', 'name': 'n'},
                                 'right': 1},
                    'then': [{'type': 'return', 'value': 1}],
                    'else': [
                        {
                            'type': 'return',
                            'value': {
                                'type': 'binary',
                                'op': '*',
                                'left': {'type': 'var', 'name': 'n'},
                                'right': {
                                    'type': 'call',
                                    'func': 'factorial',
                                    'args': [
                                        {
                                            'type': 'binary',
                                            'op': '-',
                                            'left': {'type': 'var', 'name': 'n'},
                                            'right': 1
                                        }
                                    ]
                                }
                            }
                        }
                    ]
                }
            ]
        }
    
    def test_factorial_all_languages(self):
        """Test factorial generation in all languages."""
        func_ir = self.get_factorial_ir()
        
        for target in get_supported_targets():
            with self.subTest(target=target):
                gen = get_generator(target)
                code = gen.generate_function(func_ir)
                
                # Basic checks
                self.assertIn('factorial', code)
                self.assertIn('if', code.lower())
                self.assertIn('return', code)
                
                # Should have recursive call
                self.assertIn('factorial(', code)


class TestDoWhileLoop(unittest.TestCase):
    """Test do-while loop generation (C99, C++, Java only)."""
    
    def test_do_while_c99(self):
        """Test C99 do-while loop."""
        gen = C99CodeGenerator()
        func_ir = {
            'name': 'read_until_zero',
            'params': [],
            'return_type': 'void',
            'body': [
                {
                    'type': 'var_decl',
                    'var_name': 'x',
                    'var_type': 'i32',
                    'init': 0
                },
                {
                    'type': 'do_while',
                    'condition': {'type': 'binary', 'op': '!=',
                                 'left': {'type': 'var', 'name': 'x'},
                                 'right': 0},
                    'body': [
                        {
                            'type': 'call',
                            'func': 'scanf',
                            'args': ['"%d"', '&x']
                        }
                    ]
                }
            ]
        }
        code = gen.generate_function(func_ir)
        self.assertIn('do {', code)
        self.assertIn('} while', code)
    
    def test_do_while_java(self):
        """Test Java do-while loop."""
        gen = JavaCodeGenerator()
        func_ir = {
            'name': 'processAtLeastOnce',
            'params': [],
            'return_type': 'void',
            'body': [
                {
                    'type': 'var_decl',
                    'var_name': 'count',
                    'var_type': 'i32',
                    'init': 0
                },
                {
                    'type': 'do_while',
                    'condition': {'type': 'binary', 'op': '<',
                                 'left': {'type': 'var', 'name': 'count'},
                                 'right': 10},
                    'body': [
                        {
                            'type': 'assign',
                            'target': 'count',
                            'op': '+=',
                            'value': 1
                        }
                    ]
                }
            ]
        }
        code = gen.generate_function(func_ir)
        self.assertIn('do {', code)
        self.assertIn('} while', code)


class TestInfiniteLoop(unittest.TestCase):
    """Test infinite loop generation."""
    
    def test_infinite_loop_rust(self):
        """Test Rust loop {} syntax."""
        gen = RustCodeGenerator()
        func_ir = {
            'name': 'event_loop',
            'params': [],
            'return_type': 'void',
            'body': [
                {
                    'type': 'loop',
                    'body': [
                        {
                            'type': 'call',
                            'func': 'process_event',
                            'args': []
                        }
                    ]
                }
            ]
        }
        code = gen.generate_function(func_ir)
        self.assertIn('loop {', code)
    
    def test_infinite_loop_go(self):
        """Test Go for {} syntax."""
        gen = GoCodeGenerator()
        func_ir = {
            'name': 'eventLoop',
            'params': [],
            'return_type': 'void',
            'body': [
                {
                    'type': 'loop',
                    'body': [
                        {
                            'type': 'call',
                            'func': 'processEvent',
                            'args': []
                        }
                    ]
                }
            ]
        }
        code = gen.generate_function(func_ir)
        self.assertIn('for {', code)


if __name__ == '__main__':
    unittest.main()
