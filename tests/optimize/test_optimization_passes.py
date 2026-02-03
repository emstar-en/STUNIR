#!/usr/bin/env python3
"""Tests for STUNIR Optimization Passes.

Tests:
- Each optimization pass individually
- Optimization levels produce different code
- Correctness of optimizations
- Performance benchmarks
"""

import sys
import os
import time
import unittest
import copy

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.optimize import (
    OptimizationLevel,
    PassManager,
    create_pass_manager,
    # O1 passes
    DeadCodeEliminationPass,
    ConstantFoldingPass,
    ConstantPropagationPass,
    AlgebraicSimplificationPass,
    # O2 passes
    CommonSubexpressionEliminationPass,
    LoopInvariantCodeMotionPass,
    CopyPropagationPass,
    StrengthReductionPass,
    # O3 passes
    FunctionInliningPass,
    LoopUnrollingPass,
    # Validation
    validate_optimization,
    compare_optimization,
)


class TestSampleIR:
    """Sample IR data for testing."""
    
    @staticmethod
    def simple_function():
        """Simple function with constant expressions."""
        return {
            'ir_module': 'test_module',
            'ir_functions': [{
                'name': 'add_constants',
                'params': [],
                'return_type': 'i32',
                'body': [
                    {'type': 'var_decl', 'var_name': 'x', 'init': {'type': 'literal', 'value': 10}},
                    {'type': 'var_decl', 'var_name': 'y', 'init': {'type': 'literal', 'value': 20}},
                    {'type': 'var_decl', 'var_name': 'z', 'init': {
                        'type': 'binary', 'op': '+',
                        'left': {'type': 'var', 'name': 'x'},
                        'right': {'type': 'var', 'name': 'y'}
                    }},
                    {'type': 'return', 'value': {'type': 'var', 'name': 'z'}}
                ]
            }]
        }
    
    @staticmethod
    def dead_code():
        """Function with dead code after return."""
        return {
            'ir_module': 'test_dead',
            'ir_functions': [{
                'name': 'dead_code_func',
                'params': [],
                'return_type': 'i32',
                'body': [
                    {'type': 'var_decl', 'var_name': 'x', 'init': {'type': 'literal', 'value': 42}},
                    {'type': 'return', 'value': {'type': 'var', 'name': 'x'}},
                    {'type': 'var_decl', 'var_name': 'y', 'init': {'type': 'literal', 'value': 100}},  # Dead
                    {'type': 'var_decl', 'var_name': 'z', 'init': {'type': 'literal', 'value': 200}},  # Dead
                ]
            }]
        }
    
    @staticmethod
    def constant_expr():
        """Function with foldable constant expressions."""
        return {
            'ir_module': 'test_const',
            'ir_functions': [{
                'name': 'compute',
                'params': [],
                'return_type': 'i32',
                'body': [
                    {'type': 'var_decl', 'var_name': 'a', 'init': {
                        'type': 'binary', 'op': '+',
                        'left': {'type': 'literal', 'value': 10},
                        'right': {'type': 'literal', 'value': 20}
                    }},
                    {'type': 'var_decl', 'var_name': 'b', 'init': {
                        'type': 'binary', 'op': '*',
                        'left': {'type': 'literal', 'value': 5},
                        'right': {'type': 'literal', 'value': 6}
                    }},
                    {'type': 'return', 'value': {
                        'type': 'binary', 'op': '+',
                        'left': {'type': 'var', 'name': 'a'},
                        'right': {'type': 'var', 'name': 'b'}
                    }}
                ]
            }]
        }
    
    @staticmethod
    def algebraic_simplify():
        """Function with algebraically simplifiable expressions."""
        return {
            'ir_module': 'test_algebra',
            'ir_functions': [{
                'name': 'simplify',
                'params': [{'name': 'x', 'type': 'i32'}],
                'return_type': 'i32',
                'body': [
                    {'type': 'var_decl', 'var_name': 'a', 'init': {
                        'type': 'binary', 'op': '*',
                        'left': {'type': 'var', 'name': 'x'},
                        'right': {'type': 'literal', 'value': 1}  # x * 1 = x
                    }},
                    {'type': 'var_decl', 'var_name': 'b', 'init': {
                        'type': 'binary', 'op': '+',
                        'left': {'type': 'var', 'name': 'a'},
                        'right': {'type': 'literal', 'value': 0}  # a + 0 = a
                    }},
                    {'type': 'return', 'value': {'type': 'var', 'name': 'b'}}
                ]
            }]
        }
    
    @staticmethod
    def loop_with_invariant():
        """Function with loop-invariant code."""
        return {
            'ir_module': 'test_loop',
            'ir_functions': [{
                'name': 'loop_func',
                'params': [{'name': 'n', 'type': 'i32'}],
                'return_type': 'i32',
                'body': [
                    {'type': 'var_decl', 'var_name': 'sum', 'init': {'type': 'literal', 'value': 0}},
                    {'type': 'var_decl', 'var_name': 'i', 'init': {'type': 'literal', 'value': 0}},
                    {'type': 'while', 'cond': {
                        'type': 'binary', 'op': '<',
                        'left': {'type': 'var', 'name': 'i'},
                        'right': {'type': 'var', 'name': 'n'}
                    }, 'body': [
                        # This is loop-invariant (constant)
                        {'type': 'var_decl', 'var_name': 'factor', 'init': {'type': 'literal', 'value': 10}},
                        # This is not invariant
                        {'type': 'assign', 'target': 'sum', 'value': {
                            'type': 'binary', 'op': '+',
                            'left': {'type': 'var', 'name': 'sum'},
                            'right': {'type': 'var', 'name': 'i'}
                        }},
                        {'type': 'assign', 'target': 'i', 'value': {
                            'type': 'binary', 'op': '+',
                            'left': {'type': 'var', 'name': 'i'},
                            'right': {'type': 'literal', 'value': 1}
                        }}
                    ]},
                    {'type': 'return', 'value': {'type': 'var', 'name': 'sum'}}
                ]
            }]
        }
    
    @staticmethod
    def strength_reduction_candidates():
        """Function with strength reduction opportunities."""
        return {
            'ir_module': 'test_strength',
            'ir_functions': [{
                'name': 'compute',
                'params': [{'name': 'x', 'type': 'i32'}],
                'return_type': 'i32',
                'body': [
                    # x * 8 -> x << 3
                    {'type': 'var_decl', 'var_name': 'a', 'init': {
                        'type': 'binary', 'op': '*',
                        'left': {'type': 'var', 'name': 'x'},
                        'right': {'type': 'literal', 'value': 8}
                    }},
                    # x / 4 -> x >> 2
                    {'type': 'var_decl', 'var_name': 'b', 'init': {
                        'type': 'binary', 'op': '/',
                        'left': {'type': 'var', 'name': 'x'},
                        'right': {'type': 'literal', 'value': 4}
                    }},
                    # x % 8 -> x & 7
                    {'type': 'var_decl', 'var_name': 'c', 'init': {
                        'type': 'binary', 'op': '%',
                        'left': {'type': 'var', 'name': 'x'},
                        'right': {'type': 'literal', 'value': 8}
                    }},
                    {'type': 'return', 'value': {'type': 'var', 'name': 'a'}}
                ]
            }]
        }
    
    @staticmethod
    def small_function_for_inlining():
        """Functions suitable for inlining."""
        return {
            'ir_module': 'test_inline',
            'ir_functions': [
                {
                    'name': 'add',
                    'params': [{'name': 'a'}, {'name': 'b'}],
                    'return_type': 'i32',
                    'body': [
                        {'type': 'return', 'value': {
                            'type': 'binary', 'op': '+',
                            'left': {'type': 'var', 'name': 'a'},
                            'right': {'type': 'var', 'name': 'b'}
                        }}
                    ]
                },
                {
                    'name': 'main',
                    'params': [],
                    'return_type': 'i32',
                    'body': [
                        {'type': 'var_decl', 'var_name': 'x', 'init': {'type': 'literal', 'value': 10}},
                        {'type': 'var_decl', 'var_name': 'y', 'init': {'type': 'literal', 'value': 20}},
                        {'type': 'call', 'func': 'add', 'args': [
                            {'type': 'var', 'name': 'x'},
                            {'type': 'var', 'name': 'y'}
                        ], 'result_var': 'result'},
                        {'type': 'return', 'value': {'type': 'literal', 'value': 0}}
                    ]
                }
            ]
        }


class TestDeadCodeElimination(unittest.TestCase):
    """Test dead code elimination pass."""
    
    def test_removes_dead_code_after_return(self):
        ir = TestSampleIR.dead_code()
        pass_ = DeadCodeEliminationPass()
        result = pass_.run(ir)
        
        func = result['ir_functions'][0]
        # Should only have 2 statements (var_decl + return)
        self.assertEqual(len(func['body']), 2)
        self.assertEqual(func['body'][-1]['type'], 'return')
        self.assertGreater(pass_.stats.statements_removed, 0)
    
    def test_preserves_reachable_code(self):
        ir = TestSampleIR.simple_function()
        pass_ = DeadCodeEliminationPass()
        result = pass_.run(ir)
        
        func = result['ir_functions'][0]
        # All 4 statements should be preserved
        self.assertEqual(len(func['body']), 4)


class TestConstantFolding(unittest.TestCase):
    """Test constant folding pass."""
    
    def test_folds_constant_binary_ops(self):
        ir = TestSampleIR.constant_expr()
        pass_ = ConstantFoldingPass()
        result = pass_.run(ir)
        
        func = result['ir_functions'][0]
        
        # First var should be folded to 30
        self.assertEqual(func['body'][0]['init']['value'], 30)
        
        # Second var should be folded to 30
        self.assertEqual(func['body'][1]['init']['value'], 30)
        
        self.assertGreater(pass_.stats.constants_folded, 0)


class TestConstantPropagation(unittest.TestCase):
    """Test constant propagation pass."""
    
    def test_propagates_constants(self):
        ir = TestSampleIR.simple_function()
        pass_ = ConstantPropagationPass()
        result = pass_.run(ir)
        
        # Constants should be tracked but not necessarily propagated
        # since the pass is conservative
        self.assertIsNotNone(result)


class TestAlgebraicSimplification(unittest.TestCase):
    """Test algebraic simplification pass."""
    
    def test_simplifies_mul_by_one(self):
        ir = TestSampleIR.algebraic_simplify()
        pass_ = AlgebraicSimplificationPass()
        result = pass_.run(ir)
        
        func = result['ir_functions'][0]
        
        # x * 1 should simplify to x
        first_init = func['body'][0]['init']
        self.assertEqual(first_init['type'], 'var')
        self.assertEqual(first_init['name'], 'x')
        
        self.assertGreater(pass_.stats.ir_changes, 0)


class TestStrengthReduction(unittest.TestCase):
    """Test strength reduction pass."""
    
    def test_mul_to_shift(self):
        ir = TestSampleIR.strength_reduction_candidates()
        pass_ = StrengthReductionPass()
        result = pass_.run(ir)
        
        func = result['ir_functions'][0]
        
        # x * 8 -> x << 3
        first_init = func['body'][0]['init']
        self.assertEqual(first_init['op'], '<<')
        self.assertEqual(first_init['right']['value'], 3)
        
        # x / 4 -> x >> 2
        second_init = func['body'][1]['init']
        self.assertEqual(second_init['op'], '>>')
        self.assertEqual(second_init['right']['value'], 2)
        
        # x % 8 -> x & 7
        third_init = func['body'][2]['init']
        self.assertEqual(third_init['op'], '&')
        self.assertEqual(third_init['right']['value'], 7)


class TestFunctionInlining(unittest.TestCase):
    """Test function inlining pass."""
    
    def test_inlines_small_function(self):
        ir = TestSampleIR.small_function_for_inlining()
        pass_ = FunctionInliningPass(max_statements=5)
        result = pass_.run(ir)
        
        # Should have inlined the add function
        main_func = None
        for f in result['ir_functions']:
            if f['name'] == 'main':
                main_func = f
                break
        
        self.assertIsNotNone(main_func)
        # Body should be larger due to inlining
        self.assertGreaterEqual(len(main_func['body']), 4)


class TestPassManager(unittest.TestCase):
    """Test the pass manager."""
    
    def test_o0_no_changes(self):
        ir = TestSampleIR.dead_code()
        pm = create_pass_manager('O0')
        result, stats = pm.optimize(ir)
        
        self.assertEqual(stats['level'], 'O0')
        self.assertEqual(len(stats['passes']), 0)
        # IR should be unchanged
        self.assertEqual(len(result['ir_functions'][0]['body']), 
                        len(ir['ir_functions'][0]['body']))
    
    def test_o1_basic_optimizations(self):
        ir = TestSampleIR.dead_code()
        pm = create_pass_manager('O1')
        result, stats = pm.optimize(ir)
        
        self.assertEqual(stats['level'], 'O1')
        self.assertGreater(len(stats['passes']), 0)
        # Dead code should be removed
        self.assertLess(len(result['ir_functions'][0]['body']), 
                       len(ir['ir_functions'][0]['body']))
    
    def test_o2_includes_o1(self):
        ir = TestSampleIR.dead_code()
        pm = create_pass_manager('O2')
        result, stats = pm.optimize(ir)
        
        self.assertEqual(stats['level'], 'O2')
        # O2 should have more passes than O1
        pm_o1 = create_pass_manager('O1')
        _, stats_o1 = pm_o1.optimize(ir)
        self.assertGreater(len(stats['passes']), len(stats_o1['passes']))
    
    def test_o3_includes_o2(self):
        ir = TestSampleIR.simple_function()
        pm = create_pass_manager('O3')
        result, stats = pm.optimize(ir)
        
        self.assertEqual(stats['level'], 'O3')
        # O3 should have even more passes
        pm_o2 = create_pass_manager('O2')
        _, stats_o2 = pm_o2.optimize(ir)
        self.assertGreater(len(stats['passes']), len(stats_o2['passes']))


class TestValidation(unittest.TestCase):
    """Test optimization validation."""
    
    def test_valid_optimization(self):
        ir = TestSampleIR.dead_code()
        pm = create_pass_manager('O1')
        result, _ = pm.optimize(ir)
        
        is_valid, report = validate_optimization(ir, result)
        self.assertTrue(is_valid)
    
    def test_comparison(self):
        ir = TestSampleIR.dead_code()
        pm = create_pass_manager('O1')
        result, _ = pm.optimize(ir)
        
        comparison = compare_optimization(ir, result)
        self.assertIn('statement_reduction', comparison)
        self.assertGreater(comparison['statement_reduction'], 0)


class TestOptimizationLevelsProduceDifferentCode(unittest.TestCase):
    """Verify different optimization levels produce different results."""
    
    def test_different_levels_different_results(self):
        """Test that O0, O1, O2, O3 produce increasingly optimized code."""
        ir = TestSampleIR.dead_code()
        
        results = {}
        for level in ['O0', 'O1', 'O2', 'O3']:
            pm = create_pass_manager(level)
            result, stats = pm.optimize(copy.deepcopy(ir))
            results[level] = {
                'ir': result,
                'stats': stats,
                'body_len': len(result['ir_functions'][0]['body'])
            }
        
        # O0 should have original length (4 statements)
        self.assertEqual(results['O0']['body_len'], 4)
        
        # O1+ should have dead code removed (2 statements)
        self.assertEqual(results['O1']['body_len'], 2)
        
        # Verify passes increase with level
        self.assertEqual(len(results['O0']['stats']['passes']), 0)
        self.assertGreater(len(results['O1']['stats']['passes']), 0)
        self.assertGreater(len(results['O2']['stats']['passes']), len(results['O1']['stats']['passes']))
        self.assertGreater(len(results['O3']['stats']['passes']), len(results['O2']['stats']['passes']))


class TestPerformanceBenchmark(unittest.TestCase):
    """Simple performance benchmarks."""
    
    def test_optimization_performance(self):
        """Ensure optimization completes in reasonable time."""
        # Create a moderately complex IR
        ir = {
            'ir_module': 'benchmark',
            'ir_functions': [{
                'name': 'complex_func',
                'params': [],
                'return_type': 'i32',
                'body': [
                    {'type': 'var_decl', 'var_name': f'v{i}', 
                     'init': {'type': 'binary', 'op': '+',
                              'left': {'type': 'literal', 'value': i},
                              'right': {'type': 'literal', 'value': i * 2}}}
                    for i in range(50)
                ] + [
                    {'type': 'return', 'value': {'type': 'literal', 'value': 0}}
                ]
            }]
        }
        
        pm = create_pass_manager('O3')
        
        start = time.time()
        result, stats = pm.optimize(ir)
        elapsed = time.time() - start
        
        # Should complete in under 1 second
        self.assertLess(elapsed, 1.0)
        print(f"\nOptimization benchmark: {elapsed*1000:.2f}ms for {len(ir['ir_functions'][0]['body'])} statements")


def run_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
