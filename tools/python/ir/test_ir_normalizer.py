#!/usr/bin/env python3
"""
Tests for IR Normalizer

Tests the pre-emission normalization passes:
- Switch lowering
- For loop lowering
- Break/continue lowering
- Return normalization
"""

import sys
import os
import importlib.util

# Load ir_normalizer directly without going through package imports
spec = importlib.util.spec_from_file_location(
    "ir_normalizer",
    os.path.join(os.path.dirname(__file__), "ir_normalizer.py")
)
ir_normalizer = importlib.util.module_from_spec(spec)
sys.modules["ir_normalizer"] = ir_normalizer
spec.loader.exec_module(ir_normalizer)

IRNormalizer = ir_normalizer.IRNormalizer
NormalizerConfig = ir_normalizer.NormalizerConfig
normalize_ir = ir_normalizer.normalize_ir

import unittest
import json


class TestSwitchLowering(unittest.TestCase):
    """Test switch to if/else lowering."""
    
    def test_simple_switch(self):
        """Test simple switch with two cases."""
        func = {
            "name": "test",
            "return_type": "int",
            "steps": [
                {
                    "op": "switch",
                    "expr": "x",
                    "cases": [
                        {"value": 1, "body": [{"op": "return", "value": "1"}]},
                        {"value": 2, "body": [{"op": "return", "value": "2"}]}
                    ],
                    "default": [{"op": "return", "value": "0"}]
                }
            ]
        }
        
        normalizer = IRNormalizer(NormalizerConfig(lower_switch=True))
        result = normalizer.normalize_function(func)
        
        # Should have if/else chain instead of switch
        self.assertEqual(result["steps"][0]["op"], "if")
        self.assertIn("x == 1", result["steps"][0]["condition"])
        self.assertEqual(normalizer.stats.switches_lowered, 1)
    
    def test_switch_with_default(self):
        """Test switch with default case."""
        func = {
            "name": "test",
            "return_type": "int",
            "steps": [
                {
                    "op": "switch",
                    "expr": "x",
                    "cases": [
                        {"value": 1, "body": [{"op": "return", "value": "1"}]}
                    ],
                    "default": [{"op": "return", "value": "-1"}]
                }
            ]
        }
        
        normalizer = IRNormalizer(NormalizerConfig(lower_switch=True))
        result = normalizer.normalize_function(func)
        
        # Should have if with else_block containing default
        self.assertEqual(result["steps"][0]["op"], "if")
        self.assertEqual(result["steps"][0]["else_block"][0]["op"], "return")


class TestForLoopLowering(unittest.TestCase):
    """Test for to while loop lowering."""
    
    def test_simple_for(self):
        """Test simple for loop."""
        func = {
            "name": "test",
            "return_type": "int",
            "steps": [
                {
                    "op": "for",
                    "init": "0",
                    "condition": "i < 10",
                    "increment": "i + 1",
                    "loop_var": "i",
                    "body": [
                        {"op": "assign", "target": "sum", "value": "sum + i"}
                    ]
                }
            ]
        }
        
        # Disable break/continue lowering to avoid flag init steps
        config = NormalizerConfig(lower_for=True, lower_break_continue=False)
        normalizer = IRNormalizer(config)
        result = normalizer.normalize_function(func)
        
        # Should have init assignment followed by while
        self.assertEqual(result["steps"][0]["op"], "assign")
        self.assertEqual(result["steps"][0]["target"], "i")
        self.assertEqual(result["steps"][1]["op"], "while")
        self.assertEqual(normalizer.stats.for_loops_lowered, 1)
    
    def test_for_with_body(self):
        """Test for loop with body."""
        func = {
            "name": "test",
            "return_type": "int",
            "steps": [
                {
                    "op": "for",
                    "init": "1",
                    "condition": "i <= n",
                    "increment": "i + 1",
                    "loop_var": "i",
                    "body": [
                        {"op": "assign", "target": "result", "value": "result * i"}
                    ]
                }
            ]
        }
        
        # Disable break/continue lowering to avoid flag init steps
        config = NormalizerConfig(lower_for=True, lower_break_continue=False)
        normalizer = IRNormalizer(config)
        result = normalizer.normalize_function(func)
        
        # Check that increment is appended to body
        while_step = result["steps"][1]
        self.assertEqual(while_step["op"], "while")
        # Body should include original + increment
        self.assertEqual(len(while_step["body"]), 2)


class TestBreakContinueLowering(unittest.TestCase):
    """Test break/continue to flag lowering."""
    
    def test_break_in_while(self):
        """Test break in while loop."""
        func = {
            "name": "test",
            "return_type": "int",
            "steps": [
                {
                    "op": "while",
                    "condition": "i < 10",
                    "body": [
                        {"op": "assign", "target": "i", "value": "i + 1"},
                        {"op": "break"}
                    ]
                }
            ]
        }
        
        normalizer = IRNormalizer(NormalizerConfig(lower_break_continue=True))
        result = normalizer.normalize_function(func)
        
        # Should have flag init before while
        self.assertEqual(result["steps"][0]["op"], "assign")
        self.assertIn("_break_", result["steps"][0]["target"])
        self.assertEqual(normalizer.stats.breaks_lowered, 1)
    
    def test_continue_in_while(self):
        """Test continue in while loop."""
        func = {
            "name": "test",
            "return_type": "int",
            "steps": [
                {
                    "op": "while",
                    "condition": "i < 10",
                    "body": [
                        {"op": "continue"}
                    ]
                }
            ]
        }
        
        normalizer = IRNormalizer(NormalizerConfig(lower_break_continue=True))
        result = normalizer.normalize_function(func)
        
        self.assertEqual(normalizer.stats.continues_lowered, 1)


class TestReturnNormalization(unittest.TestCase):
    """Test return normalization."""
    
    def test_add_return_int(self):
        """Test adding return for int function."""
        func = {
            "name": "test",
            "return_type": "int",
            "steps": [
                {"op": "assign", "target": "x", "value": "1"}
            ]
        }
        
        normalizer = IRNormalizer(NormalizerConfig(normalize_returns=True))
        result = normalizer.normalize_function(func)
        
        # Should have added return 0
        self.assertEqual(result["steps"][-1]["op"], "return")
        self.assertEqual(result["steps"][-1]["value"], "0")
        self.assertEqual(normalizer.stats.returns_added, 1)
    
    def test_add_return_bool(self):
        """Test adding return for bool function."""
        func = {
            "name": "test",
            "return_type": "bool",
            "steps": [
                {"op": "assign", "target": "x", "value": "true"}
            ]
        }
        
        normalizer = IRNormalizer(NormalizerConfig(normalize_returns=True))
        result = normalizer.normalize_function(func)
        
        self.assertEqual(result["steps"][-1]["value"], "false")
    
    def test_no_return_for_void(self):
        """Test void function gets empty return."""
        func = {
            "name": "test",
            "return_type": "void",
            "steps": [
                {"op": "call", "value": "do_something"}
            ]
        }
        
        normalizer = IRNormalizer(NormalizerConfig(normalize_returns=True))
        result = normalizer.normalize_function(func)
        
        self.assertEqual(result["steps"][-1]["op"], "return")
    
    def test_existing_return_not_modified(self):
        """Test that existing return is not modified."""
        func = {
            "name": "test",
            "return_type": "int",
            "steps": [
                {"op": "return", "value": "42"}
            ]
        }
        
        normalizer = IRNormalizer(NormalizerConfig(normalize_returns=True))
        result = normalizer.normalize_function(func)
        
        # Should not add another return
        self.assertEqual(len(result["steps"]), 1)
        self.assertEqual(normalizer.stats.returns_added, 0)


class TestModuleNormalization(unittest.TestCase):
    """Test full module normalization."""
    
    def test_normalize_module(self):
        """Test normalizing a full module."""
        module = {
            "ir_version": "v1",
            "module_name": "test_module",
            "functions": [
                {
                    "name": "func1",
                    "return_type": "int",
                    "steps": [
                        {"op": "for", "init": "0", "condition": "i < 10", 
                         "increment": "i + 1", "loop_var": "i", "body": []}
                    ]
                },
                {
                    "name": "func2",
                    "return_type": "int",
                    "steps": [
                        {"op": "switch", "expr": "x", "cases": [], "default": []}
                    ]
                }
            ]
        }
        
        result = normalize_ir(module)
        
        self.assertTrue(result.get("normalized", False))
        self.assertIn("normalization_stats", result)
        self.assertEqual(result["normalization_stats"]["for_loops_lowered"], 1)
        self.assertEqual(result["normalization_stats"]["switches_lowered"], 1)


class TestConfigOptions(unittest.TestCase):
    """Test configuration options."""
    
    def test_disable_switch_lowering(self):
        """Test disabling switch lowering."""
        func = {
            "name": "test",
            "return_type": "int",
            "steps": [
                {"op": "switch", "expr": "x", "cases": [], "default": []}
            ]
        }
        
        config = NormalizerConfig(lower_switch=False)
        normalizer = IRNormalizer(config)
        result = normalizer.normalize_function(func)
        
        # Switch should remain
        self.assertEqual(result["steps"][0]["op"], "switch")
        self.assertEqual(normalizer.stats.switches_lowered, 0)
    
    def test_disable_for_lowering(self):
        """Test disabling for loop lowering."""
        func = {
            "name": "test",
            "return_type": "int",
            "steps": [
                {"op": "for", "init": "0", "condition": "i < 10",
                 "increment": "i + 1", "loop_var": "i", "body": []}
            ]
        }
        
        config = NormalizerConfig(lower_for=False, lower_break_continue=False)
        normalizer = IRNormalizer(config)
        result = normalizer.normalize_function(func)
        
        # For should remain
        self.assertEqual(result["steps"][0]["op"], "for")
        self.assertEqual(normalizer.stats.for_loops_lowered, 0)


class TestTryCatchLowering(unittest.TestCase):
    """Test try/catch to error flag lowering."""
    
    def test_simple_try_catch(self):
        """Test simple try/catch lowering."""
        func = {
            "name": "test",
            "return_type": "void",
            "steps": [
                {
                    "op": "try",
                    "body": [
                        {"op": "assign", "target": "x", "value": "1"}
                    ],
                    "catch_var": "e",
                    "catch_body": [
                        {"op": "assign", "target": "x", "value": "0"}
                    ]
                }
            ]
        }
        
        normalizer = IRNormalizer()
        result = normalizer.normalize_function(func)
        
        # Should have error flag init + try body wrapper + catch conditional
        self.assertEqual(normalizer.stats.try_catch_lowered, 1)
        
        # Check for error flag initialization
        steps = result["steps"]
        self.assertEqual(steps[0]["op"], "assign")
        self.assertIn("error_flag", steps[0]["target"])
        self.assertEqual(steps[0]["value"], "false")
    
    def test_throw_conversion(self):
        """Test throw statement conversion."""
        func = {
            "name": "test",
            "return_type": "void",
            "steps": [
                {
                    "op": "try",
                    "body": [
                        {"op": "throw", "value": "error_msg"}
                    ],
                    "catch_body": []
                }
            ]
        }
        
        normalizer = IRNormalizer()
        result = normalizer.normalize_function(func)
        
        # Throw should be converted to error flag assignment
        self.assertEqual(normalizer.stats.try_catch_lowered, 1)


class TestExpressionSimplification(unittest.TestCase):
    """Test expression simplification."""
    
    def test_complex_expression_detection(self):
        """Test detection of complex expressions."""
        normalizer = IRNormalizer()
        
        # Simple expressions
        self.assertFalse(normalizer._is_complex_expression("x"))
        self.assertFalse(normalizer._is_complex_expression("x + y"))
        
        # Complex expressions (more than 2 operators)
        self.assertTrue(normalizer._is_complex_expression("a + b * c - d"))
        self.assertTrue(normalizer._is_complex_expression("x && y || z && w"))
    
    def test_expression_simplification_disabled(self):
        """Test that simplification is disabled by default."""
        func = {
            "name": "test",
            "return_type": "int",
            "steps": [
                {"op": "assign", "target": "x", "value": "a + b * c - d"}
            ]
        }
        
        # Default config has simplify_expressions=False
        normalizer = IRNormalizer()
        result = normalizer.normalize_function(func)
        
        # Expression should remain unchanged
        self.assertEqual(result["steps"][0]["value"], "a + b * c - d")
        self.assertEqual(normalizer.stats.expressions_split, 0)
    
    def test_expression_simplification_enabled(self):
        """Test expression simplification when enabled."""
        func = {
            "name": "test",
            "return_type": "int",
            "steps": [
                {"op": "assign", "target": "x", "value": "a + b * c - d"}
            ]
        }
        
        config = NormalizerConfig(simplify_expressions=True)
        normalizer = IRNormalizer(config)
        result = normalizer.normalize_function(func)
        
        # Should have attempted simplification
        self.assertEqual(normalizer.stats.expressions_split, 1)


class TestNormalizationStats(unittest.TestCase):
    """Test normalization statistics tracking."""
    
    def test_stats_accumulation(self):
        """Test that stats accumulate across multiple functions."""
        module = {
            "functions": [
                {
                    "name": "func1",
                    "return_type": "int",
                    "steps": [
                        {"op": "switch", "expr": "x", "cases": [], "default": []}
                    ]
                },
                {
                    "name": "func2",
                    "return_type": "int",
                    "steps": [
                        {"op": "switch", "expr": "y", "cases": [], "default": []}
                    ]
                }
            ]
        }
        
        normalizer = IRNormalizer()
        result = normalizer.normalize_module(module)
        
        # Should have lowered 2 switches
        self.assertEqual(result["normalization_stats"]["switches_lowered"], 2)


if __name__ == "__main__":
    unittest.main()
