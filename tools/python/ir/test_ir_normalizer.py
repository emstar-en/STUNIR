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
ValidationResult = ir_normalizer.ValidationResult

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


class TestTypeCanonicalization(unittest.TestCase):
    """Test type canonicalization."""
    
    def test_int_type_canonicalization(self):
        """Test integer type canonicalization."""
        func = {
            "name": "test",
            "return_type": "i32",
            "params": [{"name": "x", "type": "int64"}],
            "locals": [{"name": "y", "type": "integer"}],
            "steps": []
        }
        
        normalizer = IRNormalizer()
        result = normalizer.normalize_function(func)
        
        # Types should be canonicalized
        self.assertEqual(result["return_type"], "int")
        self.assertEqual(result["params"][0]["type"], "long")
        self.assertEqual(result["locals"][0]["type"], "int")
        self.assertEqual(normalizer.stats.types_canonicalized, 3)
    
    def test_float_type_canonicalization(self):
        """Test float type canonicalization."""
        func = {
            "name": "test",
            "return_type": "f64",
            "params": [{"name": "x", "type": "float32"}],
            "steps": []
        }
        
        normalizer = IRNormalizer()
        result = normalizer.normalize_function(func)
        
        self.assertEqual(result["return_type"], "double")
        self.assertEqual(result["params"][0]["type"], "float")
    
    def test_bool_type_canonicalization(self):
        """Test boolean type canonicalization."""
        func = {
            "name": "test",
            "return_type": "boolean",
            "steps": []
        }
        
        normalizer = IRNormalizer()
        result = normalizer.normalize_function(func)
        
        self.assertEqual(result["return_type"], "bool")


class TestConstantFolding(unittest.TestCase):
    """Test constant folding."""
    
    def test_arithmetic_folding(self):
        """Test arithmetic constant folding."""
        func = {
            "name": "test",
            "return_type": "int",
            "steps": [
                {"op": "assign", "target": "x", "value": "1 + 2"},
                {"op": "assign", "target": "y", "value": "10 / 2"},
                {"op": "assign", "target": "z", "value": "3 * 4"}
            ]
        }
        
        normalizer = IRNormalizer()
        result = normalizer.normalize_function(func)
        
        # Constants should be folded
        self.assertEqual(result["steps"][0]["value"], "3")
        self.assertEqual(result["steps"][1]["value"], "5")
        self.assertEqual(result["steps"][2]["value"], "12")
        self.assertEqual(normalizer.stats.constants_folded, 3)
    
    def test_boolean_folding(self):
        """Test boolean constant folding."""
        func = {
            "name": "test",
            "return_type": "bool",
            "steps": [
                {"op": "assign", "target": "a", "value": "true && false"},
                {"op": "assign", "target": "b", "value": "true || false"},
                {"op": "assign", "target": "c", "value": "!true"}
            ]
        }
        
        normalizer = IRNormalizer()
        result = normalizer.normalize_function(func)
        
        self.assertEqual(result["steps"][0]["value"], "false")
        self.assertEqual(result["steps"][1]["value"], "true")
        self.assertEqual(result["steps"][2]["value"], "false")
    
    def test_no_variable_folding(self):
        """Test that expressions with variables are not folded."""
        func = {
            "name": "test",
            "return_type": "int",
            "steps": [
                {"op": "assign", "target": "x", "value": "a + 2"}
            ]
        }
        
        normalizer = IRNormalizer()
        result = normalizer.normalize_function(func)
        
        # Should not fold expressions with variables
        self.assertEqual(result["steps"][0]["value"], "a + 2")


class TestDeadCodeRemoval(unittest.TestCase):
    """Test dead code removal."""
    
    def test_code_after_return(self):
        """Test removal of code after return."""
        func = {
            "name": "test",
            "return_type": "int",
            "steps": [
                {"op": "assign", "target": "x", "value": "1"},
                {"op": "return", "value": "x"},
                {"op": "assign", "target": "y", "value": "2"}  # Dead code
            ]
        }
        
        config = NormalizerConfig(remove_dead_code=True, normalize_returns=False)
        normalizer = IRNormalizer(config)
        result = normalizer.normalize_function(func)
        
        # Dead code should be removed
        self.assertEqual(len(result["steps"]), 2)  # assign + return
        self.assertEqual(normalizer.stats.dead_code_removed, 1)
    
    def test_empty_if_removal(self):
        """Test removal of empty if statements."""
        func = {
            "name": "test",
            "return_type": "int",
            "steps": [
                {"op": "if", "condition": "x", "then_body": [], "else_body": []}
            ]
        }
        
        config = NormalizerConfig(remove_dead_code=True, normalize_returns=False)
        normalizer = IRNormalizer(config)
        result = normalizer.normalize_function(func)
        
        # Empty if should be removed
        self.assertEqual(len(result["steps"]), 0)
    
    def test_false_condition_removal(self):
        """Test removal of if with false condition and no else."""
        func = {
            "name": "test",
            "return_type": "int",
            "steps": [
                {"op": "if", "condition": "false", "then_body": [
                    {"op": "assign", "target": "x", "value": "1"}
                ], "else_body": []}
            ]
        }
        
        config = NormalizerConfig(remove_dead_code=True, normalize_returns=False)
        normalizer = IRNormalizer(config)
        result = normalizer.normalize_function(func)
        
        # False condition with no else should be removed
        self.assertEqual(len(result["steps"]), 0)
    
    def test_true_condition_simplification(self):
        """Test simplification of if with true condition."""
        func = {
            "name": "test",
            "return_type": "int",
            "steps": [
                {"op": "if", "condition": "true", "then_body": [
                    {"op": "assign", "target": "x", "value": "1"}
                ], "else_body": [
                    {"op": "assign", "target": "y", "value": "2"}
                ]}
            ]
        }
        
        config = NormalizerConfig(remove_dead_code=True, normalize_returns=False)
        normalizer = IRNormalizer(config)
        result = normalizer.normalize_function(func)
        
        # True condition should be replaced with then body
        self.assertEqual(len(result["steps"]), 1)
        self.assertEqual(result["steps"][0]["op"], "assign")


class TestPreEmissionValidation(unittest.TestCase):
    """Test pre-emission validation guardrails."""
    
    def test_break_outside_loop(self):
        """Test detection of break outside loop."""
        func = {
            "name": "test",
            "return_type": "int",
            "steps": [
                {"op": "assign", "target": "x", "value": "1"},
                {"op": "break"}  # Error: break outside loop
            ]
        }
        
        normalizer = IRNormalizer()
        result = normalizer.validate_for_emission(func)
        
        self.assertFalse(result.valid)
        self.assertTrue(any("break" in e for e in result.errors))
    
    def test_continue_outside_loop(self):
        """Test detection of continue outside loop."""
        func = {
            "name": "test",
            "return_type": "int",
            "steps": [
                {"op": "continue"}  # Error: continue outside loop
            ]
        }
        
        normalizer = IRNormalizer()
        result = normalizer.validate_for_emission(func)
        
        self.assertFalse(result.valid)
        self.assertTrue(any("continue" in e for e in result.errors))
    
    def test_void_function_returns_value(self):
        """Test warning for void function returning value."""
        func = {
            "name": "test",
            "return_type": "void",
            "steps": [
                {"op": "return", "value": "42"}  # Warning: void returns value
            ]
        }
        
        normalizer = IRNormalizer()
        result = normalizer.validate_for_emission(func)
        
        self.assertTrue(result.valid)  # Warning, not error
        self.assertTrue(any("void" in w.lower() for w in result.warnings))
    
    def test_non_void_missing_return_value(self):
        """Test error for non-void function missing return value."""
        func = {
            "name": "test",
            "return_type": "int",
            "steps": [
                {"op": "return"}  # Error: missing return value
            ]
        }
        
        normalizer = IRNormalizer()
        result = normalizer.validate_for_emission(func)
        
        self.assertFalse(result.valid)
        self.assertTrue(any("return value" in e.lower() for e in result.errors))
    
    def test_empty_loop_body(self):
        """Test warning for empty loop body."""
        func = {
            "name": "test",
            "return_type": "int",
            "steps": [
                {"op": "while", "condition": "true", "body": []}  # Warning: empty body
            ]
        }
        
        normalizer = IRNormalizer()
        result = normalizer.validate_for_emission(func)
        
        self.assertTrue(result.valid)  # Warning, not error
        self.assertTrue(any("empty" in w.lower() for w in result.warnings))
    
    def test_valid_function(self):
        """Test that valid function passes validation."""
        func = {
            "name": "test",
            "return_type": "int",
            "params": [{"name": "x", "type": "int"}],
            "steps": [
                {"op": "assign", "target": "y", "value": "x + 1"},
                {"op": "return", "value": "y"}
            ]
        }
        
        normalizer = IRNormalizer()
        result = normalizer.validate_for_emission(func)
        
        self.assertTrue(result.valid)
        self.assertEqual(len(result.errors), 0)
    
    def test_validation_disabled(self):
        """Test that validation can be disabled."""
        func = {
            "name": "test",
            "return_type": "int",
            "steps": [
                {"op": "break"}  # Would normally be error
            ]
        }
        
        config = NormalizerConfig(validate_before_emit=False)
        normalizer = IRNormalizer(config)
        result = normalizer.validate_for_emission(func)
        
        self.assertTrue(result.valid)  # No validation run


if __name__ == "__main__":
    unittest.main()
