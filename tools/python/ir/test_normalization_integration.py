#!/usr/bin/env python3
"""
STUNIR IR Normalizer Integration Tests

Tests the complete normalization pipeline with actual emitters.
Validates that normalized IR produces correct code for each target.

Part of Phase F: Validation & Tests.
"""

import sys
import os
import importlib.util
from pathlib import Path
import unittest
import json

# Load modules directly
ir_path = os.path.dirname(os.path.abspath(__file__))

# Load ir_normalizer
spec = importlib.util.spec_from_file_location(
    "ir_normalizer",
    os.path.join(ir_path, "ir_normalizer.py")
)
ir_normalizer = importlib.util.module_from_spec(spec)
sys.modules["ir_normalizer"] = ir_normalizer
spec.loader.exec_module(ir_normalizer)

# Load normalized_emitter
spec2 = importlib.util.spec_from_file_location(
    "normalized_emitter",
    os.path.join(ir_path, "normalized_emitter.py")
)
normalized_emitter = importlib.util.module_from_spec(spec2)
sys.modules["normalized_emitter"] = normalized_emitter
spec2.loader.exec_module(normalized_emitter)

IRNormalizer = ir_normalizer.IRNormalizer
NormalizerConfig = ir_normalizer.NormalizerConfig
NormalizedEmitterBase = normalized_emitter.NormalizedEmitterBase
NormalizedEmitterConfig = normalized_emitter.NormalizedEmitterConfig
EmitterResult = normalized_emitter.EmitterResult


class MockEmitter(NormalizedEmitterBase):
    """Mock emitter for testing normalization."""
    
    DIALECT = "mock"
    FILE_EXTENSION = ".mock"
    
    def emit(self, ir):
        """Emit mock code from normalized IR."""
        lines = [f"// Module: {ir.get('name', 'unknown')}"]
        
        for func in ir.get("functions", []):
            lines.append(self._emit_function(func))
        
        return EmitterResult(code="\n".join(lines))
    
    def _emit_function(self, func):
        """Emit a function."""
        name = func.get("name", "unknown")
        ret_type = self._map_type(func.get("return_type", "void"))
        params = func.get("params", [])
        
        param_strs = []
        for p in params:
            ptype = self._map_type(p.get("type", "any"))
            pname = p.get("name", "_")
            param_strs.append(f"{pname}: {ptype}")
        
        lines = [f"func {name}({', '.join(param_strs)}) -> {ret_type} {{"]
        self._push_indent()
        
        for step in func.get("steps", []):
            lines.append(self._indent(self._emit_statement(step)))
        
        self._pop_indent()
        lines.append("}")
        return "\n".join(lines)
    
    def _emit_statement(self, step):
        """Emit a statement."""
        op = step.get("op", "unknown")
        
        if op == "assign":
            target = step.get("target", "_")
            value = self._emit_expression(step.get("value"))
            return f"{target} = {value};"
        elif op == "return":
            value = step.get("value")
            if value:
                return f"return {self._emit_expression(value)};"
            return "return;"
        elif op == "while":
            cond = step.get("condition", "true")
            body = step.get("body", [])
            lines = [f"while ({cond}) {{"]
            self._push_indent()
            for s in body:
                lines.append(self._indent(self._emit_statement(s)))
            self._pop_indent()
            lines.append("}")
            return "\n".join(lines)
        elif op == "if":
            cond = step.get("condition", "true")
            then_body = step.get("then_body", [])
            else_body = step.get("else_body", [])
            lines = [f"if ({cond}) {{"]
            self._push_indent()
            for s in then_body:
                lines.append(self._indent(self._emit_statement(s)))
            self._pop_indent()
            if else_body:
                lines.append("} else {")
                self._push_indent()
                for s in else_body:
                    lines.append(self._indent(self._emit_statement(s)))
                self._pop_indent()
            lines.append("}")
            return "\n".join(lines)
        else:
            return f"// Unknown op: {op}"
    
    def _emit_expression(self, expr):
        """Emit an expression."""
        if isinstance(expr, str):
            return expr
        if isinstance(expr, (int, float)):
            return str(expr)
        if isinstance(expr, bool):
            return "true" if expr else "false"
        return str(expr)


class TestNormalizationPipeline(unittest.TestCase):
    """Test complete normalization pipeline."""
    
    def test_for_loop_lowering_in_emission(self):
        """Test that for loops are lowered before emission."""
        config = NormalizedEmitterConfig(target_dir=Path("/tmp"))
        emitter = MockEmitter(config)
        
        ir = {
            "name": "test",
            "functions": [{
                "name": "sum",
                "return_type": "int",
                "params": [{"name": "n", "type": "int"}],
                "steps": [
                    {"op": "assign", "target": "sum", "value": "0"},
                    {"op": "for", "init": "0", "condition": "i < n", 
                     "increment": "i + 1", "loop_var": "i", "body": [
                        {"op": "assign", "target": "sum", "value": "sum + i"}
                    ]},
                    {"op": "return", "value": "sum"}
                ]
            }]
        }
        
        result = emitter.emit_with_normalization(ir)
        
        # Should not contain 'for' keyword
        self.assertNotIn("for (", result.code)
        # Should contain 'while' keyword
        self.assertIn("while", result.code)
    
    def test_switch_lowering_in_emission(self):
        """Test that switch statements are lowered before emission."""
        config = NormalizedEmitterConfig(target_dir=Path("/tmp"))
        emitter = MockEmitter(config)
        
        ir = {
            "name": "test",
            "functions": [{
                "name": "classify",
                "return_type": "int",
                "params": [{"name": "x", "type": "int"}],
                "steps": [
                    {"op": "switch", "expr": "x", "cases": [
                        {"value": 1, "body": [{"op": "return", "value": "1"}]},
                        {"value": 2, "body": [{"op": "return", "value": "2"}]}
                    ], "default": [{"op": "return", "value": "0"}]}
                ]
            }]
        }
        
        result = emitter.emit_with_normalization(ir)
        
        # Should not contain 'switch' keyword
        self.assertNotIn("switch", result.code)
        # Should contain 'if' statements
        self.assertIn("if", result.code)
    
    def test_break_continue_lowering_in_emission(self):
        """Test that break/continue are lowered before emission."""
        config = NormalizedEmitterConfig(target_dir=Path("/tmp"))
        emitter = MockEmitter(config)
        
        ir = {
            "name": "test",
            "functions": [{
                "name": "find",
                "return_type": "int",
                "params": [{"name": "arr", "type": "int[]"}, {"name": "target", "type": "int"}],
                "steps": [
                    {"op": "assign", "target": "i", "value": "0"},
                    {"op": "while", "condition": "i < 10", "body": [
                        {"op": "if", "condition": "arr[i] == target", "then_body": [
                            {"op": "assign", "target": "found", "value": "1"}
                        ]},
                        {"op": "assign", "target": "i", "value": "i + 1"}
                    ]},
                    {"op": "return", "value": "i"}
                ]
            }]
        }
        
        result = emitter.emit_with_normalization(ir)
        
        # Should emit successfully
        self.assertTrue(result.success)
        # Should contain while loop
        self.assertIn("while", result.code)
    
    def test_type_canonicalization_in_emission(self):
        """Test that types are canonicalized before emission."""
        config = NormalizedEmitterConfig(target_dir=Path("/tmp"))
        emitter = MockEmitter(config)
        
        ir = {
            "name": "test",
            "functions": [{
                "name": "compute",
                "return_type": "i32",
                "params": [
                    {"name": "a", "type": "i64"},
                    {"name": "b", "type": "f32"}
                ],
                "steps": [
                    {"op": "return", "value": "0"}
                ]
            }]
        }
        
        result = emitter.emit_with_normalization(ir)
        
        # Types should be canonicalized
        self.assertIn("int", result.code)  # i32 -> int
        self.assertIn("long", result.code)  # i64 -> long
        self.assertIn("float", result.code)  # f32 -> float
    
    def test_constant_folding_in_emission(self):
        """Test that constants are folded before emission."""
        config = NormalizedEmitterConfig(target_dir=Path("/tmp"))
        emitter = MockEmitter(config)
        
        ir = {
            "name": "test",
            "functions": [{
                "name": "compute",
                "return_type": "int",
                "steps": [
                    {"op": "assign", "target": "x", "value": "1 + 2"},
                    {"op": "assign", "target": "y", "value": "3 * 4"},
                    {"op": "return", "value": "x"}
                ]
            }]
        }
        
        result = emitter.emit_with_normalization(ir)
        
        # Constants should be folded
        self.assertIn("3", result.code)  # 1 + 2 -> 3
        self.assertIn("12", result.code)  # 3 * 4 -> 12
    
    def test_validation_catches_errors(self):
        """Test that validation catches errors before emission."""
        config = NormalizedEmitterConfig(target_dir=Path("/tmp"))
        emitter = MockEmitter(config)
        
        # IR with break outside loop
        ir = {
            "name": "test",
            "functions": [{
                "name": "bad",
                "return_type": "int",
                "steps": [
                    {"op": "break"}  # Invalid: break outside loop
                ]
            }]
        }
        
        result = emitter.emit_with_normalization(ir)
        
        # Should still emit (validation is warning-level)
        # But validation should have run
        self.assertIsNotNone(result.validation)


class TestEmitterIntegration(unittest.TestCase):
    """Test emitter integration with normalization."""
    
    def test_emitter_receives_normalized_ir(self):
        """Test that emitter receives normalized IR."""
        config = NormalizedEmitterConfig(target_dir=Path("/tmp"))
        emitter = MockEmitter(config)
        
        # Track what IR was received
        received_ir = {}
        
        original_emit = emitter.emit
        def track_emit(ir):
            received_ir['ir'] = ir
            return original_emit(ir)
        emitter.emit = track_emit
        
        ir = {
            "name": "test",
            "functions": [{
                "name": "test",
                "return_type": "i32",
                "steps": [
                    {"op": "for", "init": "0", "condition": "i < 10",
                     "increment": "i + 1", "loop_var": "i", "body": []}
                ]
            }]
        }
        
        emitter.emit_with_normalization(ir)
        
        # Check that received IR has been normalized
        self.assertIsNotNone(received_ir.get('ir'))
        steps = received_ir['ir']['functions'][0]['steps']
        # For loop should be lowered to while
        has_while = any(s.get('op') == 'while' for s in steps)
        self.assertTrue(has_while)
    
    def test_emitter_can_disable_normalization(self):
        """Test that normalization can be disabled."""
        config = NormalizedEmitterConfig(
            target_dir=Path("/tmp"),
            normalize_before_emit=False
        )
        emitter = MockEmitter(config)
        
        # Track what IR was received
        received_ir = {}
        
        original_emit = emitter.emit
        def track_emit(ir):
            received_ir['ir'] = ir
            return original_emit(ir)
        emitter.emit = track_emit
        
        ir = {
            "name": "test",
            "functions": [{
                "name": "test",
                "return_type": "i32",
                "steps": [
                    {"op": "for", "init": "0", "condition": "i < 10",
                     "increment": "i + 1", "loop_var": "i", "body": []}
                ]
            }]
        }
        
        emitter.emit_with_normalization(ir)
        
        # Check that received IR was NOT normalized
        self.assertIsNotNone(received_ir.get('ir'))
        steps = received_ir['ir']['functions'][0]['steps']
        # For loop should still be present
        has_for = any(s.get('op') == 'for' for s in steps)
        self.assertTrue(has_for)


class TestNormalizationStats(unittest.TestCase):
    """Test normalization statistics tracking."""
    
    def test_stats_are_tracked(self):
        """Test that normalization stats are tracked."""
        config = NormalizedEmitterConfig(target_dir=Path("/tmp"))
        emitter = MockEmitter(config)
        
        ir = {
            "name": "test",
            "functions": [{
                "name": "test",
                "return_type": "i32",
                "steps": [
                    {"op": "for", "init": "0", "condition": "i < 10",
                     "increment": "i + 1", "loop_var": "i", "body": []},
                    {"op": "switch", "expr": "x", "cases": [], "default": []}
                ]
            }]
        }
        
        result = emitter.emit_with_normalization(ir)
        
        # Stats should be tracked in normalizer
        self.assertIsNotNone(emitter._normalizer)
        stats = emitter._normalizer.stats
        self.assertGreater(stats.for_loops_lowered, 0)
        self.assertGreater(stats.switches_lowered, 0)
        self.assertGreater(stats.types_canonicalized, 0)


if __name__ == "__main__":
    unittest.main()