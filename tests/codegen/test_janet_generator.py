#!/usr/bin/env python3
"""Tests for STUNIR Janet Emitter.

Part of Phase 5B: Extended Lisp Implementation.
"""

import pytest
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from targets.lisp.janet import JanetEmitter, JanetConfig, JanetTypeMapper


class TestJanetTypeMapper:
    """Tests for Janet type mapping."""
    
    def test_number_types(self):
        mapper = JanetTypeMapper()
        assert mapper.map_type('i32') == ':number'
        assert mapper.map_type('f64') == ':number'
    
    def test_special_types(self):
        mapper = JanetTypeMapper()
        assert mapper.map_type('bool') == ':boolean'
        assert mapper.map_type('string') == ':string'
        assert mapper.map_type('void') == ':nil'
        assert mapper.map_type('table') == ':table'
        assert mapper.map_type('fiber') == ':fiber'


class TestJanetEmitter:
    """Tests for Janet code emission."""
    
    @pytest.fixture
    def emitter(self, tmp_path):
        config = JanetConfig(target_dir=tmp_path)
        return JanetEmitter(config)
    
    def test_basic_function(self, emitter):
        """TC-JN-001: Test basic function emission."""
        ir = {
            "module": "math",
            "functions": [{
                "name": "add",
                "params": [{"name": "a", "type": "i32"}, {"name": "b", "type": "i32"}],
                "return_type": "i32",
                "body": [{"kind": "return", "value": {"kind": "binary_op", "op": "+", "left": {"kind": "var", "name": "a"}, "right": {"kind": "var", "name": "b"}}}]
            }]
        }
        result = emitter.emit(ir)
        assert "(defn add" in result.code
        assert "[a b]" in result.code
        assert "(+ a b)" in result.code
    
    def test_boolean_literals(self, emitter):
        """TC-JN-002: Test boolean literal emission."""
        assert emitter._emit_literal(True) == "true"
        assert emitter._emit_literal(False) == "false"
        assert emitter._emit_literal(None) == "nil"
    
    def test_parameter_brackets(self, emitter):
        """TC-JN-003: Test that parameters use brackets, not parentheses."""
        func = {"name": "test", "params": [{"name": "x"}], "body": []}
        result = emitter._emit_function(func)
        assert "[x]" in result
    
    def test_struct_definition(self, emitter):
        """TC-JN-004: Test struct definition."""
        ir = {
            "module": "shapes",
            "types": [{
                "name": "point",
                "fields": [{"name": "x", "type": "f64"}, {"name": "y", "type": "f64"}]
            }],
            "functions": []
        }
        result = emitter.emit(ir)
        assert "point" in result.code.lower()
        # Janet creates constructor function
        assert "defn" in result.code.lower() or "def" in result.code.lower()
    
    def test_while_loop(self, emitter):
        """TC-JN-005: Test while loop emission."""
        stmt = {
            "kind": "while",
            "condition": {"kind": "binary_op", "op": "<", "left": {"kind": "var", "name": "i"}, "right": {"kind": "literal", "value": 10}},
            "body": []
        }
        result = emitter._emit_while(stmt)
        assert "while" in result
    
    def test_for_loop(self, emitter):
        """TC-JN-006: Test for loop emission."""
        stmt = {
            "kind": "for",
            "var": "i",
            "start": {"kind": "literal", "value": 0},
            "end": {"kind": "literal", "value": 10},
            "body": []
        }
        result = emitter._emit_for(stmt)
        assert "for" in result
    
    def test_variable_declaration(self, emitter):
        """TC-JN-007: Test variable declaration."""
        stmt = {"kind": "var_decl", "name": "x", "value": {"kind": "literal", "value": 42}}
        result = emitter._emit_var_decl(stmt)
        assert "(def x 42)" in result or "(var x 42)" in result
    
    def test_manifest_generation(self, emitter):
        """TC-JN-008: Test manifest generation."""
        ir = {"module": "test", "functions": []}
        result = emitter.emit(ir)
        assert result.manifest["dialect"] == "janet"
    
    def test_comment_syntax(self, emitter):
        """TC-JN-009: Test that Janet uses # for comments."""
        ir = {"module": "test", "functions": []}
        result = emitter.emit(ir)
        # Janet uses # for comments
        assert "# Generated" in result.code
    
    def test_each_loop(self, emitter):
        """TC-JN-010: Test each loop for iterables."""
        stmt = {
            "kind": "for",
            "var": "item",
            "iterable": {"kind": "var", "name": "items"},
            "body": []
        }
        result = emitter._emit_for(stmt)
        assert "each" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
