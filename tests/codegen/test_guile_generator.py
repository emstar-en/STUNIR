#!/usr/bin/env python3
"""Tests for STUNIR Guile Emitter.

Part of Phase 5B: Extended Lisp Implementation.
"""

import pytest
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from targets.lisp.guile import GuileEmitter, GuileConfig, GuileTypeMapper


class TestGuileTypeMapper:
    """Tests for Guile type mapping."""
    
    def test_goops_types(self):
        mapper = GuileTypeMapper(use_goops=True)
        assert mapper.map_type('i32') == '<integer>'
        assert mapper.map_type('f64') == '<real>'
        assert mapper.map_type('bool') == '<boolean>'
        assert mapper.map_type('any') == '<top>'
    
    def test_special_types(self):
        mapper = GuileTypeMapper()
        assert mapper.map_type('void') == '*unspecified*'
        assert mapper.map_type('ptr') == '<pointer>'


class TestGuileEmitter:
    """Tests for Guile code emission."""
    
    @pytest.fixture
    def emitter(self, tmp_path):
        config = GuileConfig(target_dir=tmp_path)
        return GuileEmitter(config)
    
    def test_basic_function(self, emitter):
        """TC-GU-001: Test basic function emission."""
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
        assert "(define (add a b)" in result.code
        assert "(+ a b)" in result.code
    
    def test_boolean_literals(self, emitter):
        """TC-GU-002: Test boolean literal emission."""
        assert emitter._emit_literal(True) == "#t"
        assert emitter._emit_literal(False) == "#f"
        assert emitter._emit_literal(None) == "#f"
    
    def test_module_definition(self, emitter):
        """TC-GU-003: Test module definition."""
        ir = {"module": "my-utils", "functions": [], "exports": ["func1"]}
        emitter._exports.add("func1")
        result = emitter.emit(ir)
        assert "(define-module (stunir my-utils)" in result.code
        assert "#:export" in result.code
    
    def test_goops_class(self, emitter):
        """TC-GU-004: Test GOOPS class emission."""
        ir = {
            "module": "shapes",
            "types": [{
                "name": "point",
                "fields": [{"name": "x", "type": "f64"}, {"name": "y", "type": "f64"}]
            }],
            "functions": []
        }
        result = emitter.emit(ir)
        assert "(define-class <point>" in result.code
        assert "#:accessor" in result.code
    
    def test_conditional(self, emitter):
        """TC-GU-005: Test conditional emission."""
        stmt = {
            "kind": "if",
            "condition": {"kind": "binary_op", "op": "<", "left": {"kind": "var", "name": "n"}, "right": {"kind": "literal", "value": 0}},
            "then": [{"kind": "return", "value": {"kind": "literal", "value": 0}}]
        }
        result = emitter._emit_if_stmt(stmt)
        assert "when" in result or "if" in result
        assert "< n 0" in result
    
    def test_do_loop(self, emitter):
        """TC-GU-006: Test do loop emission."""
        stmt = {
            "kind": "for",
            "var": "i",
            "start": {"kind": "literal", "value": 0},
            "end": {"kind": "literal", "value": 10},
            "body": []
        }
        result = emitter._emit_for(stmt)
        assert "do" in result
    
    def test_named_let_loop(self, emitter):
        """TC-GU-007: Test named let for while loop."""
        stmt = {
            "kind": "while",
            "condition": {"kind": "binary_op", "op": "<", "left": {"kind": "var", "name": "i"}, "right": {"kind": "literal", "value": 10}},
            "body": []
        }
        result = emitter._emit_while(stmt)
        assert "let loop" in result
    
    def test_manifest_generation(self, emitter):
        """TC-GU-008: Test manifest generation."""
        ir = {"module": "test", "functions": []}
        result = emitter.emit(ir)
        assert result.manifest["dialect"] == "guile"
    
    def test_not_equal_operator(self, emitter):
        """TC-GU-009: Test != operator special handling."""
        data = {
            "op": "!=",
            "left": {"kind": "var", "name": "a"},
            "right": {"kind": "var", "name": "b"}
        }
        result = emitter._emit_binary_op(data)
        assert "(not (= a b))" == result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
