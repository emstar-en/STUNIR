#!/usr/bin/env python3
"""Tests for STUNIR Emacs Lisp Emitter.

Part of Phase 5B: Extended Lisp Implementation.
"""

import pytest
import sys
import os
from pathlib import Path
from tempfile import TemporaryDirectory

# Add the repo to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from targets.lisp.emacs_lisp import EmacsLispEmitter, EmacsLispConfig, EmacsLispTypeMapper


class TestEmacsLispTypeMapper:
    """Tests for Emacs Lisp type mapping."""
    
    def test_integer_types(self):
        mapper = EmacsLispTypeMapper()
        assert mapper.map_type('i32') == 'integer'
        assert mapper.map_type('i64') == 'integer'
        assert mapper.map_type('u8') == 'integer'
    
    def test_float_types(self):
        mapper = EmacsLispTypeMapper()
        assert mapper.map_type('f32') == 'float'
        assert mapper.map_type('f64') == 'float'
    
    def test_other_types(self):
        mapper = EmacsLispTypeMapper()
        assert mapper.map_type('bool') == 'boolean'
        assert mapper.map_type('string') == 'string'
        assert mapper.map_type('void') == 'nil'
        assert mapper.map_type('any') == 't'


class TestEmacsLispEmitter:
    """Tests for Emacs Lisp code emission."""
    
    @pytest.fixture
    def emitter(self, tmp_path):
        config = EmacsLispConfig(target_dir=tmp_path)
        return EmacsLispEmitter(config)
    
    def test_basic_function(self, emitter):
        """TC-EL-001: Test basic function emission."""
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
        assert "(defun add (a b)" in result.code
        assert "(+ a b)" in result.code
        assert "lexical-binding: t" in result.code
    
    def test_boolean_literals(self, emitter):
        """TC-EL-002: Test boolean literal emission."""
        assert emitter._emit_literal(True) == "t"
        assert emitter._emit_literal(False) == "nil"
        assert emitter._emit_literal(None) == "nil"
    
    def test_let_binding(self, emitter):
        """TC-EL-003: Test let binding emission."""
        stmt = {"kind": "var_decl", "name": "x", "value": {"kind": "literal", "value": 42}}
        result = emitter._emit_var_decl(stmt)
        assert "let" in result
        assert "x" in result
        assert "42" in result
    
    def test_conditional(self, emitter):
        """TC-EL-004: Test conditional emission."""
        stmt = {
            "kind": "if",
            "condition": {"kind": "binary_op", "op": "<", "left": {"kind": "var", "name": "n"}, "right": {"kind": "literal", "value": 0}},
            "then": [{"kind": "return", "value": {"kind": "literal", "value": 0}}]
        }
        result = emitter._emit_if_stmt(stmt)
        # Should use when for single-branch
        assert "when" in result or "if" in result
        assert "< n 0" in result
    
    def test_module_provide(self, emitter):
        """TC-EL-005: Test module with provide statement."""
        ir = {"module": "my-utils", "functions": []}
        result = emitter.emit(ir)
        assert "(provide 'my-utils)" in result.code
        assert ";;; my-utils.el ends here" in result.code
    
    def test_while_loop(self, emitter):
        """TC-EL-006: Test while loop emission."""
        stmt = {
            "kind": "while",
            "condition": {"kind": "binary_op", "op": "<", "left": {"kind": "var", "name": "i"}, "right": {"kind": "literal", "value": 10}},
            "body": []
        }
        result = emitter._emit_while(stmt)
        assert "(while (< i 10)" in result
    
    def test_for_loop_dolist(self, emitter):
        """TC-EL-007: Test for loop with dolist."""
        stmt = {
            "kind": "for",
            "var": "item",
            "iterable": {"kind": "var", "name": "items"},
            "body": []
        }
        result = emitter._emit_for(stmt)
        assert "dolist" in result
        assert "item" in result
    
    def test_manifest_generation(self, emitter):
        """TC-EL-008: Test manifest generation."""
        ir = {"module": "test", "functions": []}
        result = emitter.emit(ir)
        assert result.manifest is not None
        assert "dialect" in result.manifest
        assert result.manifest["dialect"] == "emacs-lisp"
    
    def test_empty_function_body(self, emitter):
        """TC-EL-009: Test empty function body."""
        ir = {
            "module": "test",
            "functions": [{"name": "empty_func", "params": [], "body": []}]
        }
        result = emitter.emit(ir)
        assert "nil" in result.code
    
    def test_string_escaping(self, emitter):
        """TC-EL-010: Test string escaping."""
        assert '"hello world"' == emitter._emit_literal("hello world")
        # Note: newlines are preserved in the string
        result = emitter._emit_literal("line1\nline2")
        assert result.startswith('"') and result.endswith('"')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
