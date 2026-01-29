#!/usr/bin/env python3
"""Tests for STUNIR Hy Emitter.

Part of Phase 5B: Extended Lisp Implementation.
"""

import pytest
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from targets.lisp.hy import HyEmitter, HyConfig, HyTypeMapper


class TestHyTypeMapper:
    """Tests for Hy/Python type mapping."""
    
    def test_python_types(self):
        mapper = HyTypeMapper()
        assert mapper.map_type('i32') == 'int'
        assert mapper.map_type('f64') == 'float'
        assert mapper.map_type('bool') == 'bool'
        assert mapper.map_type('string') == 'str'
    
    def test_special_types(self):
        mapper = HyTypeMapper()
        assert mapper.map_type('void') == 'None'
        assert mapper.map_type('any') == 'typing.Any'
        assert mapper.map_type('list') == 'list'
        assert mapper.map_type('dict') == 'dict'
    
    def test_param_annotation(self):
        mapper = HyTypeMapper(emit_annotations=True)
        assert "^int x" == mapper.emit_param_annotation("x", "i32")


class TestHyEmitter:
    """Tests for Hy code emission."""
    
    @pytest.fixture
    def emitter(self, tmp_path):
        config = HyConfig(target_dir=tmp_path)
        return HyEmitter(config)
    
    def test_basic_function(self, emitter):
        """TC-HY-001: Test basic function emission."""
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
        assert "(+ a b)" in result.code
    
    def test_python_booleans(self, emitter):
        """TC-HY-002: Test Python boolean literals."""
        assert emitter._emit_literal(True) == "True"
        assert emitter._emit_literal(False) == "False"
        assert emitter._emit_literal(None) == "None"
    
    def test_type_annotations(self, emitter):
        """TC-HY-003: Test type annotations."""
        func = {
            "name": "add",
            "params": [{"name": "a", "type": "i32"}, {"name": "b", "type": "i32"}],
            "return_type": "i32",
            "body": []
        }
        result = emitter._emit_function(func)
        assert "^int" in result or "int" in result
    
    def test_python_class(self, emitter):
        """TC-HY-004: Test Python class emission."""
        ir = {
            "module": "shapes",
            "types": [{
                "name": "Point",
                "fields": [{"name": "x", "type": "f64"}, {"name": "y", "type": "f64"}]
            }],
            "functions": []
        }
        result = emitter.emit(ir)
        assert "(defclass Point" in result.code
        assert "__init__" in result.code
    
    def test_import_statement(self, emitter):
        """TC-HY-005: Test import emission."""
        ir = {
            "module": "utils",
            "imports": [{"module": "os"}],
            "functions": []
        }
        result = emitter.emit(ir)
        assert "(import os)" in result.code
    
    def test_setv_assignment(self, emitter):
        """TC-HY-006: Test setv for variable declaration."""
        stmt = {"kind": "var_decl", "name": "x", "value": {"kind": "literal", "value": 42}}
        result = emitter._emit_var_decl(stmt)
        assert "(setv x 42)" in result
    
    def test_for_loop(self, emitter):
        """TC-HY-007: Test for loop emission."""
        stmt = {
            "kind": "for",
            "var": "item",
            "iterable": {"kind": "var", "name": "items"},
            "body": []
        }
        result = emitter._emit_for(stmt)
        assert "(for [item items]" in result
    
    def test_manifest_generation(self, emitter):
        """TC-HY-008: Test manifest generation."""
        ir = {"module": "test", "functions": []}
        result = emitter.emit(ir)
        assert result.manifest["dialect"] == "hy"
    
    def test_module_filename(self, emitter):
        """TC-HY-009: Test module filename uses underscores."""
        ir = {"module": "my-module", "functions": []}
        result = emitter.emit(ir)
        # Hy files should use underscores for Python compatibility
        assert "my_module.hy" in result.files
    
    def test_from_import(self, emitter):
        """TC-HY-010: Test from X import Y syntax."""
        ir = {
            "module": "utils",
            "imports": [{"module": "pathlib", "names": ["Path"]}],
            "functions": []
        }
        result = emitter.emit(ir)
        assert "pathlib" in result.code
        assert "Path" in result.code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
