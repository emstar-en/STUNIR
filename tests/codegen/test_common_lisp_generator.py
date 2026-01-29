#!/usr/bin/env python3
"""Tests for STUNIR Common Lisp Emitter.

Part of Phase 5A: Core Lisp Implementation.
"""

import pytest
import sys
from pathlib import Path
import tempfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from targets.lisp.common_lisp.emitter import CommonLispEmitter, CommonLispConfig
from targets.lisp.common_lisp.types import CommonLispTypeMapper, COMMON_LISP_TYPES


class TestCommonLispTypeMapper:
    """Test Common Lisp type mapping."""
    
    def test_basic_types(self):
        """Test basic type mapping."""
        mapper = CommonLispTypeMapper()
        assert mapper.map_type('i32') == 'fixnum'
        assert mapper.map_type('f64') == 'double-float'
        assert mapper.map_type('bool') == 'boolean'
        assert mapper.map_type('string') == 'string'
    
    def test_declaration(self):
        """Test type declaration generation."""
        mapper = CommonLispTypeMapper()
        decl = mapper.emit_declaration('x', 'i32')
        assert 'declare' in decl
        assert 'type' in decl
        assert 'fixnum' in decl
    
    def test_ftype(self):
        """Test ftype declaration generation."""
        mapper = CommonLispTypeMapper()
        ftype = mapper.emit_ftype('add', ['i32', 'i32'], 'i32')
        assert 'declaim' in ftype
        assert 'ftype' in ftype
        assert 'function' in ftype


class TestCommonLispEmitter:
    """Test Common Lisp emitter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = CommonLispConfig(target_dir=Path(self.temp_dir))
        self.emitter = CommonLispEmitter(self.config)
    
    def test_basic_function(self):
        """TC-CL-001: Test basic function emission."""
        ir = {
            "module": "test",
            "functions": [{
                "name": "identity",
                "params": [{"name": "x", "type": "i32"}],
                "return_type": "i32",
                "body": [{"kind": "return", "value": {"kind": "var", "name": "x"}}]
            }]
        }
        result = self.emitter.emit(ir)
        
        assert result.success
        assert "(defun identity (x)" in result.code
        assert "x" in result.code
    
    def test_arithmetic(self):
        """TC-CL-002: Test arithmetic operations."""
        ir = {
            "module": "math",
            "functions": [{
                "name": "add",
                "params": [{"name": "a", "type": "i32"}, {"name": "b", "type": "i32"}],
                "return_type": "i32",
                "body": [{
                    "kind": "return",
                    "value": {
                        "kind": "binary_op",
                        "op": "+",
                        "left": {"kind": "var", "name": "a"},
                        "right": {"kind": "var", "name": "b"}
                    }
                }]
            }]
        }
        result = self.emitter.emit(ir)
        
        assert result.success
        assert "(+ a b)" in result.code
    
    def test_control_flow(self):
        """TC-CL-003: Test control flow emission."""
        ir = {
            "module": "test",
            "functions": [{
                "name": "absolute",
                "params": [{"name": "n", "type": "i32"}],
                "return_type": "i32",
                "body": [{
                    "kind": "if",
                    "condition": {
                        "kind": "binary_op",
                        "op": "<",
                        "left": {"kind": "var", "name": "n"},
                        "right": {"kind": "literal", "value": 0}
                    },
                    "then": [{
                        "kind": "return",
                        "value": {
                            "kind": "unary_op",
                            "op": "-",
                            "operand": {"kind": "var", "name": "n"}
                        }
                    }],
                    "else": [{"kind": "return", "value": {"kind": "var", "name": "n"}}]
                }]
            }]
        }
        result = self.emitter.emit(ir)
        
        assert result.success
        assert "(if (< n 0)" in result.code
    
    def test_clos_class(self):
        """TC-CL-004: Test CLOS class generation."""
        ir = {
            "module": "shapes",
            "types": [{
                "name": "Circle",
                "fields": [
                    {"name": "radius", "type": "f64"},
                    {"name": "x", "type": "f64"},
                    {"name": "y", "type": "f64"}
                ]
            }],
            "functions": []
        }
        result = self.emitter.emit(ir)
        
        assert result.success
        assert "(defclass circle" in result.code.lower()
    
    def test_asdf_system(self):
        """TC-CL-005: Test ASDF system generation."""
        ir = {"module": "mylib", "functions": []}
        result = self.emitter.emit(ir)
        
        assert result.success
        assert len(result.files) >= 1
        # Check that an .asd file was generated
        asd_files = [f for f in result.files.keys() if f.endswith('.asd')]
        assert len(asd_files) == 1
        assert "defsystem" in result.files[asd_files[0]]
    
    def test_package_definition(self):
        """Test package definition generation."""
        ir = {
            "module": "utils",
            "exports": ["helper", "util-fn"],
            "functions": [
                {"name": "helper", "params": [], "body": []},
                {"name": "util_fn", "params": [], "body": []}
            ]
        }
        result = self.emitter.emit(ir)
        
        assert result.success
        assert "defpackage" in result.code
        assert "in-package" in result.code
    
    def test_literals(self):
        """Test literal emission."""
        ir = {
            "module": "test",
            "functions": [{
                "name": "get_values",
                "params": [],
                "body": [{"kind": "return", "value": {"kind": "literal", "value": True}}]
            }]
        }
        result = self.emitter.emit(ir)
        
        assert result.success
        assert "t" in result.code  # CL uses t for true


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
