#!/usr/bin/env python3
"""Tests for STUNIR Scheme Emitter.

Part of Phase 5A: Core Lisp Implementation.
"""

import pytest
import sys
from pathlib import Path
import tempfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from targets.lisp.scheme.emitter import SchemeEmitter, SchemeEmitterConfig
from targets.lisp.scheme.types import SchemeTypeMapper, SCHEME_TYPES


class TestSchemeTypeMapper:
    """Test Scheme type mapping."""
    
    def test_basic_types(self):
        """Test basic type mapping."""
        mapper = SchemeTypeMapper()
        assert mapper.map_type('i32') == 'integer'
        assert mapper.map_type('f64') == 'real'
        assert mapper.map_type('bool') == 'boolean'
        assert mapper.map_type('string') == 'string'
    
    def test_predicate(self):
        """Test predicate generation."""
        mapper = SchemeTypeMapper()
        assert mapper.get_predicate('i32') == 'integer?'
        assert mapper.get_predicate('string') == 'string?'
    
    def test_type_comment(self):
        """Test type comment generation."""
        mapper = SchemeTypeMapper()
        comment = mapper.emit_type_comment('add', ['i32', 'i32'], 'i32')
        assert 'add' in comment
        assert 'integer' in comment


class TestSchemeEmitter:
    """Test Scheme emitter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = SchemeEmitterConfig(target_dir=Path(self.temp_dir))
        self.emitter = SchemeEmitter(self.config)
    
    def test_basic_function(self):
        """TC-SCH-001: Test basic function emission."""
        ir = {
            "module": "math",
            "functions": [{
                "name": "square",
                "params": [{"name": "x", "type": "i32"}],
                "return_type": "i32",
                "body": [{
                    "kind": "return",
                    "value": {
                        "kind": "binary_op",
                        "op": "*",
                        "left": {"kind": "var", "name": "x"},
                        "right": {"kind": "var", "name": "x"}
                    }
                }]
            }]
        }
        result = self.emitter.emit(ir)
        
        assert result.success
        assert "(define (square x)" in result.code
        assert "(* x x)" in result.code
    
    def test_boolean_emission(self):
        """TC-SCH-002: Test boolean emission uses #t/#f."""
        ir = {
            "module": "test",
            "functions": [{
                "name": "get_true",
                "params": [],
                "body": [{"kind": "return", "value": {"kind": "literal", "value": True}}]
            }]
        }
        result = self.emitter.emit(ir)
        
        assert result.success
        assert "#t" in result.code  # Not 't' like CL
    
    def test_r7rs_library(self):
        """TC-SCH-003: Test R7RS library generation."""
        ir = {
            "module": "utils",
            "functions": [{
                "name": "helper",
                "params": [],
                "exported": True,
                "body": []
            }]
        }
        result = self.emitter.emit(ir)
        
        assert result.success
        assert "(define-library (stunir utils)" in result.code
        assert "(export" in result.code
    
    def test_while_as_named_let(self):
        """Test while loop emits as named let."""
        ir = {
            "module": "test",
            "functions": [{
                "name": "count",
                "params": [],
                "body": [{
                    "kind": "while",
                    "condition": {"kind": "literal", "value": True},
                    "body": []
                }]
            }]
        }
        result = self.emitter.emit(ir)
        
        assert result.success
        assert "loop" in result.code
        assert "(when" in result.code or "loop" in result.code
    
    def test_lambda_emission(self):
        """Test lambda expression emission."""
        ir = {
            "module": "test",
            "functions": [{
                "name": "make_adder",
                "params": [{"name": "n", "type": "i32"}],
                "body": [{
                    "kind": "return",
                    "value": {
                        "kind": "lambda",
                        "params": [{"name": "x"}],
                        "body": [{
                            "kind": "return",
                            "value": {
                                "kind": "binary_op",
                                "op": "+",
                                "left": {"kind": "var", "name": "n"},
                                "right": {"kind": "var", "name": "x"}
                            }
                        }]
                    }
                }]
            }]
        }
        result = self.emitter.emit(ir)
        
        assert result.success
        assert "(lambda" in result.code


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
