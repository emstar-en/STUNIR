#!/usr/bin/env python3
"""Tests for STUNIR Racket Emitter.

Part of Phase 5A: Core Lisp Implementation.
"""

import pytest
import sys
from pathlib import Path
import tempfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from targets.lisp.racket.emitter import RacketEmitter, RacketEmitterConfig
from targets.lisp.racket.types import RacketTypeMapper, RACKET_TYPES, TYPED_RACKET_TYPES, RACKET_CONTRACTS


class TestRacketTypeMapper:
    """Test Racket type mapping."""
    
    def test_basic_types(self):
        """Test basic type mapping."""
        mapper = RacketTypeMapper(use_typed=True)
        assert mapper.map_type('i32') == 'Integer'
        assert mapper.map_type('f64') == 'Flonum'
        assert mapper.map_type('bool') == 'Boolean'
        assert mapper.map_type('string') == 'String'
    
    def test_contracts(self):
        """Test contract predicate generation."""
        mapper = RacketTypeMapper(use_contracts=True)
        assert mapper.get_contract('i32') == 'integer?'
        assert mapper.get_contract('string') == 'string?'
    
    def test_type_annotation(self):
        """Test type annotation generation."""
        mapper = RacketTypeMapper(use_typed=True)
        ann = mapper.emit_type_annotation('add', ['i32', 'i32'], 'i32')
        assert '(: add' in ann
        assert '->' in ann
        assert 'Integer' in ann
    
    def test_contract_emit(self):
        """Test contract expression generation."""
        mapper = RacketTypeMapper(use_contracts=True)
        contract = mapper.emit_contract(['i32', 'i32'], 'i32')
        assert '->' in contract
        assert 'integer?' in contract


class TestRacketEmitter:
    """Test Racket emitter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = RacketEmitterConfig(target_dir=Path(self.temp_dir))
        self.emitter = RacketEmitter(self.config)
    
    def test_lang_line(self):
        """TC-RKT-001: Test #lang directive."""
        ir = {"module": "test", "functions": []}
        result = self.emitter.emit(ir)
        
        assert result.success
        assert result.code.startswith("#lang racket")
    
    def test_contracts(self):
        """TC-RKT-002: Test contract generation."""
        ir = {
            "module": "math",
            "exports": ["add"],
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
        config = RacketEmitterConfig(target_dir=Path(self.temp_dir), use_contracts=True)
        emitter = RacketEmitter(config)
        result = emitter.emit(ir)
        
        assert result.success
        assert "contract-out" in result.code
        assert "(-> integer? integer? integer?)" in result.code
    
    def test_typed_racket(self):
        """TC-RKT-003: Test Typed Racket generation."""
        config = RacketEmitterConfig(
            target_dir=Path(self.temp_dir),
            use_typed=True,
            lang="typed/racket"
        )
        emitter = RacketEmitter(config)
        
        ir = {
            "module": "math",
            "functions": [{
                "name": "add",
                "params": [{"name": "a", "type": "i32"}, {"name": "b", "type": "i32"}],
                "return_type": "i32",
                "body": [{"kind": "return", "value": {"kind": "literal", "value": 0}}]
            }]
        }
        result = emitter.emit(ir)
        
        assert result.success
        assert "#lang typed/racket" in result.code
        assert "(: add" in result.code  # Type annotation
    
    def test_struct(self):
        """TC-RKT-004: Test struct generation."""
        ir = {
            "module": "shapes",
            "types": [{
                "name": "Point",
                "fields": [
                    {"name": "x", "type": "f64"},
                    {"name": "y", "type": "f64"}
                ]
            }],
            "functions": []
        }
        result = self.emitter.emit(ir)
        
        assert result.success
        assert "(struct point" in result.code.lower()
    
    def test_for_loop(self):
        """TC-RKT-005: Test for loop generation."""
        ir = {
            "module": "test",
            "functions": [{
                "name": "sum_range",
                "params": [],
                "body": [{
                    "kind": "for",
                    "var": "i",
                    "start": {"kind": "literal", "value": 0},
                    "end": {"kind": "literal", "value": 10},
                    "body": []
                }]
            }]
        }
        result = self.emitter.emit(ir)
        
        assert result.success
        assert "(for" in result.code
        assert "in-range" in result.code
    
    def test_boolean_emission(self):
        """Test boolean emission uses #t/#f."""
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
        assert "#t" in result.code
    
    def test_lambda_emission(self):
        """Test lambda expression emission."""
        ir = {
            "module": "test",
            "functions": [{
                "name": "get_lambda",
                "params": [],
                "body": [{
                    "kind": "return",
                    "value": {
                        "kind": "lambda",
                        "params": [{"name": "x"}],
                        "body": [{"kind": "return", "value": {"kind": "var", "name": "x"}}]
                    }
                }]
            }]
        }
        result = self.emitter.emit(ir)
        
        assert result.success
        assert "(lambda" in result.code


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
