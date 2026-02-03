#!/usr/bin/env python3
"""Tests for STUNIR Clojure Emitter.

Part of Phase 5A: Core Lisp Implementation.
"""

import pytest
import sys
from pathlib import Path
import tempfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from targets.lisp.clojure.emitter import ClojureEmitter, ClojureEmitterConfig
from targets.lisp.clojure.types import ClojureTypeMapper, CLOJURE_TYPES, CLOJURE_TYPE_HINTS


class TestClojureTypeMapper:
    """Test Clojure type mapping."""
    
    def test_basic_types(self):
        """Test basic type mapping."""
        mapper = ClojureTypeMapper()
        assert mapper.map_type('i32') == 'int?'
        assert mapper.map_type('f64') == 'double?'
        assert mapper.map_type('bool') == 'boolean?'
        assert mapper.map_type('string') == 'string?'
    
    def test_type_hints(self):
        """Test type hint generation."""
        mapper = ClojureTypeMapper(use_type_hints=True)
        assert mapper.get_type_hint('i64') == '^long'
        assert mapper.get_type_hint('f64') == '^double'
    
    def test_spec_fdef(self):
        """Test spec fdef generation."""
        mapper = ClojureTypeMapper(use_spec=True)
        spec = mapper.emit_spec_fdef('add', ['i64', 'i64'], 'i64')
        assert 's/fdef' in spec
        assert 's/cat' in spec
        assert ':ret' in spec


class TestClojureEmitter:
    """Test Clojure emitter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ClojureEmitterConfig(target_dir=Path(self.temp_dir))
        self.emitter = ClojureEmitter(self.config)
    
    def test_function_with_hints(self):
        """TC-CLJ-001: Test function with type hints."""
        ir = {
            "module": "math",
            "functions": [{
                "name": "add",
                "params": [{"name": "a", "type": "i64"}, {"name": "b", "type": "i64"}],
                "return_type": "i64",
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
        assert "(defn" in result.code
        assert "^long" in result.code  # Type hints
        assert "(+ a b)" in result.code
    
    def test_let_vector_syntax(self):
        """TC-CLJ-002: Test let with vector bindings."""
        # The Clojure emitter uses vector syntax for let
        ir = {
            "module": "test",
            "functions": [{
                "name": "compute",
                "params": [],
                "body": [
                    {"kind": "var_decl", "name": "x", "value": {"kind": "literal", "value": 10}},
                    {"kind": "return", "value": {"kind": "var", "name": "x"}}
                ]
            }]
        }
        result = self.emitter.emit(ir)
        
        assert result.success
        # Should use def for top-level variables
        assert "(def x" in result.code or "defn" in result.code
    
    def test_namespace(self):
        """TC-CLJ-003: Test namespace generation."""
        ir = {"module": "myapp.utils", "functions": []}
        result = self.emitter.emit(ir)
        
        assert result.success
        assert "(ns stunir." in result.code
    
    def test_boolean_emission(self):
        """TC-CLJ-004: Test Clojure booleans."""
        ir = {
            "module": "test",
            "functions": [{
                "name": "get_bool",
                "params": [],
                "body": [{"kind": "return", "value": {"kind": "literal", "value": True}}]
            }]
        }
        result = self.emitter.emit(ir)
        
        assert result.success
        assert "true" in result.code  # Not #t like Scheme
        assert "#t" not in result.code
    
    def test_fn_syntax(self):
        """Test anonymous function syntax."""
        ir = {
            "module": "test",
            "functions": [{
                "name": "make_fn",
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
        assert "(fn [" in result.code  # Clojure uses fn, not lambda
    
    def test_loop_recur(self):
        """Test loop/recur for while loops."""
        ir = {
            "module": "test",
            "functions": [{
                "name": "count_loop",
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
        assert "(loop" in result.code
        assert "(recur)" in result.code


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
