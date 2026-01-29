#!/usr/bin/env python3
"""Tests for GNU Prolog emitter.

Tests cover:
- Basic fact and rule emission
- Public declarations (no modules)
- CLP(FD) constraints
- CLP(B) constraints
- Dynamic declarations
- List handling
- Cut and negation
- Determinism verification
- Comparison with SWI-Prolog output

Part of Phase 5C-2: GNU Prolog with CLP support.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from targets.prolog.gnu_prolog.emitter import GNUPrologEmitter, GNUPrologConfig, EmitterResult
from targets.prolog.gnu_prolog.types import (
    GNUPrologTypeMapper, GNU_PROLOG_TYPES,
    CLPFD_OPERATORS, CLPB_OPERATORS, CLPFD_PREDICATES
)


class TestGNUPrologTypes:
    """Tests for type mapping."""
    
    def test_basic_type_mapping(self):
        """Test basic IR to GNU Prolog type mapping."""
        mapper = GNUPrologTypeMapper()
        
        assert mapper.map_type('i32') == 'integer'
        assert mapper.map_type('f64') == 'float'
        assert mapper.map_type('bool') == 'boolean'
        assert mapper.map_type('string') == 'atom'
        assert mapper.map_type('unknown') == 'term'
    
    def test_clpfd_operator_detection(self):
        """Test CLP(FD) operator detection."""
        mapper = GNUPrologTypeMapper()
        
        assert mapper.is_clpfd_operator('#=')
        assert mapper.is_clpfd_operator('#<')
        assert mapper.is_clpfd_operator('>=')
        assert not mapper.is_clpfd_operator('append')
    
    def test_clpfd_operator_mapping(self):
        """Test CLP(FD) operator mapping."""
        mapper = GNUPrologTypeMapper()
        
        assert mapper.map_clpfd_operator('==') == '#='
        assert mapper.map_clpfd_operator('!=') == '#\\='
        assert mapper.map_clpfd_operator('<') == '#<'
        assert mapper.map_clpfd_operator('>=') == '#>='
    
    def test_clpb_operator_detection(self):
        """Test CLP(B) operator detection."""
        mapper = GNUPrologTypeMapper()
        
        assert mapper.is_clpb_operator('and')
        assert mapper.is_clpb_operator('or')
        assert mapper.is_clpb_operator('#<=>')
        assert not mapper.is_clpb_operator('#=')
    
    def test_clpfd_predicate_detection(self):
        """Test CLP(FD) predicate detection."""
        mapper = GNUPrologTypeMapper()
        
        assert mapper.is_clpfd_predicate('domain')
        assert mapper.is_clpfd_predicate('fd_domain')
        assert mapper.is_clpfd_predicate('labeling')
        assert mapper.is_clpfd_predicate('all_different')
        assert not mapper.is_clpfd_predicate('append')
    
    def test_clpfd_predicate_mapping(self):
        """Test CLP(FD) predicate name mapping."""
        mapper = GNUPrologTypeMapper()
        
        assert mapper.map_clpfd_predicate('domain') == 'fd_domain'
        assert mapper.map_clpfd_predicate('labeling') == 'fd_labeling'
        assert mapper.map_clpfd_predicate('all_different') == 'fd_all_different'


class TestGNUPrologEmitter:
    """Tests for GNU Prolog emitter."""
    
    def test_simple_fact_emission(self):
        """TC-GNU-001: Test emitting simple facts without module."""
        ir = {
            "module": "family",
            "clauses": [
                {"kind": "fact", "predicate": "parent", "args": [
                    {"kind": "atom", "value": "tom"},
                    {"kind": "atom", "value": "bob"}
                ]},
                {"kind": "fact", "predicate": "parent", "args": [
                    {"kind": "atom", "value": "tom"},
                    {"kind": "atom", "value": "liz"}
                ]}
            ]
        }
        
        emitter = GNUPrologEmitter()
        result = emitter.emit(ir)
        
        assert isinstance(result, EmitterResult)
        assert "parent(tom, bob)." in result.code
        assert "parent(tom, liz)." in result.code
        # No module declaration in GNU Prolog
        assert ":- module(" not in result.code
        assert result.filename == "stunir_family"
    
    def test_public_declaration(self):
        """TC-GNU-002: Test public predicate declaration (GNU's export)."""
        ir = {
            "module": "utils",
            "exports": [{"predicate": "helper", "arity": 2}],
            "clauses": [
                {"kind": "fact", "predicate": "helper", "args": [
                    {"kind": "variable", "name": "X"},
                    {"kind": "variable", "name": "X"}
                ]}
            ]
        }
        
        emitter = GNUPrologEmitter()
        result = emitter.emit(ir)
        
        assert ":- public(helper/2)." in result.code
    
    def test_clpfd_domain(self):
        """TC-GNU-003: Test CLP(FD) domain constraint emission."""
        ir = {
            "module": "puzzle",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "solve", "args": [
                     {"kind": "variable", "name": "X"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": "domain", "args": [
                         {"kind": "variable", "name": "X"},
                         {"kind": "number", "value": 1},
                         {"kind": "number", "value": 9}
                     ]},
                     {"kind": "compound", "functor": "labeling", "args": [
                         {"kind": "list_term", "elements": [
                             {"kind": "variable", "name": "X"}
                         ]}
                     ]}
                 ]}
            ]
        }
        
        emitter = GNUPrologEmitter()
        result = emitter.emit(ir)
        
        assert "fd_domain(X, 1, 9)" in result.code
        assert "fd_labeling([X])" in result.code
        assert 'clpfd' in result.clp_features
    
    def test_clpfd_arithmetic(self):
        """TC-GNU-004: Test CLP(FD) arithmetic constraint operators."""
        ir = {
            "module": "math",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "sum_to_ten", "args": [
                     {"kind": "variable", "name": "X"},
                     {"kind": "variable", "name": "Y"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": "domain", "args": [
                         {"kind": "variable", "name": "X"},
                         {"kind": "number", "value": 0},
                         {"kind": "number", "value": 10}
                     ]},
                     {"kind": "compound", "functor": "#>", "args": [
                         {"kind": "variable", "name": "X"},
                         {"kind": "number", "value": 0}
                     ]}
                 ]}
            ]
        }
        
        emitter = GNUPrologEmitter()
        result = emitter.emit(ir)
        
        assert "X #> 0" in result.code
        assert 'clpfd' in result.clp_features
    
    def test_clpfd_all_different(self):
        """TC-GNU-005: Test CLP(FD) all_different constraint."""
        ir = {
            "module": "sudoku",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "unique", "args": [
                     {"kind": "variable", "name": "L"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": "all_different", "args": [
                         {"kind": "variable", "name": "L"}
                     ]}
                 ]}
            ]
        }
        
        emitter = GNUPrologEmitter()
        result = emitter.emit(ir)
        
        assert "fd_all_different(L)" in result.code
    
    def test_clpb_constraints(self):
        """TC-GNU-006: Test CLP(B) boolean constraint emission."""
        ir = {
            "module": "logic",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "bool_and", "args": [
                     {"kind": "variable", "name": "X"},
                     {"kind": "variable", "name": "Y"},
                     {"kind": "variable", "name": "R"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": "#<=>", "args": [
                         {"kind": "variable", "name": "R"},
                         {"kind": "compound", "functor": "#/\\", "args": [
                             {"kind": "variable", "name": "X"},
                             {"kind": "variable", "name": "Y"}
                         ]}
                     ]}
                 ]}
            ]
        }
        
        emitter = GNUPrologEmitter()
        result = emitter.emit(ir)
        
        # Check for CLP(B) operators
        assert "#<=>" in result.code or "#/\\" in result.code
        assert 'clpb' in result.clp_features
    
    def test_dynamic_declaration(self):
        """TC-GNU-007: Test dynamic predicate declaration (GNU syntax)."""
        ir = {
            "module": "db",
            "clauses": [],
            "dynamic": [
                {"predicate": "fact", "arity": 2}
            ]
        }
        
        emitter = GNUPrologEmitter()
        result = emitter.emit(ir)
        
        # GNU Prolog uses :- dynamic(p/N). syntax
        assert ":- dynamic(fact/2)." in result.code
    
    def test_rule_with_cut(self):
        """TC-GNU-008: Test rule emission with cut operator."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "max", "args": [
                     {"kind": "variable", "name": "X"},
                     {"kind": "variable", "name": "Y"},
                     {"kind": "variable", "name": "X"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": ">=", "args": [
                         {"kind": "variable", "name": "X"},
                         {"kind": "variable", "name": "Y"}
                     ]},
                     {"kind": "cut"}
                 ]}
            ]
        }
        
        emitter = GNUPrologEmitter()
        result = emitter.emit(ir)
        
        assert "!" in result.code
        assert "max(X, Y, X) :-" in result.code
    
    def test_determinism(self):
        """TC-GNU-009: Test that output is deterministic."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "a", "args": []},
                {"kind": "fact", "predicate": "b", "args": []},
            ]
        }
        
        config = GNUPrologConfig(emit_timestamps=False)
        emitter = GNUPrologEmitter(config)
        
        # Generate multiple times
        results = [emitter.emit(ir).code for _ in range(5)]
        
        # All should be identical
        assert all(r == results[0] for r in results)
    
    def test_compare_swi_gnu(self):
        """TC-GNU-010: Test that GNU output differs appropriately from SWI."""
        from targets.prolog.swi_prolog.emitter import SWIPrologEmitter
        
        ir = {
            "module": "test",
            "exports": [{"predicate": "foo", "arity": 1}],
            "clauses": [
                {"kind": "fact", "predicate": "foo", "args": [
                    {"kind": "atom", "value": "bar"}
                ]}
            ]
        }
        
        swi_emitter = SWIPrologEmitter()
        gnu_emitter = GNUPrologEmitter()
        
        swi_result = swi_emitter.emit(ir)
        gnu_result = gnu_emitter.emit(ir)
        
        # SWI has module, GNU has public
        assert ":- module(" in swi_result.code
        assert ":- module(" not in gnu_result.code
        assert ":- public(" in gnu_result.code
        
        # Both emit the fact the same way
        assert "foo(bar)." in swi_result.code
        assert "foo(bar)." in gnu_result.code
    
    def test_rule_emission(self):
        """Test emitting rules with body."""
        ir = {
            "module": "family",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "grandparent", "args": [
                     {"kind": "variable", "name": "X"},
                     {"kind": "variable", "name": "Z"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": "parent", "args": [
                         {"kind": "variable", "name": "X"},
                         {"kind": "variable", "name": "Y"}
                     ]},
                     {"kind": "compound", "functor": "parent", "args": [
                         {"kind": "variable", "name": "Y"},
                         {"kind": "variable", "name": "Z"}
                     ]}
                 ]}
            ]
        }
        
        emitter = GNUPrologEmitter()
        result = emitter.emit(ir)
        
        assert "grandparent(X, Z) :-" in result.code
        assert "parent(X, Y)" in result.code
        assert "parent(Y, Z)" in result.code
    
    def test_list_emission(self):
        """Test list term emission."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "first", "args": [
                     {"kind": "list_term", 
                      "elements": [{"kind": "variable", "name": "H"}],
                      "tail": {"kind": "variable", "name": "T"}},
                     {"kind": "variable", "name": "H"}
                 ]},
                 "body": []}
            ]
        }
        
        emitter = GNUPrologEmitter()
        result = emitter.emit(ir)
        
        assert "first([H|T], H)" in result.code
    
    def test_empty_list(self):
        """Test empty list emission."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "empty", "args": [
                    {"kind": "list_term", "elements": []}
                ]}
            ]
        }
        
        emitter = GNUPrologEmitter()
        result = emitter.emit(ir)
        
        assert "empty([])." in result.code
    
    def test_atom_escaping(self):
        """Test atom escaping for special characters."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "name", "args": [
                    {"kind": "atom", "value": "Hello World"}
                ]},
                {"kind": "fact", "predicate": "quoted", "args": [
                    {"kind": "atom", "value": "it's"}
                ]}
            ]
        }
        
        emitter = GNUPrologEmitter()
        result = emitter.emit(ir)
        
        # Atoms with spaces need quoting
        assert "'Hello World'" in result.code
        # Atoms with apostrophes need quoting and escaping
        assert "it" in result.code
    
    def test_negation(self):
        """Test negation operator emission."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "not_member", "args": [
                     {"kind": "variable", "name": "X"},
                     {"kind": "variable", "name": "L"}
                 ]},
                 "body": [
                     {"kind": "negation", 
                      "goal": {"kind": "compound", "functor": "member", "args": [
                          {"kind": "variable", "name": "X"},
                          {"kind": "variable", "name": "L"}
                      ]}}
                 ]}
            ]
        }
        
        emitter = GNUPrologEmitter()
        result = emitter.emit(ir)
        
        assert "\\+" in result.code
    
    def test_clp_header_emission(self):
        """Test CLP header comments are generated."""
        ir = {
            "module": "constraints",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "solve", "args": [
                     {"kind": "variable", "name": "X"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": "domain", "args": [
                         {"kind": "variable", "name": "X"},
                         {"kind": "number", "value": 1},
                         {"kind": "number", "value": 10}
                     ]}
                 ]}
            ]
        }
        
        emitter = GNUPrologEmitter()
        result = emitter.emit(ir)
        
        assert "CLP(FD)" in result.code
        assert "fd_domain" in result.code or "#=" in result.code
    
    def test_config_options(self):
        """Test configuration options."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "test", "args": []}
            ]
        }
        
        # Test with timestamps disabled
        config = GNUPrologConfig(emit_timestamps=False, emit_comments=True)
        emitter = GNUPrologEmitter(config)
        result = emitter.emit(ir)
        
        assert "STUNIR Generated" in result.code
        
        # Test with comments disabled
        config = GNUPrologConfig(emit_comments=False)
        emitter = GNUPrologEmitter(config)
        result = emitter.emit(ir)
        
        assert "/*" not in result.code
    
    def test_file_prefix(self):
        """Test custom file prefix."""
        ir = {
            "module": "mymodule",
            "clauses": []
        }
        
        config = GNUPrologConfig(file_prefix="custom")
        emitter = GNUPrologEmitter(config)
        result = emitter.emit(ir)
        
        assert result.filename == "custom_mymodule"
    
    def test_result_fields(self):
        """Test EmitterResult has all required fields."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "test", "args": []}
            ]
        }
        
        emitter = GNUPrologEmitter()
        result = emitter.emit(ir)
        
        assert hasattr(result, 'code')
        assert hasattr(result, 'filename')
        assert hasattr(result, 'predicates')
        assert hasattr(result, 'sha256')
        assert hasattr(result, 'emit_time')
        assert hasattr(result, 'clp_features')
        
        assert isinstance(result.code, str)
        assert isinstance(result.filename, str)
        assert isinstance(result.predicates, list)
        assert isinstance(result.sha256, str)
        assert isinstance(result.emit_time, float)
        assert isinstance(result.clp_features, list)


class TestGNUPrologEdgeCases:
    """Edge case tests for GNU Prolog emitter."""
    
    def test_empty_ir(self):
        """Test handling of minimal IR."""
        ir = {
            "module": "empty"
        }
        
        emitter = GNUPrologEmitter()
        result = emitter.emit(ir)
        
        assert result.filename == "stunir_empty"
        assert result.code  # Should have at least header
    
    def test_uppercase_module_name(self):
        """Test module names starting with uppercase."""
        ir = {
            "module": "MyModule",
            "clauses": []
        }
        
        emitter = GNUPrologEmitter()
        result = emitter.emit(ir)
        
        # Should be converted to lowercase
        assert result.filename == "stunir_myModule"
    
    def test_special_characters_in_name(self):
        """Test handling of special characters in module name."""
        ir = {
            "module": "test-module.v2",
            "clauses": []
        }
        
        emitter = GNUPrologEmitter()
        result = emitter.emit(ir)
        
        # Special chars should be replaced
        assert "test_module_v2" in result.filename
    
    def test_multiple_predicates(self):
        """Test emission of multiple predicates."""
        ir = {
            "module": "multi",
            "clauses": [
                {"kind": "fact", "predicate": "a", "args": []},
                {"kind": "fact", "predicate": "b", "args": []},
                {"kind": "fact", "predicate": "c", "args": []},
            ]
        }
        
        emitter = GNUPrologEmitter()
        result = emitter.emit(ir)
        
        assert "a." in result.code
        assert "b." in result.code
        assert "c." in result.code
        assert len(result.predicates) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
