#!/usr/bin/env python3
"""Tests for STUNIR SWI-Prolog Emitter.

Comprehensive test suite for SWI-Prolog code generation including
module declarations, facts, rules, DCG, and determinism.

Part of Phase 5C-1: Logic Programming Foundation.
"""

import pytest
import sys
import os
import re

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from targets.prolog.swi_prolog.emitter import (
    SWIPrologEmitter, SWIPrologConfig, EmitterResult
)
from targets.prolog.swi_prolog.types import (
    SWIPrologTypeMapper, SWI_PROLOG_TYPES
)


class TestSWIPrologEmitterBasic:
    """Basic emitter functionality tests."""
    
    def test_emitter_creation(self):
        """Test emitter can be created with default config."""
        emitter = SWIPrologEmitter()
        assert emitter.config.module_prefix == "stunir"
        assert emitter.config.emit_module is True
    
    def test_emitter_with_custom_config(self):
        """Test emitter with custom configuration."""
        config = SWIPrologConfig(
            module_prefix="myapp",
            emit_comments=False,
            emit_type_hints=False
        )
        emitter = SWIPrologEmitter(config)
        assert emitter.config.module_prefix == "myapp"
        assert emitter.config.emit_comments is False
    
    def test_emit_result_structure(self):
        """Test EmitterResult has correct structure."""
        ir = {"module": "test", "clauses": []}
        emitter = SWIPrologEmitter()
        result = emitter.emit(ir)
        
        assert isinstance(result, EmitterResult)
        assert isinstance(result.code, str)
        assert isinstance(result.module_name, str)
        assert isinstance(result.predicates, list)
        assert isinstance(result.sha256, str)
        assert isinstance(result.emit_time, float)


class TestSimpleFactEmission:
    """Tests for emitting simple facts."""
    
    def test_simple_fact_emission(self):
        """TC-SWI-001: Test emitting simple facts."""
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
        
        emitter = SWIPrologEmitter()
        result = emitter.emit(ir)
        
        assert "parent(tom, bob)." in result.code
        assert "parent(tom, liz)." in result.code
        assert result.module_name == "stunir_family"
    
    def test_fact_no_args(self):
        """Test fact with no arguments."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "sunny", "args": []}
            ]
        }
        
        emitter = SWIPrologEmitter()
        result = emitter.emit(ir)
        
        assert "sunny." in result.code
    
    def test_fact_with_numbers(self):
        """Test fact with numeric arguments."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "score", "args": [
                    {"kind": "atom", "value": "alice"},
                    95
                ]}
            ]
        }
        
        emitter = SWIPrologEmitter()
        result = emitter.emit(ir)
        
        assert "score(alice, 95)." in result.code


class TestRuleEmission:
    """Tests for emitting rules."""
    
    def test_rule_emission(self):
        """TC-SWI-002: Test emitting rules with body."""
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
        
        emitter = SWIPrologEmitter()
        result = emitter.emit(ir)
        
        assert "grandparent(X, Z) :-" in result.code
        assert "parent(X, Y)" in result.code
        assert "parent(Y, Z)" in result.code
    
    def test_rule_no_body(self):
        """Test rule with empty body (treated as fact)."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "true_pred", "args": []},
                 "body": []}
            ]
        }
        
        emitter = SWIPrologEmitter()
        result = emitter.emit(ir)
        
        assert "true_pred." in result.code


class TestModuleDeclaration:
    """Tests for module declarations."""
    
    def test_module_declaration(self):
        """TC-SWI-003: Test module declaration with exports."""
        ir = {
            "module": "lists",
            "exports": [{"predicate": "append", "arity": 3}],
            "clauses": [
                {"kind": "fact", "predicate": "append", "args": [
                    {"kind": "list_term", "elements": []},
                    {"kind": "variable", "name": "L"},
                    {"kind": "variable", "name": "L"}
                ]}
            ]
        }
        
        emitter = SWIPrologEmitter()
        result = emitter.emit(ir)
        
        assert ":- module(stunir_lists," in result.code
        assert "append/3" in result.code
    
    def test_module_no_exports(self):
        """Test module with no exports."""
        ir = {
            "module": "internal",
            "exports": [],
            "clauses": [
                {"kind": "fact", "predicate": "_helper", "args": []}
            ]
        }
        
        config = SWIPrologConfig(emit_module=True)
        # Override auto-export behavior by using private predicate
        emitter = SWIPrologEmitter(config)
        result = emitter.emit(ir)
        
        assert ":- module(stunir_internal," in result.code


class TestListEmission:
    """Tests for list term emission."""
    
    def test_list_emission(self):
        """TC-SWI-004: Test list term emission."""
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
        
        emitter = SWIPrologEmitter()
        result = emitter.emit(ir)
        
        assert "[H|T]" in result.code
    
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
        
        emitter = SWIPrologEmitter()
        result = emitter.emit(ir)
        
        assert "empty([])." in result.code
    
    def test_proper_list(self):
        """Test proper list emission."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "colors", "args": [
                    {"kind": "list_term", "elements": [
                        {"kind": "atom", "value": "red"},
                        {"kind": "atom", "value": "green"},
                        {"kind": "atom", "value": "blue"}
                    ]}
                ]}
            ]
        }
        
        emitter = SWIPrologEmitter()
        result = emitter.emit(ir)
        
        assert "colors([red, green, blue])." in result.code


class TestCutAndNegation:
    """Tests for cut and negation operators."""
    
    def test_cut_emission(self):
        """TC-SWI-005: Test cut and negation operators."""
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
        
        emitter = SWIPrologEmitter()
        result = emitter.emit(ir)
        
        assert "!" in result.code
        assert "max(X, Y, X) :-" in result.code
    
    def test_negation_emission(self):
        """Test negation as failure emission."""
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
        
        emitter = SWIPrologEmitter()
        result = emitter.emit(ir)
        
        assert "\\+" in result.code


class TestDCGEmission:
    """Tests for DCG rule emission."""
    
    def test_dcg_emission(self):
        """TC-SWI-006: Test DCG rule emission."""
        ir = {
            "module": "parser",
            "clauses": [],
            "dcg_rules": [
                {"head": {"kind": "compound", "functor": "sentence", "args": []},
                 "body": [
                     {"kind": "nonterminal", "term": {"kind": "compound", "functor": "noun_phrase", "args": []}},
                     {"kind": "nonterminal", "term": {"kind": "compound", "functor": "verb_phrase", "args": []}}
                 ]},
                {"head": {"kind": "compound", "functor": "noun_phrase", "args": []},
                 "body": [
                     {"kind": "terminal", "terminals": ["the"]},
                     {"kind": "nonterminal", "term": {"kind": "compound", "functor": "noun", "args": []}}
                 ]}
            ]
        }
        
        emitter = SWIPrologEmitter()
        result = emitter.emit(ir)
        
        assert "sentence -->" in result.code
        assert "noun_phrase" in result.code
        assert "[the]" in result.code
    
    def test_dcg_empty_body(self):
        """Test DCG rule with empty body."""
        ir = {
            "module": "test",
            "clauses": [],
            "dcg_rules": [
                {"head": {"kind": "compound", "functor": "epsilon", "args": []},
                 "body": []}
            ]
        }
        
        emitter = SWIPrologEmitter()
        result = emitter.emit(ir)
        
        assert "epsilon --> []." in result.code


class TestDynamicDeclaration:
    """Tests for dynamic predicate declarations."""
    
    def test_dynamic_declaration(self):
        """TC-SWI-007: Test dynamic predicate declaration."""
        ir = {
            "module": "db",
            "clauses": [],
            "dynamic": [
                {"predicate": "fact", "arity": 2}
            ]
        }
        
        emitter = SWIPrologEmitter()
        result = emitter.emit(ir)
        
        assert ":- dynamic fact/2." in result.code
    
    def test_multifile_declaration(self):
        """Test multifile predicate declaration."""
        ir = {
            "module": "test",
            "clauses": [],
            "multifile": [
                {"predicate": "handler", "arity": 2}
            ]
        }
        
        emitter = SWIPrologEmitter()
        result = emitter.emit(ir)
        
        assert ":- multifile handler/2." in result.code


class TestDeterminism:
    """Tests for deterministic output."""
    
    def test_determinism(self):
        """TC-SWI-008: Test that output is deterministic."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "a", "args": []},
                {"kind": "fact", "predicate": "b", "args": []},
            ]
        }
        
        config = SWIPrologConfig(emit_timestamps=False, emit_comments=False)
        emitter = SWIPrologEmitter(config)
        
        # Generate multiple times
        results = [emitter.emit(ir).code for _ in range(5)]
        
        # All should be identical
        assert all(r == results[0] for r in results)
    
    def test_sha256_consistency(self):
        """Test SHA256 hash is consistent."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "test", "args": []}
            ]
        }
        
        config = SWIPrologConfig(emit_timestamps=False, emit_comments=False)
        emitter = SWIPrologEmitter(config)
        
        hashes = [emitter.emit(ir).sha256 for _ in range(3)]
        assert all(h == hashes[0] for h in hashes)


class TestAtomEscaping:
    """Tests for atom escaping."""
    
    def test_lowercase_atom_no_escape(self):
        """Test lowercase atoms don't need escaping."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "atom_test", "args": [
                    {"kind": "atom", "value": "hello_world"}
                ]}
            ]
        }
        
        emitter = SWIPrologEmitter()
        result = emitter.emit(ir)
        
        assert "atom_test(hello_world)." in result.code
    
    def test_uppercase_atom_escaped(self):
        """Test atoms starting with uppercase need escaping."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "name", "args": [
                    {"kind": "atom", "value": "John"}
                ]}
            ]
        }
        
        emitter = SWIPrologEmitter()
        result = emitter.emit(ir)
        
        assert "'John'" in result.code
    
    def test_special_char_atom_escaped(self):
        """Test atoms with special characters need escaping."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "path", "args": [
                    {"kind": "atom", "value": "/usr/local/bin"}
                ]}
            ]
        }
        
        emitter = SWIPrologEmitter()
        result = emitter.emit(ir)
        
        assert "'/usr/local/bin'" in result.code


class TestQueryEmission:
    """Tests for query emission."""
    
    def test_query_emission(self):
        """Test query emission as initialization."""
        ir = {
            "module": "test",
            "clauses": [],
            "queries": [
                {"kind": "query", "goals": [
                    {"kind": "compound", "functor": "hello", "args": [
                        {"kind": "atom", "value": "world"}
                    ]}
                ]}
            ]
        }
        
        emitter = SWIPrologEmitter()
        result = emitter.emit(ir)
        
        assert ":- initialization(hello(world))." in result.code
        assert "% ?- hello(world)." in result.code


class TestTypeMapper:
    """Tests for type mapper."""
    
    def test_type_mapping(self):
        """Test basic type mapping."""
        mapper = SWIPrologTypeMapper()
        
        assert mapper.map_type('i32') == 'integer'
        assert mapper.map_type('f64') == 'float'
        assert mapper.map_type('string') == 'atom'
        assert mapper.map_type('bool') == 'boolean'
    
    def test_mode_inference(self):
        """Test mode inference."""
        mapper = SWIPrologTypeMapper()
        
        param_in = {'mode': 'input'}
        param_out = {'mode': 'output'}
        param_bi = {'mode': 'bidirectional'}
        
        assert mapper.infer_mode(param_in) == '+'
        assert mapper.infer_mode(param_out) == '-'
        assert mapper.infer_mode(param_bi) == '?'
    
    def test_determinism_format(self):
        """Test determinism declaration formatting."""
        mapper = SWIPrologTypeMapper()
        
        assert mapper.format_determinism(True, False) == 'det'
        assert mapper.format_determinism(True, True) == 'semidet'
        assert mapper.format_determinism(False, False) == 'multi'
        assert mapper.format_determinism(False, True) == 'nondet'


class TestHeaderGeneration:
    """Tests for header generation."""
    
    def test_header_with_comments(self):
        """Test header generation with comments enabled."""
        ir = {"module": "test", "clauses": []}
        
        config = SWIPrologConfig(emit_comments=True)
        emitter = SWIPrologEmitter(config)
        result = emitter.emit(ir)
        
        assert "/*" in result.code
        assert "STUNIR Generated" in result.code
        assert "*/" in result.code
    
    def test_header_without_comments(self):
        """Test header generation with comments disabled."""
        ir = {"module": "test", "clauses": []}
        
        config = SWIPrologConfig(emit_comments=False)
        emitter = SWIPrologEmitter(config)
        result = emitter.emit(ir)
        
        assert "/*" not in result.code


class TestCompoundTermEmission:
    """Tests for compound term emission."""
    
    def test_nested_compound(self):
        """Test nested compound term emission."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "expr", "args": [
                    {"kind": "compound", "functor": "add", "args": [
                        {"kind": "compound", "functor": "mul", "args": [
                            {"kind": "variable", "name": "X"},
                            {"kind": "variable", "name": "Y"}
                        ]},
                        {"kind": "variable", "name": "Z"}
                    ]}
                ]}
            ]
        }
        
        emitter = SWIPrologEmitter()
        result = emitter.emit(ir)
        
        assert "expr(add(mul(X, Y), Z))." in result.code


class TestIntegration:
    """Integration tests with complete programs."""
    
    def test_family_program(self):
        """Test complete family relationships program."""
        ir = {
            "module": "family",
            "exports": [
                {"predicate": "grandparent", "arity": 2},
                {"predicate": "sibling", "arity": 2}
            ],
            "clauses": [
                {"kind": "fact", "predicate": "parent", "args": [
                    {"kind": "atom", "value": "tom"},
                    {"kind": "atom", "value": "bob"}
                ]},
                {"kind": "fact", "predicate": "parent", "args": [
                    {"kind": "atom", "value": "tom"},
                    {"kind": "atom", "value": "liz"}
                ]},
                {"kind": "fact", "predicate": "parent", "args": [
                    {"kind": "atom", "value": "bob"},
                    {"kind": "atom", "value": "ann"}
                ]},
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
                 ]},
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "sibling", "args": [
                     {"kind": "variable", "name": "X"},
                     {"kind": "variable", "name": "Y"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": "parent", "args": [
                         {"kind": "variable", "name": "P"},
                         {"kind": "variable", "name": "X"}
                     ]},
                     {"kind": "compound", "functor": "parent", "args": [
                         {"kind": "variable", "name": "P"},
                         {"kind": "variable", "name": "Y"}
                     ]},
                     {"kind": "compound", "functor": "\\=", "args": [
                         {"kind": "variable", "name": "X"},
                         {"kind": "variable", "name": "Y"}
                     ]}
                 ]}
            ]
        }
        
        emitter = SWIPrologEmitter()
        result = emitter.emit(ir)
        
        # Verify structure
        assert "module(stunir_family" in result.code
        assert "parent(tom, bob)." in result.code
        assert "grandparent(X, Z) :-" in result.code
        assert "sibling(X, Y) :-" in result.code
        
        # Verify predicates list
        assert "parent/2" in result.predicates
        assert "grandparent/2" in result.predicates
        assert "sibling/2" in result.predicates


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
