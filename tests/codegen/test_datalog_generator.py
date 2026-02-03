#!/usr/bin/env python3
"""Tests for STUNIR Datalog Emitter.

Tests cover:
- Basic fact emission
- Rule emission with Datalog restrictions
- Stratified negation support
- Head restriction validation
- Range restriction validation
- Unstratifiable program detection
- Multiple strata computation

Part of Phase 5C-4: Datalog Emitter.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from targets.prolog.datalog import (
    DatalogEmitter,
    DatalogConfig,
    DatalogTypeMapper,
    ValidationLevel,
    DatalogRestrictionError,
    StratificationError,
    DATALOG_TYPES,
    escape_atom,
    escape_string,
    format_variable,
)


class TestDatalogTypes:
    """Tests for Datalog type mapping and utilities."""
    
    def test_type_mapping_numeric(self):
        """Test numeric type mappings."""
        mapper = DatalogTypeMapper()
        assert mapper.map_type('i32') == 'integer'
        assert mapper.map_type('i64') == 'integer'
        assert mapper.map_type('f32') == 'float'
        assert mapper.map_type('f64') == 'float'
    
    def test_type_mapping_string(self):
        """Test string type mappings."""
        mapper = DatalogTypeMapper()
        assert mapper.map_type('string') == 'symbol'
        assert mapper.map_type('str') == 'symbol'
        assert mapper.map_type('char') == 'symbol'
    
    def test_escape_atom_simple(self):
        """Test simple atom escaping."""
        assert escape_atom('hello') == 'hello'
        assert escape_atom('world123') == 'world123'
    
    def test_escape_atom_uppercase(self):
        """Test uppercase atoms need quoting."""
        assert escape_atom('Hello') == "'Hello'"
        assert escape_atom('X') == "'X'"
    
    def test_escape_atom_reserved(self):
        """Test reserved words need quoting."""
        assert escape_atom('not') == "'not'"
        assert escape_atom('true') == "'true'"
    
    def test_escape_atom_special_chars(self):
        """Test special characters need quoting."""
        assert escape_atom('hello world') == "'hello world'"
        assert escape_atom('foo-bar') == "'foo-bar'"
    
    def test_escape_string(self):
        """Test string escaping."""
        assert escape_string('hello') == 'hello'
        assert escape_string('hello\nworld') == 'hello\\nworld'
        assert escape_string('say "hi"') == 'say \\"hi\\"'
    
    def test_format_variable(self):
        """Test variable formatting."""
        assert format_variable('X') == 'X'
        assert format_variable('Var') == 'Var'
        assert format_variable('x') == 'X'  # Lowercase -> uppercase
        assert format_variable('_') == '_'
    
    def test_type_mapper_is_numeric(self):
        """Test is_numeric check."""
        mapper = DatalogTypeMapper()
        assert mapper.is_numeric('i32')
        assert mapper.is_numeric('f64')
        assert not mapper.is_numeric('string')
    
    def test_type_mapper_validate_term_for_head(self):
        """Test head term validation."""
        mapper = DatalogTypeMapper()
        assert mapper.validate_term_for_head('variable')
        assert mapper.validate_term_for_head('atom')
        assert mapper.validate_term_for_head('number')
        assert not mapper.validate_term_for_head('compound')
        assert not mapper.validate_term_for_head('list_term')


class TestDatalogEmitterBasic:
    """Tests for basic Datalog emission."""
    
    def test_emit_simple_fact(self):
        """Test basic fact emission."""
        ir = {
            'module': 'test',
            'predicates': [{
                'name': 'parent',
                'clauses': [
                    {'kind': 'fact', 'predicate': 'parent',
                     'args': [{'kind': 'atom', 'value': 'tom'},
                             {'kind': 'atom', 'value': 'bob'}]}
                ]
            }]
        }
        
        emitter = DatalogEmitter()
        result = emitter.emit(ir)
        
        assert 'parent(tom, bob).' in result.code
        assert result.facts_count == 1
        assert result.validation.is_valid
    
    def test_emit_multiple_facts(self):
        """Test multiple fact emission."""
        ir = {
            'module': 'family',
            'predicates': [{
                'name': 'parent',
                'clauses': [
                    {'kind': 'fact', 'predicate': 'parent',
                     'args': [{'kind': 'atom', 'value': 'tom'},
                             {'kind': 'atom', 'value': 'bob'}]},
                    {'kind': 'fact', 'predicate': 'parent',
                     'args': [{'kind': 'atom', 'value': 'tom'},
                             {'kind': 'atom', 'value': 'liz'}]},
                    {'kind': 'fact', 'predicate': 'parent',
                     'args': [{'kind': 'atom', 'value': 'bob'},
                             {'kind': 'atom', 'value': 'ann'}]}
                ]
            }]
        }
        
        emitter = DatalogEmitter()
        result = emitter.emit(ir)
        
        assert 'parent(tom, bob).' in result.code
        assert 'parent(tom, liz).' in result.code
        assert 'parent(bob, ann).' in result.code
        assert result.facts_count == 3
    
    def test_emit_numeric_fact(self):
        """Test fact with numeric arguments."""
        ir = {
            'module': 'numbers',
            'predicates': [{
                'name': 'age',
                'clauses': [
                    {'kind': 'fact', 'predicate': 'age',
                     'args': [{'kind': 'atom', 'value': 'tom'},
                             {'kind': 'number', 'value': 50}]}
                ]
            }]
        }
        
        emitter = DatalogEmitter()
        result = emitter.emit(ir)
        
        assert 'age(tom, 50).' in result.code
    
    def test_emit_zero_arity_fact(self):
        """Test zero-arity fact."""
        ir = {
            'module': 'test',
            'predicates': [{
                'name': 'done',
                'clauses': [
                    {'kind': 'fact', 'predicate': 'done', 'args': []}
                ]
            }]
        }
        
        emitter = DatalogEmitter()
        result = emitter.emit(ir)
        
        assert 'done.' in result.code


class TestDatalogEmitterRules:
    """Tests for Datalog rule emission."""
    
    def test_emit_simple_rule(self):
        """Test simple rule emission."""
        ir = {
            'module': 'family',
            'predicates': [{
                'name': 'ancestor',
                'clauses': [
                    {'kind': 'rule',
                     'head': {'kind': 'compound', 'functor': 'ancestor',
                             'args': [{'kind': 'variable', 'name': 'X'},
                                     {'kind': 'variable', 'name': 'Y'}]},
                     'body': [
                         {'kind': 'call',
                          'term': {'kind': 'compound', 'functor': 'parent',
                                  'args': [{'kind': 'variable', 'name': 'X'},
                                          {'kind': 'variable', 'name': 'Y'}]}}
                     ]}
                ]
            }]
        }
        
        emitter = DatalogEmitter()
        result = emitter.emit(ir)
        
        assert 'ancestor(X, Y) :- parent(X, Y).' in result.code
        assert result.rules_count == 1
    
    def test_emit_rule_multiple_body(self):
        """Test rule with multiple body goals."""
        ir = {
            'module': 'family',
            'predicates': [{
                'name': 'grandparent',
                'clauses': [
                    {'kind': 'rule',
                     'head': {'kind': 'compound', 'functor': 'grandparent',
                             'args': [{'kind': 'variable', 'name': 'X'},
                                     {'kind': 'variable', 'name': 'Z'}]},
                     'body': [
                         {'kind': 'call',
                          'term': {'kind': 'compound', 'functor': 'parent',
                                  'args': [{'kind': 'variable', 'name': 'X'},
                                          {'kind': 'variable', 'name': 'Y'}]}},
                         {'kind': 'call',
                          'term': {'kind': 'compound', 'functor': 'parent',
                                  'args': [{'kind': 'variable', 'name': 'Y'},
                                          {'kind': 'variable', 'name': 'Z'}]}}
                     ]}
                ]
            }]
        }
        
        emitter = DatalogEmitter()
        result = emitter.emit(ir)
        
        assert 'grandparent(X, Z) :- parent(X, Y), parent(Y, Z).' in result.code


class TestDatalogStratification:
    """Tests for stratified negation support."""
    
    def test_stratified_negation(self):
        """Test stratified negation support."""
        ir = {
            'module': 'birds',
            'predicates': [
                {'name': 'bird', 'clauses': [
                    {'kind': 'fact', 'predicate': 'bird',
                     'args': [{'kind': 'atom', 'value': 'tweety'}]}
                ]},
                {'name': 'penguin', 'clauses': [
                    {'kind': 'fact', 'predicate': 'penguin',
                     'args': [{'kind': 'atom', 'value': 'tux'}]}
                ]},
                {'name': 'flies', 'clauses': [
                    {'kind': 'rule',
                     'head': {'kind': 'compound', 'functor': 'flies',
                             'args': [{'kind': 'variable', 'name': 'X'}]},
                     'body': [
                         {'kind': 'call',
                          'term': {'kind': 'compound', 'functor': 'bird',
                                  'args': [{'kind': 'variable', 'name': 'X'}]}},
                         {'kind': 'negation',
                          'inner': {'kind': 'call',
                                   'term': {'kind': 'compound', 'functor': 'penguin',
                                           'args': [{'kind': 'variable', 'name': 'X'}]}}}
                     ]}
                ]}
            ]
        }
        
        emitter = DatalogEmitter()
        result = emitter.emit(ir)
        
        assert 'flies(X) :- bird(X), not penguin(X).' in result.code
        assert result.stratification is not None
        assert result.stratification.is_stratifiable
    
    def test_unstratifiable_negation(self):
        """Test that negative cycles are detected."""
        ir = {
            'module': 'unstratifiable',
            'predicates': [
                {'name': 'p', 'clauses': [
                    {'kind': 'rule',
                     'head': {'kind': 'compound', 'functor': 'p',
                             'args': [{'kind': 'variable', 'name': 'X'}]},
                     'body': [
                         {'kind': 'negation',
                          'inner': {'kind': 'call',
                                   'term': {'kind': 'compound', 'functor': 'q',
                                           'args': [{'kind': 'variable', 'name': 'X'}]}}}
                     ]}
                ]},
                {'name': 'q', 'clauses': [
                    {'kind': 'rule',
                     'head': {'kind': 'compound', 'functor': 'q',
                             'args': [{'kind': 'variable', 'name': 'X'}]},
                     'body': [
                         {'kind': 'negation',
                          'inner': {'kind': 'call',
                                   'term': {'kind': 'compound', 'functor': 'p',
                                           'args': [{'kind': 'variable', 'name': 'X'}]}}}
                     ]}
                ]}
            ]
        }
        
        emitter = DatalogEmitter()
        result = emitter.stratify(ir)
        
        assert not result.is_stratifiable
        assert len(result.errors) > 0
        assert any('negative cycle' in e.lower() for e in result.errors)
    
    def test_multiple_strata(self):
        """Test correct stratum assignment for multiple levels."""
        ir = {
            'module': 'strata',
            'predicates': [
                # Stratum 0: base facts
                {'name': 'base', 'clauses': [
                    {'kind': 'fact', 'predicate': 'base',
                     'args': [{'kind': 'number', 'value': 1}]}
                ]},
                {'name': 'excluded', 'clauses': [
                    {'kind': 'fact', 'predicate': 'excluded',
                     'args': [{'kind': 'number', 'value': 2}]}
                ]},
                # Stratum 1: depends on base, negates excluded
                {'name': 'level1', 'clauses': [
                    {'kind': 'rule',
                     'head': {'kind': 'compound', 'functor': 'level1',
                             'args': [{'kind': 'variable', 'name': 'X'}]},
                     'body': [
                         {'kind': 'call',
                          'term': {'kind': 'compound', 'functor': 'base',
                                  'args': [{'kind': 'variable', 'name': 'X'}]}},
                         {'kind': 'negation',
                          'inner': {'kind': 'call',
                                   'term': {'kind': 'compound', 'functor': 'excluded',
                                           'args': [{'kind': 'variable', 'name': 'X'}]}}}
                     ]}
                ]}
            ]
        }
        
        emitter = DatalogEmitter()
        result = emitter.stratify(ir)
        
        assert result.is_stratifiable
        assert result.strata['base'] == 0
        assert result.strata['excluded'] == 0
        assert result.strata['level1'] > result.strata['base']


class TestDatalogRestrictions:
    """Tests for Datalog restriction validation."""
    
    def test_head_restriction_violation(self):
        """Test that function symbols in heads are rejected."""
        ir = {
            'module': 'nat',
            'predicates': [{
                'name': 'nat',
                'clauses': [
                    {'kind': 'rule',
                     'head': {'kind': 'compound', 'functor': 'nat',
                             'args': [{'kind': 'compound', 'functor': 'succ',
                                      'args': [{'kind': 'variable', 'name': 'X'}]}]},
                     'body': [
                         {'kind': 'call',
                          'term': {'kind': 'compound', 'functor': 'nat',
                                  'args': [{'kind': 'variable', 'name': 'X'}]}}
                     ]}
                ]
            }]
        }
        
        config = DatalogConfig(validation_level=ValidationLevel.STRICT)
        emitter = DatalogEmitter(config)
        
        with pytest.raises(DatalogRestrictionError) as excinfo:
            emitter.emit(ir)
        
        assert "function symbol" in str(excinfo.value).lower()
    
    def test_range_restriction_violation(self):
        """Test that unsafe variables are rejected."""
        ir = {
            'module': 'unsafe',
            'predicates': [{
                'name': 'unsafe',
                'clauses': [
                    {'kind': 'rule',
                     'head': {'kind': 'compound', 'functor': 'unsafe',
                             'args': [{'kind': 'variable', 'name': 'X'}]},
                     'body': [
                         {'kind': 'negation',
                          'inner': {'kind': 'call',
                                   'term': {'kind': 'compound', 'functor': 'known',
                                           'args': [{'kind': 'variable', 'name': 'X'}]}}}
                     ]}
                ]
            }]
        }
        
        config = DatalogConfig(validation_level=ValidationLevel.STRICT)
        emitter = DatalogEmitter(config)
        
        with pytest.raises(DatalogRestrictionError) as excinfo:
            emitter.emit(ir)
        
        assert "range restriction" in str(excinfo.value).lower()
    
    def test_range_restriction_valid(self):
        """Test that safe rules pass validation."""
        ir = {
            'module': 'safe',
            'predicates': [{
                'name': 'safe',
                'clauses': [
                    {'kind': 'rule',
                     'head': {'kind': 'compound', 'functor': 'safe',
                             'args': [{'kind': 'variable', 'name': 'X'}]},
                     'body': [
                         {'kind': 'call',
                          'term': {'kind': 'compound', 'functor': 'known',
                                  'args': [{'kind': 'variable', 'name': 'X'}]}},
                         {'kind': 'negation',
                          'inner': {'kind': 'call',
                                   'term': {'kind': 'compound', 'functor': 'excluded',
                                           'args': [{'kind': 'variable', 'name': 'X'}]}}}
                     ]}
                ]
            }]
        }
        
        config = DatalogConfig(validation_level=ValidationLevel.STRICT)
        emitter = DatalogEmitter(config)
        result = emitter.emit(ir)
        
        assert result.validation.is_valid
        assert 'safe(X) :- known(X), not excluded(X).' in result.code
    
    def test_lenient_validation(self):
        """Test lenient mode allows violations with warnings."""
        ir = {
            'module': 'nat',
            'predicates': [{
                'name': 'nat',
                'clauses': [
                    {'kind': 'rule',
                     'head': {'kind': 'compound', 'functor': 'nat',
                             'args': [{'kind': 'compound', 'functor': 'succ',
                                      'args': [{'kind': 'variable', 'name': 'X'}]}]},
                     'body': [
                         {'kind': 'call',
                          'term': {'kind': 'compound', 'functor': 'nat',
                                  'args': [{'kind': 'variable', 'name': 'X'}]}}
                     ]}
                ]
            }]
        }
        
        config = DatalogConfig(validation_level=ValidationLevel.LENIENT)
        emitter = DatalogEmitter(config)
        
        # Should not raise
        result = emitter.emit(ir)
        assert not result.validation.is_valid  # Still records errors
        assert len(result.validation.errors) > 0


class TestDatalogEmitterConfig:
    """Tests for emitter configuration."""
    
    def test_emit_without_comments(self):
        """Test emission without comments."""
        ir = {
            'module': 'test',
            'predicates': [{
                'name': 'fact',
                'clauses': [
                    {'kind': 'fact', 'predicate': 'fact', 'args': []}
                ]
            }]
        }
        
        config = DatalogConfig(emit_comments=False)
        emitter = DatalogEmitter(config)
        result = emitter.emit(ir)
        
        assert '%' not in result.code
    
    def test_emitter_result_to_dict(self):
        """Test EmitterResult serialization."""
        ir = {
            'module': 'test',
            'predicates': [{
                'name': 'test',
                'clauses': [
                    {'kind': 'fact', 'predicate': 'test', 'args': []}
                ]
            }]
        }
        
        emitter = DatalogEmitter()
        result = emitter.emit(ir)
        
        d = result.to_dict()
        assert 'code' in d
        assert 'predicates' in d
        assert 'sha256' in d
        assert 'validation' in d
    
    def test_sha256_computed(self):
        """Test SHA-256 hash is computed."""
        ir = {
            'module': 'test',
            'predicates': [{
                'name': 'test',
                'clauses': [
                    {'kind': 'fact', 'predicate': 'test', 'args': []}
                ]
            }]
        }
        
        emitter = DatalogEmitter()
        result = emitter.emit(ir)
        
        assert result.sha256
        assert len(result.sha256) == 64  # SHA-256 hex string


class TestDatalogQueries:
    """Tests for query emission."""
    
    def test_emit_query(self):
        """Test query emission."""
        from tools.ir.logic_ir import Query, Goal, Compound, Variable, GoalKind
        
        ir = {
            'module': 'test',
            'predicates': []
        }
        
        # Create a query object
        query = Query(goals=[
            Goal(
                kind=GoalKind.CALL,
                term=Compound('parent', [Variable('X'), Atom('bob')])
            )
        ])
        
        emitter = DatalogEmitter()
        # Test internal query emission
        result_str = emitter._emit_queries([query])
        
        assert '?- parent(X, bob).' in result_str


class TestDatalogDependencyGraph:
    """Tests for dependency graph construction."""
    
    def test_build_dependency_graph(self):
        """Test dependency graph construction."""
        ir = {
            'module': 'deps',
            'predicates': [
                {'name': 'a', 'clauses': [
                    {'kind': 'rule',
                     'head': {'kind': 'compound', 'functor': 'a',
                             'args': [{'kind': 'variable', 'name': 'X'}]},
                     'body': [
                         {'kind': 'call',
                          'term': {'kind': 'compound', 'functor': 'b',
                                  'args': [{'kind': 'variable', 'name': 'X'}]}}
                     ]}
                ]},
                {'name': 'b', 'clauses': [
                    {'kind': 'rule',
                     'head': {'kind': 'compound', 'functor': 'b',
                             'args': [{'kind': 'variable', 'name': 'X'}]},
                     'body': [
                         {'kind': 'call',
                          'term': {'kind': 'compound', 'functor': 'c',
                                  'args': [{'kind': 'variable', 'name': 'X'}]}}
                     ]}
                ]}
            ]
        }
        
        emitter = DatalogEmitter()
        deps = emitter._build_dependency_graph(ir)
        
        assert 'a' in deps
        assert 'b' in deps
        assert ('b', False) in deps['a']  # a depends on b, not negated
        assert ('c', False) in deps['b']  # b depends on c, not negated


class TestDatalogIntegration:
    """Integration tests."""
    
    def test_complete_family_program(self):
        """Test complete family relationship program."""
        ir = {
            'module': 'family',
            'predicates': [
                # Facts
                {'name': 'parent', 'clauses': [
                    {'kind': 'fact', 'predicate': 'parent',
                     'args': [{'kind': 'atom', 'value': 'tom'},
                             {'kind': 'atom', 'value': 'bob'}]},
                    {'kind': 'fact', 'predicate': 'parent',
                     'args': [{'kind': 'atom', 'value': 'tom'},
                             {'kind': 'atom', 'value': 'liz'}]},
                    {'kind': 'fact', 'predicate': 'parent',
                     'args': [{'kind': 'atom', 'value': 'bob'},
                             {'kind': 'atom', 'value': 'ann'}]},
                    {'kind': 'fact', 'predicate': 'parent',
                     'args': [{'kind': 'atom', 'value': 'bob'},
                             {'kind': 'atom', 'value': 'pat'}]}
                ]},
                # Rules
                {'name': 'ancestor', 'clauses': [
                    # Base case
                    {'kind': 'rule',
                     'head': {'kind': 'compound', 'functor': 'ancestor',
                             'args': [{'kind': 'variable', 'name': 'X'},
                                     {'kind': 'variable', 'name': 'Y'}]},
                     'body': [
                         {'kind': 'call',
                          'term': {'kind': 'compound', 'functor': 'parent',
                                  'args': [{'kind': 'variable', 'name': 'X'},
                                          {'kind': 'variable', 'name': 'Y'}]}}
                     ]},
                    # Recursive case
                    {'kind': 'rule',
                     'head': {'kind': 'compound', 'functor': 'ancestor',
                             'args': [{'kind': 'variable', 'name': 'X'},
                                     {'kind': 'variable', 'name': 'Z'}]},
                     'body': [
                         {'kind': 'call',
                          'term': {'kind': 'compound', 'functor': 'parent',
                                  'args': [{'kind': 'variable', 'name': 'X'},
                                          {'kind': 'variable', 'name': 'Y'}]}},
                         {'kind': 'call',
                          'term': {'kind': 'compound', 'functor': 'ancestor',
                                  'args': [{'kind': 'variable', 'name': 'Y'},
                                          {'kind': 'variable', 'name': 'Z'}]}}
                     ]}
                ]}
            ]
        }
        
        emitter = DatalogEmitter()
        result = emitter.emit(ir)
        
        # Check facts
        assert 'parent(tom, bob).' in result.code
        assert 'parent(bob, ann).' in result.code
        
        # Check rules
        assert 'ancestor(X, Y) :- parent(X, Y).' in result.code
        assert 'ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).' in result.code
        
        # Check counts
        assert result.facts_count == 4
        assert result.rules_count == 2
        
        # Check validation
        assert result.validation.is_valid


# Import Atom for tests
from tools.ir.logic_ir import Atom


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
