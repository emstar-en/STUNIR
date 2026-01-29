#!/usr/bin/env python3
"""Tests for STUNIR XSB Prolog Emitter.

Comprehensive test suite for XSB Prolog code generation including
export/import directives, facts, rules, advanced tabling (incremental,
subsumptive), well-founded semantics, lattice tabling, and DCG.

Part of Phase 5D-1: XSB with Advanced Tabling.
"""

import pytest
import sys
import os
import re

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from targets.prolog.xsb.emitter import (
    XSBPrologEmitter, XSBPrologConfig, EmitterResult, TablingSpec
)
from targets.prolog.xsb.types import (
    XSBPrologTypeMapper, XSB_PROLOG_TYPES, XSB_TABLING_MODES,
    XSB_LATTICE_OPS, TablingMode
)


class TestXSBPrologEmitterBasic:
    """Basic emitter functionality tests."""
    
    def test_emitter_creation(self):
        """Test emitter can be created with default config."""
        emitter = XSBPrologEmitter()
        assert emitter.config.module_prefix == "stunir"
        assert emitter.config.emit_exports is True
        assert emitter.config.enable_tabling is True
        assert emitter.config.file_extension == ".P"
    
    def test_emitter_with_custom_config(self):
        """Test emitter with custom configuration."""
        config = XSBPrologConfig(
            module_prefix="myapp",
            emit_comments=False,
            enable_tabling=False,
            auto_incremental=False
        )
        emitter = XSBPrologEmitter(config)
        assert emitter.config.module_prefix == "myapp"
        assert emitter.config.emit_comments is False
        assert emitter.config.enable_tabling is False
        assert emitter.config.auto_incremental is False
    
    def test_emit_result_structure(self):
        """Test EmitterResult has correct structure."""
        ir = {"module": "test", "clauses": []}
        emitter = XSBPrologEmitter()
        result = emitter.emit(ir)
        
        assert isinstance(result, EmitterResult)
        assert isinstance(result.code, str)
        assert isinstance(result.module_name, str)
        assert isinstance(result.predicates, list)
        assert isinstance(result.sha256, str)
        assert isinstance(result.emit_time, float)
        assert isinstance(result.tabled_predicates, list)
        assert isinstance(result.incremental_predicates, list)
        assert isinstance(result.subsumptive_predicates, list)
        assert isinstance(result.wfs_predicates, list)


class TestSimpleFactEmission:
    """Tests for emitting simple facts."""
    
    def test_simple_fact_emission(self):
        """TC-XSB-001: Test emitting simple facts."""
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
        
        emitter = XSBPrologEmitter()
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
        
        emitter = XSBPrologEmitter()
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
        
        emitter = XSBPrologEmitter()
        result = emitter.emit(ir)
        
        assert "score(alice, 95)." in result.code


class TestRuleEmission:
    """Tests for emitting rules."""
    
    def test_rule_emission(self):
        """TC-XSB-002: Test emitting rules with body."""
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
        
        emitter = XSBPrologEmitter()
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
        
        emitter = XSBPrologEmitter()
        result = emitter.emit(ir)
        
        assert "true_pred." in result.code


class TestExportDirectives:
    """Tests for XSB-specific export directives."""
    
    def test_export_directives(self):
        """TC-XSB-003: Test export directives."""
        ir = {
            "module": "lists",
            "exports": [
                {"predicate": "append", "arity": 3},
                {"predicate": "member", "arity": 2}
            ],
            "clauses": [
                {"kind": "fact", "predicate": "append", "args": [
                    {"kind": "list_term", "elements": []},
                    {"kind": "variable", "name": "L"},
                    {"kind": "variable", "name": "L"}
                ]}
            ]
        }
        
        emitter = XSBPrologEmitter()
        result = emitter.emit(ir)
        
        assert ":- export(append/3)." in result.code
        assert ":- export(member/2)." in result.code
    
    def test_auto_export_public(self):
        """Test auto-export of public predicates."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "public_pred", "args": []},
                {"kind": "fact", "predicate": "_private_pred", "args": []}
            ]
        }
        
        emitter = XSBPrologEmitter()
        result = emitter.emit(ir)
        
        # Public predicate should be exported
        assert ":- export(public_pred/0)." in result.code
        # Private predicate (starting with _) should not be exported
        assert ":- export(_private_pred" not in result.code


class TestImportDirectives:
    """Tests for XSB-specific import directives."""
    
    def test_import_directives(self):
        """TC-XSB-004: Test import directives."""
        ir = {
            "module": "mymodule",
            "imports": [
                {"predicate": "member", "arity": 2, "from": "basics"},
                {"predicate": "append", "arity": 3, "from": "basics"}
            ],
            "clauses": []
        }
        
        emitter = XSBPrologEmitter()
        result = emitter.emit(ir)
        
        assert ":- import(member/2 from basics)." in result.code
        assert ":- import(append/3 from basics)." in result.code


class TestBasicTabling:
    """Tests for basic tabling."""
    
    def test_basic_tabling(self):
        """TC-XSB-005: Test basic tabling for recursive predicate."""
        ir = {
            "module": "fibonacci",
            "clauses": [
                {"kind": "fact", "predicate": "fib", "args": [0, 0]},
                {"kind": "fact", "predicate": "fib", "args": [1, 1]},
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "fib", "args": [
                     {"kind": "variable", "name": "N"},
                     {"kind": "variable", "name": "F"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": ">", "args": [
                         {"kind": "variable", "name": "N"}, 1
                     ]},
                     {"kind": "compound", "functor": "is", "args": [
                         {"kind": "variable", "name": "N1"},
                         {"kind": "compound", "functor": "-", "args": [
                             {"kind": "variable", "name": "N"}, 1
                         ]}
                     ]},
                     {"kind": "compound", "functor": "fib", "args": [
                         {"kind": "variable", "name": "N1"},
                         {"kind": "variable", "name": "F1"}
                     ]}
                 ]}
            ]
        }
        
        emitter = XSBPrologEmitter()
        result = emitter.emit(ir)
        
        # Recursive predicate should be auto-tabled
        assert ":- table fib/2" in result.code
        assert "fib/2" in result.tabled_predicates


class TestIncrementalTabling:
    """Tests for incremental tabling."""
    
    def test_incremental_tabling(self):
        """TC-XSB-006: Test incremental tabling."""
        ir = {
            "module": "graph",
            "dynamic": [{"predicate": "edge", "arity": 2}],
            "tabled": [{"predicate": "reachable", "arity": 2, "mode": "incremental"}],
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "reachable", "args": [
                     {"kind": "variable", "name": "X"},
                     {"kind": "variable", "name": "Y"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": "edge", "args": [
                         {"kind": "variable", "name": "X"},
                         {"kind": "variable", "name": "Y"}
                     ]}
                 ]},
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "reachable", "args": [
                     {"kind": "variable", "name": "X"},
                     {"kind": "variable", "name": "Z"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": "edge", "args": [
                         {"kind": "variable", "name": "X"},
                         {"kind": "variable", "name": "Y"}
                     ]},
                     {"kind": "compound", "functor": "reachable", "args": [
                         {"kind": "variable", "name": "Y"},
                         {"kind": "variable", "name": "Z"}
                     ]}
                 ]}
            ]
        }
        
        emitter = XSBPrologEmitter()
        result = emitter.emit(ir)
        
        assert ":- table reachable/2 as incremental." in result.code
        assert "reachable/2" in result.incremental_predicates


class TestSubsumptiveTabling:
    """Tests for subsumptive tabling."""
    
    def test_subsumptive_tabling(self):
        """TC-XSB-007: Test subsumptive tabling."""
        ir = {
            "module": "ancestor",
            "tabled": [{"predicate": "ancestor", "arity": 2, "mode": "subsumptive"}],
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "ancestor", "args": [
                     {"kind": "variable", "name": "X"},
                     {"kind": "variable", "name": "Y"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": "parent", "args": [
                         {"kind": "variable", "name": "X"},
                         {"kind": "variable", "name": "Y"}
                     ]}
                 ]}
            ]
        }
        
        emitter = XSBPrologEmitter()
        result = emitter.emit(ir)
        
        assert ":- table ancestor/2 as subsumptive." in result.code
        assert "ancestor/2" in result.subsumptive_predicates


class TestCombinedTablingModes:
    """Tests for combined tabling modes."""
    
    def test_combined_modes(self):
        """TC-XSB-008: Test combined tabling modes."""
        ir = {
            "module": "combined",
            "tabled": [{
                "predicate": "path",
                "arity": 2,
                "mode": "incremental",
                "additional_modes": ["subsumptive"]
            }],
            "clauses": [
                {"kind": "fact", "predicate": "path", "args": [
                    {"kind": "atom", "value": "a"},
                    {"kind": "atom", "value": "b"}
                ]}
            ]
        }
        
        emitter = XSBPrologEmitter()
        result = emitter.emit(ir)
        
        # Should have combined modes
        assert "as (incremental, subsumptive)" in result.code or \
               "as (subsumptive, incremental)" in result.code


class TestLatticeTabling:
    """Tests for lattice tabling."""
    
    def test_lattice_tabling(self):
        """TC-XSB-009: Test lattice tabling."""
        ir = {
            "module": "shortest_path",
            "tabled": [{
                "predicate": "shortest",
                "arity": 3,
                "mode": "variant",
                "lattice_op": "min"
            }],
            "clauses": [
                {"kind": "fact", "predicate": "shortest", "args": [
                    {"kind": "atom", "value": "a"},
                    {"kind": "atom", "value": "a"},
                    0
                ]}
            ]
        }
        
        emitter = XSBPrologEmitter()
        result = emitter.emit(ir)
        
        # Should have lattice tabling with min operation
        assert "lattice(min/3)" in result.code


class TestWellFoundedSemantics:
    """Tests for well-founded semantics."""
    
    def test_wfs_detection(self):
        """TC-XSB-010: Test WFS detection for predicates with negation."""
        ir = {
            "module": "game",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "win", "args": [
                     {"kind": "variable", "name": "X"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": "move", "args": [
                         {"kind": "variable", "name": "X"},
                         {"kind": "variable", "name": "Y"}
                     ]},
                     {"kind": "negation",
                      "goal": {"kind": "compound", "functor": "win", "args": [
                          {"kind": "variable", "name": "Y"}
                      ]}}
                 ]}
            ]
        }
        
        emitter = XSBPrologEmitter()
        result = emitter.emit(ir)
        
        # Should detect WFS requirement
        assert "win/1" in result.wfs_predicates
        # Should have WFS comment
        assert "WFS" in result.code or "well-founded" in result.code.lower()


class TestListEmission:
    """Tests for list emission."""
    
    def test_list_emission(self):
        """TC-XSB-011: Test list emission."""
        ir = {
            "module": "lists",
            "clauses": [
                {"kind": "fact", "predicate": "mylist", "args": [
                    {"kind": "list_term", "elements": [
                        {"kind": "atom", "value": "a"},
                        {"kind": "atom", "value": "b"},
                        {"kind": "atom", "value": "c"}
                    ]}
                ]}
            ]
        }
        
        emitter = XSBPrologEmitter()
        result = emitter.emit(ir)
        
        assert "[a, b, c]" in result.code
    
    def test_list_with_tail(self):
        """Test list with tail variable."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "split", "args": [
                    {"kind": "list_term", 
                     "elements": [{"kind": "variable", "name": "H"}],
                     "tail": {"kind": "variable", "name": "T"}}
                ]}
            ]
        }
        
        emitter = XSBPrologEmitter()
        result = emitter.emit(ir)
        
        assert "[H|T]" in result.code
    
    def test_empty_list(self):
        """Test empty list."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "empty", "args": [
                    {"kind": "list_term", "elements": []}
                ]}
            ]
        }
        
        emitter = XSBPrologEmitter()
        result = emitter.emit(ir)
        
        assert "[]" in result.code


class TestDCGRules:
    """Tests for DCG rules."""
    
    def test_dcg_emission(self):
        """TC-XSB-012: Test DCG rule emission."""
        ir = {
            "module": "grammar",
            "dcg_rules": [
                {
                    "head": {"kind": "compound", "functor": "sentence", "args": []},
                    "body": [
                        {"kind": "nonterminal", "term": {"kind": "compound", "functor": "noun", "args": []}},
                        {"kind": "nonterminal", "term": {"kind": "compound", "functor": "verb", "args": []}}
                    ]
                }
            ],
            "clauses": []
        }
        
        emitter = XSBPrologEmitter()
        result = emitter.emit(ir)
        
        assert "-->" in result.code
        assert "sentence -->" in result.code


class TestDeterminism:
    """Tests for deterministic output."""
    
    def test_deterministic_emission(self):
        """TC-XSB-013: Test deterministic emission."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "foo", "args": [
                    {"kind": "atom", "value": "bar"}
                ]},
                {"kind": "fact", "predicate": "baz", "args": [1, 2, 3]}
            ]
        }
        
        emitter = XSBPrologEmitter()
        result1 = emitter.emit(ir)
        result2 = emitter.emit(ir)
        
        # Emission should be deterministic
        assert result1.sha256 == result2.sha256
    
    def test_sha256_consistency(self):
        """Test SHA256 hash consistency."""
        ir = {"module": "test", "clauses": []}
        
        emitter = XSBPrologEmitter()
        result = emitter.emit(ir)
        
        # SHA256 should be 64 hex characters
        assert len(result.sha256) == 64
        assert all(c in '0123456789abcdef' for c in result.sha256)


class TestMaxAnswers:
    """Tests for max_answers tabling option."""
    
    def test_max_answers(self):
        """TC-XSB-014: Test max_answers annotation."""
        ir = {
            "module": "limited",
            "tabled": [{
                "predicate": "top_results",
                "arity": 2,
                "mode": "variant",
                "max_answers": 100
            }],
            "clauses": [
                {"kind": "fact", "predicate": "top_results", "args": [
                    {"kind": "atom", "value": "item"},
                    1
                ]}
            ]
        }
        
        emitter = XSBPrologEmitter()
        result = emitter.emit(ir)
        
        assert "max_answers(100)" in result.code


class TestPrivateTabling:
    """Tests for private tabling."""
    
    def test_private_tabling(self):
        """TC-XSB-015: Test private tabling."""
        ir = {
            "module": "internal",
            "tabled": [{
                "predicate": "_helper",
                "arity": 2,
                "mode": "private"
            }],
            "clauses": [
                {"kind": "fact", "predicate": "_helper", "args": [
                    {"kind": "atom", "value": "x"},
                    {"kind": "atom", "value": "y"}
                ]}
            ]
        }
        
        emitter = XSBPrologEmitter()
        result = emitter.emit(ir)
        
        assert ":- table _helper/2 as private." in result.code


class TestTypeMapper:
    """Tests for XSBPrologTypeMapper."""
    
    def test_type_mapping(self):
        """Test IR type to XSB type mapping."""
        mapper = XSBPrologTypeMapper()
        
        assert mapper.map_type('i32') == 'integer'
        assert mapper.map_type('f64') == 'float'
        assert mapper.map_type('string') == 'atom'
        assert mapper.map_type('bool') == 'atom'  # XSB uses atoms for booleans
        assert mapper.map_type('list') == 'list'
        assert mapper.map_type('unknown') == 'term'
    
    def test_compound_type_mapping(self):
        """Test compound type mapping."""
        mapper = XSBPrologTypeMapper()
        
        assert mapper.map_type('list(i32)') == 'list'
        assert mapper.map_type('array(f64)') == 'list'
    
    def test_tabling_mode_conversion(self):
        """Test tabling mode string to enum conversion."""
        mapper = XSBPrologTypeMapper()
        
        assert mapper.get_tabling_mode('variant') == TablingMode.VARIANT
        assert mapper.get_tabling_mode('incremental') == TablingMode.INCREMENTAL
        assert mapper.get_tabling_mode('subsumptive') == TablingMode.SUBSUMPTIVE
        assert mapper.get_tabling_mode('opaque') == TablingMode.OPAQUE
        assert mapper.get_tabling_mode('private') == TablingMode.PRIVATE
        assert mapper.get_tabling_mode('unknown') == TablingMode.VARIANT
    
    def test_export_directive_format(self):
        """Test export directive formatting."""
        mapper = XSBPrologTypeMapper()
        
        directive = mapper.format_export_directive('parent', 2)
        assert directive == ':- export(parent/2).'
    
    def test_import_directive_format(self):
        """Test import directive formatting."""
        mapper = XSBPrologTypeMapper()
        
        directive = mapper.format_import_directive('member', 2, 'basics')
        assert directive == ':- import(member/2 from basics).'
    
    def test_tabling_directive_format(self):
        """Test tabling directive formatting."""
        mapper = XSBPrologTypeMapper()
        
        # Basic tabling
        directive = mapper.format_tabling_directive('fib', 2)
        assert directive == ':- table fib/2.'
        
        # Incremental tabling
        directive = mapper.format_tabling_directive('edge', 2, TablingMode.INCREMENTAL)
        assert directive == ':- table edge/2 as incremental.'
        
        # Subsumptive tabling
        directive = mapper.format_tabling_directive('ancestor', 2, TablingMode.SUBSUMPTIVE)
        assert directive == ':- table ancestor/2 as subsumptive.'
    
    def test_lattice_op_check(self):
        """Test lattice operation validation."""
        mapper = XSBPrologTypeMapper()
        
        assert mapper.is_lattice_op('min')
        assert mapper.is_lattice_op('max')
        assert mapper.is_lattice_op('join')
        assert not mapper.is_lattice_op('unknown_op')


class TestTablingSpec:
    """Tests for TablingSpec dataclass."""
    
    def test_tabling_spec_creation(self):
        """Test TablingSpec creation."""
        spec = TablingSpec(name='test', arity=2)
        
        assert spec.name == 'test'
        assert spec.arity == 2
        assert spec.mode == TablingMode.VARIANT
        assert spec.lattice_op is None
        assert spec.max_answers is None
    
    def test_tabling_spec_to_directive(self):
        """Test TablingSpec directive generation."""
        mapper = XSBPrologTypeMapper()
        
        # Basic spec
        spec = TablingSpec(name='fib', arity=2)
        assert ':- table fib/2.' == spec.to_directive(mapper)
        
        # Incremental spec
        spec = TablingSpec(name='edge', arity=2, mode=TablingMode.INCREMENTAL)
        assert ':- table edge/2 as incremental.' == spec.to_directive(mapper)
        
        # Spec with max_answers
        spec = TablingSpec(name='results', arity=2, max_answers=50)
        assert 'max_answers(50)' in spec.to_directive(mapper)


class TestGoalEmission:
    """Tests for goal emission."""
    
    def test_cut_emission(self):
        """Test cut emission."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "first", "args": [
                     {"kind": "variable", "name": "X"}
                 ]},
                 "body": [
                     {"kind": "compound", "functor": "member", "args": [
                         {"kind": "variable", "name": "X"},
                         {"kind": "variable", "name": "L"}
                     ]},
                     {"kind": "cut"}
                 ]}
            ]
        }
        
        emitter = XSBPrologEmitter()
        result = emitter.emit(ir)
        
        assert "!" in result.code
    
    def test_negation_emission(self):
        """Test negation emission."""
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
        
        emitter = XSBPrologEmitter()
        result = emitter.emit(ir)
        
        # Should use \+ for negation
        assert "\\+" in result.code
    
    def test_unification_emission(self):
        """Test unification goal emission."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "rule",
                 "head": {"kind": "compound", "functor": "same", "args": [
                     {"kind": "variable", "name": "X"}
                 ]},
                 "body": [
                     {"kind": "unification",
                      "left": {"kind": "variable", "name": "X"},
                      "right": {"kind": "atom", "value": "hello"}}
                 ]}
            ]
        }
        
        emitter = XSBPrologEmitter()
        result = emitter.emit(ir)
        
        assert "X = hello" in result.code


class TestAtomEscaping:
    """Tests for atom escaping."""
    
    def test_simple_atom(self):
        """Test simple atom doesn't need escaping."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "atom", "args": [
                    {"kind": "atom", "value": "hello"}
                ]}
            ]
        }
        
        emitter = XSBPrologEmitter()
        result = emitter.emit(ir)
        
        assert "atom(hello)." in result.code
    
    def test_atom_with_spaces(self):
        """Test atom with spaces needs quoting."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "quoted", "args": [
                    {"kind": "atom", "value": "hello world"}
                ]}
            ]
        }
        
        emitter = XSBPrologEmitter()
        result = emitter.emit(ir)
        
        assert "'hello world'" in result.code
    
    def test_atom_uppercase_start(self):
        """Test atom starting with uppercase needs quoting."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "name", "args": [
                    {"kind": "atom", "value": "Alice"}
                ]}
            ]
        }
        
        emitter = XSBPrologEmitter()
        result = emitter.emit(ir)
        
        assert "'Alice'" in result.code


class TestDynamicDeclarations:
    """Tests for dynamic predicate declarations."""
    
    def test_dynamic_declaration(self):
        """Test dynamic predicate declaration."""
        ir = {
            "module": "dynamic_test",
            "dynamic": [{"predicate": "fact_db", "arity": 2}],
            "clauses": []
        }
        
        emitter = XSBPrologEmitter()
        result = emitter.emit(ir)
        
        assert ":- dynamic fact_db/2." in result.code
    
    def test_incremental_dynamic(self):
        """Test incremental dynamic declaration."""
        ir = {
            "module": "incr_test",
            "dynamic": [{"predicate": "edge", "arity": 2}],
            "tabled": [{"predicate": "edge", "arity": 2, "mode": "incremental"}],
            "clauses": []
        }
        
        emitter = XSBPrologEmitter()
        result = emitter.emit(ir)
        
        # XSB requires dynamic predicates to be marked as incremental
        # if they are used with incremental tabling
        assert "as incremental" in result.code


class TestConfigOptions:
    """Tests for configuration options."""
    
    def test_disable_exports(self):
        """Test disabling exports."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "foo", "args": []}
            ]
        }
        
        config = XSBPrologConfig(emit_exports=False)
        emitter = XSBPrologEmitter(config)
        result = emitter.emit(ir)
        
        assert ":- export(" not in result.code
    
    def test_disable_comments(self):
        """Test disabling comments."""
        ir = {"module": "test", "clauses": []}
        
        config = XSBPrologConfig(emit_comments=False)
        emitter = XSBPrologEmitter(config)
        result = emitter.emit(ir)
        
        assert "/*" not in result.code
        assert "STUNIR" not in result.code
    
    def test_custom_prefix(self):
        """Test custom module prefix."""
        ir = {"module": "mymod", "clauses": []}
        
        config = XSBPrologConfig(module_prefix="custom")
        emitter = XSBPrologEmitter(config)
        result = emitter.emit(ir)
        
        assert result.module_name == "custom_mymod"


class TestEmitToFile:
    """Tests for file emission."""
    
    def test_emit_to_file(self, tmp_path):
        """Test emitting to file."""
        ir = {
            "module": "test",
            "clauses": [
                {"kind": "fact", "predicate": "test", "args": []}
            ]
        }
        
        emitter = XSBPrologEmitter()
        output_path = tmp_path / "test.P"
        result = emitter.emit_to_file(ir, output_path)
        
        assert output_path.exists()
        content = output_path.read_text()
        assert "test." in content
        assert result.code in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
