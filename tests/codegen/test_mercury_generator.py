#!/usr/bin/env python3
"""Tests for STUNIR Mercury Code Generator.

Tests the Mercury emitter including:
- Module structure (interface/implementation sections)
- Type declarations
- Mode declarations
- Determinism inference
- Predicate emission
- Function emission
- Term conversion

Part of Phase 5D-3: Mercury Emitter.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from targets.prolog.mercury import (
    MercuryEmitter,
    MercuryConfig,
    MercuryTypeMapper,
    MercuryMode,
    Determinism,
    Purity,
    MERCURY_TYPES,
)
from tools.ir.logic_ir import (
    Variable, Atom, Number, StringTerm, Compound, ListTerm, Anonymous,
    Fact, Rule, Goal, Predicate, GoalKind
)


class TestMercuryTypeMapper:
    """Tests for Mercury type mapping."""
    
    def test_basic_type_mapping(self):
        """Test basic IR to Mercury type mapping."""
        mapper = MercuryTypeMapper()
        
        assert mapper.map_type('i32') == 'int'
        assert mapper.map_type('i64') == 'int'
        assert mapper.map_type('f64') == 'float'
        assert mapper.map_type('bool') == 'bool'
        assert mapper.map_type('string') == 'string'
        assert mapper.map_type('char') == 'char'
    
    def test_list_type_mapping(self):
        """Test list type mapping with parameters."""
        mapper = MercuryTypeMapper()
        
        assert mapper.map_type('list') == 'list(T)'
        assert mapper.map_type('list(i32)') == 'list(int)'
        assert mapper.map_type('list(string)') == 'list(string)'
    
    def test_unknown_type_fallback(self):
        """Test unknown types fall back to univ."""
        mapper = MercuryTypeMapper()
        
        assert mapper.map_type('unknown_type') == 'univ'
        assert mapper.map_type('') == 'univ'
        assert mapper.map_type(None) == 'univ'
    
    def test_mode_mapping(self):
        """Test mode string to MercuryMode mapping."""
        mapper = MercuryTypeMapper()
        
        assert mapper.map_mode('input') == MercuryMode.IN
        assert mapper.map_mode('output') == MercuryMode.OUT
        assert mapper.map_mode('in') == MercuryMode.IN
        assert mapper.map_mode('out') == MercuryMode.OUT
        assert mapper.map_mode('bidirectional') == MercuryMode.IN_OUT
    
    def test_mode_inference(self):
        """Test mode inference from parameter info."""
        mapper = MercuryTypeMapper()
        
        # Explicit mode
        assert mapper.infer_mode_from_param({'mode': 'input'}) == MercuryMode.IN
        
        # Name-based inference
        assert mapper.infer_mode_from_param({'name': 'Result'}) == MercuryMode.OUT
        assert mapper.infer_mode_from_param({'name': 'Input'}) == MercuryMode.IN
        
        # Position-based inference
        assert mapper.infer_mode_from_param({}, 'first') == MercuryMode.IN
        assert mapper.infer_mode_from_param({}, 'last') == MercuryMode.OUT
    
    def test_determinism_inference_single_clause(self):
        """Test determinism inference for single clause predicates."""
        mapper = MercuryTypeMapper()
        
        # Single clause without guards → det
        fact = Fact(predicate='foo', args=[Atom('a')])
        det = mapper.infer_determinism_from_clauses([fact])
        assert det == Determinism.DET
    
    def test_determinism_inference_empty(self):
        """Test determinism inference for no clauses."""
        mapper = MercuryTypeMapper()
        
        det = mapper.infer_determinism_from_clauses([])
        assert det == Determinism.FAILURE
    
    def test_determinism_with_cut(self):
        """Test determinism with cut present."""
        mapper = MercuryTypeMapper()
        
        fact = Fact(predicate='foo', args=[])
        det = mapper.infer_determinism_from_clauses([fact], has_cut=True)
        assert det == Determinism.SEMIDET
    
    def test_pred_declaration_format(self):
        """Test predicate declaration formatting."""
        mapper = MercuryTypeMapper()
        
        params = [
            {'type': 'i32', 'mode': 'input'},
            {'type': 'i32', 'mode': 'input'},
            {'type': 'i32', 'mode': 'output'},
        ]
        
        decl = mapper.format_pred_declaration('add', params, Determinism.DET)
        
        assert ':- pred add(' in decl
        assert 'int::in' in decl
        assert 'int::out' in decl
        assert 'is det' in decl
    
    def test_func_declaration_format(self):
        """Test function declaration formatting."""
        mapper = MercuryTypeMapper()
        
        decl = mapper.format_func_declaration(
            'double', ['i32'], 'i32', Determinism.DET
        )
        
        assert ':- func double(int)' in decl
        assert '= int' in decl
        assert 'is det' in decl


class TestMercuryEmitter:
    """Tests for Mercury code emitter."""
    
    def test_basic_module_emission(self):
        """Test basic module structure emission."""
        emitter = MercuryEmitter()
        
        ir = {'module': 'test', 'predicates': []}
        result = emitter.emit(ir)
        
        assert ':- module stunir_test.' in result.code
        assert ':- interface.' in result.code
        assert ':- implementation.' in result.code
        assert ':- end_module stunir_test.' in result.code
    
    def test_module_name_conversion(self):
        """Test module name is properly converted."""
        emitter = MercuryEmitter()
        
        # Uppercase start
        ir = {'module': 'TestModule'}
        result = emitter.emit(ir)
        assert ':- module stunir_testModule.' in result.code
        
        # Special characters
        ir = {'module': 'test-module'}
        result = emitter.emit(ir)
        assert ':- module stunir_test_module.' in result.code
    
    def test_predicate_emission(self):
        """Test predicate with clauses emission."""
        emitter = MercuryEmitter()
        
        # IR uses 'clauses' at the top level, not 'predicates'
        ir = {
            'module': 'math',
            'clauses': [{
                'kind': 'fact',
                'predicate': 'double',
                'args': [{'kind': 'number', 'value': 2}, {'kind': 'number', 'value': 4}]
            }]
        }
        
        result = emitter.emit(ir)
        
        assert 'double(2, 4).' in result.code
        assert 'double/2' in result.predicates
    
    def test_rule_emission(self):
        """Test rule emission with head and body."""
        emitter = MercuryEmitter()
        
        # Create a simple rule: add(X, Y, Z) :- Z = X + Y
        head = Compound('add', [Variable('X'), Variable('Y'), Variable('Z')])
        body_goal = Goal.unification(
            Variable('Z'),
            Compound('+', [Variable('X'), Variable('Y')])
        )
        rule = Rule(head=head, body=[body_goal])
        pred = Predicate('add', 3, clauses=[rule])
        
        emitter._predicates = {('add', 3): pred}
        emitter._exports = {('add', 3)}
        
        result = emitter._emit_predicate(pred)
        
        assert 'add(X, Y, Z)' in result
        assert ':-' in result
    
    def test_type_declaration_enum(self):
        """Test enum type declaration emission."""
        emitter = MercuryEmitter()
        
        type_def = {
            'name': 'color',
            'kind': 'enum',
            'constructors': ['red', 'green', 'blue']
        }
        
        result = emitter._emit_type_declaration(type_def)
        
        assert ':- type color --->' in result
        assert 'red' in result
        assert 'green' in result
        assert 'blue' in result
    
    def test_type_declaration_parametric(self):
        """Test parametric type declaration."""
        emitter = MercuryEmitter()
        
        type_def = {
            'name': 'maybe',
            'kind': 'enum',
            'params': ['T'],
            'constructors': ['yes(T)', 'no']
        }
        
        result = emitter._emit_type_declaration(type_def)
        
        assert ':- type maybe(T)' in result
    
    def test_term_emission_variable(self):
        """Test variable term emission."""
        emitter = MercuryEmitter()
        
        assert emitter._emit_term(Variable('X')) == 'X'
        assert emitter._emit_term(Variable('_')) == '_'
        assert emitter._emit_term(Variable('Result')) == 'Result'
    
    def test_term_emission_atom(self):
        """Test atom term emission."""
        emitter = MercuryEmitter()
        
        assert emitter._emit_term(Atom('hello')) == 'hello'
        assert emitter._emit_term(Atom('Hello')) == "'Hello'"  # Needs quoting
    
    def test_term_emission_number(self):
        """Test number term emission."""
        emitter = MercuryEmitter()
        
        assert emitter._emit_term(Number(42)) == '42'
        assert emitter._emit_term(Number(3.14)) == '3.14'
    
    def test_term_emission_string(self):
        """Test string term emission."""
        emitter = MercuryEmitter()
        
        assert emitter._emit_term(StringTerm('hello')) == '"hello"'
        assert emitter._emit_term(StringTerm('say "hi"')) == '"say \\"hi\\""'
    
    def test_term_emission_list(self):
        """Test list term emission."""
        emitter = MercuryEmitter()
        
        # Empty list
        assert emitter._emit_term(ListTerm([], None)) == '[]'
        
        # Simple list
        lst = ListTerm([Number(1), Number(2), Number(3)], None)
        assert emitter._emit_term(lst) == '[1, 2, 3]'
        
        # List with tail
        lst_tail = ListTerm([Number(1)], Variable('T'))
        assert emitter._emit_term(lst_tail) == '[1 | T]'
    
    def test_term_emission_compound(self):
        """Test compound term emission."""
        emitter = MercuryEmitter()
        
        # Simple compound
        comp = Compound('foo', [Atom('bar'), Number(42)])
        assert emitter._emit_term(comp) == 'foo(bar, 42)'
        
        # No args
        comp_no_args = Compound('atom', [])
        assert emitter._emit_term(comp_no_args) == 'atom'
    
    def test_goal_emission_unification(self):
        """Test unification goal emission."""
        emitter = MercuryEmitter()
        
        goal = Goal.unification(Variable('X'), Number(42))
        assert emitter._emit_goal(goal) == 'X = 42'
    
    def test_goal_emission_negation(self):
        """Test negation goal emission."""
        emitter = MercuryEmitter()
        
        inner = Goal.call(Compound('foo', [Variable('X')]))
        goal = Goal.negation(inner)
        result = emitter._emit_goal(goal)
        
        assert 'not' in result
        assert 'foo(X)' in result
    
    def test_goal_emission_conjunction(self):
        """Test conjunction goal emission."""
        emitter = MercuryEmitter()
        
        g1 = Goal.call(Compound('foo', []))
        g2 = Goal.call(Compound('bar', []))
        # Create conjunction goal directly
        goal = Goal(kind=GoalKind.CONJUNCTION, goals=[g1, g2])
        
        result = emitter._emit_goal(goal)
        assert 'foo' in result
        assert 'bar' in result
        assert ',' in result
    
    def test_goal_emission_disjunction(self):
        """Test disjunction goal emission."""
        emitter = MercuryEmitter()
        
        g1 = Goal.call(Compound('foo', []))
        g2 = Goal.call(Compound('bar', []))
        # Create disjunction goal directly
        goal = Goal(kind=GoalKind.DISJUNCTION, goals=[g1, g2])
        
        result = emitter._emit_goal(goal)
        assert 'foo' in result
        assert 'bar' in result
        assert ';' in result
    
    def test_goal_emission_if_then_else(self):
        """Test if-then-else goal emission."""
        emitter = MercuryEmitter()
        
        cond = Goal.call(Compound('test', []))
        then_g = Goal.call(Compound('success', []))
        else_g = Goal.call(Compound('failure', []))
        # Create if-then-else goal directly
        goal = Goal(kind=GoalKind.IF_THEN_ELSE, goals=[cond, then_g, else_g])
        
        result = emitter._emit_goal(goal)
        assert 'if' in result
        assert 'then' in result
        assert 'else' in result
    
    def test_function_emission(self):
        """Test function emission."""
        emitter = MercuryEmitter()
        
        func = {
            'name': 'double',
            'params': [{'name': 'X', 'type': 'i32'}],
            'return_type': 'i32',
            'body': {
                'kind': 'binary_op',
                'op': '*',
                'left': {'kind': 'var', 'name': 'X'},
                'right': {'kind': 'literal', 'value': 2}
            }
        }
        
        result = emitter._emit_function(func)
        
        assert 'double(X)' in result
        assert '=' in result
        assert '*' in result
    
    def test_config_options(self):
        """Test configuration options affect output."""
        config = MercuryConfig(
            emit_comments=False,
            emit_timestamps=False,
            emit_end_module=False
        )
        emitter = MercuryEmitter(config)
        
        ir = {'module': 'test'}
        result = emitter.emit(ir)
        
        assert '%' not in result.code.split('\n')[0] or ':- module' in result.code.split('\n')[0]
        assert ':- end_module' not in result.code
    
    def test_emitter_result_structure(self):
        """Test EmitterResult has all required fields."""
        emitter = MercuryEmitter()
        
        ir = {'module': 'test'}
        result = emitter.emit(ir)
        
        assert hasattr(result, 'code')
        assert hasattr(result, 'module_name')
        assert hasattr(result, 'predicates')
        assert hasattr(result, 'functions')
        assert hasattr(result, 'types')
        assert hasattr(result, 'sha256')
        assert hasattr(result, 'emit_time')
        
        assert result.module_name == 'stunir_test'
        assert len(result.sha256) == 64  # SHA256 hex length
    
    def test_complex_ir_emission(self):
        """Test emission of complex IR with multiple components."""
        emitter = MercuryEmitter()
        
        ir = {
            'module': 'example',
            'types': [
                {
                    'name': 'result',
                    'kind': 'enum',
                    'constructors': ['ok', 'error']
                }
            ],
            'predicates': [
                {
                    'name': 'member',
                    'clauses': [
                        {
                            'kind': 'fact',
                            'predicate': 'member',
                            'args': [
                                {'kind': 'variable', 'name': 'X'},
                                {'kind': 'list_term', 'elements': [
                                    {'kind': 'variable', 'name': 'X'}
                                ], 'tail': {'kind': 'variable', 'name': '_'}}
                            ]
                        }
                    ]
                }
            ],
            'functions': [
                {
                    'name': 'identity',
                    'params': [{'name': 'X', 'type': 'any'}],
                    'return_type': 'any',
                    'body': {'kind': 'var', 'name': 'X'}
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        # Check module structure
        assert ':- module stunir_example.' in result.code
        assert ':- interface.' in result.code
        assert ':- implementation.' in result.code
        
        # Check type declaration
        assert ':- type result' in result.code
        
        # Check function
        assert ':- func identity' in result.code


class TestDeterminismEnum:
    """Tests for Determinism enum properties."""
    
    def test_can_fail_property(self):
        """Test can_fail property."""
        assert not Determinism.DET.can_fail
        assert Determinism.SEMIDET.can_fail
        assert not Determinism.MULTI.can_fail
        assert Determinism.NONDET.can_fail
        assert Determinism.FAILURE.can_fail
        assert not Determinism.ERRONEOUS.can_fail
    
    def test_at_most_one_property(self):
        """Test at_most_one property."""
        assert Determinism.DET.at_most_one
        assert Determinism.SEMIDET.at_most_one
        assert not Determinism.MULTI.at_most_one
        assert not Determinism.NONDET.at_most_one
        assert Determinism.FAILURE.at_most_one
    
    def test_string_representation(self):
        """Test string representation."""
        assert str(Determinism.DET) == 'det'
        assert str(Determinism.SEMIDET) == 'semidet'
        assert str(Determinism.MULTI) == 'multi'
        assert str(Determinism.NONDET) == 'nondet'


class TestMercuryModeEnum:
    """Tests for MercuryMode enum."""
    
    def test_string_representation(self):
        """Test string representation of modes."""
        assert str(MercuryMode.IN) == 'in'
        assert str(MercuryMode.OUT) == 'out'
        assert str(MercuryMode.IN_OUT) == 'in_out'
        assert str(MercuryMode.UI) == 'ui'
        assert str(MercuryMode.UO) == 'uo'


class TestPurityEnum:
    """Tests for Purity enum."""
    
    def test_string_representation(self):
        """Test string representation of purity levels."""
        assert str(Purity.PURE) == 'pure'
        assert str(Purity.SEMIPURE) == 'semipure'
        assert str(Purity.IMPURE) == 'impure'


class TestMercuryReservedWords:
    """Tests for Mercury reserved word handling."""
    
    def test_reserved_word_escaping(self):
        """Test that reserved words are properly handled."""
        from targets.prolog.mercury.types import MERCURY_RESERVED
        
        assert 'module' in MERCURY_RESERVED
        assert 'interface' in MERCURY_RESERVED
        assert 'implementation' in MERCURY_RESERVED
        assert 'pred' in MERCURY_RESERVED
        assert 'func' in MERCURY_RESERVED
        assert 'det' in MERCURY_RESERVED
    
    def test_module_name_avoids_reserved(self):
        """Test module naming avoids reserved words."""
        emitter = MercuryEmitter()
        
        # 'module' is reserved
        name = emitter._mercury_name('module')
        assert name == 'module_'  # Underscore added


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
