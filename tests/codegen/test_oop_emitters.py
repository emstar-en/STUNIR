#!/usr/bin/env python3
"""Tests for STUNIR OOP Emitters (Smalltalk and ALGOL).

Tests cover:
- Smalltalk class definitions
- Smalltalk method definitions
- Message passing (unary, binary, keyword)
- Block emission
- Cascading
- ALGOL block structure
- ALGOL procedures and functions
- Call-by-name and call-by-value
- For loops with step
- Arrays and switches
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from targets.oop import SmalltalkEmitter, ALGOLEmitter


# =============================================================================
# Smalltalk Emitter Tests
# =============================================================================

class TestSmalltalkEmitter:
    """Test Smalltalk code emission."""
    
    @pytest.fixture
    def emitter(self):
        return SmalltalkEmitter()
    
    def test_class_definition(self, emitter):
        """TC-ST-001: Class definition emission."""
        ir = {
            'kind': 'class_def',
            'name': 'Point',
            'superclass': 'Object',
            'instance_variables': ['x', 'y'],
            'class_variables': [],
            'category': 'Graphics'
        }
        result = emitter.emit_class(ir)
        assert "Object subclass: #Point" in result
        assert "instanceVariableNames: 'x y'" in result
        assert "category: 'Graphics'" in result
    
    def test_unary_message(self, emitter):
        """TC-ST-002: Unary message emission."""
        ir = {
            'kind': 'message',
            'receiver': {'kind': 'variable', 'name': 'collection'},
            'selector': 'size',
            'message_type': 'unary'
        }
        result = emitter._emit_message(ir)
        assert result == "collection size"
    
    def test_binary_message(self, emitter):
        """TC-ST-003: Binary message emission."""
        ir = {
            'kind': 'message',
            'receiver': {'kind': 'literal', 'value': 3, 'literal_type': 'integer'},
            'selector': '+',
            'message_type': 'binary',
            'arguments': [
                {'kind': 'literal', 'value': 4, 'literal_type': 'integer'}
            ]
        }
        result = emitter._emit_message(ir)
        assert "3 + 4" in result
    
    def test_keyword_message(self, emitter):
        """TC-ST-004: Keyword message emission."""
        ir = {
            'kind': 'message',
            'receiver': {'kind': 'variable', 'name': 'dict'},
            'selector': 'at:put:',
            'message_type': 'keyword',
            'arguments': [
                {'kind': 'symbol', 'value': 'key'},
                {'kind': 'literal', 'value': 'value', 'literal_type': 'string'}
            ]
        }
        result = emitter._emit_message(ir)
        assert "dict at: #key put: 'value'" in result
    
    def test_block_emission_simple(self, emitter):
        """TC-ST-005: Simple block emission."""
        ir = {
            'kind': 'block',
            'parameters': [],
            'statements': [
                {'kind': 'literal', 'value': 42, 'literal_type': 'integer'}
            ]
        }
        result = emitter._emit_block(ir)
        assert "[" in result
        assert "42" in result
        assert "]" in result
    
    def test_block_with_parameters(self, emitter):
        """TC-ST-006: Block with parameters."""
        ir = {
            'kind': 'block',
            'parameters': ['x'],
            'statements': [
                {'kind': 'message', 
                 'receiver': {'kind': 'variable', 'name': 'x'},
                 'selector': 'squared', 
                 'message_type': 'unary'}
            ]
        }
        result = emitter._emit_block(ir)
        assert ":x |" in result
        assert "x squared" in result
    
    def test_cascade_emission(self, emitter):
        """TC-ST-007: Cascaded messages."""
        ir = {
            'kind': 'cascade',
            'receiver': {'kind': 'variable', 'name': 'stream'},
            'messages': [
                {'selector': 'nextPutAll:', 'message_type': 'keyword',
                 'arguments': [{'kind': 'literal', 'value': 'Hello', 'literal_type': 'string'}]},
                {'selector': 'cr', 'message_type': 'unary'},
                {'selector': 'flush', 'message_type': 'unary'}
            ]
        }
        result = emitter._emit_cascade(ir)
        assert "stream nextPutAll: 'Hello'" in result
        assert "cr" in result
        assert "flush" in result
        assert ";" in result
    
    def test_method_definition_unary(self, emitter):
        """TC-ST-008: Unary method definition."""
        ir = {
            'kind': 'method_def',
            'selector': 'x',
            'message_type': 'unary',
            'parameters': [],
            'temporaries': [],
            'statements': [
                {'kind': 'return', 'value': {'kind': 'variable', 'name': 'x'}}
            ]
        }
        result = emitter.emit_method(ir)
        assert "x\n" in result
        assert "^x" in result
    
    def test_method_definition_keyword(self, emitter):
        """TC-ST-009: Keyword method definition."""
        ir = {
            'kind': 'method_def',
            'selector': 'at:put:',
            'message_type': 'keyword',
            'parameters': ['index', 'value'],
            'temporaries': [],
            'statements': [
                {'kind': 'message',
                 'receiver': {'kind': 'variable', 'name': 'array'},
                 'selector': 'at:put:',
                 'message_type': 'keyword',
                 'arguments': [
                     {'kind': 'variable', 'name': 'index'},
                     {'kind': 'variable', 'name': 'value'}
                 ]}
            ]
        }
        result = emitter.emit_method(ir)
        assert "at: index put: value" in result
    
    def test_conditional_ifTrue(self, emitter):
        """TC-ST-010: Conditional ifTrue emission."""
        ir = {
            'kind': 'conditional',
            'condition': {'kind': 'variable', 'name': 'flag'},
            'true_block': {
                'kind': 'block',
                'statements': [{'kind': 'literal', 'value': 'yes', 'literal_type': 'string'}]
            },
            'false_block': None
        }
        result = emitter._emit_conditional(ir)
        assert "flag ifTrue:" in result
        assert "'yes'" in result
    
    def test_conditional_ifTrue_ifFalse(self, emitter):
        """TC-ST-011: Conditional ifTrue:ifFalse emission."""
        ir = {
            'kind': 'conditional',
            'condition': {'kind': 'variable', 'name': 'flag'},
            'true_block': {
                'kind': 'block',
                'statements': [{'kind': 'literal', 'value': 1}]
            },
            'false_block': {
                'kind': 'block',
                'statements': [{'kind': 'literal', 'value': 0}]
            }
        }
        result = emitter._emit_conditional(ir)
        assert "ifTrue:" in result
        assert "ifFalse:" in result
    
    def test_loop_whileTrue(self, emitter):
        """TC-ST-012: whileTrue loop emission."""
        ir = {
            'kind': 'loop',
            'loop_type': 'whileTrue:',
            'condition_or_count': {
                'kind': 'binary_op',
                'operator': '<',
                'left': {'kind': 'variable', 'name': 'i'},
                'right': {'kind': 'literal', 'value': 10}
            },
            'body': {
                'kind': 'block',
                'statements': [
                    {'kind': 'assignment', 
                     'target': 'i',
                     'value': {'kind': 'binary_op', 'operator': '+',
                              'left': {'kind': 'variable', 'name': 'i'},
                              'right': {'kind': 'literal', 'value': 1}}}
                ]
            }
        }
        result = emitter._emit_loop(ir)
        assert "whileTrue:" in result
    
    def test_loop_timesRepeat(self, emitter):
        """TC-ST-013: timesRepeat loop emission."""
        ir = {
            'kind': 'loop',
            'loop_type': 'timesRepeat:',
            'condition_or_count': {'kind': 'literal', 'value': 5},
            'body': {
                'kind': 'block',
                'statements': [{'kind': 'variable', 'name': 'body'}]
            }
        }
        result = emitter._emit_loop(ir)
        assert "5 timesRepeat:" in result
    
    def test_array_literal(self, emitter):
        """TC-ST-014: Array literal emission."""
        ir = {
            'kind': 'array_literal',
            'elements': [
                {'kind': 'literal', 'value': 1},
                {'kind': 'literal', 'value': 2},
                {'kind': 'literal', 'value': 3}
            ]
        }
        result = emitter._emit_array_literal(ir)
        assert "#(" in result
        assert "1" in result
        assert "2" in result
        assert "3" in result
        assert ")" in result
    
    def test_symbol_literal(self, emitter):
        """TC-ST-015: Symbol literal emission."""
        ir = {'kind': 'symbol', 'value': 'mySymbol'}
        result = emitter._emit_expression(ir)
        assert result == "#mySymbol"
    
    def test_character_literal(self, emitter):
        """TC-ST-016: Character literal emission."""
        ir = {'kind': 'character', 'value': 'a'}
        result = emitter._emit_expression(ir)
        assert result == "$a"
    
    def test_self_reference(self, emitter):
        """TC-ST-017: Self reference emission."""
        ir = {'kind': 'self'}
        result = emitter._emit_expression(ir)
        assert result == "self"
    
    def test_super_reference(self, emitter):
        """TC-ST-018: Super reference emission."""
        ir = {'kind': 'super'}
        result = emitter._emit_expression(ir)
        assert result == "super"
    
    def test_assignment(self, emitter):
        """TC-ST-019: Assignment emission."""
        ir = {
            'kind': 'assignment',
            'target': 'x',
            'value': {'kind': 'literal', 'value': 42}
        }
        result = emitter._emit_assignment(ir)
        assert "x := 42" in result
    
    def test_return(self, emitter):
        """TC-ST-020: Return emission."""
        ir = {
            'kind': 'return',
            'value': {'kind': 'variable', 'name': 'result'}
        }
        result = emitter._emit_return(ir)
        assert "^result" in result
    
    def test_collection_new(self, emitter):
        """TC-ST-021: Collection creation."""
        ir = {
            'kind': 'collection_new',
            'collection_type': 'OrderedCollection',
            'initial_size': 10
        }
        result = emitter._emit_collection_new(ir)
        assert "OrderedCollection new: 10" in result
    
    def test_iteration_do(self, emitter):
        """TC-ST-022: Collection iteration do:."""
        ir = {
            'kind': 'iteration',
            'collection': {'kind': 'variable', 'name': 'items'},
            'iterator': 'do:',
            'block': {
                'kind': 'block',
                'parameters': ['each'],
                'statements': [{'kind': 'variable', 'name': 'each'}]
            }
        }
        result = emitter._emit_iteration(ir)
        assert "items do:" in result
        assert ":each" in result
    
    def test_full_emit(self, emitter):
        """TC-ST-023: Full program emission."""
        ir = {
            'kind': 'smalltalk_program',
            'classes': [
                {
                    'kind': 'class_def',
                    'name': 'Counter',
                    'superclass': 'Object',
                    'instance_variables': ['count'],
                    'class_variables': [],
                    'category': 'Examples'
                }
            ],
            'methods': []
        }
        result = emitter.emit(ir)
        assert "Generated by STUNIR" in result.code
        assert "Object subclass: #Counter" in result.code
        assert result.manifest['dialect'] == 'smalltalk'


# =============================================================================
# ALGOL Emitter Tests
# =============================================================================

class TestALGOLEmitter:
    """Test ALGOL code emission."""
    
    @pytest.fixture
    def emitter(self):
        return ALGOLEmitter()
    
    def test_block_structure(self, emitter):
        """TC-ALG-001: Block structure emission."""
        ir = {
            'kind': 'algol_block',
            'declarations': [
                {'kind': 'var_decl', 'name': 'x', 'var_type': 'integer'}
            ],
            'statements': [
                {'kind': 'assignment', 'target': 'x', 
                 'value': {'kind': 'literal', 'value': 10}}
            ]
        }
        result = emitter._emit_block(ir)
        assert 'begin' in result
        assert 'integer x' in result
        assert 'x := 10' in result
        assert 'end' in result
    
    def test_procedure_emission(self, emitter):
        """TC-ALG-002: Procedure with parameters."""
        ir = {
            'kind': 'algol_procedure',
            'name': 'swap',
            'parameters': [
                {'name': 'a', 'param_type': 'integer', 'mode': 'name'},
                {'name': 'b', 'param_type': 'integer', 'mode': 'name'}
            ],
            'body': {
                'kind': 'algol_block',
                'declarations': [],
                'statements': []
            }
        }
        result = emitter._emit_procedure(ir)
        assert 'procedure swap(a, b)' in result
        assert 'integer a' in result
        assert 'integer b' in result
    
    def test_procedure_with_value_params(self, emitter):
        """TC-ALG-003: Procedure with value parameters."""
        ir = {
            'kind': 'algol_procedure',
            'name': 'double',
            'parameters': [
                {'name': 'x', 'param_type': 'real', 'mode': 'value'}
            ],
            'result_type': 'real',
            'body': {
                'kind': 'algol_block',
                'declarations': [],
                'statements': []
            }
        }
        result = emitter._emit_procedure(ir)
        assert 'real procedure double(x)' in result
        assert 'value x' in result
        assert 'real x' in result
    
    def test_for_loop(self, emitter):
        """TC-ALG-004: For loop with step."""
        ir = {
            'kind': 'algol_for',
            'variable': 'i',
            'init_value': {'kind': 'literal', 'value': 1},
            'step': {'kind': 'literal', 'value': 2},
            'until_value': {'kind': 'literal', 'value': 10},
            'body': {'kind': 'variable', 'name': 'sum'}
        }
        result = emitter._emit_for_loop(ir)
        assert 'for i := 1 step 2 until 10 do' in result
    
    def test_for_loop_while(self, emitter):
        """TC-ALG-005: For loop with while."""
        ir = {
            'kind': 'algol_for',
            'variable': 'i',
            'init_value': {'kind': 'literal', 'value': 1},
            'while_condition': {
                'kind': 'binary_op',
                'operator': '<',
                'left': {'kind': 'variable', 'name': 'i'},
                'right': {'kind': 'literal', 'value': 100}
            },
            'body': {'kind': 'variable', 'name': 'body'}
        }
        result = emitter._emit_for_loop(ir)
        assert 'for i := 1 while' in result
    
    def test_array_declaration(self, emitter):
        """TC-ALG-006: Array with dynamic bounds."""
        ir = {
            'kind': 'algol_array',
            'name': 'matrix',
            'element_type': 'real',
            'bounds': [
                ({'kind': 'literal', 'value': 1}, {'kind': 'variable', 'name': 'n'}),
                ({'kind': 'literal', 'value': 1}, {'kind': 'variable', 'name': 'm'})
            ]
        }
        result = emitter._emit_array_declaration(ir)
        assert 'array matrix[1:n, 1:m]' in result
    
    def test_own_variable(self, emitter):
        """TC-ALG-007: Own variable declaration."""
        ir = {
            'kind': 'algol_procedure',
            'name': 'counter',
            'parameters': [],
            'result_type': 'integer',
            'own_variables': [
                {'name': 'count', 'var_type': 'integer',
                 'initial_value': {'kind': 'literal', 'value': 0}}
            ],
            'body': {'kind': 'algol_block', 'declarations': [], 'statements': []}
        }
        result = emitter._emit_procedure(ir)
        assert 'own integer count := 0' in result
    
    def test_switch_declaration(self, emitter):
        """TC-ALG-008: Switch declaration."""
        ir = {
            'kind': 'algol_switch',
            'name': 'S',
            'labels': ['L1', 'L2', 'L3', 'L4']
        }
        result = emitter._emit_switch(ir)
        assert 'switch S := L1, L2, L3, L4' in result
    
    def test_if_statement(self, emitter):
        """TC-ALG-009: If statement."""
        ir = {
            'kind': 'algol_if',
            'condition': {
                'kind': 'binary_op',
                'operator': '<',
                'left': {'kind': 'variable', 'name': 'x'},
                'right': {'kind': 'literal', 'value': 0}
            },
            'then_branch': {
                'kind': 'assignment',
                'target': 'x',
                'value': {'kind': 'literal', 'value': 0}
            },
            'else_branch': None
        }
        result = emitter._emit_if_statement(ir)
        assert 'if (x < 0) then x := 0' in result
    
    def test_if_else_statement(self, emitter):
        """TC-ALG-010: If-else statement."""
        ir = {
            'kind': 'algol_if',
            'condition': {
                'kind': 'binary_op',
                'operator': '>',
                'left': {'kind': 'variable', 'name': 'a'},
                'right': {'kind': 'variable', 'name': 'b'}
            },
            'then_branch': {
                'kind': 'assignment',
                'target': 'max',
                'value': {'kind': 'variable', 'name': 'a'}
            },
            'else_branch': {
                'kind': 'assignment',
                'target': 'max',
                'value': {'kind': 'variable', 'name': 'b'}
            }
        }
        result = emitter._emit_if_statement(ir)
        assert 'if' in result
        assert 'then' in result
        assert 'else' in result
    
    def test_goto_statement(self, emitter):
        """TC-ALG-011: Goto statement."""
        ir = {
            'kind': 'algol_goto',
            'target': 'L1'
        }
        result = emitter._emit_goto(ir)
        assert 'goto L1' in result
    
    def test_switch_goto(self, emitter):
        """TC-ALG-012: Switch-based goto."""
        ir = {
            'kind': 'algol_goto',
            'switch_name': 'S',
            'switch_index': {'kind': 'variable', 'name': 'i'}
        }
        result = emitter._emit_goto(ir)
        assert 'goto S[i]' in result
    
    def test_procedure_call(self, emitter):
        """TC-ALG-013: Procedure call."""
        ir = {
            'kind': 'algol_call',
            'name': 'print',
            'arguments': [
                {'kind': 'literal', 'value': 'Hello', 'literal_type': 'string'},
                {'kind': 'variable', 'name': 'x'}
            ]
        }
        result = emitter._emit_procedure_call(ir)
        assert 'print("Hello", x)' in result
    
    def test_boolean_literal_true(self, emitter):
        """TC-ALG-014: Boolean true literal."""
        ir = {'kind': 'literal', 'value': True}
        result = emitter._emit_literal(ir)
        assert result == 'true'
    
    def test_boolean_literal_false(self, emitter):
        """TC-ALG-015: Boolean false literal."""
        ir = {'kind': 'literal', 'value': False}
        result = emitter._emit_literal(ir)
        assert result == 'false'
    
    def test_string_literal(self, emitter):
        """TC-ALG-016: String literal."""
        ir = {'kind': 'literal', 'value': 'hello', 'literal_type': 'string'}
        result = emitter._emit_literal(ir)
        assert result == '"hello"'
    
    def test_binary_operation(self, emitter):
        """TC-ALG-017: Binary operation."""
        ir = {
            'kind': 'binary_op',
            'operator': '+',
            'left': {'kind': 'variable', 'name': 'a'},
            'right': {'kind': 'variable', 'name': 'b'}
        }
        result = emitter._emit_binary_op(ir)
        assert '(a + b)' in result
    
    def test_unary_operation(self, emitter):
        """TC-ALG-018: Unary operation."""
        ir = {
            'kind': 'unary_op',
            'operator': 'not',
            'operand': {'kind': 'variable', 'name': 'flag'}
        }
        result = emitter._emit_unary_op(ir)
        assert 'not flag' in result
    
    def test_labeled_block(self, emitter):
        """TC-ALG-019: Labeled block."""
        ir = {
            'kind': 'algol_block',
            'label': 'L1',
            'declarations': [],
            'statements': []
        }
        result = emitter._emit_block(ir)
        assert 'L1:' in result
        assert 'begin' in result
        assert 'end' in result
    
    def test_full_emit(self, emitter):
        """TC-ALG-020: Full program emission."""
        ir = {
            'kind': 'algol_program',
            'name': 'test',
            'main_block': {
                'kind': 'algol_block',
                'declarations': [
                    {'kind': 'var_decl', 'name': 'x', 'var_type': 'integer'}
                ],
                'statements': [
                    {'kind': 'assignment', 'target': 'x', 
                     'value': {'kind': 'literal', 'value': 42}}
                ]
            }
        }
        result = emitter.emit(ir)
        assert 'STUNIR Generated' in result.code
        assert 'begin' in result.code
        assert 'integer x' in result.code
        assert 'x := 42' in result.code
        assert result.manifest['dialect'] == 'algol'
    
    def test_type_mapping(self, emitter):
        """TC-ALG-021: Type mapping."""
        assert emitter._map_type('i32') == 'integer'
        assert emitter._map_type('f64') == 'real'
        assert emitter._map_type('bool') == 'Boolean'
        assert emitter._map_type('unknown') == 'unknown'


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for OOP emitters."""
    
    def test_smalltalk_point_class(self):
        """TC-INT-001: Complete Point class in Smalltalk."""
        emitter = SmalltalkEmitter()
        
        ir = {
            'kind': 'class_def',
            'name': 'Point',
            'superclass': 'Object',
            'instance_variables': ['x', 'y'],
            'class_variables': ['Origin'],
            'category': 'Graphics-Primitives'
        }
        
        result = emitter.emit_class(ir)
        
        assert "Object subclass: #Point" in result
        assert "instanceVariableNames: 'x y'" in result
        assert "classVariableNames: 'Origin'" in result
    
    def test_algol_jensens_device(self):
        """TC-INT-002: Jensen's device in ALGOL."""
        emitter = ALGOLEmitter()
        
        ir = {
            'kind': 'algol_procedure',
            'name': 'sum',
            'result_type': 'real',
            'parameters': [
                {'name': 'i', 'param_type': 'integer', 'mode': 'name'},
                {'name': 'lo', 'param_type': 'integer', 'mode': 'value'},
                {'name': 'hi', 'param_type': 'integer', 'mode': 'value'},
                {'name': 'term', 'param_type': 'real', 'mode': 'name'}
            ],
            'body': {
                'kind': 'algol_block',
                'declarations': [
                    {'kind': 'var_decl', 'name': 'temp', 'var_type': 'real'}
                ],
                'statements': [
                    {'kind': 'assignment', 'target': 'temp',
                     'value': {'kind': 'literal', 'value': 0}},
                    {
                        'kind': 'algol_for',
                        'variable': 'i',
                        'init_value': {'kind': 'variable', 'name': 'lo'},
                        'step': {'kind': 'literal', 'value': 1},
                        'until_value': {'kind': 'variable', 'name': 'hi'},
                        'body': {
                            'kind': 'assignment',
                            'target': 'temp',
                            'value': {
                                'kind': 'binary_op',
                                'operator': '+',
                                'left': {'kind': 'variable', 'name': 'temp'},
                                'right': {'kind': 'variable', 'name': 'term'}
                            }
                        }
                    }
                ]
            }
        }
        
        result = emitter._emit_procedure(ir)
        
        assert 'real procedure sum' in result
        assert 'value lo, hi' in result
        assert 'integer i' in result
        assert 'real term' in result
        assert 'for i := lo step 1 until hi do' in result
    
    def test_smalltalk_factorial_method(self):
        """TC-INT-003: Factorial method in Smalltalk."""
        emitter = SmalltalkEmitter()
        
        ir = {
            'kind': 'method_def',
            'selector': 'factorial',
            'message_type': 'unary',
            'parameters': [],
            'temporaries': [],
            'statements': [
                {
                    'kind': 'conditional',
                    'condition': {
                        'kind': 'message',
                        'receiver': {'kind': 'self'},
                        'selector': '<=',
                        'message_type': 'binary',
                        'arguments': [{'kind': 'literal', 'value': 1}]
                    },
                    'true_block': {
                        'kind': 'block',
                        'statements': [{'kind': 'literal', 'value': 1}]
                    },
                    'false_block': {
                        'kind': 'block',
                        'statements': [
                            {
                                'kind': 'message',
                                'receiver': {'kind': 'self'},
                                'selector': '*',
                                'message_type': 'binary',
                                'arguments': [
                                    {
                                        'kind': 'message',
                                        'receiver': {
                                            'kind': 'message',
                                            'receiver': {'kind': 'self'},
                                            'selector': '-',
                                            'message_type': 'binary',
                                            'arguments': [{'kind': 'literal', 'value': 1}]
                                        },
                                        'selector': 'factorial',
                                        'message_type': 'unary'
                                    }
                                ]
                            }
                        ]
                    }
                }
            ]
        }
        
        result = emitter.emit_method(ir)
        
        assert 'factorial' in result
        assert 'ifTrue:' in result
        assert 'ifFalse:' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
