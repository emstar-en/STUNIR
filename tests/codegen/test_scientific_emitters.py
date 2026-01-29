#!/usr/bin/env python3
"""Tests for STUNIR Scientific Language Emitters.

Tests cover:
- Fortran code generation
- Pascal code generation
- Array operations
- Numerical computing
- Parallel constructs
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from targets.scientific import FortranEmitter, PascalEmitter


class TestFortranEmitter:
    """Test Fortran code generation."""
    
    @pytest.fixture
    def emitter(self):
        return FortranEmitter()
    
    def test_simple_program(self, emitter):
        """Test simple program generation."""
        ir = {
            'kind': 'program',
            'name': 'hello',
            'body': [
                {
                    'kind': 'call_statement',
                    'name': 'PRINT',
                    'arguments': [{'kind': 'literal', 'value': 'Hello, Fortran!'}]
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        assert 'PROGRAM hello' in result.code
        assert 'END PROGRAM hello' in result.code
        assert 'IMPLICIT NONE' in result.code
    
    def test_module_generation(self, emitter):
        """Test module generation."""
        ir = {
            'kind': 'module',
            'name': 'math_utils',
            'exports': ['add_arrays'],
            'subprograms': [
                {
                    'kind': 'subprogram',
                    'name': 'add_arrays',
                    'is_function': True,
                    'is_pure': True,
                    'parameters': [
                        {'name': 'a', 'type_ref': {'name': 'f64'}, 'intent': 'in'},
                        {'name': 'b', 'type_ref': {'name': 'f64'}, 'intent': 'in'}
                    ],
                    'return_type': {'name': 'f64'},
                    'body': [
                        {
                            'kind': 'return_statement',
                            'value': {
                                'kind': 'binary_op',
                                'op': '+',
                                'left': {'kind': 'var_ref', 'name': 'a'},
                                'right': {'kind': 'var_ref', 'name': 'b'}
                            }
                        }
                    ]
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        assert 'MODULE math_utils' in result.code
        assert 'PUBLIC :: add_arrays' in result.code
        assert 'PURE' in result.code
        assert 'FUNCTION add_arrays' in result.code
        assert 'END MODULE math_utils' in result.code
    
    def test_derived_type(self, emitter):
        """Test derived type generation."""
        ir = {
            'kind': 'module',
            'name': 'types_mod',
            'types': [
                {
                    'kind': 'record_type',
                    'name': 'point_t',
                    'fields': [
                        {'name': 'x', 'type_ref': {'name': 'f64'}},
                        {'name': 'y', 'type_ref': {'name': 'f64'}},
                        {'name': 'z', 'type_ref': {'name': 'f64'}}
                    ]
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        assert 'TYPE :: point_t' in result.code
        assert 'REAL(KIND=8) :: x' in result.code
        assert 'END TYPE point_t' in result.code
    
    def test_array_declaration(self, emitter):
        """Test array variable declaration."""
        ir = {
            'kind': 'program',
            'name': 'array_test',
            'variables': [
                {
                    'name': 'matrix',
                    'type_ref': {
                        'kind': 'array_type',
                        'element_type': {'name': 'f64'},
                        'dimensions': [
                            {'lower': {'value': 1}, 'upper': {'value': 10}},
                            {'lower': {'value': 1}, 'upper': {'value': 10}}
                        ]
                    }
                }
            ],
            'body': []
        }
        
        result = emitter.emit(ir)
        
        assert 'DIMENSION' in result.code
        assert '1:10' in result.code
    
    def test_do_concurrent(self, emitter):
        """Test DO CONCURRENT loop generation."""
        ir = {
            'kind': 'program',
            'name': 'parallel_test',
            'variables': [
                {'name': 'n', 'type_ref': {'name': 'i32'}},
                {'name': 'i', 'type_ref': {'name': 'i32'}}
            ],
            'body': [
                {
                    'kind': 'do_concurrent',
                    'indices': [
                        {
                            'variable': 'i',
                            'start': {'value': 1},
                            'end': {'kind': 'var_ref', 'name': 'n'}
                        }
                    ],
                    'body': [
                        {
                            'kind': 'call_statement',
                            'name': 'process',
                            'arguments': [{'kind': 'var_ref', 'name': 'i'}]
                        }
                    ]
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        assert 'DO CONCURRENT' in result.code
        assert 'END DO' in result.code
    
    def test_if_statement(self, emitter):
        """Test IF statement generation."""
        ir = {
            'kind': 'program',
            'name': 'if_test',
            'body': [
                {
                    'kind': 'if_statement',
                    'condition': {
                        'kind': 'binary_op',
                        'op': '>',
                        'left': {'kind': 'var_ref', 'name': 'x'},
                        'right': {'value': 0}
                    },
                    'then_body': [
                        {'kind': 'call_statement', 'name': 'positive', 'arguments': []}
                    ],
                    'else_body': [
                        {'kind': 'call_statement', 'name': 'negative', 'arguments': []}
                    ]
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        assert 'IF' in result.code
        assert 'THEN' in result.code
        assert 'ELSE' in result.code
        assert 'END IF' in result.code
    
    def test_for_loop(self, emitter):
        """Test DO loop generation."""
        ir = {
            'kind': 'program',
            'name': 'loop_test',
            'body': [
                {
                    'kind': 'for_loop',
                    'variable': 'i',
                    'start': {'value': 1},
                    'end': {'value': 10},
                    'body': [
                        {'kind': 'call_statement', 'name': 'print_i', 'arguments': []}
                    ]
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        assert 'DO i = 1, 10' in result.code
        assert 'END DO' in result.code
    
    def test_array_slice(self, emitter):
        """Test array slice expression."""
        ir = {
            'kind': 'program',
            'name': 'slice_test',
            'body': [
                {
                    'kind': 'assignment',
                    'target': {'kind': 'var_ref', 'name': 'result'},
                    'value': {
                        'kind': 'array_slice',
                        'array': {'kind': 'var_ref', 'name': 'arr'},
                        'slices': [
                            {'start': {'value': 1}, 'stop': {'value': 5}},
                            {}  # Full slice
                        ]
                    }
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        assert 'arr(1:5, :)' in result.code
    
    def test_complex_literal(self, emitter):
        """Test complex number literal."""
        ir = {
            'kind': 'program',
            'name': 'complex_test',
            'body': [
                {
                    'kind': 'assignment',
                    'target': {'kind': 'var_ref', 'name': 'z'},
                    'value': {
                        'kind': 'complex_literal',
                        'real_part': {'value': 1.0},
                        'imag_part': {'value': 2.0}
                    }
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        assert '(1.0, 2.0)' in result.code
    
    def test_interface_block(self, emitter):
        """Test interface block generation."""
        ir = {
            'kind': 'module',
            'name': 'generic_mod',
            'interfaces': [
                {
                    'kind': 'interface',
                    'name': 'add',
                    'procedures': ['add_int', 'add_real']
                }
            ],
            'subprograms': []
        }
        
        result = emitter.emit(ir)
        
        assert 'INTERFACE add' in result.code
        assert 'MODULE PROCEDURE add_int' in result.code
        assert 'END INTERFACE add' in result.code
    
    def test_type_mappings(self, emitter):
        """Test Fortran type mappings."""
        ir = {
            'kind': 'program',
            'name': 'types_test',
            'variables': [
                {'name': 'a', 'type_ref': {'name': 'i32'}},
                {'name': 'b', 'type_ref': {'name': 'f64'}},
                {'name': 'c', 'type_ref': {'name': 'bool'}},
                {'name': 'd', 'type_ref': {'name': 'string'}}
            ],
            'body': []
        }
        
        result = emitter.emit(ir)
        
        assert 'INTEGER(KIND=4)' in result.code
        assert 'REAL(KIND=8)' in result.code
        assert 'LOGICAL' in result.code
        assert 'CHARACTER' in result.code
    
    def test_manifest_generation(self, emitter):
        """Test manifest is generated."""
        ir = {'kind': 'program', 'name': 'test', 'body': []}
        
        result = emitter.emit(ir)
        
        assert 'schema' in result.manifest
        assert result.manifest['schema'] == 'stunir.manifest.targets.v1'
        assert 'ir_hash' in result.manifest
        assert 'output' in result.manifest


class TestPascalEmitter:
    """Test Pascal code generation."""
    
    @pytest.fixture
    def emitter(self):
        return PascalEmitter()
    
    def test_simple_program(self, emitter):
        """Test simple program generation."""
        ir = {
            'kind': 'program',
            'name': 'Hello',
            'body': [
                {
                    'kind': 'call_statement',
                    'name': 'WriteLn',
                    'arguments': [{'kind': 'literal', 'value': 'Hello, Pascal!'}]
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        assert 'program Hello;' in result.code
        assert 'begin' in result.code
        assert 'end.' in result.code
    
    def test_unit_generation(self, emitter):
        """Test unit generation."""
        ir = {
            'kind': 'module',
            'name': 'MathUtils',
            'exports': ['Add'],
            'subprograms': [
                {
                    'kind': 'subprogram',
                    'name': 'Add',
                    'is_function': True,
                    'parameters': [
                        {'name': 'A', 'type_ref': {'name': 'i32'}, 'mode': 'value'},
                        {'name': 'B', 'type_ref': {'name': 'i32'}, 'mode': 'value'}
                    ],
                    'return_type': {'name': 'i32'},
                    'body': [
                        {
                            'kind': 'return_statement',
                            'value': {
                                'kind': 'binary_op',
                                'op': '+',
                                'left': {'kind': 'var_ref', 'name': 'A'},
                                'right': {'kind': 'var_ref', 'name': 'B'}
                            }
                        }
                    ]
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        assert 'unit MathUtils;' in result.code
        assert 'interface' in result.code
        assert 'implementation' in result.code
        assert 'function Add' in result.code
    
    def test_record_type(self, emitter):
        """Test record type generation."""
        ir = {
            'kind': 'program',
            'name': 'RecordTest',
            'types': [
                {
                    'kind': 'record_type',
                    'name': 'TPoint',
                    'fields': [
                        {'name': 'X', 'type_ref': {'name': 'f64'}},
                        {'name': 'Y', 'type_ref': {'name': 'f64'}}
                    ]
                }
            ],
            'body': []
        }
        
        result = emitter.emit(ir)
        
        assert 'TPoint = record' in result.code
        assert 'X: Double;' in result.code
        assert 'Y: Double;' in result.code
        assert 'end;' in result.code
    
    def test_procedure_with_var_param(self, emitter):
        """Test procedure with VAR parameter."""
        ir = {
            'kind': 'program',
            'name': 'VarTest',
            'subprograms': [
                {
                    'kind': 'subprogram',
                    'name': 'Swap',
                    'is_function': False,
                    'parameters': [
                        {'name': 'A', 'type_ref': {'name': 'i32'}, 'mode': 'var'},
                        {'name': 'B', 'type_ref': {'name': 'i32'}, 'mode': 'var'}
                    ],
                    'local_vars': [
                        {'name': 'Temp', 'type_ref': {'name': 'i32'}}
                    ],
                    'body': [
                        {'kind': 'assignment', 'target': {'kind': 'var_ref', 'name': 'Temp'}, 'value': {'kind': 'var_ref', 'name': 'A'}},
                        {'kind': 'assignment', 'target': {'kind': 'var_ref', 'name': 'A'}, 'value': {'kind': 'var_ref', 'name': 'B'}},
                        {'kind': 'assignment', 'target': {'kind': 'var_ref', 'name': 'B'}, 'value': {'kind': 'var_ref', 'name': 'Temp'}}
                    ]
                }
            ],
            'body': []
        }
        
        result = emitter.emit(ir)
        
        assert 'procedure Swap' in result.code
        assert 'var A: LongInt' in result.code
        assert 'var B: LongInt' in result.code
    
    def test_for_loop(self, emitter):
        """Test FOR loop generation."""
        ir = {
            'kind': 'program',
            'name': 'LoopTest',
            'body': [
                {
                    'kind': 'for_loop',
                    'variable': 'I',
                    'start': {'value': 1},
                    'end': {'value': 10},
                    'body': [
                        {'kind': 'call_statement', 'name': 'WriteLn', 'arguments': [{'kind': 'var_ref', 'name': 'I'}]}
                    ]
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        assert 'for I := 1 to 10 do' in result.code
        assert 'begin' in result.code
        assert 'end;' in result.code
    
    def test_for_downto(self, emitter):
        """Test FOR DOWNTO loop."""
        ir = {
            'kind': 'program',
            'name': 'DowntoTest',
            'body': [
                {
                    'kind': 'for_loop',
                    'variable': 'I',
                    'start': {'value': 10},
                    'end': {'value': 1},
                    'is_downto': True,
                    'body': []
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        assert 'for I := 10 downto 1 do' in result.code
    
    def test_repeat_until(self, emitter):
        """Test REPEAT-UNTIL loop."""
        ir = {
            'kind': 'program',
            'name': 'RepeatTest',
            'body': [
                {
                    'kind': 'repeat_loop',
                    'condition': {
                        'kind': 'binary_op',
                        'op': '>=',
                        'left': {'kind': 'var_ref', 'name': 'X'},
                        'right': {'value': 10}
                    },
                    'body': [
                        {'kind': 'call_statement', 'name': 'Inc', 'arguments': [{'kind': 'var_ref', 'name': 'X'}]}
                    ]
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        assert 'repeat' in result.code
        assert 'until' in result.code
    
    def test_case_statement(self, emitter):
        """Test CASE statement generation."""
        ir = {
            'kind': 'program',
            'name': 'CaseTest',
            'body': [
                {
                    'kind': 'case_statement',
                    'selector': {'kind': 'var_ref', 'name': 'Option'},
                    'cases': [
                        {'values': [{'value': 1}], 'body': [{'kind': 'call_statement', 'name': 'DoOne', 'arguments': []}]},
                        {'values': [{'value': 2}], 'body': [{'kind': 'call_statement', 'name': 'DoTwo', 'arguments': []}]}
                    ],
                    'default_body': [
                        {'kind': 'call_statement', 'name': 'DoDefault', 'arguments': []}
                    ]
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        assert 'case Option of' in result.code
        assert '1:' in result.code
        assert '2:' in result.code
        assert 'else' in result.code
    
    def test_set_expression(self, emitter):
        """Test set expression generation."""
        ir = {
            'kind': 'program',
            'name': 'SetTest',
            'body': [
                {
                    'kind': 'assignment',
                    'target': {'kind': 'var_ref', 'name': 'MySet'},
                    'value': {
                        'kind': 'set_expr',
                        'elements': [
                            {'value': 1},
                            {'value': 2},
                            {'value': 3}
                        ]
                    }
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        assert '[1, 2, 3]' in result.code
    
    def test_set_type(self, emitter):
        """Test set type generation."""
        ir = {
            'kind': 'program',
            'name': 'SetTypeTest',
            'types': [
                {
                    'kind': 'set_type',
                    'name': 'TDigits',
                    'base_type': {'name': 'i8'}
                }
            ],
            'body': []
        }
        
        result = emitter.emit(ir)
        
        assert 'TDigits = set of ShortInt;' in result.code
    
    def test_variant_record(self, emitter):
        """Test variant record generation."""
        ir = {
            'kind': 'program',
            'name': 'VariantTest',
            'types': [
                {
                    'kind': 'variant_record',
                    'name': 'TShape',
                    'fixed_fields': [
                        {'name': 'X', 'type_ref': {'name': 'f64'}},
                        {'name': 'Y', 'type_ref': {'name': 'f64'}}
                    ],
                    'tag_field': {'name': 'Kind', 'type_ref': {'name': 'i32'}},
                    'variants': [
                        {
                            'tag_values': [{'value': 1}],
                            'fields': [{'name': 'Radius', 'type_ref': {'name': 'f64'}}]
                        }
                    ]
                }
            ],
            'body': []
        }
        
        result = emitter.emit(ir)
        
        assert 'TShape = record' in result.code
        assert 'case Kind: LongInt of' in result.code
        assert 'Radius: Double;' in result.code
    
    def test_with_statement(self, emitter):
        """Test WITH statement generation."""
        ir = {
            'kind': 'program',
            'name': 'WithTest',
            'body': [
                {
                    'kind': 'with_statement',
                    'record_vars': [{'kind': 'var_ref', 'name': 'Point'}],
                    'body': [
                        {'kind': 'assignment', 'target': {'kind': 'var_ref', 'name': 'X'}, 'value': {'value': 10}}
                    ]
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        assert 'with Point do' in result.code
    
    def test_try_except(self, emitter):
        """Test TRY-EXCEPT generation."""
        ir = {
            'kind': 'program',
            'name': 'TryTest',
            'body': [
                {
                    'kind': 'try_statement',
                    'try_body': [
                        {'kind': 'call_statement', 'name': 'RiskyOperation', 'arguments': []}
                    ],
                    'handlers': [
                        {
                            'exception_type': 'Exception',
                            'variable': 'E',
                            'body': [
                                {'kind': 'call_statement', 'name': 'HandleError', 'arguments': []}
                            ]
                        }
                    ],
                    'finally_body': []
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        assert 'try' in result.code
        assert 'except' in result.code
        assert 'on E: Exception do' in result.code
    
    def test_class_type(self, emitter):
        """Test Object Pascal class generation."""
        ir = {
            'kind': 'program',
            'name': 'ClassTest',
            'types': [
                {
                    'kind': 'class_type',
                    'name': 'TVector',
                    'parent': 'TObject',
                    'fields': [
                        {'name': 'FX', 'type_ref': {'name': 'f64'}, 'visibility': 'private'},
                        {'name': 'FY', 'type_ref': {'name': 'f64'}, 'visibility': 'private'}
                    ],
                    'methods': [
                        {
                            'kind': 'method_decl',
                            'name': 'GetLength',
                            'is_function': True,
                            'return_type': {'name': 'f64'},
                            'parameters': [],
                            'visibility': 'public'
                        }
                    ],
                    'properties': [
                        {
                            'kind': 'property_decl',
                            'name': 'X',
                            'type_ref': {'name': 'f64'},
                            'getter': 'FX',
                            'setter': 'FX',
                            'visibility': 'public'
                        }
                    ]
                }
            ],
            'body': []
        }
        
        result = emitter.emit(ir)
        
        assert 'TVector = class(TObject)' in result.code
        assert 'private' in result.code
        assert 'public' in result.code
        assert 'property X' in result.code
    
    def test_type_mappings(self, emitter):
        """Test Pascal type mappings."""
        ir = {
            'kind': 'program',
            'name': 'TypesTest',
            'variables': [
                {'name': 'A', 'type_ref': {'name': 'i32'}},
                {'name': 'B', 'type_ref': {'name': 'f64'}},
                {'name': 'C', 'type_ref': {'name': 'bool'}},
                {'name': 'D', 'type_ref': {'name': 'string'}}
            ],
            'body': []
        }
        
        result = emitter.emit(ir)
        
        assert 'LongInt' in result.code
        assert 'Double' in result.code
        assert 'Boolean' in result.code
        assert 'String' in result.code
    
    def test_pointer_operations(self, emitter):
        """Test pointer operations."""
        ir = {
            'kind': 'program',
            'name': 'PointerTest',
            'types': [
                {
                    'kind': 'pointer_type',
                    'name': 'PInteger',
                    'target_type': {'name': 'i32'}
                }
            ],
            'body': [
                {
                    'kind': 'assignment',
                    'target': {'kind': 'var_ref', 'name': 'P'},
                    'value': {
                        'kind': 'address_of',
                        'operand': {'kind': 'var_ref', 'name': 'X'}
                    }
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        assert 'PInteger = ^LongInt;' in result.code
        assert '@X' in result.code
    
    def test_manifest_generation(self, emitter):
        """Test manifest is generated."""
        ir = {'kind': 'program', 'name': 'Test', 'body': []}
        
        result = emitter.emit(ir)
        
        assert 'schema' in result.manifest
        assert result.manifest['schema'] == 'stunir.manifest.targets.v1'
        assert 'ir_hash' in result.manifest
        assert 'output' in result.manifest


class TestExpressionEmission:
    """Test expression emission for both emitters."""
    
    @pytest.fixture
    def fortran(self):
        return FortranEmitter()
    
    @pytest.fixture
    def pascal(self):
        return PascalEmitter()
    
    def test_boolean_literals(self, fortran, pascal):
        """Test boolean literal emission."""
        ir = {
            'kind': 'program',
            'name': 'test',
            'body': [
                {
                    'kind': 'assignment',
                    'target': {'kind': 'var_ref', 'name': 'x'},
                    'value': {'kind': 'literal', 'value': True}
                }
            ]
        }
        
        fortran_result = fortran.emit(ir)
        pascal_result = pascal.emit(ir)
        
        assert '.TRUE.' in fortran_result.code
        assert 'True' in pascal_result.code
    
    def test_string_literals(self, fortran, pascal):
        """Test string literal emission."""
        ir = {
            'kind': 'program',
            'name': 'test',
            'body': [
                {
                    'kind': 'assignment',
                    'target': {'kind': 'var_ref', 'name': 'x'},
                    'value': {'kind': 'literal', 'value': 'hello'}
                }
            ]
        }
        
        fortran_result = fortran.emit(ir)
        pascal_result = pascal.emit(ir)
        
        assert "'hello'" in fortran_result.code
        assert "'hello'" in pascal_result.code
    
    def test_comparison_operators(self, fortran, pascal):
        """Test comparison operator emission."""
        ir = {
            'kind': 'program',
            'name': 'test',
            'body': [
                {
                    'kind': 'if_statement',
                    'condition': {
                        'kind': 'binary_op',
                        'op': '!=',
                        'left': {'kind': 'var_ref', 'name': 'a'},
                        'right': {'kind': 'var_ref', 'name': 'b'}
                    },
                    'then_body': [],
                    'else_body': []
                }
            ]
        }
        
        fortran_result = fortran.emit(ir)
        pascal_result = pascal.emit(ir)
        
        assert '/=' in fortran_result.code
        assert '<>' in pascal_result.code
    
    def test_logical_operators(self, fortran, pascal):
        """Test logical operator emission."""
        ir = {
            'kind': 'program',
            'name': 'test',
            'body': [
                {
                    'kind': 'if_statement',
                    'condition': {
                        'kind': 'binary_op',
                        'op': 'and',
                        'left': {'kind': 'var_ref', 'name': 'a'},
                        'right': {'kind': 'var_ref', 'name': 'b'}
                    },
                    'then_body': [],
                    'else_body': []
                }
            ]
        }
        
        fortran_result = fortran.emit(ir)
        pascal_result = pascal.emit(ir)
        
        assert '.AND.' in fortran_result.code
        assert 'and' in pascal_result.code


class TestMathIntrinsics:
    """Test mathematical intrinsic function emission."""
    
    @pytest.fixture
    def fortran(self):
        return FortranEmitter()
    
    @pytest.fixture
    def pascal(self):
        return PascalEmitter()
    
    def test_sin_function(self, fortran, pascal):
        """Test SIN function emission."""
        ir = {
            'kind': 'program',
            'name': 'test',
            'body': [
                {
                    'kind': 'assignment',
                    'target': {'kind': 'var_ref', 'name': 'y'},
                    'value': {
                        'kind': 'math_intrinsic',
                        'name': 'sin',
                        'arguments': [{'kind': 'var_ref', 'name': 'x'}]
                    }
                }
            ]
        }
        
        fortran_result = fortran.emit(ir)
        pascal_result = pascal.emit(ir)
        
        assert 'SIN(x)' in fortran_result.code
        assert 'Sin(x)' in pascal_result.code
    
    def test_sqrt_function(self, fortran, pascal):
        """Test SQRT function emission."""
        ir = {
            'kind': 'program',
            'name': 'test',
            'body': [
                {
                    'kind': 'assignment',
                    'target': {'kind': 'var_ref', 'name': 'y'},
                    'value': {
                        'kind': 'math_intrinsic',
                        'name': 'sqrt',
                        'arguments': [{'kind': 'var_ref', 'name': 'x'}]
                    }
                }
            ]
        }
        
        fortran_result = fortran.emit(ir)
        pascal_result = pascal.emit(ir)
        
        assert 'SQRT(x)' in fortran_result.code
        assert 'Sqrt(x)' in pascal_result.code


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
