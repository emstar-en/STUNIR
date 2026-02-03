#!/usr/bin/env python3
"""Tests for STUNIR Business Language Emitters.

Tests cover:
- COBOL emitter (divisions, records, file handling, PERFORM)
- BASIC emitter (line numbers, control flow, I/O)
- Example programs (payroll processing, inventory management)
"""

import sys
from pathlib import Path

# Add repository root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from targets.business import COBOLEmitter, BASICEmitter, EmitterResult


class TestCOBOLEmitter:
    """Test COBOL code emitter."""
    
    @pytest.fixture
    def emitter(self):
        """Create COBOL emitter instance."""
        return COBOLEmitter()
    
    def test_emit_simple_program(self, emitter):
        """Test emitting a simple COBOL program."""
        ir = {
            'name': 'HELLO',
            'paragraphs': [
                {
                    'name': 'MAIN-PARA',
                    'statements': [
                        {'kind': 'display_statement', 'items': [
                            {'kind': 'literal', 'value': 'HELLO WORLD'}
                        ]},
                        {'kind': 'stop_statement', 'run': True}
                    ]
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        assert isinstance(result, EmitterResult)
        assert 'IDENTIFICATION DIVISION' in result.code
        assert 'PROGRAM-ID. HELLO' in result.code
        assert 'PROCEDURE DIVISION' in result.code
        assert 'DISPLAY "HELLO WORLD"' in result.code
        assert 'STOP RUN' in result.code
    
    def test_emit_four_divisions(self, emitter):
        """Test all four COBOL divisions are generated."""
        ir = {
            'name': 'TEST',
            'files': [],
            'data_items': [],
            'paragraphs': []
        }
        
        result = emitter.emit(ir)
        
        assert 'IDENTIFICATION DIVISION' in result.code
        assert 'ENVIRONMENT DIVISION' in result.code
        assert 'DATA DIVISION' in result.code
        assert 'PROCEDURE DIVISION' in result.code
    
    def test_emit_data_items(self, emitter):
        """Test data item emission."""
        ir = {
            'name': 'DATA-TEST',
            'data_items': [
                {'name': 'WS-COUNTER', 'level': 1, 'picture': {'pattern': '9(5)'}},
                {'name': 'WS-NAME', 'level': 1, 'picture': {'pattern': 'X(30)'}},
                {'name': 'WS-AMOUNT', 'level': 1, 'picture': {'pattern': '9(7)V99'}, 'value': 0},
            ],
            'paragraphs': []
        }
        
        result = emitter.emit(ir)
        
        assert 'WORKING-STORAGE SECTION' in result.code
        assert '01  WS-COUNTER PIC 9(5)' in result.code
        assert '01  WS-NAME PIC X(30)' in result.code
        assert '01  WS-AMOUNT PIC 9(7)V99 VALUE 0' in result.code
    
    def test_emit_record_structure(self, emitter):
        """Test hierarchical record structure emission."""
        ir = {
            'name': 'RECORD-TEST',
            'data_items': [
                {
                    'name': 'EMPLOYEE-RECORD',
                    'level': 1,
                    'children': [
                        {'name': 'EMP-ID', 'level': 5, 'picture': {'pattern': '9(5)'}},
                        {'name': 'EMP-NAME', 'level': 5, 'picture': {'pattern': 'X(30)'}},
                        {
                            'name': 'EMP-ADDRESS',
                            'level': 5,
                            'children': [
                                {'name': 'ADDR-STREET', 'level': 10, 'picture': {'pattern': 'X(30)'}},
                                {'name': 'ADDR-CITY', 'level': 10, 'picture': {'pattern': 'X(20)'}},
                            ]
                        }
                    ]
                }
            ],
            'paragraphs': []
        }
        
        result = emitter.emit(ir)
        
        assert '01  EMPLOYEE-RECORD' in result.code
        assert '05  EMP-ID PIC 9(5)' in result.code
        assert '05  EMP-NAME PIC X(30)' in result.code
        assert '05  EMP-ADDRESS' in result.code
        assert '10  ADDR-STREET PIC X(30)' in result.code
        assert '10  ADDR-CITY PIC X(20)' in result.code
    
    def test_emit_occurs_clause(self, emitter):
        """Test OCCURS clause emission."""
        ir = {
            'name': 'OCCURS-TEST',
            'data_items': [
                {
                    'name': 'ITEM-TABLE',
                    'level': 1,
                    'children': [
                        {
                            'name': 'ITEM-ENTRY',
                            'level': 5,
                            'picture': {'pattern': 'X(20)'},
                            'occurs': {
                                'times': 100,
                                'indexed_by': ['IDX-1']
                            }
                        }
                    ]
                }
            ],
            'paragraphs': []
        }
        
        result = emitter.emit(ir)
        
        assert 'OCCURS 100 TIMES' in result.code
        assert 'INDEXED BY IDX-1' in result.code
    
    def test_emit_file_control(self, emitter):
        """Test FILE-CONTROL emission."""
        ir = {
            'name': 'FILE-TEST',
            'files': [
                {
                    'name': 'EMPLOYEE-FILE',
                    'assign_to': 'EMPLOYEE.DAT',
                    'organization': 'indexed',
                    'access': 'sequential',
                    'record_key': 'EMP-ID',
                    'file_status': 'WS-STATUS',
                    'record': {
                        'name': 'EMPLOYEE-RECORD',
                        'level': 1,
                        'children': [
                            {'name': 'EMP-ID', 'level': 5, 'picture': {'pattern': '9(5)'}}
                        ]
                    }
                }
            ],
            'data_items': [
                {'name': 'WS-STATUS', 'level': 1, 'picture': {'pattern': 'XX'}}
            ],
            'paragraphs': []
        }
        
        result = emitter.emit(ir)
        
        assert 'INPUT-OUTPUT SECTION' in result.code
        assert 'FILE-CONTROL' in result.code
        assert 'SELECT EMPLOYEE-FILE' in result.code
        assert 'ASSIGN TO "EMPLOYEE.DAT"' in result.code
        assert 'ORGANIZATION IS INDEXED' in result.code
        assert 'RECORD KEY IS EMP-ID' in result.code
        assert 'FILE STATUS IS WS-STATUS' in result.code
        assert 'FILE SECTION' in result.code
        assert 'FD  EMPLOYEE-FILE' in result.code
    
    def test_emit_move_statement(self, emitter):
        """Test MOVE statement emission."""
        ir = {
            'name': 'MOVE-TEST',
            'paragraphs': [
                {
                    'name': 'MAIN-PARA',
                    'statements': [
                        {
                            'kind': 'move_statement',
                            'source': {'kind': 'literal', 'value': 'SPACES'},
                            'destinations': ['WS-NAME', 'WS-ADDRESS']
                        }
                    ]
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        assert 'MOVE SPACES TO WS-NAME WS-ADDRESS' in result.code
    
    def test_emit_compute_statement(self, emitter):
        """Test COMPUTE statement emission."""
        ir = {
            'name': 'COMPUTE-TEST',
            'paragraphs': [
                {
                    'name': 'CALC-PARA',
                    'statements': [
                        {
                            'kind': 'compute_statement',
                            'target': 'WS-TOTAL',
                            'expression': {
                                'kind': 'binary_expr',
                                'left': {'kind': 'identifier', 'name': 'QTY'},
                                'op': 'mul',
                                'right': {'kind': 'identifier', 'name': 'PRICE'}
                            },
                            'rounded': True
                        }
                    ]
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        assert 'COMPUTE WS-TOTAL ROUNDED = QTY * PRICE' in result.code
    
    def test_emit_perform_until(self, emitter):
        """Test PERFORM UNTIL emission."""
        ir = {
            'name': 'PERFORM-TEST',
            'paragraphs': [
                {
                    'name': 'MAIN-PARA',
                    'statements': [
                        {
                            'kind': 'perform_statement',
                            'paragraph_name': 'PROCESS-RECORDS',
                            'until': {
                                'kind': 'condition',
                                'left': {'kind': 'identifier', 'name': 'WS-EOF'},
                                'op': '=',
                                'right': {'kind': 'literal', 'value': 1}
                            }
                        }
                    ]
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        assert 'PERFORM PROCESS-RECORDS UNTIL WS-EOF = 1' in result.code
    
    def test_emit_perform_varying(self, emitter):
        """Test PERFORM VARYING emission."""
        ir = {
            'name': 'VARYING-TEST',
            'paragraphs': [
                {
                    'name': 'MAIN-PARA',
                    'statements': [
                        {
                            'kind': 'perform_statement',
                            'paragraph_name': 'PROCESS-TABLE',
                            'varying': {
                                'identifier': 'IDX',
                                'from_value': {'kind': 'literal', 'value': 1},
                                'by_value': {'kind': 'literal', 'value': 1},
                                'until_value': {
                                    'kind': 'condition',
                                    'left': {'kind': 'identifier', 'name': 'IDX'},
                                    'op': '>',
                                    'right': {'kind': 'literal', 'value': 100}
                                }
                            }
                        }
                    ]
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        assert 'PERFORM PROCESS-TABLE VARYING IDX FROM 1 BY 1 UNTIL IDX > 100' in result.code
    
    def test_emit_if_statement(self, emitter):
        """Test IF statement emission."""
        ir = {
            'name': 'IF-TEST',
            'paragraphs': [
                {
                    'name': 'MAIN-PARA',
                    'statements': [
                        {
                            'kind': 'if_statement',
                            'condition': {
                                'kind': 'condition',
                                'left': {'kind': 'identifier', 'name': 'AMOUNT'},
                                'op': '>',
                                'right': {'kind': 'literal', 'value': 1000}
                            },
                            'then_statements': [
                                {
                                    'kind': 'move_statement',
                                    'source': {'kind': 'literal', 'value': 'HIGH'},
                                    'destinations': ['WS-LEVEL']
                                }
                            ],
                            'else_statements': [
                                {
                                    'kind': 'move_statement',
                                    'source': {'kind': 'literal', 'value': 'LOW'},
                                    'destinations': ['WS-LEVEL']
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        assert 'IF AMOUNT > 1000' in result.code
        assert 'MOVE "HIGH" TO WS-LEVEL' in result.code
        assert 'ELSE' in result.code
        assert 'MOVE "LOW" TO WS-LEVEL' in result.code
        assert 'END-IF' in result.code
    
    def test_emit_evaluate_statement(self, emitter):
        """Test EVALUATE statement emission."""
        ir = {
            'name': 'EVALUATE-TEST',
            'paragraphs': [
                {
                    'name': 'MAIN-PARA',
                    'statements': [
                        {
                            'kind': 'evaluate_statement',
                            'subjects': [{'kind': 'identifier', 'name': 'WS-CODE'}],
                            'when_clauses': [
                                {
                                    'conditions': [{'kind': 'when_condition', 'value': {'kind': 'literal', 'value': 'A'}}],
                                    'statements': [
                                        {'kind': 'display_statement', 'items': [{'kind': 'literal', 'value': 'Active'}]}
                                    ]
                                },
                                {
                                    'conditions': [{'kind': 'when_condition', 'value': {'kind': 'literal', 'value': 'I'}}],
                                    'statements': [
                                        {'kind': 'display_statement', 'items': [{'kind': 'literal', 'value': 'Inactive'}]}
                                    ]
                                }
                            ],
                            'when_other': [
                                {'kind': 'display_statement', 'items': [{'kind': 'literal', 'value': 'Unknown'}]}
                            ]
                        }
                    ]
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        assert 'EVALUATE WS-CODE' in result.code
        assert 'WHEN "A"' in result.code
        assert 'WHEN "I"' in result.code
        assert 'WHEN OTHER' in result.code
        assert 'END-EVALUATE' in result.code
    
    def test_emit_file_operations(self, emitter):
        """Test file I/O statement emission."""
        ir = {
            'name': 'FILE-IO-TEST',
            'paragraphs': [
                {
                    'name': 'FILE-PARA',
                    'statements': [
                        {
                            'kind': 'open_statement',
                            'files': [{'name': 'EMPLOYEE-FILE', 'mode': 'input'}]
                        },
                        {
                            'kind': 'read_statement',
                            'file_name': 'EMPLOYEE-FILE',
                            'at_end': [
                                {'kind': 'move_statement', 'source': {'kind': 'literal', 'value': 1}, 'destinations': ['WS-EOF']}
                            ]
                        },
                        {
                            'kind': 'close_statement',
                            'files': ['EMPLOYEE-FILE']
                        }
                    ]
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        assert 'OPEN INPUT EMPLOYEE-FILE' in result.code
        assert 'READ EMPLOYEE-FILE' in result.code
        assert 'AT END' in result.code
        assert 'CLOSE EMPLOYEE-FILE' in result.code
    
    def test_manifest_generation(self, emitter):
        """Test manifest generation."""
        ir = {
            'name': 'MANIFEST-TEST',
            'files': [{'name': 'FILE1'}],
            'data_items': [{'name': 'ITEM1', 'level': 1}],
            'paragraphs': [{'name': 'PARA1', 'statements': []}]
        }
        
        result = emitter.emit(ir)
        
        assert result.manifest['schema'] == 'stunir.codegen.cobol.v1'
        assert result.manifest['program_name'] == 'MANIFEST-TEST'
        assert result.manifest['dialect'] == 'standard'
        assert result.manifest['files_count'] == 1
        assert result.manifest['data_items_count'] == 1
        assert result.manifest['paragraphs_count'] == 1
        assert 'code_hash' in result.manifest


class TestBASICEmitter:
    """Test BASIC code emitter."""
    
    @pytest.fixture
    def emitter(self):
        """Create BASIC emitter instance."""
        return BASICEmitter()
    
    def test_emit_simple_program(self, emitter):
        """Test emitting a simple BASIC program."""
        ir = {
            'name': 'HELLO',
            'statements': [
                {'kind': 'basic_print_screen', 'items': [{'value': {'kind': 'literal', 'value': 'HELLO WORLD'}}]},
            ]
        }
        
        result = emitter.emit(ir)
        
        assert hasattr(result, 'code')
        assert hasattr(result, 'manifest')
        assert 'REM HELLO' in result.code
        assert 'PRINT "HELLO WORLD"' in result.code
        assert 'END' in result.code
    
    def test_emit_line_numbers(self, emitter):
        """Test line number generation."""
        ir = {
            'name': 'LINES',
            'line_numbers': True,
            'statements': [
                {'kind': 'assignment', 'variable': 'X', 'value': {'kind': 'literal', 'value': 10}},
                {'kind': 'assignment', 'variable': 'Y', 'value': {'kind': 'literal', 'value': 20}},
            ]
        }
        
        result = emitter.emit(ir)
        lines = result.code.split('\n')
        
        # Check line numbers increment
        line_nums = [int(line.split()[0]) for line in lines if line and line[0].isdigit()]
        assert all(b - a == 10 for a, b in zip(line_nums, line_nums[1:]))
    
    def test_emit_dim_statements(self, emitter):
        """Test DIM statement emission."""
        ir = {
            'name': 'ARRAYS',
            'dim_statements': [
                {'variable': 'ITEM$', 'dimensions': [100]},
                {'variable': 'MATRIX', 'dimensions': [10, 10]},
            ],
            'statements': []
        }
        
        result = emitter.emit(ir)
        
        assert 'DIM ITEM$(100)' in result.code
        assert 'DIM MATRIX(10, 10)' in result.code
    
    def test_emit_def_functions(self, emitter):
        """Test DEF FN emission."""
        ir = {
            'name': 'FUNCTIONS',
            'def_functions': [
                {
                    'name': 'SQUARE',
                    'parameter': 'X',
                    'expression': {
                        'kind': 'binary_expr',
                        'left': {'kind': 'identifier', 'name': 'X'},
                        'op': 'mul',
                        'right': {'kind': 'identifier', 'name': 'X'}
                    }
                }
            ],
            'statements': []
        }
        
        result = emitter.emit(ir)
        
        assert 'DEF FNSQUARE(X) = (X * X)' in result.code
    
    def test_emit_data_statements(self, emitter):
        """Test DATA statement emission."""
        ir = {
            'name': 'DATA-TEST',
            'data_statements': [
                {'values': ['Apple', 1.50, 'Banana', 0.75]}
            ],
            'statements': []
        }
        
        result = emitter.emit(ir)
        
        assert 'DATA "Apple", 1.5, "Banana", 0.75' in result.code
    
    def test_emit_assignment(self, emitter):
        """Test LET statement emission."""
        ir = {
            'name': 'ASSIGN',
            'statements': [
                {'kind': 'assignment', 'variable': 'COUNT', 'value': {'kind': 'literal', 'value': 0}},
                {
                    'kind': 'assignment',
                    'variable': 'TOTAL',
                    'value': {
                        'kind': 'binary_expr',
                        'left': {'kind': 'identifier', 'name': 'A'},
                        'op': 'add',
                        'right': {'kind': 'identifier', 'name': 'B'}
                    }
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        assert 'LET COUNT = 0' in result.code
        assert 'LET TOTAL = (A + B)' in result.code
    
    def test_emit_for_loop(self, emitter):
        """Test FOR...NEXT loop emission."""
        ir = {
            'name': 'FOR-TEST',
            'statements': [
                {
                    'kind': 'for_loop',
                    'variable': 'I',
                    'start': {'kind': 'literal', 'value': 1},
                    'end': {'kind': 'literal', 'value': 10},
                    'step': {'kind': 'literal', 'value': 2},
                    'statements': [
                        {'kind': 'basic_print_screen', 'items': [{'value': {'kind': 'identifier', 'name': 'I'}}]}
                    ]
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        assert 'FOR I = 1 TO 10 STEP 2' in result.code
        assert 'PRINT I' in result.code
        assert 'NEXT I' in result.code
    
    def test_emit_while_loop(self, emitter):
        """Test WHILE...WEND loop emission."""
        ir = {
            'name': 'WHILE-TEST',
            'statements': [
                {
                    'kind': 'while_loop',
                    'condition': {
                        'kind': 'condition',
                        'left': {'kind': 'identifier', 'name': 'X'},
                        'op': '<',
                        'right': {'kind': 'literal', 'value': 100}
                    },
                    'statements': [
                        {'kind': 'assignment', 'variable': 'X', 'value': {
                            'kind': 'binary_expr',
                            'left': {'kind': 'identifier', 'name': 'X'},
                            'op': 'add',
                            'right': {'kind': 'literal', 'value': 1}
                        }}
                    ]
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        assert 'WHILE X < 100' in result.code
        assert 'WEND' in result.code
    
    def test_emit_if_single_line(self, emitter):
        """Test single-line IF emission."""
        ir = {
            'name': 'IF-TEST',
            'statements': [
                {
                    'kind': 'if_statement',
                    'condition': {
                        'kind': 'condition',
                        'left': {'kind': 'identifier', 'name': 'X'},
                        'op': '=',
                        'right': {'kind': 'literal', 'value': 1}
                    },
                    'then_statements': [
                        {'kind': 'basic_print_screen', 'items': [{'value': {'kind': 'literal', 'value': 'YES'}}]}
                    ]
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        assert 'IF X = 1 THEN PRINT "YES"' in result.code
    
    def test_emit_goto_gosub(self, emitter):
        """Test GOTO and GOSUB emission."""
        ir = {
            'name': 'JUMP-TEST',
            'statements': [
                {'kind': 'goto_statement', 'target': '100'},
                {'kind': 'gosub_statement', 'target': 500},
                {'kind': 'return_statement'}
            ]
        }
        
        result = emitter.emit(ir)
        
        assert 'GOTO 100' in result.code
        assert 'GOSUB 500' in result.code
        assert 'RETURN' in result.code
    
    def test_emit_print_input(self, emitter):
        """Test PRINT and INPUT emission."""
        ir = {
            'name': 'IO-TEST',
            'statements': [
                {'kind': 'basic_print_screen', 'items': [
                    {'value': {'kind': 'literal', 'value': 'Enter name:'}},
                ]},
                {'kind': 'basic_input_user', 'prompt': 'NAME', 'variables': ['NAME$']},
                {'kind': 'basic_print_screen', 'items': [
                    {'value': {'kind': 'literal', 'value': 'Hello, '}},
                    {'value': {'kind': 'identifier', 'name': 'NAME$'}}
                ]}
            ]
        }
        
        result = emitter.emit(ir)
        
        assert 'PRINT "Enter name:"' in result.code
        assert 'INPUT "NAME"; NAME$' in result.code
    
    def test_emit_read_data(self, emitter):
        """Test READ/DATA emission."""
        ir = {
            'name': 'READ-TEST',
            'data_statements': [
                {'values': ['John', 25, 'Jane', 30]}
            ],
            'statements': [
                {'kind': 'read_data_statement', 'variables': ['NAME$', 'AGE']},
                {'kind': 'restore_statement', 'target_line': 100}
            ]
        }
        
        result = emitter.emit(ir)
        
        assert 'DATA "John", 25, "Jane", 30' in result.code
        assert 'READ NAME$, AGE' in result.code
        assert 'RESTORE 100' in result.code
    
    def test_emit_file_operations(self, emitter):
        """Test BASIC file operations emission."""
        ir = {
            'name': 'FILE-TEST',
            'statements': [
                {'kind': 'basic_open', 'file_number': 1, 'filename': 'DATA.TXT', 'mode': 'input'},
                {'kind': 'basic_input_file', 'file_number': 1, 'variables': ['NAME$', 'VALUE']},
                {'kind': 'basic_close', 'file_numbers': [1]}
            ]
        }
        
        result = emitter.emit(ir)
        
        assert 'OPEN "DATA.TXT" FOR INPUT AS #1' in result.code
        assert 'INPUT #1, NAME$, VALUE' in result.code
        assert 'CLOSE #1' in result.code
    
    def test_emit_rem_comments(self, emitter):
        """Test REM comment emission."""
        ir = {
            'name': 'COMMENTS',
            'statements': [
                {'kind': 'rem_statement', 'text': 'This is a comment'},
                {'kind': 'assignment', 'variable': 'X', 'value': {'kind': 'literal', 'value': 1}}
            ]
        }
        
        result = emitter.emit(ir)
        
        assert 'REM This is a comment' in result.code
    
    def test_emit_array_access(self, emitter):
        """Test array subscript emission."""
        ir = {
            'name': 'ARRAY-ACCESS',
            'dim_statements': [
                {'variable': 'ITEMS', 'dimensions': [10]}
            ],
            'statements': [
                {
                    'kind': 'assignment',
                    'variable': 'ITEMS(5)',
                    'value': {'kind': 'literal', 'value': 100}
                },
                {
                    'kind': 'basic_print_screen',
                    'items': [{
                        'value': {
                            'kind': 'identifier',
                            'name': 'ITEMS',
                            'subscripts': [{'kind': 'identifier', 'name': 'I'}]
                        }
                    }]
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        assert 'DIM ITEMS(10)' in result.code
        assert 'PRINT ITEMS(I)' in result.code
    
    def test_manifest_generation(self, emitter):
        """Test manifest generation."""
        ir = {
            'name': 'MANIFEST-TEST',
            'dim_statements': [{'variable': 'A', 'dimensions': [10]}],
            'def_functions': [{'name': 'X', 'parameter': 'N', 'expression': {'kind': 'literal', 'value': 1}}],
            'data_statements': [{'values': [1, 2, 3]}],
            'statements': [{'kind': 'end_statement'}]
        }
        
        result = emitter.emit(ir)
        
        assert result.manifest['schema'] == 'stunir.codegen.basic.v1'
        assert result.manifest['program_name'] == 'MANIFEST-TEST'
        assert result.manifest['dim_count'] == 1
        assert result.manifest['def_fn_count'] == 1
        assert result.manifest['data_count'] == 1
        assert result.manifest['statement_count'] == 1
        assert 'code_hash' in result.manifest


class TestExamplePrograms:
    """Test complete example programs."""
    
    def test_cobol_payroll_program(self):
        """Test complete COBOL payroll program."""
        emitter = COBOLEmitter()
        
        ir = {
            'name': 'PAYROLL',
            'author': 'STUNIR',
            'files': [
                {
                    'name': 'EMPLOYEE-FILE',
                    'assign_to': 'EMPLOYEE.DAT',
                    'organization': 'indexed',
                    'access': 'sequential',
                    'record_key': 'EMP-ID',
                    'file_status': 'WS-FILE-STATUS',
                    'record': {
                        'name': 'EMPLOYEE-RECORD',
                        'level': 1,
                        'children': [
                            {'name': 'EMP-ID', 'level': 5, 'picture': {'pattern': '9(5)'}},
                            {'name': 'EMP-NAME', 'level': 5, 'picture': {'pattern': 'X(30)'}},
                            {'name': 'EMP-SALARY', 'level': 5, 'picture': {'pattern': '9(7)V99'}},
                        ]
                    }
                }
            ],
            'data_items': [
                {'name': 'WS-FILE-STATUS', 'level': 1, 'picture': {'pattern': 'XX'}},
                {'name': 'WS-EOF', 'level': 1, 'picture': {'pattern': '9'}, 'value': 0},
                {'name': 'WS-TOTAL-SALARY', 'level': 1, 'picture': {'pattern': '9(10)V99'}, 'value': 0},
            ],
            'paragraphs': [
                {
                    'name': 'MAIN-PARA',
                    'statements': [
                        {'kind': 'open_statement', 'files': [{'name': 'EMPLOYEE-FILE', 'mode': 'input'}]},
                        {
                            'kind': 'perform_statement',
                            'paragraph_name': 'PROCESS-RECORDS',
                            'until': {
                                'kind': 'condition',
                                'left': {'kind': 'identifier', 'name': 'WS-EOF'},
                                'op': '=',
                                'right': {'kind': 'literal', 'value': 1}
                            }
                        },
                        {'kind': 'close_statement', 'files': ['EMPLOYEE-FILE']},
                        {
                            'kind': 'display_statement',
                            'items': [
                                {'kind': 'literal', 'value': 'TOTAL SALARY: '},
                                {'kind': 'identifier', 'name': 'WS-TOTAL-SALARY'}
                            ]
                        },
                    ]
                },
                {
                    'name': 'PROCESS-RECORDS',
                    'statements': [
                        {
                            'kind': 'read_statement',
                            'file_name': 'EMPLOYEE-FILE',
                            'at_end': [
                                {'kind': 'move_statement', 'source': {'kind': 'literal', 'value': 1}, 'destinations': ['WS-EOF']}
                            ],
                            'not_at_end': [
                                {'kind': 'add_statement', 'values': [{'kind': 'identifier', 'name': 'EMP-SALARY'}], 'to_value': 'WS-TOTAL-SALARY'}
                            ]
                        }
                    ]
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        # Verify structure
        assert 'PROGRAM-ID. PAYROLL' in result.code
        assert 'AUTHOR. STUNIR' in result.code
        assert 'SELECT EMPLOYEE-FILE' in result.code
        assert 'FD  EMPLOYEE-FILE' in result.code
        assert 'PERFORM PROCESS-RECORDS' in result.code
        assert 'READ EMPLOYEE-FILE' in result.code
        assert 'ADD EMP-SALARY TO WS-TOTAL-SALARY' in result.code
    
    def test_basic_inventory_program(self):
        """Test complete BASIC inventory program."""
        emitter = BASICEmitter()
        
        ir = {
            'name': 'INVENTORY',
            'line_numbers': True,
            'dim_statements': [
                {'variable': 'ITEM$', 'dimensions': [100]},
                {'variable': 'QTY', 'dimensions': [100]},
                {'variable': 'PRICE', 'dimensions': [100]},
            ],
            'def_functions': [
                {
                    'name': 'VALUE',
                    'parameter': 'I',
                    'expression': {
                        'kind': 'binary_expr',
                        'op': 'mul',
                        'left': {'kind': 'identifier', 'name': 'QTY', 'subscripts': [{'kind': 'identifier', 'name': 'I'}]},
                        'right': {'kind': 'identifier', 'name': 'PRICE', 'subscripts': [{'kind': 'identifier', 'name': 'I'}]}
                    }
                }
            ],
            'statements': [
                {'kind': 'assignment', 'variable': 'COUNT', 'value': {'kind': 'literal', 'value': 0}},
                {'kind': 'rem_statement', 'text': 'MAIN MENU'},
                {'kind': 'basic_print_screen', 'items': [{'value': {'kind': 'literal', 'value': 'INVENTORY SYSTEM'}}]},
                {'kind': 'basic_print_screen', 'items': [{'value': {'kind': 'literal', 'value': '1. ADD ITEM'}}]},
                {'kind': 'basic_print_screen', 'items': [{'value': {'kind': 'literal', 'value': '2. LIST ITEMS'}}]},
                {'kind': 'basic_print_screen', 'items': [{'value': {'kind': 'literal', 'value': '3. EXIT'}}]},
                {'kind': 'basic_input_user', 'prompt': 'CHOICE', 'variables': ['C']},
                {
                    'kind': 'if_statement',
                    'condition': {'kind': 'condition', 'left': {'kind': 'identifier', 'name': 'C'}, 'op': '=', 'right': {'kind': 'literal', 'value': 3}},
                    'then_statements': [{'kind': 'end_statement'}]
                },
                {'kind': 'goto_statement', 'target': '30'},
            ]
        }
        
        result = emitter.emit(ir)
        
        # Verify structure
        assert 'REM INVENTORY' in result.code
        assert 'DIM ITEM$(100)' in result.code
        assert 'DIM QTY(100)' in result.code
        assert 'DEF FNVALUE(I)' in result.code
        assert 'LET COUNT = 0' in result.code
        assert 'PRINT "INVENTORY SYSTEM"' in result.code
        assert 'INPUT "CHOICE"; C' in result.code
        assert 'GOTO 30' in result.code


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_cobol_empty_program(self):
        """Test COBOL emitter with minimal program."""
        emitter = COBOLEmitter()
        ir = {'name': 'EMPTY'}
        
        result = emitter.emit(ir)
        
        assert 'PROGRAM-ID. EMPTY' in result.code
        assert 'STOP RUN' in result.code
    
    def test_basic_empty_program(self):
        """Test BASIC emitter with minimal program."""
        emitter = BASICEmitter()
        ir = {'name': 'EMPTY', 'statements': []}
        
        result = emitter.emit(ir)
        
        assert 'REM EMPTY' in result.code
    
    def test_cobol_special_characters(self):
        """Test handling of special characters in literals."""
        emitter = COBOLEmitter()
        ir = {
            'name': 'SPECIAL',
            'paragraphs': [
                {
                    'name': 'MAIN-PARA',
                    'statements': [
                        {'kind': 'display_statement', 'items': [
                            {'kind': 'literal', 'value': 'Hello "World"'}
                        ]}
                    ]
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        # Should escape quotes
        assert 'DISPLAY "Hello ""World"""' in result.code
    
    def test_basic_special_characters(self):
        """Test handling of special characters in BASIC."""
        emitter = BASICEmitter()
        ir = {
            'name': 'SPECIAL',
            'statements': [
                {'kind': 'basic_print_screen', 'items': [
                    {'value': {'kind': 'literal', 'value': 'Say "Hello"'}}
                ]}
            ]
        }
        
        result = emitter.emit(ir)
        
        # Should escape quotes
        assert 'PRINT "Say ""Hello"""' in result.code
    
    def test_cobol_figurative_constants(self):
        """Test COBOL figurative constants."""
        emitter = COBOLEmitter()
        ir = {
            'name': 'FIG-CONST',
            'paragraphs': [
                {
                    'name': 'MAIN-PARA',
                    'statements': [
                        {'kind': 'move_statement', 'source': {'kind': 'literal', 'value': 'SPACES'}, 'destinations': ['WS-NAME']},
                        {'kind': 'move_statement', 'source': {'kind': 'literal', 'value': 'ZEROS'}, 'destinations': ['WS-AMOUNT']}
                    ]
                }
            ]
        }
        
        result = emitter.emit(ir)
        
        assert 'MOVE SPACES TO WS-NAME' in result.code
        assert 'MOVE ZEROS TO WS-AMOUNT' in result.code
    
    def test_basic_boolean_values(self):
        """Test BASIC boolean value handling."""
        emitter = BASICEmitter()
        ir = {
            'name': 'BOOL',
            'statements': [
                {'kind': 'assignment', 'variable': 'FLAG', 'value': {'kind': 'literal', 'value': True}}
            ]
        }
        
        result = emitter.emit(ir)
        
        # BASIC uses -1 for TRUE
        assert 'LET FLAG = -1' in result.code


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
