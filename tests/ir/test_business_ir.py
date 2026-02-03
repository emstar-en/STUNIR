#!/usr/bin/env python3
"""Tests for STUNIR Business IR.

Tests cover:
- BusinessNode base class and to_dict()
- Record structures with level numbers
- PICTURE clause parsing
- File definitions and operations
- Data processing statements
- BASIC-specific constructs
"""

import sys
from pathlib import Path

# Add repository root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from ir.business import (
    # Enumerations
    FileOrganization, FileAccess, DataUsage, PictureType, OpenMode,
    # Base
    BusinessNode, BusinessProgram, Division, Section, Paragraph,
    # Expressions
    Literal, Identifier, BinaryExpr, UnaryExpr, Condition,
    # Control flow
    IfStatement, EvaluateStatement, WhenClause, PerformStatement,
    VaryingClause, GotoStatement, ForLoop, WhileLoop,
    # Data processing
    MoveStatement, ComputeStatement, AddStatement, Assignment,
    # I/O
    DisplayStatement, BasicPrintStatement, PrintItem, DataStatement,
    # BASIC specific
    DimStatement, DefFunction,
)
from ir.business.records import (
    PictureClause, OccursClause, SortKey, DataItem, RecordStructure,
    ConditionName, create_numeric_field, create_alphanumeric_field,
)
from ir.business.files import (
    FileDefinition, FileControl, AlternateKey,
    OpenStatement, OpenFile, ReadStatement, WriteStatement,
    BasicOpenStatement, BasicCloseStatement,
)


class TestBusinessNode:
    """Test BusinessNode base class."""
    
    def test_to_dict_simple(self):
        """Test simple to_dict conversion."""
        lit = Literal(value=42, literal_type='numeric')
        result = lit.to_dict()
        
        assert result['kind'] == 'literal'
        assert result['value'] == 42
        assert result['literal_type'] == 'numeric'
    
    def test_to_dict_nested(self):
        """Test nested to_dict conversion."""
        expr = BinaryExpr(
            left=Literal(value=10),
            op='add',
            right=Literal(value=20)
        )
        result = expr.to_dict()
        
        assert result['kind'] == 'binary_expr'
        assert result['left']['kind'] == 'literal'
        assert result['left']['value'] == 10
        assert result['right']['value'] == 20
    
    def test_to_dict_list(self):
        """Test list handling in to_dict."""
        stmt = DisplayStatement(items=[
            Literal(value='Hello'),
            Identifier(name='WS-NAME')
        ])
        result = stmt.to_dict()
        
        assert len(result['items']) == 2
        assert result['items'][0]['value'] == 'Hello'
        assert result['items'][1]['name'] == 'WS-NAME'
    
    def test_to_dict_enum(self):
        """Test enum handling in to_dict."""
        file = FileDefinition(
            name='TEST-FILE',
            organization=FileOrganization.INDEXED,
            access=FileAccess.RANDOM
        )
        result = file.to_dict()
        
        assert result['organization'] == 'indexed'
        assert result['access'] == 'random'


class TestPictureClause:
    """Test PICTURE clause functionality."""
    
    def test_numeric_picture(self):
        """Test numeric PICTURE clause."""
        pic = PictureClause(pattern='9(5)')
        
        assert pic.data_type == PictureType.NUMERIC
        assert pic.size == 5
        assert pic.decimal_places == 0
        assert not pic.is_signed
    
    def test_numeric_with_decimals(self):
        """Test numeric with decimal places."""
        pic = PictureClause(pattern='9(5)V99')
        
        assert pic.data_type == PictureType.NUMERIC
        assert pic.size == 7  # 5 + 2 (V doesn't count)
        assert pic.decimal_places == 2
    
    def test_signed_numeric(self):
        """Test signed numeric PICTURE."""
        pic = PictureClause(pattern='S9(7)V99')
        
        assert pic.is_signed
        assert pic.decimal_places == 2
    
    def test_alphanumeric(self):
        """Test alphanumeric PICTURE clause."""
        pic = PictureClause(pattern='X(20)')
        
        assert pic.data_type == PictureType.ALPHANUMERIC
        assert pic.size == 20
    
    def test_alphabetic(self):
        """Test alphabetic PICTURE clause."""
        pic = PictureClause(pattern='A(10)')
        
        assert pic.data_type == PictureType.ALPHABETIC
        assert pic.size == 10
    
    def test_edited_numeric(self):
        """Test edited numeric PICTURE."""
        pic = PictureClause(pattern='ZZZ,ZZ9.99')
        
        assert pic.data_type == PictureType.EDITED
        assert pic.size == 10  # Z*3 + , + Z*2 + 9 + . + 9*2
    
    def test_expand_pattern(self):
        """Test pattern expansion."""
        pic = PictureClause(pattern='9(5)V9(2)')
        expanded = pic.expand_pattern()
        
        assert expanded == '99999V99'


class TestRecordStructure:
    """Test record structure functionality."""
    
    def test_simple_record(self):
        """Test simple record structure."""
        record = RecordStructure(
            name='EMPLOYEE-RECORD',
            level=1,
            fields=[
                DataItem(name='EMP-ID', level=5, picture=PictureClause(pattern='9(5)')),
                DataItem(name='EMP-NAME', level=5, picture=PictureClause(pattern='X(30)')),
            ]
        )
        
        assert record.name == 'EMPLOYEE-RECORD'
        assert len(record.fields) == 2
        assert record.fields[0].name == 'EMP-ID'
    
    def test_get_field(self):
        """Test finding field by name."""
        record = RecordStructure(
            name='TEST-RECORD',
            fields=[
                DataItem(name='FIELD-A', level=5),
                DataItem(name='FIELD-B', level=5, children=[
                    DataItem(name='FIELD-B1', level=10),
                    DataItem(name='FIELD-B2', level=10),
                ]),
            ]
        )
        
        # Direct field
        field_a = record.get_field('FIELD-A')
        assert field_a is not None
        assert field_a.name == 'FIELD-A'
        
        # Nested field
        field_b2 = record.get_field('FIELD-B2')
        assert field_b2 is not None
        assert field_b2.name == 'FIELD-B2'
        
        # Non-existent
        assert record.get_field('NONEXISTENT') is None
    
    def test_get_all_fields(self):
        """Test getting all fields flattened."""
        record = RecordStructure(
            name='TEST-RECORD',
            fields=[
                DataItem(name='A', level=5),
                DataItem(name='B', level=5, children=[
                    DataItem(name='B1', level=10),
                    DataItem(name='B2', level=10),
                ]),
            ]
        )
        
        all_fields = record.get_all_fields()
        assert len(all_fields) == 4
        names = [f.name for f in all_fields]
        assert 'A' in names
        assert 'B' in names
        assert 'B1' in names
        assert 'B2' in names
    
    def test_nested_record(self):
        """Test hierarchical record structure."""
        record = RecordStructure(
            name='CUSTOMER-RECORD',
            level=1,
            fields=[
                DataItem(name='CUST-ID', level=5, picture=PictureClause(pattern='9(8)')),
                DataItem(
                    name='CUST-ADDRESS',
                    level=5,
                    children=[
                        DataItem(name='ADDR-STREET', level=10, picture=PictureClause(pattern='X(30)')),
                        DataItem(name='ADDR-CITY', level=10, picture=PictureClause(pattern='X(20)')),
                        DataItem(name='ADDR-STATE', level=10, picture=PictureClause(pattern='XX')),
                        DataItem(name='ADDR-ZIP', level=10, picture=PictureClause(pattern='9(5)')),
                    ]
                ),
            ]
        )
        
        result = record.to_dict()
        assert result['kind'] == 'record_structure'
        assert len(result['fields']) == 2
        assert len(result['fields'][1]['children']) == 4


class TestOccursClause:
    """Test OCCURS clause (arrays)."""
    
    def test_fixed_occurs(self):
        """Test fixed-size OCCURS."""
        occurs = OccursClause(times=10)
        result = occurs.to_dict()
        
        assert result['times'] == 10
    
    def test_variable_occurs(self):
        """Test variable-length OCCURS."""
        occurs = OccursClause(
            min_times=1,
            max_times=100,
            depending_on='ITEM-COUNT'
        )
        result = occurs.to_dict()
        
        assert result['min_times'] == 1
        assert result['max_times'] == 100
        assert result['depending_on'] == 'ITEM-COUNT'
    
    def test_indexed_occurs(self):
        """Test OCCURS with indexes and keys."""
        occurs = OccursClause(
            times=50,
            indexed_by=['IDX-1', 'IDX-2'],
            keys=[SortKey(name='ITEM-KEY', ascending=True)]
        )
        result = occurs.to_dict()
        
        assert result['indexed_by'] == ['IDX-1', 'IDX-2']
        assert len(result['keys']) == 1
        assert result['keys'][0]['ascending'] == True


class TestFileDefinition:
    """Test file definition classes."""
    
    def test_sequential_file(self):
        """Test sequential file definition."""
        file = FileDefinition(
            name='REPORT-FILE',
            organization=FileOrganization.SEQUENTIAL,
            access=FileAccess.SEQUENTIAL
        )
        result = file.to_dict()
        
        assert result['name'] == 'REPORT-FILE'
        assert result['organization'] == 'sequential'
    
    def test_indexed_file(self):
        """Test indexed file definition."""
        file = FileDefinition(
            name='CUSTOMER-FILE',
            organization=FileOrganization.INDEXED,
            access=FileAccess.DYNAMIC,
            record_key='CUST-ID',
            alternate_keys=[
                AlternateKey(name='CUST-NAME', with_duplicates=True)
            ],
            status='WS-FILE-STATUS'
        )
        result = file.to_dict()
        
        assert result['organization'] == 'indexed'
        assert result['access'] == 'dynamic'
        assert result['record_key'] == 'CUST-ID'
        assert len(result['alternate_keys']) == 1
        assert result['alternate_keys'][0]['with_duplicates'] == True
    
    def test_file_control(self):
        """Test FILE-CONTROL entry."""
        fc = FileControl(
            select_name='EMPLOYEE-FILE',
            assign_to='EMPLOYEE.DAT',
            organization=FileOrganization.INDEXED,
            access=FileAccess.SEQUENTIAL,
            record_key='EMP-ID',
            file_status='WS-STATUS'
        )
        result = fc.to_dict()
        
        assert result['select_name'] == 'EMPLOYEE-FILE'
        assert result['assign_to'] == 'EMPLOYEE.DAT'


class TestStatements:
    """Test statement classes."""
    
    def test_move_statement(self):
        """Test MOVE statement."""
        stmt = MoveStatement(
            source=Literal(value='SPACES'),
            destinations=['WS-NAME', 'WS-ADDRESS']
        )
        result = stmt.to_dict()
        
        assert result['kind'] == 'move_statement'
        assert len(result['destinations']) == 2
    
    def test_compute_statement(self):
        """Test COMPUTE statement."""
        stmt = ComputeStatement(
            target='WS-RESULT',
            expression=BinaryExpr(
                left=Identifier(name='A'),
                op='mul',
                right=Identifier(name='B')
            ),
            rounded=True
        )
        result = stmt.to_dict()
        
        assert result['kind'] == 'compute_statement'
        assert result['target'] == 'WS-RESULT'
        assert result['rounded'] == True
    
    def test_perform_statement(self):
        """Test PERFORM statement."""
        stmt = PerformStatement(
            paragraph_name='PROCESS-RECORDS',
            until=Condition(
                left=Identifier(name='WS-EOF'),
                op='eq',
                right=Literal(value=1)
            )
        )
        result = stmt.to_dict()
        
        assert result['kind'] == 'perform_statement'
        assert result['paragraph_name'] == 'PROCESS-RECORDS'
        assert result['until']['kind'] == 'condition'
    
    def test_perform_varying(self):
        """Test PERFORM VARYING statement."""
        stmt = PerformStatement(
            paragraph_name='PROCESS-TABLE',
            varying=VaryingClause(
                identifier='IDX',
                from_value=Literal(value=1),
                by_value=Literal(value=1),
                until_value=Condition(
                    left=Identifier(name='IDX'),
                    op='gt',
                    right=Literal(value=100)
                )
            )
        )
        result = stmt.to_dict()
        
        assert result['varying']['identifier'] == 'IDX'
        assert result['varying']['from_value']['value'] == 1
    
    def test_if_statement(self):
        """Test IF statement."""
        stmt = IfStatement(
            condition=Condition(
                left=Identifier(name='AMOUNT'),
                op='gt',
                right=Literal(value=1000)
            ),
            then_statements=[
                MoveStatement(
                    source=Literal(value='HIGH'),
                    destinations=['WS-CATEGORY']
                )
            ],
            else_statements=[
                MoveStatement(
                    source=Literal(value='LOW'),
                    destinations=['WS-CATEGORY']
                )
            ]
        )
        result = stmt.to_dict()
        
        assert result['kind'] == 'if_statement'
        assert len(result['then_statements']) == 1
        assert len(result['else_statements']) == 1
    
    def test_evaluate_statement(self):
        """Test EVALUATE statement."""
        stmt = EvaluateStatement(
            subjects=[Identifier(name='WS-CODE')],
            when_clauses=[
                WhenClause(
                    conditions=[{'kind': 'when_condition', 'value': Literal(value='A')}],
                    statements=[DisplayStatement(items=[Literal(value='Active')])]
                ),
                WhenClause(
                    conditions=[{'kind': 'when_condition', 'value': Literal(value='I')}],
                    statements=[DisplayStatement(items=[Literal(value='Inactive')])]
                ),
            ],
            when_other=[DisplayStatement(items=[Literal(value='Unknown')])]
        )
        result = stmt.to_dict()
        
        assert result['kind'] == 'evaluate_statement'
        assert len(result['when_clauses']) == 2
        assert len(result['when_other']) == 1


class TestBASICConstructs:
    """Test BASIC-specific IR constructs."""
    
    def test_dim_statement(self):
        """Test DIM statement."""
        stmt = DimStatement(
            variable='ITEM$',
            dimensions=[100]
        )
        result = stmt.to_dict()
        
        assert result['kind'] == 'dim_statement'
        assert result['variable'] == 'ITEM$'
        assert result['dimensions'] == [100]
    
    def test_multi_dimensional_array(self):
        """Test multi-dimensional DIM."""
        stmt = DimStatement(
            variable='MATRIX',
            dimensions=[10, 10]
        )
        result = stmt.to_dict()
        
        assert result['dimensions'] == [10, 10]
    
    def test_def_function(self):
        """Test DEF FN statement."""
        stmt = DefFunction(
            name='VALUE',
            parameter='I',
            expression=BinaryExpr(
                left=Identifier(name='QTY', subscripts=[Identifier(name='I')]),
                op='mul',
                right=Identifier(name='PRICE', subscripts=[Identifier(name='I')])
            )
        )
        result = stmt.to_dict()
        
        assert result['kind'] == 'def_function'
        assert result['name'] == 'VALUE'
        assert result['parameter'] == 'I'
    
    def test_for_loop(self):
        """Test FOR...NEXT loop."""
        stmt = ForLoop(
            variable='I',
            start=Literal(value=1),
            end=Literal(value=10),
            step=Literal(value=2),
            statements=[
                BasicPrintStatement(items=[
                    PrintItem(value=Identifier(name='I'))
                ])
            ]
        )
        result = stmt.to_dict()
        
        assert result['kind'] == 'for_loop'
        assert result['variable'] == 'I'
        assert result['step']['value'] == 2
    
    def test_while_loop(self):
        """Test WHILE...WEND loop."""
        stmt = WhileLoop(
            condition=Condition(
                left=Identifier(name='X'),
                op='lt',
                right=Literal(value=100)
            ),
            statements=[
                Assignment(variable='X', value=BinaryExpr(
                    left=Identifier(name='X'),
                    op='add',
                    right=Literal(value=1)
                ))
            ]
        )
        result = stmt.to_dict()
        
        assert result['kind'] == 'while_loop'
    
    def test_gosub(self):
        """Test GOSUB statement."""
        stmt = GotoStatement(target='500')
        result = stmt.to_dict()
        
        assert result['target'] == '500'
    
    def test_data_statement(self):
        """Test DATA statement."""
        stmt = DataStatement(
            values=[
                Literal(value='Apple'),
                Literal(value=1.50),
                Literal(value='Banana'),
                Literal(value=0.75)
            ]
        )
        result = stmt.to_dict()
        
        assert result['kind'] == 'data_statement'
        assert len(result['values']) == 4


class TestBasicFileOperations:
    """Test BASIC file operation classes."""
    
    def test_basic_open(self):
        """Test BASIC OPEN statement."""
        stmt = BasicOpenStatement(
            file_number=1,
            filename='DATA.TXT',
            mode='input'
        )
        result = stmt.to_dict()
        
        assert result['kind'] == 'basic_open'
        assert result['file_number'] == 1
        assert result['filename'] == 'DATA.TXT'
    
    def test_basic_random_access(self):
        """Test BASIC random access file."""
        stmt = BasicOpenStatement(
            file_number=2,
            filename='RECORDS.DAT',
            mode='random',
            record_length=80
        )
        result = stmt.to_dict()
        
        assert result['mode'] == 'random'
        assert result['record_length'] == 80


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_numeric_field(self):
        """Test create_numeric_field utility."""
        field = create_numeric_field('AMOUNT', 5, 9, 2, True)
        
        assert field.name == 'AMOUNT'
        assert field.level == 5
        assert field.picture.pattern == 'S9(7)V9(2)'
    
    def test_create_alphanumeric_field(self):
        """Test create_alphanumeric_field utility."""
        field = create_alphanumeric_field('DESCRIPTION', 10, 50)
        
        assert field.name == 'DESCRIPTION'
        assert field.level == 10
        assert field.picture.pattern == 'X(50)'


class TestBusinessProgram:
    """Test complete business program structure."""
    
    def test_cobol_program_structure(self):
        """Test COBOL program structure."""
        program = BusinessProgram(
            name='PAYROLL',
            author='Test Author',
            divisions=[
                Division(name='IDENTIFICATION'),
                Division(name='ENVIRONMENT'),
                Division(name='DATA'),
                Division(name='PROCEDURE'),
            ],
            files=[
                FileDefinition(
                    name='EMPLOYEE-FILE',
                    organization=FileOrganization.INDEXED
                )
            ],
            data_items=[
                DataItem(name='WS-COUNTER', level=1, picture=PictureClause(pattern='9(5)'))
            ],
            paragraphs=[
                Paragraph(
                    name='MAIN-PARA',
                    statements=[
                        DisplayStatement(items=[Literal(value='Hello COBOL')])
                    ]
                )
            ]
        )
        
        result = program.to_dict()
        
        assert result['kind'] == 'business_program'
        assert result['name'] == 'PAYROLL'
        assert len(result['divisions']) == 4
        assert len(result['files']) == 1
        assert len(result['paragraphs']) == 1
    
    def test_basic_program_structure(self):
        """Test BASIC program structure."""
        program = BusinessProgram(
            name='INVENTORY',
            line_numbers=True,
            dim_statements=[
                DimStatement(variable='ITEM$', dimensions=[100]),
                DimStatement(variable='PRICE', dimensions=[100]),
            ],
            def_functions=[
                DefFunction(
                    name='TOTAL',
                    parameter='I',
                    expression=Literal(value=0)
                )
            ],
            statements=[
                Assignment(variable='COUNT', value=Literal(value=0)),
                ForLoop(
                    variable='I',
                    start=Literal(value=1),
                    end=Literal(value=10),
                    statements=[]
                )
            ]
        )
        
        result = program.to_dict()
        
        assert result['line_numbers'] == True
        assert len(result['dim_statements']) == 2
        assert len(result['def_functions']) == 1
        assert len(result['statements']) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
