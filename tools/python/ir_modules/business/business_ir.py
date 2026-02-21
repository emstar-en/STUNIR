#!/usr/bin/env python3
"""STUNIR Business IR - Core business programming constructs.

This module defines IR nodes for business-oriented programming
languages like COBOL and BASIC, including record structures,
file operations, and data processing constructs.

Usage:
    from ir.business.business_ir import BusinessProgram, Division, DataItem
    from ir.business import FileOrganization, DataUsage
    
    # Create a COBOL program
    prog = BusinessProgram(
        name='PAYROLL',
        divisions=[
            Division(name='IDENTIFICATION'),
            Division(name='ENVIRONMENT'),
            Division(name='DATA'),
            Division(name='PROCEDURE')
        ]
    )
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from abc import ABC
from enum import Enum


# =============================================================================
# Enumerations
# =============================================================================

class FileOrganization(Enum):
    """File organization types for COBOL."""
    SEQUENTIAL = 'sequential'
    INDEXED = 'indexed'
    RELATIVE = 'relative'
    LINE_SEQUENTIAL = 'line_sequential'


class FileAccess(Enum):
    """File access modes."""
    SEQUENTIAL = 'sequential'
    RANDOM = 'random'
    DYNAMIC = 'dynamic'


class DataUsage(Enum):
    """COBOL data usage types."""
    DISPLAY = 'display'
    BINARY = 'binary'
    COMP = 'comp'
    COMP_1 = 'comp-1'
    COMP_2 = 'comp-2'
    COMP_3 = 'comp-3'  # Packed decimal
    COMP_4 = 'comp-4'
    COMP_5 = 'comp-5'
    INDEX = 'index'
    POINTER = 'pointer'


class PictureType(Enum):
    """COBOL PICTURE clause types."""
    NUMERIC = 'numeric'         # 9
    ALPHABETIC = 'alphabetic'   # A
    ALPHANUMERIC = 'alphanumeric'  # X
    EDITED = 'edited'           # Z, *, $, etc.


class OpenMode(Enum):
    """File open modes."""
    INPUT = 'input'
    OUTPUT = 'output'
    IO = 'i-o'
    EXTEND = 'extend'


class BasicVarType(Enum):
    """BASIC variable types."""
    NUMERIC = 'numeric'
    STRING = 'string'
    INTEGER = 'integer'


# =============================================================================
# Base Class
# =============================================================================

@dataclass
class BusinessNode(ABC):
    """Base class for all business IR nodes."""
    kind: str = 'business_node'
    
    def to_dict(self) -> dict:
        """Convert node to dictionary representation."""
        result = {'kind': self.kind}
        for key, value in self.__dict__.items():
            if key != 'kind' and value is not None:
                if isinstance(value, BusinessNode):
                    result[key] = value.to_dict()
                elif isinstance(value, list):
                    result[key] = [
                        v.to_dict() if isinstance(v, BusinessNode) else 
                        (v.value if isinstance(v, Enum) else v)
                        for v in value
                    ]
                elif isinstance(value, dict):
                    result[key] = {
                        k: (v.to_dict() if isinstance(v, BusinessNode) else 
                            (v.value if isinstance(v, Enum) else v))
                        for k, v in value.items()
                    }
                elif isinstance(value, Enum):
                    result[key] = value.value
                else:
                    result[key] = value
        return result


# =============================================================================
# Program Structure
# =============================================================================

@dataclass
class BusinessProgram(BusinessNode):
    """Business program definition (COBOL program, BASIC program)."""
    name: str = ''
    kind: str = 'business_program'
    divisions: List['Division'] = field(default_factory=list)  # COBOL
    sections: List['Section'] = field(default_factory=list)
    data_items: List['DataItem'] = field(default_factory=list)
    files: List['FileDefinition'] = field(default_factory=list)
    procedures: List['Procedure'] = field(default_factory=list)
    paragraphs: List['Paragraph'] = field(default_factory=list)
    statements: List['Statement'] = field(default_factory=list)  # BASIC
    dim_statements: List['DimStatement'] = field(default_factory=list)
    def_functions: List['DefFunction'] = field(default_factory=list)
    data_statements: List['DataStatement'] = field(default_factory=list)
    line_numbers: bool = False  # BASIC line-numbered
    author: Optional[str] = None
    date_written: Optional[str] = None
    special_names: Dict[str, str] = field(default_factory=dict)


@dataclass 
class Division(BusinessNode):
    """COBOL division."""
    name: str = ''  # IDENTIFICATION, ENVIRONMENT, DATA, PROCEDURE
    kind: str = 'division'
    sections: List['Section'] = field(default_factory=list)
    paragraphs: List['Paragraph'] = field(default_factory=list)
    entries: List['Entry'] = field(default_factory=list)


@dataclass
class Section(BusinessNode):
    """COBOL section within a division."""
    name: str = ''
    kind: str = 'section'
    paragraphs: List['Paragraph'] = field(default_factory=list)
    entries: List['Entry'] = field(default_factory=list)


@dataclass
class Paragraph(BusinessNode):
    """COBOL paragraph (procedure)."""
    name: str = ''
    kind: str = 'paragraph'
    statements: List['Statement'] = field(default_factory=list)


@dataclass
class Entry(BusinessNode):
    """Generic entry in a section."""
    name: str = ''
    kind: str = 'entry'
    value: Any = None


@dataclass
class Procedure(BusinessNode):
    """BASIC subroutine/procedure."""
    name: str = ''
    kind: str = 'procedure'
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    statements: List['Statement'] = field(default_factory=list)


# =============================================================================
# Expressions
# =============================================================================

@dataclass
class Literal(BusinessNode):
    """Literal value."""
    value: Any = None
    kind: str = 'literal'
    literal_type: str = 'numeric'  # 'numeric', 'string', 'figurative'
    # Figurative constants: ZERO, SPACES, HIGH-VALUES, LOW-VALUES, QUOTES


@dataclass
class Identifier(BusinessNode):
    """Variable/field reference."""
    name: str = ''
    kind: str = 'identifier'
    qualifiers: List[str] = field(default_factory=list)  # COBOL OF/IN
    subscripts: List['Expr'] = field(default_factory=list)
    reference_mod: Optional['ReferenceMod'] = None


@dataclass
class ReferenceMod(BusinessNode):
    """COBOL reference modification (substring)."""
    start: 'Expr' = None
    kind: str = 'reference_mod'
    length: Optional['Expr'] = None


@dataclass
class BinaryExpr(BusinessNode):
    """Binary expression."""
    left: 'Expr' = None
    kind: str = 'binary_expr'
    op: str = ''  # +, -, *, /, **, AND, OR, etc.
    right: 'Expr' = None


@dataclass
class UnaryExpr(BusinessNode):
    """Unary expression."""
    op: str = ''  # NOT, -, +
    kind: str = 'unary_expr'
    operand: 'Expr' = None


@dataclass
class Condition(BusinessNode):
    """Condition expression."""
    kind: str = 'condition'
    condition_type: str = ''  # 'comparison', 'class', 'sign', 'condition_name'
    left: Optional['Expr'] = None
    op: Optional[str] = None
    right: Optional['Expr'] = None
    negated: bool = False


@dataclass
class FunctionCall(BusinessNode):
    """Function call (BASIC DEF FN, intrinsic functions)."""
    name: str = ''
    kind: str = 'function_call'
    arguments: List['Expr'] = field(default_factory=list)
    line_number: Optional[int] = None


# =============================================================================
# Statements - Control Flow
# =============================================================================

@dataclass
class IfStatement(BusinessNode):
    """IF statement."""
    condition: 'Expr' = None
    kind: str = 'if_statement'
    then_statements: List['Statement'] = field(default_factory=list)
    else_statements: List['Statement'] = field(default_factory=list)
    line_number: Optional[int] = None  # BASIC


@dataclass
class EvaluateStatement(BusinessNode):
    """COBOL EVALUATE statement (CASE/SWITCH)."""
    subjects: List['Expr'] = field(default_factory=list)
    kind: str = 'evaluate_statement'
    when_clauses: List['WhenClause'] = field(default_factory=list)
    when_other: List['Statement'] = field(default_factory=list)


@dataclass
class WhenClause(BusinessNode):
    """WHEN clause in EVALUATE."""
    conditions: List['WhenCondition'] = field(default_factory=list)
    kind: str = 'when_clause'
    statements: List['Statement'] = field(default_factory=list)


@dataclass
class WhenCondition(BusinessNode):
    """Condition in WHEN clause."""
    value: 'Expr' = None
    kind: str = 'when_condition'
    is_any: bool = False
    is_true: bool = False
    is_false: bool = False
    thru_value: Optional['Expr'] = None


@dataclass
class PerformStatement(BusinessNode):
    """COBOL PERFORM statement."""
    paragraph_name: str = ''
    kind: str = 'perform_statement'
    through: Optional[str] = None  # PERFORM ... THRU ...
    times: Optional['Expr'] = None  # PERFORM ... TIMES
    until: Optional['Expr'] = None  # PERFORM ... UNTIL
    varying: Optional['VaryingClause'] = None
    with_test: str = 'before'  # 'before' or 'after'
    inline_statements: List['Statement'] = field(default_factory=list)


@dataclass
class VaryingClause(BusinessNode):
    """VARYING clause in PERFORM."""
    identifier: str = ''
    kind: str = 'varying_clause'
    from_value: 'Expr' = None
    by_value: 'Expr' = None
    until_value: 'Expr' = None
    after_clauses: List['VaryingClause'] = field(default_factory=list)


@dataclass
class GotoStatement(BusinessNode):
    """GOTO statement (COBOL GO TO, BASIC GOTO)."""
    target: str = ''  # Paragraph name or line number
    kind: str = 'goto_statement'
    depending_on: Optional[str] = None  # GO TO ... DEPENDING ON
    targets: List[str] = field(default_factory=list)  # Multiple targets
    line_number: Optional[int] = None  # BASIC


@dataclass
class GosubStatement(BusinessNode):
    """BASIC GOSUB statement."""
    target: int = 0  # Line number
    kind: str = 'gosub_statement'
    line_number: Optional[int] = None


@dataclass
class ReturnStatement(BusinessNode):
    """BASIC RETURN statement."""
    kind: str = 'return_statement'
    line_number: Optional[int] = None


@dataclass
class ForLoop(BusinessNode):
    """FOR loop (BASIC FOR...NEXT)."""
    variable: str = ''
    kind: str = 'for_loop'
    start: 'Expr' = None
    end: 'Expr' = None
    step: Optional['Expr'] = None
    statements: List['Statement'] = field(default_factory=list)
    line_number: Optional[int] = None


@dataclass
class WhileLoop(BusinessNode):
    """WHILE loop (BASIC WHILE...WEND)."""
    condition: 'Expr' = None
    kind: str = 'while_loop'
    statements: List['Statement'] = field(default_factory=list)
    line_number: Optional[int] = None


@dataclass
class StopStatement(BusinessNode):
    """COBOL STOP RUN statement."""
    kind: str = 'stop_statement'
    run: bool = True
    literal: Optional[str] = None


@dataclass
class EndStatement(BusinessNode):
    """BASIC END statement."""
    kind: str = 'end_statement'
    line_number: Optional[int] = None


# =============================================================================
# Statements - Data Processing
# =============================================================================

@dataclass
class MoveStatement(BusinessNode):
    """COBOL MOVE statement."""
    source: 'Expr' = None
    kind: str = 'move_statement'
    destinations: List[str] = field(default_factory=list)
    corresponding: bool = False  # MOVE CORRESPONDING


@dataclass
class ComputeStatement(BusinessNode):
    """COBOL COMPUTE statement."""
    target: str = ''
    kind: str = 'compute_statement'
    expression: 'Expr' = None
    rounded: bool = False
    on_size_error: List['Statement'] = field(default_factory=list)
    not_on_size_error: List['Statement'] = field(default_factory=list)


@dataclass
class AddStatement(BusinessNode):
    """COBOL ADD statement."""
    values: List['Expr'] = field(default_factory=list)
    kind: str = 'add_statement'
    to_value: Optional[str] = None
    giving: Optional[str] = None
    rounded: bool = False
    on_size_error: List['Statement'] = field(default_factory=list)


@dataclass
class SubtractStatement(BusinessNode):
    """COBOL SUBTRACT statement."""
    values: List['Expr'] = field(default_factory=list)
    kind: str = 'subtract_statement'
    from_value: str = ''
    giving: Optional[str] = None
    rounded: bool = False
    on_size_error: List['Statement'] = field(default_factory=list)


@dataclass
class MultiplyStatement(BusinessNode):
    """COBOL MULTIPLY statement."""
    value1: 'Expr' = None
    kind: str = 'multiply_statement'
    by_value: 'Expr' = None
    giving: Optional[str] = None
    rounded: bool = False
    on_size_error: List['Statement'] = field(default_factory=list)


@dataclass
class DivideStatement(BusinessNode):
    """COBOL DIVIDE statement."""
    value1: 'Expr' = None
    kind: str = 'divide_statement'
    into_value: 'Expr' = None
    giving: Optional[str] = None
    remainder: Optional[str] = None
    rounded: bool = False
    on_size_error: List['Statement'] = field(default_factory=list)


@dataclass
class StringStatement(BusinessNode):
    """COBOL STRING statement."""
    sources: List['StringSource'] = field(default_factory=list)
    kind: str = 'string_statement'
    into: str = ''
    pointer: Optional[str] = None
    on_overflow: List['Statement'] = field(default_factory=list)


@dataclass
class StringSource(BusinessNode):
    """Source for STRING statement."""
    value: 'Expr' = None
    kind: str = 'string_source'
    delimited_by: Optional['Expr'] = None  # SIZE, SPACE, or literal


@dataclass
class UnstringStatement(BusinessNode):
    """COBOL UNSTRING statement."""
    source: str = ''
    kind: str = 'unstring_statement'
    delimiters: List['DelimiterSpec'] = field(default_factory=list)
    into_fields: List['UnstringField'] = field(default_factory=list)
    pointer: Optional[str] = None
    tallying: Optional[str] = None
    on_overflow: List['Statement'] = field(default_factory=list)


@dataclass
class DelimiterSpec(BusinessNode):
    """Delimiter specification for UNSTRING."""
    value: 'Expr' = None
    kind: str = 'delimiter_spec'
    all_delimiters: bool = False


@dataclass
class UnstringField(BusinessNode):
    """Field specification for UNSTRING INTO."""
    name: str = ''
    kind: str = 'unstring_field'
    delimiter_in: Optional[str] = None
    count_in: Optional[str] = None


@dataclass
class InspectStatement(BusinessNode):
    """COBOL INSPECT statement."""
    identifier: str = ''
    kind: str = 'inspect_statement'
    tallying: Optional['TallyingClause'] = None
    replacing: Optional['ReplacingClause'] = None
    converting: Optional['ConvertingClause'] = None


@dataclass
class TallyingClause(BusinessNode):
    """TALLYING clause for INSPECT."""
    counter: str = ''
    kind: str = 'tallying_clause'
    for_items: List['TallyItem'] = field(default_factory=list)


@dataclass
class TallyItem(BusinessNode):
    """Item to tally in INSPECT."""
    tally_type: str = ''  # 'characters', 'all', 'leading'
    kind: str = 'tally_item'
    value: Optional['Expr'] = None
    before: Optional['Expr'] = None
    after: Optional['Expr'] = None


@dataclass
class ReplacingClause(BusinessNode):
    """REPLACING clause for INSPECT."""
    items: List['ReplaceItem'] = field(default_factory=list)
    kind: str = 'replacing_clause'


@dataclass
class ReplaceItem(BusinessNode):
    """Item to replace in INSPECT."""
    replace_type: str = ''  # 'characters', 'all', 'leading', 'first'
    kind: str = 'replace_item'
    from_value: 'Expr' = None
    to_value: 'Expr' = None
    before: Optional['Expr'] = None
    after: Optional['Expr'] = None


@dataclass
class ConvertingClause(BusinessNode):
    """CONVERTING clause for INSPECT."""
    from_chars: str = ''
    kind: str = 'converting_clause'
    to_chars: str = ''
    before: Optional['Expr'] = None
    after: Optional['Expr'] = None


@dataclass
class Assignment(BusinessNode):
    """BASIC assignment statement (LET)."""
    variable: str = ''
    kind: str = 'assignment'
    value: 'Expr' = None
    line_number: Optional[int] = None


# =============================================================================
# Statements - I/O
# =============================================================================

@dataclass
class DisplayStatement(BusinessNode):
    """COBOL DISPLAY statement."""
    items: List['Expr'] = field(default_factory=list)
    kind: str = 'display_statement'
    upon: Optional[str] = None  # Device name
    with_no_advancing: bool = False


@dataclass
class AcceptStatement(BusinessNode):
    """COBOL ACCEPT statement."""
    identifier: str = ''
    kind: str = 'accept_statement'
    from_source: Optional[str] = None  # DATE, TIME, etc.


@dataclass
class BasicInputStatement(BusinessNode):
    """BASIC INPUT statement (from user)."""
    prompt: Optional[str] = None
    kind: str = 'basic_input_user'
    variables: List[str] = field(default_factory=list)
    line_number: Optional[int] = None


@dataclass
class BasicPrintStatement(BusinessNode):
    """BASIC PRINT statement."""
    items: List['PrintItem'] = field(default_factory=list)
    kind: str = 'basic_print_screen'
    line_number: Optional[int] = None


@dataclass
class PrintItem(BusinessNode):
    """Item in PRINT statement."""
    value: 'Expr' = None
    kind: str = 'print_item'
    separator: str = ''  # '', ',', ';'


@dataclass
class DataStatement(BusinessNode):
    """BASIC DATA statement."""
    values: List['Literal'] = field(default_factory=list)
    kind: str = 'data_statement'
    line_number: Optional[int] = None


@dataclass
class ReadDataStatement(BusinessNode):
    """BASIC READ statement (from DATA)."""
    variables: List[str] = field(default_factory=list)
    kind: str = 'read_data_statement'
    line_number: Optional[int] = None


@dataclass
class RestoreStatement(BusinessNode):
    """BASIC RESTORE statement."""
    target_line: Optional[int] = None
    kind: str = 'restore_statement'
    line_number: Optional[int] = None


@dataclass
class RemStatement(BusinessNode):
    """BASIC REM (comment) statement."""
    text: str = ''
    kind: str = 'rem_statement'
    line_number: Optional[int] = None


# =============================================================================
# BASIC Specific
# =============================================================================

@dataclass
class BasicVariable(BusinessNode):
    """BASIC variable (dynamic typing)."""
    name: str = ''
    kind: str = 'basic_variable'
    var_type: BasicVarType = BasicVarType.NUMERIC
    is_array: bool = False
    dimensions: List[int] = field(default_factory=list)


@dataclass
class DimStatement(BusinessNode):
    """BASIC DIM statement for arrays."""
    variable: str = ''
    kind: str = 'dim_statement'
    dimensions: List[int] = field(default_factory=list)
    line_number: Optional[int] = None


@dataclass
class DefFunction(BusinessNode):
    """BASIC DEF FN definition."""
    name: str = ''  # e.g., 'FNA'
    kind: str = 'def_function'
    parameter: Optional[str] = None
    expression: 'Expr' = None
    line_number: Optional[int] = None


# Type aliases for convenience
Expr = Union[Literal, Identifier, BinaryExpr, UnaryExpr, Condition, FunctionCall]
Statement = Union[
    IfStatement, EvaluateStatement, PerformStatement, GotoStatement,
    GosubStatement, ReturnStatement, ForLoop, WhileLoop,
    MoveStatement, ComputeStatement, AddStatement, SubtractStatement,
    MultiplyStatement, DivideStatement, StringStatement, UnstringStatement,
    InspectStatement, Assignment, DisplayStatement, AcceptStatement,
    BasicInputStatement, BasicPrintStatement, ReadDataStatement,
    RestoreStatement, RemStatement, StopStatement, EndStatement
]


# =============================================================================
# Export all public symbols
# =============================================================================

__all__ = [
    # Enumerations
    'FileOrganization', 'FileAccess', 'DataUsage', 'PictureType',
    'OpenMode', 'BasicVarType',
    # Base
    'BusinessNode',
    # Program structure
    'BusinessProgram', 'Division', 'Section', 'Paragraph', 'Entry', 'Procedure',
    # Expressions
    'Literal', 'Identifier', 'ReferenceMod', 'BinaryExpr', 'UnaryExpr',
    'Condition', 'FunctionCall',
    # Control flow
    'IfStatement', 'EvaluateStatement', 'WhenClause', 'WhenCondition',
    'PerformStatement', 'VaryingClause', 'GotoStatement', 'GosubStatement',
    'ReturnStatement', 'ForLoop', 'WhileLoop', 'StopStatement', 'EndStatement',
    # Data processing
    'MoveStatement', 'ComputeStatement', 'AddStatement', 'SubtractStatement',
    'MultiplyStatement', 'DivideStatement', 'StringStatement', 'StringSource',
    'UnstringStatement', 'DelimiterSpec', 'UnstringField', 'InspectStatement',
    'TallyingClause', 'TallyItem', 'ReplacingClause', 'ReplaceItem',
    'ConvertingClause', 'Assignment',
    # I/O
    'DisplayStatement', 'AcceptStatement', 'BasicInputStatement',
    'BasicPrintStatement', 'PrintItem', 'DataStatement', 'ReadDataStatement',
    'RestoreStatement', 'RemStatement',
    # BASIC specific
    'BasicVariable', 'DimStatement', 'DefFunction',
    # Type aliases
    'Expr', 'Statement',
]
