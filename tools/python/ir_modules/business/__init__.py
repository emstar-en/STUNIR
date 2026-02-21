#!/usr/bin/env python3
"""STUNIR Business IR Package.

Provides intermediate representation for business-oriented programming
languages including COBOL and BASIC.

Features:
- Record structures with hierarchical level numbers
- PICTURE clauses for data formatting
- File operations (sequential, indexed, relative)
- Data processing statements
- Control flow constructs

Usage:
    from ir.business import (
        # Program structure
        BusinessProgram, Division, Section, Paragraph,
        # Enumerations
        FileOrganization, FileAccess, DataUsage, PictureType,
        # Records
        RecordStructure, DataItem, PictureClause, OccursClause,
        # Files
        FileDefinition, FileControl, OpenStatement, ReadStatement,
        # Statements
        MoveStatement, ComputeStatement, PerformStatement,
    )
"""

# Import from business_ir module
from .business_ir import (
    # Enumerations
    FileOrganization,
    FileAccess,
    DataUsage,
    PictureType,
    OpenMode,
    BasicVarType,
    # Base class
    BusinessNode,
    # Program structure
    BusinessProgram,
    Division,
    Section,
    Paragraph,
    Entry,
    Procedure,
    # Expressions
    Literal,
    Identifier,
    ReferenceMod,
    BinaryExpr,
    UnaryExpr,
    Condition,
    FunctionCall,
    # Control flow
    IfStatement,
    EvaluateStatement,
    WhenClause,
    WhenCondition,
    PerformStatement,
    VaryingClause,
    GotoStatement,
    GosubStatement,
    ReturnStatement,
    ForLoop,
    WhileLoop,
    StopStatement,
    EndStatement,
    # Data processing
    MoveStatement,
    ComputeStatement,
    AddStatement,
    SubtractStatement,
    MultiplyStatement,
    DivideStatement,
    StringStatement,
    StringSource,
    UnstringStatement,
    DelimiterSpec,
    UnstringField,
    InspectStatement,
    TallyingClause,
    TallyItem,
    ReplacingClause,
    ReplaceItem,
    ConvertingClause,
    Assignment,
    # I/O
    DisplayStatement,
    AcceptStatement,
    BasicInputStatement,
    BasicPrintStatement,
    PrintItem,
    DataStatement,
    ReadDataStatement,
    RestoreStatement,
    RemStatement,
    # BASIC specific
    BasicVariable,
    DimStatement,
    DefFunction,
)

# Import from records module
from .records import (
    PictureClause,
    OccursClause,
    SortKey,
    DataItem,
    RecordStructure,
    ConditionName,
    RenamesClause,
    CopyStatement,
    ReplaceSpec,
    WorkingStorageSection,
    LocalStorageSection,
    LinkageSection,
    # Utility functions
    parse_picture,
    create_numeric_field,
    create_alphanumeric_field,
    create_edited_field,
)

# Import from files module
from .files import (
    FileDefinition,
    AlternateKey,
    RecordContains,
    LinageClause,
    FileControl,
    OpenStatement,
    OpenFile,
    CloseStatement,
    CloseFile,
    ReadStatement,
    WriteStatement,
    AdvanceSpec,
    RewriteStatement,
    DeleteStatement,
    StartStatement,
    SortStatement,
    MergeStatement,
    ReleaseStatement,
    # BASIC file operations
    BasicOpenStatement,
    BasicCloseStatement,
    BasicInputFileStatement,
    BasicPrintFileStatement,
    BasicLineInputStatement,
    BasicWriteStatement,
    BasicGetStatement,
    BasicPutStatement,
    BasicFieldStatement,
    BasicFieldSpec,
    BasicLsetStatement,
    BasicRsetStatement,
    BasicEofFunction,
    BasicLofFunction,
    BasicLocFunction,
    # I/O Control
    IOControlSection,
    IOControlEntry,
    TapeFileSpec,
    RerunSpec,
)

__all__ = [
    # Enumerations
    'FileOrganization',
    'FileAccess',
    'DataUsage',
    'PictureType',
    'OpenMode',
    'BasicVarType',
    # Base
    'BusinessNode',
    # Program structure
    'BusinessProgram',
    'Division',
    'Section',
    'Paragraph',
    'Entry',
    'Procedure',
    # Expressions
    'Literal',
    'Identifier',
    'ReferenceMod',
    'BinaryExpr',
    'UnaryExpr',
    'Condition',
    'FunctionCall',
    # Control flow
    'IfStatement',
    'EvaluateStatement',
    'WhenClause',
    'WhenCondition',
    'PerformStatement',
    'VaryingClause',
    'GotoStatement',
    'GosubStatement',
    'ReturnStatement',
    'ForLoop',
    'WhileLoop',
    'StopStatement',
    'EndStatement',
    # Data processing
    'MoveStatement',
    'ComputeStatement',
    'AddStatement',
    'SubtractStatement',
    'MultiplyStatement',
    'DivideStatement',
    'StringStatement',
    'StringSource',
    'UnstringStatement',
    'DelimiterSpec',
    'UnstringField',
    'InspectStatement',
    'TallyingClause',
    'TallyItem',
    'ReplacingClause',
    'ReplaceItem',
    'ConvertingClause',
    'Assignment',
    # I/O
    'DisplayStatement',
    'AcceptStatement',
    'BasicInputStatement',
    'BasicPrintStatement',
    'PrintItem',
    'DataStatement',
    'ReadDataStatement',
    'RestoreStatement',
    'RemStatement',
    # BASIC specific
    'BasicVariable',
    'DimStatement',
    'DefFunction',
    # Records
    'PictureClause',
    'OccursClause',
    'SortKey',
    'DataItem',
    'RecordStructure',
    'ConditionName',
    'RenamesClause',
    'CopyStatement',
    'ReplaceSpec',
    'WorkingStorageSection',
    'LocalStorageSection',
    'LinkageSection',
    # Utility functions
    'parse_picture',
    'create_numeric_field',
    'create_alphanumeric_field',
    'create_edited_field',
    # Files
    'FileDefinition',
    'AlternateKey',
    'RecordContains',
    'LinageClause',
    'FileControl',
    'OpenStatement',
    'OpenFile',
    'CloseStatement',
    'CloseFile',
    'ReadStatement',
    'WriteStatement',
    'AdvanceSpec',
    'RewriteStatement',
    'DeleteStatement',
    'StartStatement',
    'SortStatement',
    'MergeStatement',
    'ReleaseStatement',
    # BASIC file operations
    'BasicOpenStatement',
    'BasicCloseStatement',
    'BasicInputFileStatement',
    'BasicPrintFileStatement',
    'BasicLineInputStatement',
    'BasicWriteStatement',
    'BasicGetStatement',
    'BasicPutStatement',
    'BasicFieldStatement',
    'BasicFieldSpec',
    'BasicLsetStatement',
    'BasicRsetStatement',
    'BasicEofFunction',
    'BasicLofFunction',
    'BasicLocFunction',
    # I/O Control
    'IOControlSection',
    'IOControlEntry',
    'TapeFileSpec',
    'RerunSpec',
]
