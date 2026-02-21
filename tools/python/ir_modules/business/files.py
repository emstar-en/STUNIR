#!/usr/bin/env python3
"""STUNIR Business IR - File operations.

This module defines IR nodes for file definitions and file I/O
operations for COBOL and BASIC programs.

Usage:
    from ir.business.files import FileDefinition, FileControl, OpenStatement
    from ir.business import FileOrganization, FileAccess
    
    # Define a COBOL indexed file
    file_def = FileDefinition(
        name='EMPLOYEE-FILE',
        organization=FileOrganization.INDEXED,
        access=FileAccess.SEQUENTIAL,
        record_key='EMP-ID'
    )
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any

from .business_ir import (
    BusinessNode, FileOrganization, FileAccess, OpenMode
)
from .records import RecordStructure


# =============================================================================
# COBOL File Definitions
# =============================================================================

@dataclass
class FileDefinition(BusinessNode):
    """File definition (COBOL FD/SD entry).
    
    FD = File Description
    SD = Sort/Merge Description
    """
    name: str = ''
    kind: str = 'file_definition'
    organization: FileOrganization = FileOrganization.SEQUENTIAL
    access: FileAccess = FileAccess.SEQUENTIAL
    record_key: Optional[str] = None  # Primary key for indexed
    alternate_keys: List['AlternateKey'] = field(default_factory=list)
    status: Optional[str] = None  # File status variable
    record: Optional[RecordStructure] = None
    records: List[RecordStructure] = field(default_factory=list)  # Multiple record types
    block_contains: Optional[int] = None  # BLOCK CONTAINS clause
    record_contains: Optional['RecordContains'] = None
    label_records: str = 'standard'  # STANDARD, OMITTED
    value_of_file_id: Optional[str] = None
    data_records: List[str] = field(default_factory=list)
    linage: Optional['LinageClause'] = None
    code_set: Optional[str] = None  # CODE-SET clause
    is_sort_file: bool = False  # SD instead of FD


@dataclass
class AlternateKey(BusinessNode):
    """Alternate key for indexed files."""
    name: str = ''
    kind: str = 'alternate_key'
    with_duplicates: bool = False


@dataclass
class RecordContains(BusinessNode):
    """RECORD CONTAINS clause."""
    min_chars: int = 0
    kind: str = 'record_contains'
    max_chars: Optional[int] = None
    depending_on: Optional[str] = None


@dataclass
class LinageClause(BusinessNode):
    """LINAGE clause for report files."""
    lines_per_page: int = 0
    kind: str = 'linage_clause'
    footing_at: Optional[int] = None
    top: Optional[int] = None
    bottom: Optional[int] = None


@dataclass
class FileControl(BusinessNode):
    """COBOL FILE-CONTROL entry (SELECT statement)."""
    select_name: str = ''
    kind: str = 'file_control'
    assign_to: str = ''
    organization: FileOrganization = FileOrganization.SEQUENTIAL
    access: FileAccess = FileAccess.SEQUENTIAL
    record_key: Optional[str] = None
    alternate_keys: List[AlternateKey] = field(default_factory=list)
    file_status: Optional[str] = None
    relative_key: Optional[str] = None  # For RELATIVE organization
    padding_character: Optional[str] = None
    record_delimiter: Optional[str] = None
    reserve: Optional[int] = None  # RESERVE clause
    lock_mode: Optional[str] = None  # LOCK MODE clause


# =============================================================================
# COBOL File I/O Statements
# =============================================================================

@dataclass
class OpenStatement(BusinessNode):
    """File OPEN statement."""
    files: List['OpenFile'] = field(default_factory=list)
    kind: str = 'open_statement'


@dataclass
class OpenFile(BusinessNode):
    """File entry in OPEN statement."""
    name: str = ''
    kind: str = 'open_file'
    mode: OpenMode = OpenMode.INPUT


@dataclass
class CloseStatement(BusinessNode):
    """File CLOSE statement."""
    files: List['CloseFile'] = field(default_factory=list)
    kind: str = 'close_statement'


@dataclass
class CloseFile(BusinessNode):
    """File entry in CLOSE statement."""
    name: str = ''
    kind: str = 'close_file'
    with_lock: bool = False
    reel_unit: Optional[str] = None  # REEL, UNIT, NO REWIND


@dataclass
class ReadStatement(BusinessNode):
    """File READ statement."""
    file_name: str = ''
    kind: str = 'read_statement'
    into: Optional[str] = None
    key: Optional[str] = None  # For indexed access
    key_is: Optional[str] = None  # Key field name
    next_record: bool = False  # READ NEXT
    previous_record: bool = False  # READ PREVIOUS
    at_end: List['Statement'] = field(default_factory=list)
    not_at_end: List['Statement'] = field(default_factory=list)
    invalid_key: List['Statement'] = field(default_factory=list)
    not_invalid_key: List['Statement'] = field(default_factory=list)


@dataclass
class WriteStatement(BusinessNode):
    """File WRITE statement."""
    record_name: str = ''
    kind: str = 'write_statement'
    from_value: Optional[str] = None
    after_advancing: Optional['AdvanceSpec'] = None
    before_advancing: Optional['AdvanceSpec'] = None
    invalid_key: List['Statement'] = field(default_factory=list)
    not_invalid_key: List['Statement'] = field(default_factory=list)
    end_of_page: List['Statement'] = field(default_factory=list)
    not_end_of_page: List['Statement'] = field(default_factory=list)


@dataclass
class AdvanceSpec(BusinessNode):
    """Advancing specification for WRITE."""
    kind: str = 'advance_spec'
    lines: Optional[int] = None  # Number of lines
    identifier: Optional[str] = None  # Identifier for lines
    mnemonic_name: Optional[str] = None  # PAGE, etc.


@dataclass
class RewriteStatement(BusinessNode):
    """File REWRITE statement (update)."""
    record_name: str = ''
    kind: str = 'rewrite_statement'
    from_value: Optional[str] = None
    invalid_key: List['Statement'] = field(default_factory=list)
    not_invalid_key: List['Statement'] = field(default_factory=list)


@dataclass
class DeleteStatement(BusinessNode):
    """File DELETE statement."""
    file_name: str = ''
    kind: str = 'delete_statement'
    invalid_key: List['Statement'] = field(default_factory=list)
    not_invalid_key: List['Statement'] = field(default_factory=list)


@dataclass
class StartStatement(BusinessNode):
    """Position for indexed/relative files."""
    file_name: str = ''
    kind: str = 'start_statement'
    key_condition: str = 'equal'  # 'equal', 'greater', 'not-less', 'less', 'not-greater'
    key_name: Optional[str] = None
    invalid_key: List['Statement'] = field(default_factory=list)
    not_invalid_key: List['Statement'] = field(default_factory=list)


# =============================================================================
# COBOL Sort/Merge
# =============================================================================

@dataclass
class SortStatement(BusinessNode):
    """COBOL SORT statement."""
    file_name: str = ''
    kind: str = 'sort_statement'
    keys: List['SortKey'] = field(default_factory=list)
    with_duplicates: bool = False
    input_procedure: Optional[str] = None
    input_files: List[str] = field(default_factory=list)  # USING
    output_procedure: Optional[str] = None
    output_files: List[str] = field(default_factory=list)  # GIVING


@dataclass
class SortKey(BusinessNode):
    """Sort key for SORT statement."""
    name: str = ''
    kind: str = 'sort_key'
    ascending: bool = True


@dataclass
class MergeStatement(BusinessNode):
    """COBOL MERGE statement."""
    file_name: str = ''
    kind: str = 'merge_statement'
    keys: List[SortKey] = field(default_factory=list)
    input_files: List[str] = field(default_factory=list)  # USING
    output_procedure: Optional[str] = None
    output_files: List[str] = field(default_factory=list)  # GIVING


@dataclass
class ReleaseStatement(BusinessNode):
    """COBOL RELEASE statement (for sort input procedure)."""
    record_name: str = ''
    kind: str = 'release_statement'
    from_value: Optional[str] = None


@dataclass
class ReturnStatement(BusinessNode):
    """COBOL RETURN statement (for sort output procedure)."""
    file_name: str = ''
    kind: str = 'return_cobol_statement'
    into: Optional[str] = None
    at_end: List['Statement'] = field(default_factory=list)
    not_at_end: List['Statement'] = field(default_factory=list)


# =============================================================================
# BASIC File Operations
# =============================================================================

@dataclass
class BasicOpenStatement(BusinessNode):
    """BASIC OPEN statement."""
    file_number: int = 1
    kind: str = 'basic_open'
    filename: str = ''
    mode: str = 'input'  # 'input', 'output', 'append', 'random', 'binary'
    record_length: Optional[int] = None
    access: Optional[str] = None  # 'read', 'write', 'read write'
    lock: Optional[str] = None  # 'shared', 'lock read', 'lock write'
    line_number: Optional[int] = None


@dataclass
class BasicCloseStatement(BusinessNode):
    """BASIC CLOSE statement."""
    file_numbers: List[int] = field(default_factory=list)
    kind: str = 'basic_close'
    line_number: Optional[int] = None


@dataclass
class BasicInputFileStatement(BusinessNode):
    """BASIC INPUT# statement for file reading."""
    file_number: int = 1
    kind: str = 'basic_input_file'
    variables: List[str] = field(default_factory=list)
    line_number: Optional[int] = None


@dataclass
class BasicPrintFileStatement(BusinessNode):
    """BASIC PRINT# statement for file writing."""
    file_number: int = 1
    kind: str = 'basic_print_file'
    expressions: List[Any] = field(default_factory=list)
    line_number: Optional[int] = None


@dataclass
class BasicLineInputStatement(BusinessNode):
    """BASIC LINE INPUT# statement."""
    file_number: int = 1
    kind: str = 'basic_line_input'
    variable: str = ''
    line_number: Optional[int] = None


@dataclass
class BasicWriteStatement(BusinessNode):
    """BASIC WRITE# statement."""
    file_number: int = 1
    kind: str = 'basic_write'
    expressions: List[Any] = field(default_factory=list)
    line_number: Optional[int] = None


@dataclass
class BasicGetStatement(BusinessNode):
    """BASIC GET# statement for random access."""
    file_number: int = 1
    kind: str = 'basic_get'
    record_number: Optional[int] = None
    line_number: Optional[int] = None


@dataclass
class BasicPutStatement(BusinessNode):
    """BASIC PUT# statement for random access."""
    file_number: int = 1
    kind: str = 'basic_put'
    record_number: Optional[int] = None
    line_number: Optional[int] = None


@dataclass
class BasicFieldStatement(BusinessNode):
    """BASIC FIELD statement for random file buffers."""
    file_number: int = 1
    kind: str = 'basic_field'
    fields: List['BasicFieldSpec'] = field(default_factory=list)
    line_number: Optional[int] = None


@dataclass
class BasicFieldSpec(BusinessNode):
    """Field specification in FIELD statement."""
    size: int = 0
    kind: str = 'basic_field_spec'
    variable: str = ''


@dataclass
class BasicLsetStatement(BusinessNode):
    """BASIC LSET statement."""
    variable: str = ''
    kind: str = 'basic_lset'
    value: Any = None
    line_number: Optional[int] = None


@dataclass
class BasicRsetStatement(BusinessNode):
    """BASIC RSET statement."""
    variable: str = ''
    kind: str = 'basic_rset'
    value: Any = None
    line_number: Optional[int] = None


@dataclass
class BasicEofFunction(BusinessNode):
    """BASIC EOF function."""
    file_number: int = 1
    kind: str = 'basic_eof'


@dataclass
class BasicLofFunction(BusinessNode):
    """BASIC LOF (length of file) function."""
    file_number: int = 1
    kind: str = 'basic_lof'


@dataclass
class BasicLocFunction(BusinessNode):
    """BASIC LOC (location in file) function."""
    file_number: int = 1
    kind: str = 'basic_loc'


# =============================================================================
# I/O Control
# =============================================================================

@dataclass
class IOControlSection(BusinessNode):
    """COBOL I-O-CONTROL section."""
    kind: str = 'io_control_section'
    entries: List['IOControlEntry'] = field(default_factory=list)


@dataclass
class IOControlEntry(BusinessNode):
    """I-O-CONTROL entry."""
    kind: str = 'io_control_entry'
    same_area: List[str] = field(default_factory=list)
    same_record_area: List[str] = field(default_factory=list)
    multiple_file_tape: List['TapeFileSpec'] = field(default_factory=list)
    rerun: Optional['RerunSpec'] = None


@dataclass
class TapeFileSpec(BusinessNode):
    """Multiple file tape specification."""
    file_name: str = ''
    kind: str = 'tape_file_spec'
    position: Optional[int] = None


@dataclass
class RerunSpec(BusinessNode):
    """RERUN specification."""
    on_file: Optional[str] = None
    kind: str = 'rerun_spec'
    every_records: Optional[int] = None
    every_end_of_reel: bool = False


# =============================================================================
# Export all public symbols
# =============================================================================

__all__ = [
    # COBOL file definitions
    'FileDefinition', 'AlternateKey', 'RecordContains', 'LinageClause',
    'FileControl',
    # COBOL file I/O
    'OpenStatement', 'OpenFile', 'CloseStatement', 'CloseFile',
    'ReadStatement', 'WriteStatement', 'AdvanceSpec',
    'RewriteStatement', 'DeleteStatement', 'StartStatement',
    # Sort/Merge
    'SortStatement', 'SortKey', 'MergeStatement',
    'ReleaseStatement', 'ReturnStatement',
    # BASIC file operations
    'BasicOpenStatement', 'BasicCloseStatement',
    'BasicInputFileStatement', 'BasicPrintFileStatement',
    'BasicLineInputStatement', 'BasicWriteStatement',
    'BasicGetStatement', 'BasicPutStatement',
    'BasicFieldStatement', 'BasicFieldSpec',
    'BasicLsetStatement', 'BasicRsetStatement',
    'BasicEofFunction', 'BasicLofFunction', 'BasicLocFunction',
    # I/O Control
    'IOControlSection', 'IOControlEntry', 'TapeFileSpec', 'RerunSpec',
]
