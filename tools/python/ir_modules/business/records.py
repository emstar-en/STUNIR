#!/usr/bin/env python3
"""STUNIR Business IR - Record structures and data definitions.

This module defines IR nodes for COBOL record structures with
hierarchical level numbers, PICTURE clauses, and data items.

Usage:
    from ir.business.records import RecordStructure, DataItem, PictureClause
    
    # Create a COBOL record
    record = RecordStructure(
        name='EMPLOYEE-RECORD',
        level=1,
        fields=[
            DataItem(name='EMP-ID', level=5, picture=PictureClause(pattern='9(5)')),
            DataItem(name='EMP-NAME', level=5, picture=PictureClause(pattern='X(30)')),
        ]
    )
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Any
from enum import Enum

from .business_ir import BusinessNode, DataUsage, PictureType


# =============================================================================
# PICTURE Clause
# =============================================================================

@dataclass
class PictureClause(BusinessNode):
    """COBOL PICTURE clause specification.
    
    PICTURE clause symbols:
    - 9: Numeric digit
    - A: Alphabetic
    - X: Alphanumeric
    - V: Implied decimal point
    - S: Sign (leading)
    - P: Scaling position
    - Z: Zero suppression
    - *: Check protection (asterisk fill)
    - $: Currency symbol
    - ,: Comma
    - .: Decimal point
    - -: Minus sign
    - +: Plus sign
    - CR: Credit symbol
    - DB: Debit symbol
    - B: Blank insertion
    - 0: Zero insertion
    - /: Slash insertion
    """
    pattern: str = ''  # e.g., '9(5)V99', 'X(20)', 'ZZZ,ZZ9.99'
    kind: str = 'picture_clause'
    
    @property
    def data_type(self) -> PictureType:
        """Determine data type from pattern."""
        pattern_upper = self.pattern.upper()
        # Check for edit characters first
        edit_chars = set('Z*$,./+-')
        if any(c in pattern_upper for c in edit_chars):
            return PictureType.EDITED
        if 'CR' in pattern_upper or 'DB' in pattern_upper:
            return PictureType.EDITED
        # Check for alphabetic only
        if 'A' in pattern_upper and '9' not in pattern_upper and 'X' not in pattern_upper:
            return PictureType.ALPHABETIC
        # Check for alphanumeric
        if 'X' in pattern_upper:
            return PictureType.ALPHANUMERIC
        # Default to numeric
        return PictureType.NUMERIC
    
    @property
    def size(self) -> int:
        """Calculate storage size from pattern."""
        expanded = self.expand_pattern()
        # Count storage positions (exclude V, S, P)
        count = 0
        i = 0
        while i < len(expanded):
            c = expanded[i].upper()
            if c in '9AXZB0/':
                count += 1
            elif c in '*.':
                count += 1
            elif c == '$':
                count += 1
            elif c == ',':
                count += 1
            elif c in '+-':
                count += 1
            elif c == 'C' and i + 1 < len(expanded) and expanded[i + 1].upper() == 'R':
                count += 2
                i += 1
            elif c == 'D' and i + 1 < len(expanded) and expanded[i + 1].upper() == 'B':
                count += 2
                i += 1
            # V, S, P don't take storage
            i += 1
        return count
    
    @property
    def decimal_places(self) -> int:
        """Get number of decimal places."""
        expanded = self.expand_pattern().upper()
        if 'V' not in expanded:
            return 0
        v_pos = expanded.index('V')
        # Count digits after V
        return sum(1 for c in expanded[v_pos + 1:] if c == '9')
    
    @property
    def is_signed(self) -> bool:
        """Check if numeric is signed."""
        return 'S' in self.pattern.upper() or '+' in self.pattern or '-' in self.pattern
    
    def expand_pattern(self) -> str:
        """Expand shorthand notation (e.g., 9(5) -> 99999)."""
        def expand_match(match):
            char = match.group(1)
            count = int(match.group(2))
            return char * count
        
        return re.sub(r'(\w)\((\d+)\)', expand_match, self.pattern)


# =============================================================================
# OCCURS Clause (Arrays)
# =============================================================================

@dataclass
class OccursClause(BusinessNode):
    """COBOL OCCURS clause (arrays/tables).
    
    Examples:
        OCCURS 10 TIMES
        OCCURS 1 TO 100 TIMES DEPENDING ON ITEM-COUNT
        OCCURS 5 TIMES INDEXED BY IDX-1
        OCCURS 10 TIMES ASCENDING KEY IS ITEM-KEY
    """
    times: int = 1  # Fixed occurrences
    kind: str = 'occurs_clause'
    min_times: Optional[int] = None  # OCCURS DEPENDING ON
    max_times: Optional[int] = None
    depending_on: Optional[str] = None
    indexed_by: List[str] = field(default_factory=list)
    keys: List['SortKey'] = field(default_factory=list)


@dataclass
class SortKey(BusinessNode):
    """Sort key specification for indexed tables."""
    name: str = ''
    ascending: bool = True
    kind: str = 'sort_key'


# =============================================================================
# Data Items
# =============================================================================

@dataclass
class DataItem(BusinessNode):
    """Individual data item (COBOL data description entry).
    
    Level numbers:
    - 01-49: Regular data items
    - 66: RENAMES clause
    - 77: Independent data items
    - 88: Condition names
    """
    name: str = ''
    kind: str = 'data_item'
    level: int = 1  # Level number (01-49, 66, 77, 88)
    picture: Optional[PictureClause] = None
    usage: DataUsage = DataUsage.DISPLAY
    value: Optional[Any] = None  # VALUE clause
    occurs: Optional[OccursClause] = None
    redefines: Optional[str] = None  # REDEFINES clause
    children: List['DataItem'] = field(default_factory=list)
    justified: bool = False  # JUSTIFIED RIGHT
    blank_when_zero: bool = False  # BLANK WHEN ZERO
    synchronized: bool = False  # SYNCHRONIZED/SYNC
    sign_leading: bool = False  # SIGN LEADING
    sign_separate: bool = False  # SIGN SEPARATE


@dataclass
class RecordStructure(BusinessNode):
    """Hierarchical record structure (COBOL 01-level group)."""
    name: str = ''
    kind: str = 'record_structure'
    level: int = 1  # Always 01 for records
    fields: List[DataItem] = field(default_factory=list)
    redefines: Optional[str] = None  # REDEFINES clause
    occurs: Optional[OccursClause] = None
    external: bool = False  # EXTERNAL clause
    global_record: bool = False  # GLOBAL clause
    
    def get_field(self, name: str) -> Optional[DataItem]:
        """Find a field by name (recursive)."""
        def search(items: List[DataItem]) -> Optional[DataItem]:
            for item in items:
                if item.name.upper() == name.upper():
                    return item
                if item.children:
                    found = search(item.children)
                    if found:
                        return found
            return None
        return search(self.fields)
    
    def get_all_fields(self) -> List[DataItem]:
        """Get all fields flattened."""
        result = []
        def collect(items: List[DataItem]):
            for item in items:
                result.append(item)
                if item.children:
                    collect(item.children)
        collect(self.fields)
        return result
    
    def calculate_size(self) -> int:
        """Calculate total record size."""
        total = 0
        for item in self.get_all_fields():
            if item.picture:
                size = item.picture.size
                if item.occurs:
                    size *= item.occurs.times
                total += size
        return total


@dataclass
class ConditionName(BusinessNode):
    """COBOL 88-level condition name.
    
    Example:
        88 VALID-STATUS VALUES ARE 'A' 'I' 'P'.
        88 INVALID-STATUS VALUE IS 'X'.
    """
    name: str = ''
    kind: str = 'condition_name'
    level: int = 88
    values: List[Any] = field(default_factory=list)
    through_values: List[tuple] = field(default_factory=list)  # VALUE ... THRU ...
    false_value: Optional[Any] = None  # SET ... TO FALSE


@dataclass
class RenamesClause(BusinessNode):
    """COBOL 66-level RENAMES clause.
    
    Example:
        66 FULL-NAME RENAMES FIRST-NAME THRU LAST-NAME.
    """
    name: str = ''
    kind: str = 'renames_clause'
    level: int = 66
    from_name: str = ''
    through_name: Optional[str] = None


@dataclass
class CopyStatement(BusinessNode):
    """COBOL COPY statement for copybooks."""
    copybook_name: str = ''
    kind: str = 'copy_statement'
    library: Optional[str] = None
    replacing: List['ReplaceSpec'] = field(default_factory=list)
    suppress: bool = False


@dataclass
class ReplaceSpec(BusinessNode):
    """Replacement specification in COPY statement."""
    from_text: str = ''
    kind: str = 'replace_spec'
    to_text: str = ''


# =============================================================================
# Working Storage / File Section Items
# =============================================================================

@dataclass
class WorkingStorageSection(BusinessNode):
    """COBOL WORKING-STORAGE SECTION."""
    kind: str = 'working_storage_section'
    items: List[DataItem] = field(default_factory=list)
    records: List[RecordStructure] = field(default_factory=list)


@dataclass
class LocalStorageSection(BusinessNode):
    """COBOL LOCAL-STORAGE SECTION."""
    kind: str = 'local_storage_section'
    items: List[DataItem] = field(default_factory=list)
    records: List[RecordStructure] = field(default_factory=list)


@dataclass
class LinkageSection(BusinessNode):
    """COBOL LINKAGE SECTION."""
    kind: str = 'linkage_section'
    items: List[DataItem] = field(default_factory=list)
    records: List[RecordStructure] = field(default_factory=list)


# =============================================================================
# Utility Functions
# =============================================================================

def parse_picture(pattern: str) -> PictureClause:
    """Parse a PICTURE clause pattern string."""
    return PictureClause(pattern=pattern)


def create_numeric_field(name: str, level: int, digits: int, 
                         decimals: int = 0, signed: bool = False) -> DataItem:
    """Create a numeric data item."""
    if decimals > 0:
        pattern = f'9({digits - decimals})V9({decimals})'
    else:
        pattern = f'9({digits})'
    if signed:
        pattern = 'S' + pattern
    return DataItem(
        name=name,
        level=level,
        picture=PictureClause(pattern=pattern)
    )


def create_alphanumeric_field(name: str, level: int, size: int) -> DataItem:
    """Create an alphanumeric data item."""
    return DataItem(
        name=name,
        level=level,
        picture=PictureClause(pattern=f'X({size})')
    )


def create_edited_field(name: str, level: int, pattern: str) -> DataItem:
    """Create an edited data item."""
    return DataItem(
        name=name,
        level=level,
        picture=PictureClause(pattern=pattern)
    )


# =============================================================================
# Export all public symbols
# =============================================================================

__all__ = [
    # Core classes
    'PictureClause', 'OccursClause', 'SortKey', 'DataItem', 'RecordStructure',
    # Special levels
    'ConditionName', 'RenamesClause',
    # COPY statement
    'CopyStatement', 'ReplaceSpec',
    # Sections
    'WorkingStorageSection', 'LocalStorageSection', 'LinkageSection',
    # Utility functions
    'parse_picture', 'create_numeric_field', 'create_alphanumeric_field',
    'create_edited_field',
]
