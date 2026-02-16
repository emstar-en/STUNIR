#!/usr/bin/env python3
"""Type mapping for Datalog.

Maps STUNIR IR types to Datalog types and provides type conversion
utilities with Datalog-specific restrictions.

Datalog is a subset of Prolog with key restrictions:
- No function symbols in rule heads
- Range restriction for safety
- Stratified negation only
- Bottom-up evaluation semantics

Part of Phase 5C-4: Datalog Emitter.
"""

from typing import Dict, Any, Optional, List, Set
import re


# Type mapping from IR types to Datalog constants
DATALOG_TYPES: Dict[str, str] = {
    # Numeric types - all map to constants
    'i8': 'integer',
    'i16': 'integer',
    'i32': 'integer',
    'i64': 'integer',
    'u8': 'integer',
    'u16': 'integer',
    'u32': 'integer',
    'u64': 'integer',
    'f32': 'float',
    'f64': 'float',
    'number': 'number',
    
    # Boolean
    'bool': 'symbol',
    'boolean': 'symbol',
    
    # String types - map to symbols in Datalog
    'string': 'symbol',
    'str': 'symbol',
    'char': 'symbol',
    
    # Special types
    'void': 'symbol',
    'any': 'any',
}


# Reserved keywords that need quoting
DATALOG_RESERVED: Set[str] = {
    'not', 'and', 'or', 'true', 'false', 'nil',
    'if', 'then', 'else', 'end',
}


def escape_atom(value: str) -> str:
    """Escape an atom for Datalog output.
    
    Atoms that:
    - Start with uppercase
    - Contain special characters
    - Are reserved words
    need to be quoted.
    
    Args:
        value: The atom string
        
    Returns:
        Properly escaped atom
    """
    if not value:
        return "''"
    
    # Check if needs quoting
    needs_quote = (
        value[0].isupper() or
        value[0] == '_' or
        value.lower() in DATALOG_RESERVED or
        not re.match(r'^[a-z][a-zA-Z0-9_]*$', value) or
        ' ' in value
    )
    
    if needs_quote:
        # Escape single quotes within
        escaped = value.replace("'", "\\'")
        return f"'{escaped}'"
    
    return value


def escape_string(value: str) -> str:
    """Escape a string for Datalog output.
    
    Args:
        value: The string to escape
        
    Returns:
        Escaped string (without outer quotes)
    """
    return (
        value
        .replace('\\', '\\\\')
        .replace('"', '\\"')
        .replace('\n', '\\n')
        .replace('\t', '\\t')
        .replace('\r', '\\r')
    )


def format_variable(name: str) -> str:
    """Format a variable name for Datalog.
    
    Variables must start with uppercase or underscore.
    
    Args:
        name: Variable name
        
    Returns:
        Properly formatted variable
    """
    if not name:
        return '_'
    
    # Ensure starts with uppercase
    if name[0].islower():
        return name[0].upper() + name[1:]
    
    return name


class DatalogTypeMapper:
    """Maps STUNIR types to Datalog types.
    
    Provides type conversion and validation for Datalog code generation,
    enforcing Datalog-specific restrictions.
    """
    
    def __init__(self):
        """Initialize type mapper."""
        self.type_map = dict(DATALOG_TYPES)
        self._custom_types: Dict[str, str] = {}
    
    def map_type(self, ir_type: str) -> str:
        """Map IR type to Datalog type.
        
        Args:
            ir_type: STUNIR IR type name
            
        Returns:
            Datalog type name
        """
        # Handle compound types like list(i32)
        if '(' in ir_type:
            base = ir_type.split('(')[0]
            return self.type_map.get(base, 'symbol')
        
        return self.type_map.get(ir_type, 'symbol')
    
    def register_type(self, ir_type: str, datalog_type: str) -> None:
        """Register a custom type mapping.
        
        Args:
            ir_type: STUNIR IR type name
            datalog_type: Datalog type name
        """
        self._custom_types[ir_type] = datalog_type
        self.type_map[ir_type] = datalog_type
    
    def is_numeric(self, ir_type: str) -> bool:
        """Check if type is numeric.
        
        Args:
            ir_type: IR type name
            
        Returns:
            True if numeric type
        """
        return ir_type in (
            'i8', 'i16', 'i32', 'i64',
            'u8', 'u16', 'u32', 'u64',
            'f32', 'f64', 'number'
        )
    
    def is_string(self, ir_type: str) -> bool:
        """Check if type is string-like.
        
        Args:
            ir_type: IR type name
            
        Returns:
            True if string-like type
        """
        return ir_type in ('string', 'str', 'char')
    
    def format_constant(self, value: Any, ir_type: Optional[str] = None) -> str:
        """Format a constant value for Datalog.
        
        Args:
            value: The value to format
            ir_type: Optional IR type hint
            
        Returns:
            Formatted constant string
        """
        if value is None:
            return 'nil'
        
        if isinstance(value, bool):
            return 'true' if value else 'false'
        
        if isinstance(value, (int, float)):
            return str(value)
        
        if isinstance(value, str):
            # Check if it should be an atom or string
            if ir_type and self.is_string(ir_type):
                return f'"{escape_string(value)}"'
            return escape_atom(value)
        
        # Fallback
        return escape_atom(str(value))
    
    def validate_term_for_head(self, term_kind: str) -> bool:
        """Check if term kind is valid for Datalog rule head.
        
        Datalog restricts what can appear in rule heads:
        - Variables: OK
        - Atoms/constants: OK
        - Compounds (function symbols): NOT OK
        - Lists: NOT OK
        
        Args:
            term_kind: Kind of term
            
        Returns:
            True if valid for head position
        """
        # Only simple terms allowed in heads
        return term_kind in ('variable', 'atom', 'number', 'string_term')
    
    def get_supported_types(self) -> List[str]:
        """Get list of supported IR types.
        
        Returns:
            List of type names
        """
        return list(self.type_map.keys())


__all__ = [
    'DatalogTypeMapper',
    'DATALOG_TYPES',
    'DATALOG_RESERVED',
    'escape_atom',
    'escape_string',
    'format_variable',
]
