#!/usr/bin/env python3
"""Emacs Lisp Type Mapping.

Maps STUNIR IR types to Emacs Lisp types.
Part of Phase 5B: Extended Lisp Implementation.
"""

from typing import Dict


# Emacs Lisp type mapping
# Note: Elisp is dynamically typed, these are for documentation/comments
EMACS_LISP_TYPES: Dict[str, str] = {
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
    'bool': 'boolean',
    'string': 'string',
    'void': 'nil',
    'any': 't',
    'ptr': 'integer',
    'array': 'vector',
    'list': 'list',
    'symbol': 'symbol',
}


class EmacsLispTypeMapper:
    """Maps IR types to Emacs Lisp types."""
    
    def __init__(self, emit_type_comments: bool = True):
        """Initialize type mapper.
        
        Args:
            emit_type_comments: Whether to emit type comments.
        """
        self.emit_type_comments = emit_type_comments
    
    def map_type(self, ir_type: str) -> str:
        """Map IR type to Emacs Lisp type.
        
        Args:
            ir_type: IR type string.
            
        Returns:
            Emacs Lisp type string.
        """
        return EMACS_LISP_TYPES.get(ir_type, 't')
    
    def emit_type_comment(self, var_name: str, ir_type: str) -> str:
        """Emit a type comment for a variable.
        
        Args:
            var_name: Variable name.
            ir_type: IR type.
            
        Returns:
            Type comment string or empty string.
        """
        if not self.emit_type_comments:
            return ""
        el_type = self.map_type(ir_type)
        return f";; {var_name}: {el_type}"
