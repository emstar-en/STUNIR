#!/usr/bin/env python3
"""Scheme Type Mapping.

Maps STUNIR IR types to Scheme type comments and predicates.
Part of Phase 5A: Core Lisp Implementation.
"""

from typing import Dict


# Scheme type names (for comments, as Scheme is dynamically typed)
SCHEME_TYPES: Dict[str, str] = {
    'i8': 'integer',
    'i16': 'integer',
    'i32': 'integer',
    'i64': 'integer',
    'u8': 'integer',
    'u16': 'integer',
    'u32': 'integer',
    'u64': 'integer',
    'f32': 'real',
    'f64': 'real',
    'bool': 'boolean',
    'string': 'string',
    'void': 'void',
    'any': 'any',
    'list': 'list',
    'pair': 'pair',
    'symbol': 'symbol',
    'procedure': 'procedure',
}

# Scheme type predicates
SCHEME_PREDICATES: Dict[str, str] = {
    'i8': 'integer?',
    'i16': 'integer?',
    'i32': 'integer?',
    'i64': 'integer?',
    'u8': 'integer?',
    'u16': 'integer?',
    'u32': 'integer?',
    'u64': 'integer?',
    'f32': 'real?',
    'f64': 'real?',
    'bool': 'boolean?',
    'string': 'string?',
    'void': 'void?',
    'any': 'any?',
    'list': 'list?',
    'pair': 'pair?',
    'symbol': 'symbol?',
    'procedure': 'procedure?',
}


class SchemeTypeMapper:
    """Maps IR types to Scheme types."""
    
    def __init__(self, emit_comments: bool = True):
        """Initialize type mapper.
        
        Args:
            emit_comments: Whether to emit type comments.
        """
        self.emit_comments = emit_comments
    
    def map_type(self, ir_type: str) -> str:
        """Map IR type to Scheme type name.
        
        Args:
            ir_type: IR type string.
            
        Returns:
            Scheme type name.
        """
        return SCHEME_TYPES.get(ir_type, 'any')
    
    def get_predicate(self, ir_type: str) -> str:
        """Get Scheme type predicate.
        
        Args:
            ir_type: IR type string.
            
        Returns:
            Scheme predicate function name.
        """
        return SCHEME_PREDICATES.get(ir_type, 'any?')
    
    def emit_type_comment(self, name: str, param_types: list, return_type: str) -> str:
        """Emit a type signature comment.
        
        Args:
            name: Function name.
            param_types: List of parameter types.
            return_type: Return type.
            
        Returns:
            Type signature comment.
        """
        if not self.emit_comments:
            return ""
        
        param_str = ' '.join(self.map_type(t) for t in param_types)
        ret_str = self.map_type(return_type)
        
        return f";; {name} : {param_str} -> {ret_str}"
