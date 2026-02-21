#!/usr/bin/env python3
"""Guile (GNU Scheme) Type Mapping.

Maps STUNIR IR types to Guile/GOOPS types.
Part of Phase 5B: Extended Lisp Implementation.
"""

from typing import Dict


# Guile type mapping (GOOPS types for OO, standard Scheme for primitives)
GUILE_TYPES: Dict[str, str] = {
    'i8': '<integer>',
    'i16': '<integer>',
    'i32': '<integer>',
    'i64': '<integer>',
    'u8': '<integer>',
    'u16': '<integer>',
    'u32': '<integer>',
    'u64': '<integer>',
    'f32': '<real>',
    'f64': '<real>',
    'bool': '<boolean>',
    'string': '<string>',
    'void': '*unspecified*',
    'any': '<top>',
    'ptr': '<pointer>',
    'array': '<array>',
    'list': '<list>',
    'symbol': '<symbol>',
}


class GuileTypeMapper:
    """Maps IR types to Guile types."""
    
    def __init__(self, use_goops: bool = True):
        """Initialize type mapper.
        
        Args:
            use_goops: Whether to use GOOPS types.
        """
        self.use_goops = use_goops
    
    def map_type(self, ir_type: str) -> str:
        """Map IR type to Guile type.
        
        Args:
            ir_type: IR type string.
            
        Returns:
            Guile type string.
        """
        return GUILE_TYPES.get(ir_type, '<top>')
    
    def emit_slot_definition(self, name: str, ir_type: str, 
                              init_keyword: bool = True,
                              accessor: bool = True) -> str:
        """Emit a GOOPS slot definition.
        
        Args:
            name: Slot name.
            ir_type: IR type.
            init_keyword: Include #:init-keyword.
            accessor: Include #:accessor.
            
        Returns:
            Slot definition string.
        """
        guile_type = self.map_type(ir_type)
        parts = [f"({name}"]
        
        if init_keyword:
            parts.append(f"#:init-keyword #{name}")
        if accessor:
            parts.append(f"#:accessor {name}-ref")
        
        parts.append(")")
        return " ".join(parts)
