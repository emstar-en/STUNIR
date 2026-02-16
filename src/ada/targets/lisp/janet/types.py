#!/usr/bin/env python3
"""Janet Type Mapping.

Maps STUNIR IR types to Janet types.
Part of Phase 5B: Extended Lisp Implementation.
"""

from typing import Dict


# Janet type mapping
# Janet uses keywords for type annotations
JANET_TYPES: Dict[str, str] = {
    'i8': ':number',
    'i16': ':number',
    'i32': ':number',
    'i64': ':number',
    'u8': ':number',
    'u16': ':number',
    'u32': ':number',
    'u64': ':number',
    'f32': ':number',
    'f64': ':number',
    'bool': ':boolean',
    'string': ':string',
    'void': ':nil',
    'any': ':any',
    'ptr': ':pointer',
    'array': ':array',
    'list': ':tuple',
    'tuple': ':tuple',
    'table': ':table',
    'struct': ':struct',
    'symbol': ':symbol',
    'keyword': ':keyword',
    'buffer': ':buffer',
    'fiber': ':fiber',
}


class JanetTypeMapper:
    """Maps IR types to Janet types."""
    
    def __init__(self, use_keywords: bool = True):
        """Initialize type mapper.
        
        Args:
            use_keywords: Whether to use keyword type annotations.
        """
        self.use_keywords = use_keywords
    
    def map_type(self, ir_type: str) -> str:
        """Map IR type to Janet type keyword.
        
        Args:
            ir_type: IR type string.
            
        Returns:
            Janet type keyword.
        """
        return JANET_TYPES.get(ir_type, ':any')
    
    def emit_struct_field(self, name: str, ir_type: str) -> str:
        """Emit a struct field definition.
        
        Args:
            name: Field name.
            ir_type: IR type.
            
        Returns:
            Struct field definition.
        """
        janet_type = self.map_type(ir_type)
        return f":{name} {janet_type}"
