#!/usr/bin/env python3
"""Hy (Python Lisp) Type Mapping.

Maps STUNIR IR types to Python/Hy types.
Part of Phase 5B: Extended Lisp Implementation.
"""

from typing import Dict


# Hy/Python type mapping
HY_TYPES: Dict[str, str] = {
    'i8': 'int',
    'i16': 'int',
    'i32': 'int',
    'i64': 'int',
    'u8': 'int',
    'u16': 'int',
    'u32': 'int',
    'u64': 'int',
    'f32': 'float',
    'f64': 'float',
    'bool': 'bool',
    'string': 'str',
    'void': 'None',
    'any': 'typing.Any',
    'ptr': 'int',
    'array': 'list',
    'list': 'list',
    'tuple': 'tuple',
    'dict': 'dict',
    'set': 'set',
    'bytes': 'bytes',
}


class HyTypeMapper:
    """Maps IR types to Hy/Python types."""
    
    def __init__(self, emit_annotations: bool = True):
        """Initialize type mapper.
        
        Args:
            emit_annotations: Whether to emit type annotations.
        """
        self.emit_annotations = emit_annotations
    
    def map_type(self, ir_type: str) -> str:
        """Map IR type to Hy/Python type.
        
        Args:
            ir_type: IR type string.
            
        Returns:
            Python type string.
        """
        return HY_TYPES.get(ir_type, 'typing.Any')
    
    def emit_param_annotation(self, name: str, ir_type: str) -> str:
        """Emit a parameter with type annotation.
        
        Args:
            name: Parameter name.
            ir_type: IR type.
            
        Returns:
            Annotated parameter string for Hy.
        """
        if not self.emit_annotations:
            return name
        py_type = self.map_type(ir_type)
        return f"^{py_type} {name}"
    
    def emit_return_annotation(self, ir_type: str) -> str:
        """Emit a return type annotation.
        
        Args:
            ir_type: IR type.
            
        Returns:
            Return type annotation string.
        """
        if not self.emit_annotations:
            return ""
        return f"-> {self.map_type(ir_type)}"
