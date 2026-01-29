#!/usr/bin/env python3
"""Common Lisp Type Mapping.

Maps STUNIR IR types to Common Lisp type specifiers.
Part of Phase 5A: Core Lisp Implementation.
"""

from typing import Dict, Optional


# Common Lisp type specifiers
COMMON_LISP_TYPES: Dict[str, str] = {
    'i8': '(signed-byte 8)',
    'i16': '(signed-byte 16)',
    'i32': 'fixnum',
    'i64': '(signed-byte 64)',
    'u8': '(unsigned-byte 8)',
    'u16': '(unsigned-byte 16)',
    'u32': '(unsigned-byte 32)',
    'u64': '(unsigned-byte 64)',
    'f32': 'single-float',
    'f64': 'double-float',
    'bool': 'boolean',
    'string': 'string',
    'void': '(values)',
    'any': 't',
    'ptr': '(or null fixnum)',
    'array': 'array',
    'list': 'list',
}


class CommonLispTypeMapper:
    """Maps IR types to Common Lisp types."""
    
    def __init__(self, use_declarations: bool = True):
        """Initialize type mapper.
        
        Args:
            use_declarations: Whether to emit type declarations.
        """
        self.use_declarations = use_declarations
    
    def map_type(self, ir_type: str) -> str:
        """Map IR type to Common Lisp type specifier.
        
        Args:
            ir_type: IR type string.
            
        Returns:
            Common Lisp type specifier.
        """
        return COMMON_LISP_TYPES.get(ir_type, 't')
    
    def emit_declaration(self, var_name: str, ir_type: str) -> str:
        """Emit a type declaration.
        
        Args:
            var_name: Variable name.
            ir_type: IR type.
            
        Returns:
            Type declaration string.
        """
        if not self.use_declarations:
            return ""
        cl_type = self.map_type(ir_type)
        return f"(declare (type {cl_type} {var_name}))"
    
    def emit_ftype(self, func_name: str, param_types: list, return_type: str) -> str:
        """Emit a function type declaration.
        
        Args:
            func_name: Function name.
            param_types: List of parameter IR types.
            return_type: Return IR type.
            
        Returns:
            ftype declaration string.
        """
        if not self.use_declarations:
            return ""
        
        param_cl_types = ' '.join(self.map_type(t) for t in param_types)
        ret_cl_type = self.map_type(return_type)
        
        return f"(declaim (ftype (function ({param_cl_types}) {ret_cl_type}) {func_name}))"
    
    def emit_the(self, ir_type: str, expr: str) -> str:
        """Emit a THE form for type assertion.
        
        Args:
            ir_type: IR type.
            expr: Expression string.
            
        Returns:
            THE form string.
        """
        cl_type = self.map_type(ir_type)
        return f"(the {cl_type} {expr})"
