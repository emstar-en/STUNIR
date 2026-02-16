#!/usr/bin/env python3
"""Type mapping for GNU Prolog.

Maps STUNIR IR types to GNU Prolog types and provides
type conversion utilities with CLP support.

Part of Phase 5C-2: GNU Prolog with CLP support.
"""

from typing import Dict, Any, Optional


# Type mapping from IR types to GNU Prolog types
GNU_PROLOG_TYPES: Dict[str, str] = {
    # Numeric types
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
    'bool': 'boolean',
    'boolean': 'boolean',
    
    # String types
    'string': 'atom',
    'str': 'atom',
    'char': 'atom',
    
    # Collections
    'list': 'list',
    'array': 'list',
    
    # Special
    'void': 'true',
    'any': 'any',
    'term': 'term',
    
    # CLP types
    'fd_var': 'fd_variable',     # Finite domain variable
    'bool_var': 'bool_variable', # Boolean CLP variable
    'real_var': 'real_variable', # Real CLP variable
}


# CLP(FD) operator mapping: IR operators -> GNU Prolog CLP(FD)
CLPFD_OPERATORS: Dict[str, str] = {
    '==': '#=',      # Equality
    '=': '#=',       # Equality (alternate)
    '#=': '#=',      # Direct CLP(FD) equality
    '!=': '#\\=',   # Not equal
    '\\=': '#\\=', # Not equal (alternate)
    '#\\=': '#\\=', # Direct CLP(FD) not equal
    '<': '#<',       # Less than
    '#<': '#<',      # Direct CLP(FD) less than
    '>': '#>',       # Greater than
    '#>': '#>',      # Direct CLP(FD) greater than
    '<=': '#=<',     # Less or equal
    '=<': '#=<',     # Less or equal (Prolog style)
    '#=<': '#=<',    # Direct CLP(FD) less or equal
    '>=': '#>=',     # Greater or equal
    '#>=': '#>=',    # Direct CLP(FD) greater or equal
}


# CLP(B) operator mapping: IR operators -> GNU Prolog CLP(B)
CLPB_OPERATORS: Dict[str, str] = {
    'and': '#/\\',     # Boolean AND
    '#/\\': '#/\\',   # Direct
    'or': '#\\/',      # Boolean OR
    '#\\/': '#\\/',   # Direct
    'not': '#\\',      # Boolean NOT
    '#\\': '#\\',     # Direct
    'xor': '#\\',      # XOR (same as not in some contexts)
    'equiv': '#<=>',   # Equivalence
    '#<=>': '#<=>',    # Direct
    'implies': '#==>',  # Implication
    '#==>': '#==>',    # Direct
}


# CLP(FD) predicate names: Generic -> GNU Prolog
CLPFD_PREDICATES: Dict[str, str] = {
    'domain': 'fd_domain',
    'fd_domain': 'fd_domain',
    'all_different': 'fd_all_different',
    'fd_all_different': 'fd_all_different',
    'labeling': 'fd_labeling',
    'fd_labeling': 'fd_labeling',
    'label': 'fd_labeling',
    'minimize': 'fd_minimize',
    'maximize': 'fd_maximize',
}


class GNUPrologTypeMapper:
    """Maps STUNIR types to GNU Prolog types.
    
    Provides type conversion and CLP detection for
    generating proper GNU Prolog code.
    """
    
    def __init__(self, config: Any = None):
        """Initialize type mapper.
        
        Args:
            config: Emitter configuration
        """
        self.config = config
        self.type_map = dict(GNU_PROLOG_TYPES)
    
    def map_type(self, ir_type: str) -> str:
        """Map IR type to Prolog type.
        
        Args:
            ir_type: STUNIR IR type name
            
        Returns:
            GNU Prolog type name
        """
        # Handle compound types like list(i32)
        if '(' in ir_type:
            base = ir_type.split('(')[0]
            return self.type_map.get(base, 'term')
        
        return self.type_map.get(ir_type, 'term')
    
    def is_clpfd_operator(self, op: str) -> bool:
        """Check if operator is a CLP(FD) operator.
        
        Args:
            op: Operator string
            
        Returns:
            True if CLP(FD) operator
        """
        return op in CLPFD_OPERATORS or op.startswith('#')
    
    def map_clpfd_operator(self, op: str) -> str:
        """Map operator to CLP(FD) equivalent.
        
        Args:
            op: Input operator
            
        Returns:
            GNU Prolog CLP(FD) operator
        """
        return CLPFD_OPERATORS.get(op, op)
    
    def is_clpb_operator(self, op: str) -> bool:
        """Check if operator is a CLP(B) operator.
        
        Args:
            op: Operator string
            
        Returns:
            True if CLP(B) operator
        """
        return op in CLPB_OPERATORS
    
    def map_clpb_operator(self, op: str) -> str:
        """Map operator to CLP(B) equivalent.
        
        Args:
            op: Input operator
            
        Returns:
            GNU Prolog CLP(B) operator
        """
        return CLPB_OPERATORS.get(op, op)
    
    def is_clpfd_predicate(self, pred: str) -> bool:
        """Check if predicate is a CLP(FD) predicate.
        
        Args:
            pred: Predicate name
            
        Returns:
            True if CLP(FD) predicate
        """
        return pred in CLPFD_PREDICATES or pred.startswith('fd_')
    
    def map_clpfd_predicate(self, pred: str) -> str:
        """Map predicate to GNU Prolog CLP(FD) name.
        
        Args:
            pred: Input predicate name
            
        Returns:
            GNU Prolog CLP(FD) predicate name
        """
        return CLPFD_PREDICATES.get(pred, pred)
    
    def is_numeric(self, ir_type: str) -> bool:
        """Check if type is numeric.
        
        Args:
            ir_type: IR type name
            
        Returns:
            True if numeric type
        """
        return ir_type in ('i8', 'i16', 'i32', 'i64', 
                          'u8', 'u16', 'u32', 'u64',
                          'f32', 'f64', 'number')
    
    def is_string(self, ir_type: str) -> bool:
        """Check if type is string-like.
        
        Args:
            ir_type: IR type name
            
        Returns:
            True if string-like type
        """
        return ir_type in ('string', 'str', 'char', 'atom')


__all__ = [
    'GNUPrologTypeMapper', 
    'GNU_PROLOG_TYPES', 
    'CLPFD_OPERATORS',
    'CLPB_OPERATORS',
    'CLPFD_PREDICATES'
]
