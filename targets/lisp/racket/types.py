#!/usr/bin/env python3
"""Racket Type Mapping.

Maps STUNIR IR types to Racket types and contracts.
Part of Phase 5A: Core Lisp Implementation.
"""

from typing import Dict, List


# Typed Racket types
TYPED_RACKET_TYPES: Dict[str, str] = {
    'i8': 'Byte',
    'i16': 'Fixnum',
    'i32': 'Integer',
    'i64': 'Integer',
    'u8': 'Byte',
    'u16': 'Nonnegative-Fixnum',
    'u32': 'Nonnegative-Integer',
    'u64': 'Nonnegative-Integer',
    'f32': 'Flonum',
    'f64': 'Flonum',
    'bool': 'Boolean',
    'string': 'String',
    'void': 'Void',
    'any': 'Any',
    'list': '(Listof Any)',
    'symbol': 'Symbol',
}

# Racket contract predicates (for untyped Racket)
RACKET_CONTRACTS: Dict[str, str] = {
    'i8': 'byte?',
    'i16': 'fixnum?',
    'i32': 'integer?',
    'i64': 'integer?',
    'u8': 'byte?',
    'u16': 'exact-nonnegative-integer?',
    'u32': 'exact-nonnegative-integer?',
    'u64': 'exact-nonnegative-integer?',
    'f32': 'flonum?',
    'f64': 'flonum?',
    'bool': 'boolean?',
    'string': 'string?',
    'void': 'void?',
    'any': 'any/c',
    'list': 'list?',
    'symbol': 'symbol?',
}

# Simple type names for comments
RACKET_TYPES: Dict[str, str] = {
    'i8': 'integer',
    'i16': 'integer',
    'i32': 'integer',
    'i64': 'integer',
    'f32': 'float',
    'f64': 'float',
    'bool': 'boolean',
    'string': 'string',
    'void': 'void',
    'any': 'any',
}


class RacketTypeMapper:
    """Maps IR types to Racket types and contracts."""
    
    def __init__(self, use_typed: bool = False, use_contracts: bool = True):
        """Initialize type mapper.
        
        Args:
            use_typed: Whether to emit Typed Racket types.
            use_contracts: Whether to emit contracts.
        """
        self.use_typed = use_typed
        self.use_contracts = use_contracts
    
    def map_type(self, ir_type: str) -> str:
        """Map IR type to Racket type.
        
        Args:
            ir_type: IR type string.
            
        Returns:
            Racket type string.
        """
        if self.use_typed:
            return TYPED_RACKET_TYPES.get(ir_type, 'Any')
        return RACKET_TYPES.get(ir_type, 'any')
    
    def get_contract(self, ir_type: str) -> str:
        """Get Racket contract predicate.
        
        Args:
            ir_type: IR type string.
            
        Returns:
            Racket contract predicate.
        """
        return RACKET_CONTRACTS.get(ir_type, 'any/c')
    
    def emit_type_annotation(self, name: str, param_types: List[str], return_type: str) -> str:
        """Emit Typed Racket type annotation.
        
        Args:
            name: Function name.
            param_types: List of parameter IR types.
            return_type: Return IR type.
            
        Returns:
            Type annotation string.
        """
        if not self.use_typed:
            return ""
        
        typed_params = ' '.join(TYPED_RACKET_TYPES.get(t, 'Any') for t in param_types)
        typed_ret = TYPED_RACKET_TYPES.get(return_type, 'Any')
        
        return f"(: {name} (-> {typed_params} {typed_ret}))"
    
    def emit_contract(self, param_types: List[str], return_type: str) -> str:
        """Emit Racket contract.
        
        Args:
            param_types: List of parameter IR types.
            return_type: Return IR type.
            
        Returns:
            Contract expression.
        """
        if not self.use_contracts:
            return ""
        
        contract_params = ' '.join(RACKET_CONTRACTS.get(t, 'any/c') for t in param_types)
        contract_ret = RACKET_CONTRACTS.get(return_type, 'any/c')
        
        return f"(-> {contract_params} {contract_ret})"
