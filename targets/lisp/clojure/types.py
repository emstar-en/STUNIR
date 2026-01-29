#!/usr/bin/env python3
"""Clojure Type Mapping.

Maps STUNIR IR types to Clojure type hints and specs.
Part of Phase 5A: Core Lisp Implementation.
"""

from typing import Dict, List


# Clojure type hints for performance
CLOJURE_TYPE_HINTS: Dict[str, str] = {
    'i8': '^byte',
    'i16': '^short',
    'i32': '^int',
    'i64': '^long',
    'u8': '^byte',
    'u16': '^short',
    'u32': '^int',
    'u64': '^long',
    'f32': '^float',
    'f64': '^double',
    'bool': '^boolean',
    'string': '^String',
    'void': '',  # No hint
    'any': '^Object',
    'array': '^"[Ljava.lang.Object;"',
}

# Clojure spec predicates
CLOJURE_TYPES: Dict[str, str] = {
    'i8': 'int?',
    'i16': 'int?',
    'i32': 'int?',
    'i64': 'int?',
    'u8': 'nat-int?',
    'u16': 'nat-int?',
    'u32': 'nat-int?',
    'u64': 'nat-int?',
    'f32': 'float?',
    'f64': 'double?',
    'bool': 'boolean?',
    'string': 'string?',
    'void': 'nil?',
    'any': 'any?',
    'list': 'sequential?',
    'map': 'map?',
    'set': 'set?',
    'keyword': 'keyword?',
    'symbol': 'symbol?',
    'fn': 'fn?',
}


class ClojureTypeMapper:
    """Maps IR types to Clojure types and hints."""
    
    def __init__(self, use_type_hints: bool = True, use_spec: bool = True):
        """Initialize type mapper.
        
        Args:
            use_type_hints: Whether to emit Java type hints.
            use_spec: Whether to generate clojure.spec definitions.
        """
        self.use_type_hints = use_type_hints
        self.use_spec = use_spec
    
    def map_type(self, ir_type: str) -> str:
        """Map IR type to Clojure spec predicate.
        
        Args:
            ir_type: IR type string.
            
        Returns:
            Clojure spec predicate.
        """
        return CLOJURE_TYPES.get(ir_type, 'any?')
    
    def get_type_hint(self, ir_type: str) -> str:
        """Get Clojure type hint.
        
        Args:
            ir_type: IR type string.
            
        Returns:
            Clojure type hint string (empty if no hint needed).
        """
        if not self.use_type_hints:
            return ""
        return CLOJURE_TYPE_HINTS.get(ir_type, '')
    
    def emit_spec_fdef(self, func_name: str, param_types: List[str], return_type: str) -> str:
        """Emit clojure.spec.alpha fdef.
        
        Args:
            func_name: Function name.
            param_types: List of parameter IR types.
            return_type: Return IR type.
            
        Returns:
            s/fdef form string.
        """
        if not self.use_spec:
            return ""
        
        # Build s/cat for args
        args_parts = []
        for i, ptype in enumerate(param_types):
            pred = self.map_type(ptype)
            args_parts.append(f":arg{i} {pred}")
        
        args_spec = f"(s/cat {' '.join(args_parts)})" if args_parts else "(s/cat)"
        ret_spec = self.map_type(return_type)
        
        return f"(s/fdef {func_name}\n  :args {args_spec}\n  :ret {ret_spec})"
    
    def emit_typed_param(self, param_name: str, ir_type: str) -> str:
        """Emit a parameter with type hint.
        
        Args:
            param_name: Parameter name.
            ir_type: IR type.
            
        Returns:
            Parameter string with optional type hint.
        """
        hint = self.get_type_hint(ir_type)
        if hint:
            return f"{hint} {param_name}"
        return param_name
