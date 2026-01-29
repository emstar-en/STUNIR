#!/usr/bin/env python3
"""Type mapping for YAP Prolog.

Maps STUNIR IR types to YAP Prolog types and provides
tabling/indexing utilities.

Part of Phase 5C-3: YAP with Tabling Support.
"""

from typing import Dict, Any, Optional, List


# Type mapping from IR types to YAP Prolog types
YAP_PROLOG_TYPES: Dict[str, str] = {
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
}


# Tabling modes supported by YAP
TABLING_MODES: Dict[str, str] = {
    'variant': 'variant',          # Default - cache based on variant calls
    'subsumptive': 'subsumptive',  # More aggressive caching
    'lattice': 'lattice',          # Aggregation tabling
    'monotonic': 'monotonic',      # Incremental tabling
}


# Mode declarations for predicate arguments (PlDoc-style)
MODE_DECLARATIONS: Dict[str, str] = {
    'input': '+',
    'output': '-',
    'bidirectional': '?',
    'ground': '@',
    'nonvar': '!',
}


class YAPPrologTypeMapper:
    """Maps STUNIR types to YAP Prolog types.
    
    Provides type conversion and tabling detection for
    generating proper YAP Prolog code.
    """
    
    def __init__(self, emit_declarations: bool = True):
        """Initialize type mapper.
        
        Args:
            emit_declarations: Whether to emit type declarations
        """
        self.emit_declarations = emit_declarations
        self.type_map = dict(YAP_PROLOG_TYPES)
    
    def map_type(self, ir_type: str) -> str:
        """Map IR type to Prolog type.
        
        Args:
            ir_type: STUNIR IR type name
            
        Returns:
            YAP Prolog type name
        """
        # Handle compound types like list(i32)
        if '(' in ir_type:
            base = ir_type.split('(')[0]
            return self.type_map.get(base, 'term')
        
        return self.type_map.get(ir_type, 'term')
    
    def infer_mode(self, param: Dict[str, Any]) -> str:
        """Infer argument mode from parameter info.
        
        Args:
            param: Parameter dictionary with optional 'mode' field
            
        Returns:
            Mode character (+, -, ?, etc.)
        """
        mode = param.get('mode', 'bidirectional')
        return MODE_DECLARATIONS.get(mode, '?')
    
    def format_type_declaration(self, name: str, params: list) -> str:
        """Format a type declaration for a predicate.
        
        Args:
            name: Predicate name
            params: List of parameter dictionaries
            
        Returns:
            PlDoc-style type declaration string
        """
        if not self.emit_declarations:
            return ''
        
        modes = []
        for param in params:
            mode_char = self.infer_mode(param)
            type_name = self.map_type(param.get('type', 'any'))
            modes.append(f"{mode_char}{type_name}")
        
        return f"%% {name}({', '.join(modes)})"
    
    def format_determinism(self, is_deterministic: bool, 
                           can_fail: bool = False) -> str:
        """Format determinism declaration.
        
        Args:
            is_deterministic: Whether predicate is deterministic
            can_fail: Whether predicate can fail
            
        Returns:
            Determinism indicator (det, nondet, semidet, etc.)
        """
        if is_deterministic and not can_fail:
            return 'det'
        elif is_deterministic and can_fail:
            return 'semidet'
        elif not is_deterministic and not can_fail:
            return 'multi'
        else:
            return 'nondet'
    
    def is_tabling_mode_valid(self, mode: str) -> bool:
        """Check if tabling mode is valid.
        
        Args:
            mode: Tabling mode string
            
        Returns:
            True if valid mode
        """
        return mode in TABLING_MODES
    
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
    'YAPPrologTypeMapper', 
    'YAP_PROLOG_TYPES', 
    'TABLING_MODES',
    'MODE_DECLARATIONS'
]
