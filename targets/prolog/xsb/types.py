#!/usr/bin/env python3
"""Type mapping for XSB Prolog.

Maps STUNIR IR types to XSB Prolog types and provides
advanced tabling mode utilities.

Key XSB features:
- Incremental tabling (`:- table pred/N as incremental`)
- Answer subsumption (`:- table pred/N as subsumptive`)
- Well-founded semantics (WFS)
- Lattice tabling operations
- Different module syntax from SWI/YAP

Part of Phase 5D-1: XSB with Advanced Tabling.
"""

from enum import Enum, auto
from typing import Dict, Any, Optional, List, Set


# Type mapping from IR types to XSB Prolog types
XSB_PROLOG_TYPES: Dict[str, str] = {
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
    
    # Boolean (XSB uses atoms true/false)
    'bool': 'atom',
    'boolean': 'atom',
    
    # String types
    'string': 'atom',
    'str': 'atom',
    'char': 'atom',
    
    # Collections
    'list': 'list',
    'array': 'list',
    
    # Special
    'void': 'true',
    'any': 'term',
    'term': 'term',
}


class TablingMode(Enum):
    """XSB tabling modes.
    
    XSB supports several tabling modes that can be combined:
    - VARIANT: Default call variance (standard tabling)
    - INCREMENTAL: Supports dynamic updates to tabled facts
    - SUBSUMPTIVE: More aggressive answer caching
    - OPAQUE: Treats calls as black boxes
    - PRIVATE: Not exported from module
    """
    VARIANT = auto()       # Default call variance
    INCREMENTAL = auto()   # Incremental evaluation
    SUBSUMPTIVE = auto()   # Answer subsumption
    OPAQUE = auto()        # Opaque tabling
    PRIVATE = auto()       # Private to module


# Tabling mode strings for XSB directives
XSB_TABLING_MODES: Dict[str, Optional[str]] = {
    'variant': None,           # Default, no modifier needed
    'incremental': 'incremental',
    'subsumptive': 'subsumptive',
    'opaque': 'opaque',
    'private': 'private',
}


# Lattice operations for answer subsumption
XSB_LATTICE_OPS: Dict[str, str] = {
    'min': 'min/3',
    'max': 'max/3',
    'or': 'or/3',
    'and': 'and/3',
    'join': 'join/3',
    'lub': 'lub/3',
    'glb': 'glb/3',
    'sum': 'sum/3',
    'count': 'count/3',
}


# Mode declarations for predicate arguments
MODE_DECLARATIONS: Dict[str, str] = {
    'input': '+',
    'output': '-',
    'bidirectional': '?',
    'ground': '@',
    'nonvar': '!',
}


# XSB built-in predicates that should not be redefined
XSB_BUILTINS: Set[str] = {
    # Tabling predicates
    'abolish_all_tables',
    'abolish_table_call',
    'abolish_table_pred',
    'get_calls',
    'get_returns',
    'table_state',
    'tfindall',
    'tbagof',
    'tsetof',
    'tnot',
    'sk_not',
    
    # Incremental tabling
    'incr_assert',
    'incr_retract',
    'incr_retractall',
    'incr_invalid_subgoals',
    'incr_table_update',
    
    # Module predicates
    'import',
    'export',
    
    # HiLog (Higher-order)
    'apply',
    'hilog',
    
    # Standard predicates
    'assert', 'asserta', 'assertz',
    'retract', 'retractall', 'abolish',
    'findall', 'bagof', 'setof',
    'functor', 'arg', 'copy_term',
    'call', 'once',
}


class XSBPrologTypeMapper:
    """Maps STUNIR types to XSB Prolog types.
    
    Provides type conversion and advanced tabling utilities
    for generating proper XSB Prolog code with:
    - Incremental tabling
    - Answer subsumption
    - Well-founded semantics
    - Lattice operations
    """
    
    def __init__(self, emit_declarations: bool = True):
        """Initialize type mapper.
        
        Args:
            emit_declarations: Whether to emit type declarations
        """
        self.emit_declarations = emit_declarations
        self.type_map = dict(XSB_PROLOG_TYPES)
    
    def map_type(self, ir_type: str) -> str:
        """Map IR type to XSB Prolog type.
        
        Args:
            ir_type: STUNIR IR type name
            
        Returns:
            XSB Prolog type name
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
            XSB-style type declaration string
        """
        if not self.emit_declarations:
            return ''
        
        modes = []
        for param in params:
            mode_char = self.infer_mode(param)
            type_name = self.map_type(param.get('type', 'any'))
            modes.append(f"{mode_char}{type_name}")
        
        return f"%% {name}({', '.join(modes)})"
    
    def format_tabling_directive(self, name: str, arity: int,
                                  mode: TablingMode = TablingMode.VARIANT,
                                  lattice_op: Optional[str] = None,
                                  max_answers: Optional[int] = None,
                                  additional_modes: Optional[List[TablingMode]] = None) -> str:
        """Format a tabling directive for XSB.
        
        Args:
            name: Predicate name
            arity: Predicate arity
            mode: Primary tabling mode
            lattice_op: Optional lattice operation
            max_answers: Optional answer limit
            additional_modes: Optional additional modes to combine
            
        Returns:
            XSB tabling directive string
        """
        # Collect all modes
        all_modes = [mode]
        if additional_modes:
            all_modes.extend(additional_modes)
        
        # Build modifier strings
        modifiers = []
        for m in all_modes:
            mode_str = XSB_TABLING_MODES.get(m.name.lower())
            if mode_str and mode_str not in modifiers:
                modifiers.append(mode_str)
        
        # Add max_answers if specified
        if max_answers:
            modifiers.append(f"max_answers({max_answers})")
        
        # Format output
        if lattice_op:
            # Lattice tabling with answer template
            op_str = XSB_LATTICE_OPS.get(lattice_op, f"{lattice_op}/3")
            args = ', '.join(['_'] * (arity - 1) + [f"lattice({op_str})"])
            return f":- table {name}({args})."
        elif modifiers:
            mod_str = ', '.join(modifiers)
            if len(modifiers) > 1:
                return f":- table {name}/{arity} as ({mod_str})."
            else:
                return f":- table {name}/{arity} as {mod_str}."
        else:
            return f":- table {name}/{arity}."
    
    def format_export_directive(self, name: str, arity: int) -> str:
        """Format an export directive for XSB.
        
        Args:
            name: Predicate name
            arity: Predicate arity
            
        Returns:
            XSB export directive
        """
        return f":- export({name}/{arity})."
    
    def format_import_directive(self, name: str, arity: int, module: str) -> str:
        """Format an import directive for XSB.
        
        Args:
            name: Predicate name
            arity: Predicate arity
            module: Source module name
            
        Returns:
            XSB import directive
        """
        return f":- import({name}/{arity} from {module})."
    
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
    
    def get_tabling_mode(self, mode_str: str) -> TablingMode:
        """Convert string to TablingMode enum.
        
        Args:
            mode_str: Mode string (e.g., 'incremental')
            
        Returns:
            TablingMode enum value
        """
        mode_map = {
            'variant': TablingMode.VARIANT,
            'incremental': TablingMode.INCREMENTAL,
            'subsumptive': TablingMode.SUBSUMPTIVE,
            'opaque': TablingMode.OPAQUE,
            'private': TablingMode.PRIVATE,
        }
        return mode_map.get(mode_str.lower(), TablingMode.VARIANT)
    
    def is_builtin(self, predicate: str) -> bool:
        """Check if predicate is an XSB builtin.
        
        Args:
            predicate: Predicate name
            
        Returns:
            True if XSB builtin
        """
        return predicate in XSB_BUILTINS
    
    def is_lattice_op(self, op: str) -> bool:
        """Check if operation is a valid lattice operation.
        
        Args:
            op: Operation name
            
        Returns:
            True if valid lattice operation
        """
        return op in XSB_LATTICE_OPS


__all__ = [
    'XSBPrologTypeMapper',
    'XSB_PROLOG_TYPES',
    'XSB_TABLING_MODES',
    'XSB_LATTICE_OPS',
    'MODE_DECLARATIONS',
    'XSB_BUILTINS',
    'TablingMode',
]
