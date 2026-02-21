#!/usr/bin/env python3
"""Mercury Type System and Mode/Determinism Mapping.

Maps STUNIR IR types to Mercury types, handles mode declarations,
and provides determinism inference for Mercury predicates.

Mercury requires mandatory type, mode, and determinism declarations
for all predicates and functions, unlike traditional Prolog.

Part of Phase 5D-3: Mercury Emitter.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class MercuryMode(Enum):
    """Mercury mode declarations for predicate arguments.
    
    Modes describe the instantiation state of arguments on call and success.
    """
    IN = "in"           # Input: must be ground on call
    OUT = "out"         # Output: free on call, ground on success
    IN_OUT = "in_out"   # Bidirectional: ground on call, may be more ground
    UI = "ui"           # Unique input: unique on call, dead after
    UO = "uo"           # Unique output: free on call, unique on success
    DI = "di"           # Dead input: dead after call
    UU = "uu"           # Unused: unused on success
    
    def __str__(self) -> str:
        return self.value


class Determinism(Enum):
    """Mercury determinism categories.
    
    Describes how many solutions a predicate can produce and whether it can fail.
    """
    DET = "det"             # Exactly one solution, cannot fail
    SEMIDET = "semidet"     # At most one solution, can fail
    MULTI = "multi"         # At least one solution, cannot fail
    NONDET = "nondet"       # Any number of solutions (0+), can fail
    FAILURE = "failure"     # Always fails, no solutions
    ERRONEOUS = "erroneous" # Never returns (raises exception)
    CC_MULTI = "cc_multi"   # Committed choice multi
    CC_NONDET = "cc_nondet" # Committed choice nondet
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def can_fail(self) -> bool:
        """Whether this determinism category can fail."""
        return self in (Determinism.SEMIDET, Determinism.NONDET, 
                       Determinism.FAILURE, Determinism.CC_NONDET)
    
    @property
    def at_most_one(self) -> bool:
        """Whether this determinism produces at most one solution."""
        return self in (Determinism.DET, Determinism.SEMIDET,
                       Determinism.FAILURE, Determinism.ERRONEOUS)


class Purity(Enum):
    """Mercury purity annotations.
    
    Mercury tracks side effects through purity declarations.
    """
    PURE = "pure"           # No side effects
    SEMIPURE = "semipure"   # Reads global state only
    IMPURE = "impure"       # Full side effects
    
    def __str__(self) -> str:
        return self.value


# Type mapping from IR types to Mercury types
MERCURY_TYPES: Dict[str, str] = {
    # Integer types (Mercury has arbitrary precision integers)
    'i8': 'int',
    'i16': 'int',
    'i32': 'int',
    'i64': 'int',
    'u8': 'int',
    'u16': 'int',
    'u32': 'int',
    'u64': 'int',
    'int': 'int',
    'integer': 'int',
    
    # Floating point
    'f32': 'float',
    'f64': 'float',
    'float': 'float',
    'double': 'float',
    'number': 'float',
    
    # Boolean
    'bool': 'bool',
    'boolean': 'bool',
    
    # String types
    'string': 'string',
    'str': 'string',
    'char': 'char',
    'atom': 'string',  # Prolog atoms map to Mercury strings
    
    # Collections
    'list': 'list(T)',
    'array': 'array(T)',
    
    # Special types
    'void': '{}',       # Unit type
    'unit': '{}',
    'any': 'univ',      # Universal type
    'term': 'univ',
    
    # Higher-order types
    'pred': 'pred',
    'func': 'func',
}


# Mode mapping from IR modes to Mercury modes
MODE_MAPPING: Dict[str, MercuryMode] = {
    'input': MercuryMode.IN,
    'in': MercuryMode.IN,
    'output': MercuryMode.OUT,
    'out': MercuryMode.OUT,
    'bidirectional': MercuryMode.IN_OUT,
    'inout': MercuryMode.IN_OUT,
    'in_out': MercuryMode.IN_OUT,
    'ground': MercuryMode.IN,
    'free': MercuryMode.OUT,
    'unique_in': MercuryMode.UI,
    'unique_out': MercuryMode.UO,
    'dead': MercuryMode.DI,
    'unused': MercuryMode.UU,
}


# Standard Mercury library imports
MERCURY_IMPORTS: List[str] = [
    'int',
    'float', 
    'bool',
    'string',
    'char',
    'list',
    'io',
]


# Mercury reserved words (cannot be used as identifiers)
MERCURY_RESERVED: Set[str] = {
    'module', 'interface', 'implementation', 'import_module',
    'use_module', 'include_module', 'end_module',
    'type', 'pred', 'func', 'mode', 'inst', 'solver',
    'is', 'det', 'semidet', 'multi', 'nondet', 'cc_multi', 'cc_nondet',
    'failure', 'erroneous',
    'in', 'out', 'di', 'uo', 'ui', 'mdi', 'muo',
    'if', 'then', 'else', 'true', 'fail', 'false',
    'not', 'some', 'all', 'require_complete_switch',
    'promise_pure', 'promise_semipure', 'promise_impure',
    'impure', 'semipure',
    'where', 'with_type', 'with_inst',
}


class MercuryTypeMapper:
    """Maps STUNIR IR types to Mercury types with mode and determinism inference.
    
    Provides comprehensive type conversion, mode inference based on usage patterns,
    and determinism inference based on clause structure.
    """
    
    def __init__(self, infer_types: bool = True, infer_modes: bool = True,
                 infer_determinism: bool = True):
        """Initialize type mapper.
        
        Args:
            infer_types: Whether to infer types from structure
            infer_modes: Whether to infer modes from usage
            infer_determinism: Whether to infer determinism from clauses
        """
        self.infer_types = infer_types
        self.infer_modes = infer_modes
        self.infer_determinism = infer_determinism
        self.type_map = dict(MERCURY_TYPES)
        self.mode_map = dict(MODE_MAPPING)
    
    def map_type(self, ir_type: str) -> str:
        """Map IR type to Mercury type.
        
        Args:
            ir_type: STUNIR IR type name
            
        Returns:
            Mercury type string
        """
        if not ir_type:
            return 'univ'
        
        ir_type_lower = ir_type.lower()
        
        # Direct mapping
        if ir_type_lower in self.type_map:
            return self.type_map[ir_type_lower]
        
        # Handle parametric types like list(i32)
        if '(' in ir_type:
            base, rest = ir_type.split('(', 1)
            params = rest.rstrip(')')
            
            # For list type, get base without the (T)
            if base.lower() == 'list':
                base_mapped = 'list'
            elif base.lower() == 'array':
                base_mapped = 'array'
            else:
                base_mapped = self.type_map.get(base.lower(), base)
            
            # Map parameter types
            param_types = []
            for param in params.split(','):
                param = param.strip()
                param_mapped = self.map_type(param) if param else 'T'
                param_types.append(param_mapped)
            
            if param_types:
                return f"{base_mapped}({', '.join(param_types)})"
            return base_mapped
        
        # Handle higher-order types: pred(T) or func(T1, T2)
        if ir_type.startswith('pred'):
            return ir_type  # Pass through
        if ir_type.startswith('func'):
            return ir_type  # Pass through
        
        # Unknown type - use univ
        return 'univ'
    
    def map_mode(self, mode_str: str) -> MercuryMode:
        """Map mode string to MercuryMode.
        
        Args:
            mode_str: Mode string from IR
            
        Returns:
            MercuryMode enum value
        """
        if not mode_str:
            return MercuryMode.IN_OUT
        
        mode_lower = mode_str.lower()
        return self.mode_map.get(mode_lower, MercuryMode.IN_OUT)
    
    def infer_mode_from_param(self, param: Dict[str, Any], 
                               position: str = 'any') -> MercuryMode:
        """Infer mode from parameter information.
        
        Args:
            param: Parameter dictionary with optional 'mode', 'name' fields
            position: Position hint ('first', 'last', 'any')
            
        Returns:
            Inferred MercuryMode
        """
        # Explicit mode takes precedence
        if 'mode' in param:
            return self.map_mode(param['mode'])
        
        if not self.infer_modes:
            return MercuryMode.IN_OUT
        
        name = param.get('name', '').lower()
        
        # Heuristic: names suggest mode
        if name.startswith('result') or name.startswith('out'):
            return MercuryMode.OUT
        if name.startswith('in') or name.startswith('input'):
            return MercuryMode.IN
        
        # Positional heuristic
        if position == 'last':
            return MercuryMode.OUT
        if position == 'first':
            return MercuryMode.IN
        
        return MercuryMode.IN_OUT
    
    def infer_determinism_from_clauses(self, clauses: List[Any], 
                                        has_cut: bool = False) -> Determinism:
        """Infer determinism from clause structure.
        
        Args:
            clauses: List of clause objects or dicts
            has_cut: Whether predicate contains cut
            
        Returns:
            Inferred Determinism
        """
        if not self.infer_determinism:
            return Determinism.NONDET
        
        if not clauses:
            return Determinism.FAILURE
        
        num_clauses = len(clauses)
        
        # Single clause analysis
        if num_clauses == 1:
            clause = clauses[0]
            
            # Check for guards/conditions
            body = self._get_clause_body(clause)
            has_guard = self._has_guard(body)
            
            if has_guard or has_cut:
                return Determinism.SEMIDET
            return Determinism.DET
        
        # Multiple clauses
        if has_cut:
            return Determinism.SEMIDET
        
        # Check if patterns are disjoint (exhaustive enum)
        if self._patterns_exhaustive(clauses):
            return Determinism.MULTI
        
        return Determinism.NONDET
    
    def _get_clause_body(self, clause: Any) -> List[Any]:
        """Extract body from clause."""
        if hasattr(clause, 'body'):
            return clause.body or []
        if isinstance(clause, dict):
            return clause.get('body', [])
        return []
    
    def _has_guard(self, body: List[Any]) -> bool:
        """Check if body contains guards (conditionals that can fail)."""
        for goal in body:
            if hasattr(goal, 'kind'):
                kind = goal.kind.value if hasattr(goal.kind, 'value') else str(goal.kind)
            elif isinstance(goal, dict):
                kind = goal.get('kind', '')
            else:
                continue
            
            if kind in ('negation', 'if_then_else', 'unification'):
                return True
        return False
    
    def _patterns_exhaustive(self, clauses: List[Any]) -> bool:
        """Check if clause patterns are exhaustive (cover all cases)."""
        # Simplified check - assume not exhaustive unless proven
        return False
    
    def format_type_annotation(self, type_str: str, mode: MercuryMode) -> str:
        """Format a type with mode annotation.
        
        Args:
            type_str: Mercury type string
            mode: Mode for this argument
            
        Returns:
            Formatted "type::mode" string
        """
        return f"{type_str}::{mode}"
    
    def format_pred_declaration(self, name: str, 
                                 params: List[Dict[str, Any]],
                                 determinism: Optional[Determinism] = None) -> str:
        """Format a predicate declaration with types, modes, determinism.
        
        Args:
            name: Predicate name
            params: List of parameter dicts with 'type', 'mode'
            determinism: Determinism category
            
        Returns:
            Mercury predicate declaration string
        """
        if not params:
            det = determinism or Determinism.DET
            return f":- pred {name} is {det}."
        
        # Build argument list with types and modes
        arg_strs = []
        for i, param in enumerate(params):
            ir_type = param.get('type', 'any')
            merc_type = self.map_type(ir_type)
            
            position = 'first' if i == 0 else ('last' if i == len(params) - 1 else 'any')
            mode = self.infer_mode_from_param(param, position)
            
            arg_strs.append(f"{merc_type}::{mode}")
        
        det = determinism or Determinism.DET
        args = ', '.join(arg_strs)
        return f":- pred {name}({args}) is {det}."
    
    def format_func_declaration(self, name: str,
                                 arg_types: List[str],
                                 return_type: str,
                                 determinism: Optional[Determinism] = None) -> str:
        """Format a function declaration.
        
        Args:
            name: Function name
            arg_types: List of argument types
            return_type: Return type
            determinism: Determinism (functions are usually det)
            
        Returns:
            Mercury function declaration string
        """
        args = ', '.join(self.map_type(t) for t in arg_types) if arg_types else ''
        ret = self.map_type(return_type)
        det = determinism or Determinism.DET
        
        if args:
            return f":- func {name}({args}) = {ret} is {det}."
        return f":- func {name} = {ret} is {det}."
    
    def is_numeric(self, ir_type: str) -> bool:
        """Check if type is numeric."""
        return ir_type.lower() in (
            'i8', 'i16', 'i32', 'i64',
            'u8', 'u16', 'u32', 'u64',
            'f32', 'f64', 'int', 'integer',
            'float', 'double', 'number'
        )
    
    def is_string(self, ir_type: str) -> bool:
        """Check if type is string-like."""
        return ir_type.lower() in ('string', 'str', 'char', 'atom')
    
    def is_list(self, ir_type: str) -> bool:
        """Check if type is a list type."""
        return ir_type.lower().startswith('list')


__all__ = [
    'MercuryMode',
    'Determinism', 
    'Purity',
    'MERCURY_TYPES',
    'MODE_MAPPING',
    'MERCURY_IMPORTS',
    'MERCURY_RESERVED',
    'MercuryTypeMapper',
]
