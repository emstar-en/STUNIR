#!/usr/bin/env python3
"""Type mapping for ECLiPSe Constraint Logic Programming System.

Maps STUNIR IR types to ECLiPSe types and provides type conversion
utilities with support for multiple constraint libraries (IC, FD, R, Q).

Key ECLiPSe features:
- IC (Interval Constraints): Most powerful, unified integers/reals
- FD (Finite Domains): Classic CLP(FD) with # operators
- R (Real numbers): Real arithmetic constraints
- Q (Rationals): Rational number constraints

Part of Phase 5D-2: ECLiPSe with Constraint Optimization.
"""

from typing import Dict, Any, Optional


# Type mapping from IR types to ECLiPSe types
ECLIPSE_TYPES: Dict[str, str] = {
    # Numeric types (integers)
    'i8': 'integer',
    'i16': 'integer',
    'i32': 'integer',
    'i64': 'integer',
    'u8': 'integer',
    'u16': 'integer',
    'u32': 'integer',
    'u64': 'integer',
    
    # Floating point (real in ECLiPSe)
    'f32': 'real',
    'f64': 'real',
    'number': 'number',
    
    # Boolean (represented as 0/1 integers in ECLiPSe)
    'bool': 'integer',
    'boolean': 'integer',
    
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
    
    # CLP variable types
    'fd_var': 'ic_variable',      # Finite domain variable (using IC)
    'ic_var': 'ic_variable',      # Interval constraint variable
    'real_var': 'ic_variable',    # Real variable (IC handles both)
    'int_var': 'ic_variable',     # Integer variable
}


# IC library operators (recommended, most powerful)
# IC uses $ prefix for constraint operators
IC_OPERATORS: Dict[str, str] = {
    # Equality
    '==': '$=',
    '=': '$=',
    '#=': '$=',
    'eq': '$=',
    
    # Not equal
    '!=': '$\\=',
    '\\=': '$\\=',
    '#\\=': '$\\=',
    'neq': '$\\=',
    
    # Less than
    '<': '$<',
    '#<': '$<',
    'lt': '$<',
    
    # Greater than
    '>': '$>',
    '#>': '$>',
    'gt': '$>',
    
    # Less or equal
    '<=': '$=<',
    '=<': '$=<',
    '#=<': '$=<',
    'le': '$=<',
    'leq': '$=<',
    
    # Greater or equal
    '>=': '$>=',
    '#>=': '$>=',
    'ge': '$>=',
    'geq': '$>=',
}


# FD library operators (alternative, classic CLP(FD))
# FD uses # prefix like GNU Prolog
FD_OPERATORS: Dict[str, str] = {
    # Equality
    '==': '#=',
    '=': '#=',
    '$=': '#=',
    'eq': '#=',
    
    # Not equal
    '!=': '#\\=',
    '\\=': '#\\=',
    '$\\=': '#\\=',
    'neq': '#\\=',
    
    # Less than
    '<': '#<',
    '$<': '#<',
    'lt': '#<',
    
    # Greater than
    '>': '#>',
    '$>': '#>',
    'gt': '#>',
    
    # Less or equal
    '<=': '#=<',
    '=<': '#=<',
    '$=<': '#=<',
    'le': '#=<',
    
    # Greater or equal
    '>=': '#>=',
    '$>=': '#>=',
    'ge': '#>=',
}


# Global constraint mapping (require lib(ic_global))
ECLIPSE_GLOBALS: Dict[str, str] = {
    # All different
    'all_different': 'alldifferent',
    'fd_all_different': 'alldifferent',
    'alldistinct': 'alldifferent',
    
    # Element constraint
    'element': 'element',
    'nth': 'element',
    
    # Scheduling constraints
    'cumulative': 'cumulative',
    'disjunctive': 'disjunctive',
    
    # Circuit constraints
    'circuit': 'circuit',
    'subcircuit': 'subcircuit',
    
    # Cardinality constraints  
    'gcc': 'gcc',
    'global_cardinality': 'gcc',
    
    # Sorting
    'sorted': 'sorted',
    'ordered': 'ordered',
    
    # Lex ordering
    'lex_le': 'lex_le',
    'lex_lt': 'lex_lt',
}


# Optimization predicates mapping
ECLIPSE_OPTIMIZATION: Dict[str, str] = {
    # Simple minimize/maximize
    'minimize': 'minimize',
    'maximize': 'maximize',
    
    # From GNU Prolog
    'fd_minimize': 'bb_min',
    'fd_maximize': 'bb_max',
    
    # Branch-and-bound
    'bb_min': 'bb_min',
    'bb_max': 'bb_max',
    
    # Cost-based
    'cost_minimize': 'bb_min',
}


# Search predicates mapping
ECLIPSE_SEARCH: Dict[str, str] = {
    # Simple labeling
    'labeling': 'labeling',
    'fd_labeling': 'labeling',
    'label': 'labeling',
    
    # Advanced search
    'search': 'search',
}


# Search variable selection strategies
ECLIPSE_SELECT_METHODS: Dict[str, str] = {
    # Most constrained first
    'first_fail': 'first_fail',
    'ff': 'first_fail',
    'smallest': 'first_fail',
    
    # Most constrained (by constraint count)
    'most_constrained': 'most_constrained',
    'mc': 'most_constrained',
    
    # By occurrence
    'occurrence': 'occurrence',
    'occ': 'occurrence',
    
    # Input order
    'input_order': 'input_order',
    'leftmost': 'input_order',
    
    # Max regret
    'max_regret': 'max_regret',
    'regret': 'max_regret',
    
    # Anti first fail
    'anti_first_fail': 'anti_first_fail',
    'largest': 'anti_first_fail',
}


# Search value choice strategies
ECLIPSE_CHOICE_METHODS: Dict[str, str] = {
    # Simple indomain
    'indomain': 'indomain',
    
    # Start from middle
    'indomain_middle': 'indomain_middle',
    'middle': 'indomain_middle',
    
    # Start from minimum
    'indomain_min': 'indomain_min',
    'min': 'indomain_min',
    
    # Start from maximum
    'indomain_max': 'indomain_max',
    'max': 'indomain_max',
    
    # Random
    'indomain_random': 'indomain_random',
    'random': 'indomain_random',
    
    # Split domain
    'indomain_split': 'indomain_split',
    'split': 'indomain_split',
    
    # Reverse split
    'indomain_reverse_split': 'indomain_reverse_split',
    'rsplit': 'indomain_reverse_split',
}


# ECLiPSe library names
ECLIPSE_LIBRARIES: Dict[str, str] = {
    'ic': 'ic',                         # Interval Constraints (main)
    'fd': 'fd',                         # Finite Domains
    'ic_global': 'ic_global',           # Global constraints for IC
    'ic_search': 'ic_search',           # Advanced search for IC
    'branch_and_bound': 'branch_and_bound',  # Optimization
    'listut': 'listut',                 # List utilities
    'suspend': 'suspend',               # Suspension/delay
    'ic_symbolic': 'ic_symbolic',       # Symbolic constraints
}


class ECLiPSeTypeMapper:
    """Maps STUNIR types to ECLiPSe types with CLP support.
    
    Provides type conversion and constraint operator mapping
    for generating proper ECLiPSe code with IC or FD libraries.
    """
    
    def __init__(self, config: Any = None):
        """Initialize type mapper.
        
        Args:
            config: Emitter configuration (ECLiPSeConfig)
        """
        self.config = config
        self.type_map = dict(ECLIPSE_TYPES)
        
        # Determine which library to use (default IC)
        self._library = 'ic'
        if config and hasattr(config, 'default_library'):
            self._library = config.default_library
    
    @property
    def library(self) -> str:
        """Get active constraint library."""
        return self._library
    
    @library.setter
    def library(self, value: str) -> None:
        """Set active constraint library."""
        if value in ('ic', 'fd', 'r', 'q'):
            self._library = value
    
    def map_type(self, ir_type: str) -> str:
        """Map IR type to ECLiPSe type.
        
        Args:
            ir_type: STUNIR IR type name
            
        Returns:
            ECLiPSe type name
        """
        # Handle compound types like list(i32)
        if '(' in ir_type:
            base = ir_type.split('(')[0]
            return self.type_map.get(base, 'term')
        
        return self.type_map.get(ir_type, 'term')
    
    def is_constraint_operator(self, op: str) -> bool:
        """Check if operator is a constraint operator.
        
        Args:
            op: Operator string
            
        Returns:
            True if constraint operator
        """
        return op in IC_OPERATORS or op in FD_OPERATORS or op.startswith('$') or op.startswith('#')
    
    def map_constraint_operator(self, op: str) -> str:
        """Map operator to ECLiPSe constraint operator.
        
        Uses IC library by default, FD if configured.
        
        Args:
            op: Input operator
            
        Returns:
            ECLiPSe constraint operator
        """
        if self._library == 'fd':
            return FD_OPERATORS.get(op, op)
        return IC_OPERATORS.get(op, op)
    
    def is_global_constraint(self, pred: str) -> bool:
        """Check if predicate is a global constraint.
        
        Args:
            pred: Predicate name
            
        Returns:
            True if global constraint
        """
        return pred.lower() in ECLIPSE_GLOBALS or pred in ECLIPSE_GLOBALS.values()
    
    def map_global_constraint(self, pred: str) -> str:
        """Map predicate to ECLiPSe global constraint name.
        
        Args:
            pred: Input predicate name
            
        Returns:
            ECLiPSe global constraint name
        """
        return ECLIPSE_GLOBALS.get(pred.lower(), ECLIPSE_GLOBALS.get(pred, pred))
    
    def is_optimization_predicate(self, pred: str) -> bool:
        """Check if predicate is an optimization predicate.
        
        Args:
            pred: Predicate name
            
        Returns:
            True if optimization predicate
        """
        return pred.lower() in ECLIPSE_OPTIMIZATION or pred in ECLIPSE_OPTIMIZATION.values()
    
    def map_optimization_predicate(self, pred: str) -> str:
        """Map predicate to ECLiPSe optimization predicate.
        
        Args:
            pred: Input predicate name
            
        Returns:
            ECLiPSe optimization predicate name
        """
        return ECLIPSE_OPTIMIZATION.get(pred.lower(), ECLIPSE_OPTIMIZATION.get(pred, pred))
    
    def is_search_predicate(self, pred: str) -> bool:
        """Check if predicate is a search predicate.
        
        Args:
            pred: Predicate name
            
        Returns:
            True if search predicate
        """
        return pred.lower() in ECLIPSE_SEARCH
    
    def map_select_method(self, method: str) -> str:
        """Map variable selection method.
        
        Args:
            method: Selection method name
            
        Returns:
            ECLiPSe selection method
        """
        return ECLIPSE_SELECT_METHODS.get(method.lower(), method)
    
    def map_choice_method(self, method: str) -> str:
        """Map value choice method.
        
        Args:
            method: Choice method name
            
        Returns:
            ECLiPSe choice method
        """
        return ECLIPSE_CHOICE_METHODS.get(method.lower(), method)
    
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
    
    def is_real(self, ir_type: str) -> bool:
        """Check if type is real (floating point).
        
        Args:
            ir_type: IR type name
            
        Returns:
            True if real type
        """
        return ir_type in ('f32', 'f64', 'real_var')
    
    def is_integer(self, ir_type: str) -> bool:
        """Check if type is integer.
        
        Args:
            ir_type: IR type name
            
        Returns:
            True if integer type
        """
        return ir_type in ('i8', 'i16', 'i32', 'i64',
                          'u8', 'u16', 'u32', 'u64',
                          'fd_var', 'ic_var', 'int_var')
    
    def get_required_libraries(self, has_constraints: bool = False,
                               has_globals: bool = False,
                               has_optimization: bool = False,
                               has_search: bool = False) -> list:
        """Get list of required ECLiPSe libraries.
        
        Args:
            has_constraints: Whether constraints are used
            has_globals: Whether global constraints are used
            has_optimization: Whether optimization is used
            has_search: Whether advanced search is used
            
        Returns:
            List of library names to import
        """
        libs = []
        
        if has_constraints:
            libs.append(self._library)  # ic or fd
        
        if has_globals and self._library == 'ic':
            libs.append('ic_global')
        
        if has_search and self._library == 'ic':
            libs.append('ic_search')
        
        if has_optimization:
            libs.append('branch_and_bound')
        
        return libs


__all__ = [
    'ECLiPSeTypeMapper',
    'ECLIPSE_TYPES',
    'IC_OPERATORS',
    'FD_OPERATORS',
    'ECLIPSE_GLOBALS',
    'ECLIPSE_OPTIMIZATION',
    'ECLIPSE_SEARCH',
    'ECLIPSE_SELECT_METHODS',
    'ECLIPSE_CHOICE_METHODS',
    'ECLIPSE_LIBRARIES',
]
