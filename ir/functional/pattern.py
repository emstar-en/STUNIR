#!/usr/bin/env python3
"""STUNIR Functional IR - Pattern Matching Definitions.

This module defines pattern types for functional pattern matching.

Usage:
    from ir.functional.pattern import Pattern, VarPattern, ConstructorPattern
    
    # Create a constructor pattern for Maybe type
    just_pattern = ConstructorPattern(
        constructor='Just',
        args=[VarPattern(name='x')]
    )
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any
from ir.functional.functional_ir import FunctionalNode


@dataclass
class Pattern(FunctionalNode):
    """Base class for patterns."""
    kind: str = 'pattern'


@dataclass
class WildcardPattern(Pattern):
    """Wildcard pattern (_).
    
    Matches any value and binds nothing.
    """
    kind: str = 'wildcard'


@dataclass
class VarPattern(Pattern):
    """Variable pattern (binds to name).
    
    Matches any value and binds it to the given name.
    """
    name: str = ''
    kind: str = 'var_pattern'


@dataclass
class LiteralPattern(Pattern):
    """Literal pattern (match exact value).
    
    Matches if the scrutinee equals the literal value.
    """
    value: Any = None
    literal_type: str = 'int'  # 'int', 'float', 'bool', 'string', 'char'
    kind: str = 'literal_pattern'


@dataclass
class ConstructorPattern(Pattern):
    """Constructor pattern (e.g., Just x, Cons h t).
    
    Matches a data constructor and its arguments.
    """
    constructor: str = ''
    args: List['Pattern'] = field(default_factory=list)
    kind: str = 'constructor_pattern'


@dataclass
class TuplePattern(Pattern):
    """Tuple pattern (a, b, c).
    
    Matches a tuple with the given element patterns.
    """
    elements: List['Pattern'] = field(default_factory=list)
    kind: str = 'tuple_pattern'


@dataclass
class ListPattern(Pattern):
    """List pattern [a, b, c] or (h:t).
    
    Can match literal lists or cons patterns.
    
    Attributes:
        elements: Patterns for list elements
        rest: Optional tail pattern for cons (h:t)
    """
    elements: List['Pattern'] = field(default_factory=list)
    rest: Optional['Pattern'] = None  # For cons patterns
    kind: str = 'list_pattern'


@dataclass
class AsPattern(Pattern):
    """As pattern (x@pattern).
    
    Matches the pattern and binds the whole value to name.
    """
    name: str = ''
    pattern: 'Pattern' = None
    kind: str = 'as_pattern'


@dataclass
class RecordPattern(Pattern):
    """Record pattern { field1 = p1; field2 = p2 }.
    
    Matches a record with specific field patterns.
    """
    fields: dict = field(default_factory=dict)  # field_name -> Pattern
    kind: str = 'record_pattern'


@dataclass
class OrPattern(Pattern):
    """Or pattern (p1 | p2).
    
    Matches if either pattern matches (OCaml-specific).
    """
    left: 'Pattern' = None
    right: 'Pattern' = None
    kind: str = 'or_pattern'


# =============================================================================
# Pattern Utilities
# =============================================================================

def get_pattern_variables(pattern: Pattern) -> List[str]:
    """Extract all variable names bound by a pattern.
    
    Args:
        pattern: Pattern to analyze
        
    Returns:
        List of bound variable names
    """
    if isinstance(pattern, WildcardPattern):
        return []
    elif isinstance(pattern, VarPattern):
        return [pattern.name]
    elif isinstance(pattern, LiteralPattern):
        return []
    elif isinstance(pattern, ConstructorPattern):
        vars = []
        for arg in pattern.args:
            vars.extend(get_pattern_variables(arg))
        return vars
    elif isinstance(pattern, TuplePattern):
        vars = []
        for elem in pattern.elements:
            vars.extend(get_pattern_variables(elem))
        return vars
    elif isinstance(pattern, ListPattern):
        vars = []
        for elem in pattern.elements:
            vars.extend(get_pattern_variables(elem))
        if pattern.rest:
            vars.extend(get_pattern_variables(pattern.rest))
        return vars
    elif isinstance(pattern, AsPattern):
        vars = [pattern.name]
        if pattern.pattern:
            vars.extend(get_pattern_variables(pattern.pattern))
        return vars
    elif isinstance(pattern, RecordPattern):
        vars = []
        for field_pattern in pattern.fields.values():
            vars.extend(get_pattern_variables(field_pattern))
        return vars
    elif isinstance(pattern, OrPattern):
        # Both alternatives must bind the same variables
        return get_pattern_variables(pattern.left)
    return []


def is_exhaustive(patterns: List[Pattern], adt_constructors: List[str]) -> bool:
    """Check if a list of patterns is exhaustive for given constructors.
    
    Args:
        patterns: List of patterns to check
        adt_constructors: List of all constructor names for the type
        
    Returns:
        True if patterns cover all cases
    """
    # Simple check - look for wildcard or variable patterns
    for pattern in patterns:
        if isinstance(pattern, (WildcardPattern, VarPattern)):
            return True
    
    # Check if all constructors are covered
    covered = set()
    for pattern in patterns:
        if isinstance(pattern, ConstructorPattern):
            covered.add(pattern.constructor)
    
    return covered >= set(adt_constructors)


def simplify_pattern(pattern: Pattern) -> Pattern:
    """Simplify a pattern by removing redundant structure.
    
    Args:
        pattern: Pattern to simplify
        
    Returns:
        Simplified pattern
    """
    if isinstance(pattern, TuplePattern) and len(pattern.elements) == 1:
        return simplify_pattern(pattern.elements[0])
    elif isinstance(pattern, ListPattern) and not pattern.elements and not pattern.rest:
        return ConstructorPattern(constructor='[]')
    elif isinstance(pattern, AsPattern) and isinstance(pattern.pattern, WildcardPattern):
        return VarPattern(name=pattern.name)
    return pattern
