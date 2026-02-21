"""Pattern matching for rule-based systems.

This module provides pattern elements and pattern matching
for the forward chaining inference engine.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from .rule_ir import PatternType


# Pattern Elements
@dataclass(frozen=True)
class PatternElement:
    """Base class for pattern elements."""
    pattern_type: PatternType


@dataclass(frozen=True)
class LiteralPattern(PatternElement):
    """Literal match pattern."""
    value: Any
    pattern_type: PatternType = field(default=PatternType.LITERAL, init=False)


@dataclass(frozen=True)
class VariablePattern(PatternElement):
    """Variable binding pattern (e.g., ?name)."""
    name: str
    constraint: Optional[str] = None  # Constraint expression
    pattern_type: PatternType = field(default=PatternType.VARIABLE, init=False)


@dataclass(frozen=True)
class WildcardPattern(PatternElement):
    """Single wildcard pattern (?)."""
    pattern_type: PatternType = field(default=PatternType.WILDCARD, init=False)


@dataclass(frozen=True)
class MultifieldPattern(PatternElement):
    """Multifield wildcard pattern ($?)."""
    name: Optional[str] = None  # Optional name for binding
    pattern_type: PatternType = field(default=PatternType.MULTIFIELD, init=False)


# Type alias for any pattern element
AnyPatternElement = Union[LiteralPattern, VariablePattern, WildcardPattern, MultifieldPattern]


class PatternMatcher:
    """Pattern matching for rule conditions."""
    
    def match_pattern(
        self,
        pattern: 'PatternCondition',
        fact: 'Fact',
        bindings: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Match a pattern against a fact.
        
        Returns updated bindings if match succeeds, None otherwise.
        
        Args:
            pattern: The pattern condition to match
            fact: The fact to match against
            bindings: Current variable bindings
            
        Returns:
            Updated bindings dict if match succeeds, None otherwise
        """
        # Check template name
        if pattern.template_name != fact.template_name:
            return None
        
        new_bindings = dict(bindings)
        
        if fact.is_template():
            # Match template slots
            for slot_name, pat_elem in pattern.patterns:
                if slot_name not in fact.slots:
                    return None
                value = fact.slots[slot_name]
                result = self._match_element(pat_elem, value, new_bindings)
                if result is None:
                    return None
                new_bindings = result
        else:
            # Match ordered fact values
            if len(pattern.ordered_patterns) != len(fact.values):
                return None
            for pat_elem, value in zip(pattern.ordered_patterns, fact.values):
                result = self._match_element(pat_elem, value, new_bindings)
                if result is None:
                    return None
                new_bindings = result
        
        # Bind the fact if requested
        if pattern.binding_name:
            new_bindings[pattern.binding_name] = fact
        
        return new_bindings
    
    def _match_element(
        self,
        pattern: AnyPatternElement,
        value: Any,
        bindings: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Match a single pattern element against a value."""
        
        if isinstance(pattern, LiteralPattern):
            if pattern.value == value:
                return bindings
            return None
        
        elif isinstance(pattern, VariablePattern):
            var_name = pattern.name
            if var_name in bindings:
                # Variable already bound, check consistency
                if bindings[var_name] == value:
                    return bindings
                return None
            else:
                # Check constraint if present
                if pattern.constraint:
                    if not self._evaluate_constraint(pattern.constraint, value, bindings):
                        return None
                # Bind variable
                new_bindings = dict(bindings)
                new_bindings[var_name] = value
                return new_bindings
        
        elif isinstance(pattern, WildcardPattern):
            return bindings  # Wildcard matches anything
        
        elif isinstance(pattern, MultifieldPattern):
            # Multifield matches any sequence (simplified)
            if pattern.name:
                new_bindings = dict(bindings)
                new_bindings[pattern.name] = value
                return new_bindings
            return bindings
        
        return None
    
    def _evaluate_constraint(
        self,
        constraint: str,
        value: Any,
        bindings: Dict[str, Any]
    ) -> bool:
        """Evaluate a constraint expression.
        
        Args:
            constraint: The constraint expression string
            value: The current value being matched
            bindings: Current variable bindings
            
        Returns:
            True if constraint is satisfied, False otherwise
        """
        # Simple constraint evaluation
        # In production, use a proper expression evaluator
        try:
            # Replace variable references
            expr = constraint
            for var, val in bindings.items():
                expr = expr.replace(f"?{var}", repr(val))
            expr = expr.replace("?", repr(value))
            # Safe evaluation with restricted builtins
            allowed_builtins = {
                'abs': abs, 'min': min, 'max': max,
                'len': len, 'str': str, 'int': int, 'float': float,
                'True': True, 'False': False, 'None': None
            }
            return bool(eval(expr, {"__builtins__": allowed_builtins}, {}))
        except Exception:
            return False


# Import Fact and PatternCondition for type hints (avoid circular import)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .fact import Fact
    from .rule import PatternCondition
