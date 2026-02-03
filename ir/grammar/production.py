#!/usr/bin/env python3
"""Production rule representation for grammars.

This module defines:
- ProductionRule: A single production rule A → α
- EBNF operators: OptionalOp, Repetition, OneOrMore, Group, Alternation
"""

from dataclasses import dataclass, field
from typing import Optional as OptionalType, Tuple, Union, List, Any

from ir.grammar.symbol import Symbol, SymbolType, EPSILON


@dataclass(frozen=True)
class EBNFOperator:
    """Base class for EBNF extension operators."""
    pass


@dataclass(frozen=True)
class OptionalOp(EBNFOperator):
    """Optional element: [x] or x?
    
    Represents an element that can appear zero or one time.
    
    Example:
        >>> opt_num = OptionalOp(num)  # num?
    """
    element: Union[Symbol, 'Group', 'EBNFOperator']

# Alias for backward compatibility
Optional = OptionalOp


@dataclass(frozen=True)
class Repetition(EBNFOperator):
    """Zero or more repetitions: {x} or x*
    
    Represents an element that can appear zero or more times.
    
    Example:
        >>> nums = Repetition(num)  # num*
    """
    element: Union[Symbol, 'Group', 'EBNFOperator']


@dataclass(frozen=True)
class OneOrMore(EBNFOperator):
    """One or more repetitions: x+
    
    Represents an element that must appear at least once.
    
    Example:
        >>> digits = OneOrMore(digit)  # digit+
    """
    element: Union[Symbol, 'Group', 'EBNFOperator']


@dataclass(frozen=True)
class Group(EBNFOperator):
    """Grouped elements: (a b c)
    
    Represents a sequence of elements treated as a unit.
    
    Example:
        >>> group = Group((a, b, c))  # (a b c)
    """
    elements: Tuple[Union[Symbol, 'EBNFOperator'], ...]


@dataclass(frozen=True)
class Alternation(EBNFOperator):
    """Alternation: a | b | c
    
    Represents a choice between alternatives.
    
    Example:
        >>> alt = Alternation((a, b, c))  # a | b | c
    """
    alternatives: Tuple[Union[Symbol, 'EBNFOperator', Tuple], ...]


# Type alias for production body elements
BodyElement = Union[Symbol, EBNFOperator]


@dataclass
class ProductionRule:
    """A single production rule: A → α.
    
    Represents a context-free grammar production rule where:
    - head: The left-hand side non-terminal
    - body: The right-hand side sequence of symbols
    - label: Optional name for the production (useful for AST construction)
    - action: Optional semantic action code
    - precedence: Optional precedence level for conflict resolution
    
    Attributes:
        head: Left-hand side non-terminal symbol
        body: Right-hand side sequence of symbols (can be empty for ε production)
        label: Optional production label for identification
        action: Optional semantic action code string
        precedence: Optional precedence level (higher = tighter binding)
    
    Examples:
        >>> E = Symbol("E", SymbolType.NONTERMINAL)
        >>> num = Symbol("num", SymbolType.TERMINAL)
        >>> plus = Symbol("+", SymbolType.TERMINAL)
        
        >>> # E → E + num
        >>> rule1 = ProductionRule(E, (E, plus, num))
        
        >>> # E → num (with label)
        >>> rule2 = ProductionRule(E, (num,), label="literal")
        
        >>> # E → ε (epsilon production)
        >>> rule3 = ProductionRule(E, ())
    """
    head: Symbol
    body: Tuple[BodyElement, ...] = field(default_factory=tuple)
    label: OptionalType[str] = None
    action: OptionalType[str] = None
    precedence: OptionalType[int] = None
    
    def __post_init__(self):
        """Validate the production rule after initialization."""
        if not self.head.is_nonterminal():
            raise ValueError(f"Production head must be a non-terminal, got: {self.head}")
        # Ensure body is a tuple
        if not isinstance(self.body, tuple):
            object.__setattr__(self, 'body', tuple(self.body))
    
    def is_epsilon_production(self) -> bool:
        """Check if this is an epsilon (empty) production.
        
        Returns:
            True if the production body is empty or contains only epsilon.
        """
        if len(self.body) == 0:
            return True
        if len(self.body) == 1 and isinstance(self.body[0], Symbol) and self.body[0].is_epsilon():
            return True
        return False
    
    def body_symbols(self) -> List[Symbol]:
        """Get all symbols in the body (flattening EBNF operators).
        
        Returns:
            List of Symbol objects in the body.
        """
        symbols = []
        
        def extract_symbols(elem: BodyElement):
            if isinstance(elem, Symbol):
                symbols.append(elem)
            elif isinstance(elem, (Optional, Repetition, OneOrMore)):
                extract_symbols(elem.element)
            elif isinstance(elem, Group):
                for e in elem.elements:
                    extract_symbols(e)
            elif isinstance(elem, Alternation):
                for alt in elem.alternatives:
                    if isinstance(alt, tuple):
                        for e in alt:
                            extract_symbols(e)
                    else:
                        extract_symbols(alt)
        
        for elem in self.body:
            extract_symbols(elem)
        
        return symbols
    
    def terminals(self) -> List[Symbol]:
        """Get all terminal symbols in the body.
        
        Returns:
            List of terminal Symbol objects.
        """
        return [s for s in self.body_symbols() if s.is_terminal()]
    
    def nonterminals(self) -> List[Symbol]:
        """Get all non-terminal symbols in the body.
        
        Returns:
            List of non-terminal Symbol objects.
        """
        return [s for s in self.body_symbols() if s.is_nonterminal()]
    
    def contains_ebnf(self) -> bool:
        """Check if the body contains any EBNF operators.
        
        Returns:
            True if any element in the body is an EBNF operator.
        """
        return any(isinstance(elem, EBNFOperator) for elem in self.body)
    
    def __str__(self) -> str:
        """Return BNF-style string representation.
        
        Returns:
            String like "A → B C" or "A → ε" for empty productions.
        """
        if self.is_epsilon_production():
            body_str = "ε"
        else:
            body_str = " ".join(_element_to_str(elem) for elem in self.body)
        
        result = f"{self.head.name} → {body_str}"
        if self.label:
            result += f"  # {self.label}"
        return result
    
    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        return f"ProductionRule({self.head.name!r}, {self.body!r})"
    
    def __eq__(self, other: object) -> bool:
        """Check equality with another production rule."""
        if not isinstance(other, ProductionRule):
            return False
        return (self.head == other.head and 
                self.body == other.body and
                self.label == other.label)
    
    def __hash__(self) -> int:
        """Return hash value for use in sets and dicts."""
        return hash((self.head, self.body, self.label))


def _element_to_str(elem: BodyElement) -> str:
    """Convert a body element to string representation."""
    if isinstance(elem, Symbol):
        return str(elem)
    elif isinstance(elem, Optional):
        return f"[{_element_to_str(elem.element)}]"
    elif isinstance(elem, Repetition):
        return f"{{{_element_to_str(elem.element)}}}"
    elif isinstance(elem, OneOrMore):
        return f"{_element_to_str(elem.element)}+"
    elif isinstance(elem, Group):
        inner = " ".join(_element_to_str(e) for e in elem.elements)
        return f"({inner})"
    elif isinstance(elem, Alternation):
        parts = []
        for alt in elem.alternatives:
            if isinstance(alt, tuple):
                parts.append(" ".join(_element_to_str(e) for e in alt))
            else:
                parts.append(_element_to_str(alt))
        return f"({' | '.join(parts)})"
    else:
        return str(elem)
