"""ASP Aggregate definitions.

This module defines aggregate expressions used in ASP rules,
including #count, #sum, #min, #max operations.

Part of Phase 7D: Answer Set Programming
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Union
from .asp_ir import AggregateFunction, ComparisonOp
from .atom import Term, Literal


@dataclass
class AggregateElement:
    """An element in an aggregate.
    
    Attributes:
        terms: Terms to aggregate (the 'weight' or value terms)
        condition: Literals forming the condition
    
    Examples:
        - { X : p(X) } - terms=[X], condition=[p(X)]
        - { W,I : cost(I,W), selected(I) } - terms=[W,I], condition=[cost(I,W), selected(I)]
    """
    terms: List[Term] = field(default_factory=list)
    condition: List[Literal] = field(default_factory=list)
    
    def get_variables(self) -> List[Term]:
        """Get all variables in this element."""
        result = [t for t in self.terms if t.is_variable]
        for lit in self.condition:
            result.extend(lit.get_variables())
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "terms": [t.to_dict() for t in self.terms],
            "condition": [lit.to_dict() for lit in self.condition]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AggregateElement':
        """Create from dictionary representation."""
        terms = [Term.from_dict(t) for t in data.get("terms", [])]
        condition = [Literal.from_dict(lit) for lit in data.get("condition", [])]
        return cls(terms=terms, condition=condition)
    
    def __str__(self) -> str:
        """Return string representation."""
        terms_str = ", ".join(str(t) for t in self.terms)
        if self.condition:
            cond_str = ", ".join(str(lit) for lit in self.condition)
            return f"{terms_str} : {cond_str}"
        return terms_str


@dataclass
class Guard:
    """A guard (bound) for an aggregate.
    
    Attributes:
        op: Comparison operator
        term: The bound value
    """
    op: ComparisonOp
    term: Term
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "op": self.op.value,
            "term": self.term.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Guard':
        """Create from dictionary representation."""
        op = ComparisonOp(data["op"])
        term = Term.from_dict(data["term"])
        return cls(op=op, term=term)
    
    def __str__(self) -> str:
        return f"{self.op.value} {self.term}"


@dataclass
class Aggregate:
    """An aggregate expression.
    
    Attributes:
        function: Aggregate function (#count, #sum, etc.)
        elements: Elements in the aggregate
        left_guard: Optional left comparison (e.g., 2 < #count{...})
        right_guard: Optional right comparison (e.g., #count{...} < 5)
    
    Examples:
        - #count { X : node(X) }
        - 2 <= #count { X : color(X,C) } <= 3
        - #sum { W,X : weight(X,W), selected(X) } < 100
    """
    function: AggregateFunction
    elements: List[AggregateElement] = field(default_factory=list)
    left_guard: Optional[Guard] = None
    right_guard: Optional[Guard] = None
    
    def add_element(self, terms: List[Term], condition: List[Literal]) -> 'Aggregate':
        """Add an element to the aggregate."""
        self.elements.append(AggregateElement(terms=terms, condition=condition))
        return self
    
    def set_left_guard(self, term: Term, op: ComparisonOp) -> 'Aggregate':
        """Set left guard (lower bound)."""
        self.left_guard = Guard(op=op, term=term)
        return self
    
    def set_right_guard(self, op: ComparisonOp, term: Term) -> 'Aggregate':
        """Set right guard (upper bound)."""
        self.right_guard = Guard(op=op, term=term)
        return self
    
    def get_variables(self) -> List[Term]:
        """Get all variables in this aggregate."""
        result = []
        for elem in self.elements:
            result.extend(elem.get_variables())
        if self.left_guard and self.left_guard.term.is_variable:
            result.append(self.left_guard.term)
        if self.right_guard and self.right_guard.term.is_variable:
            result.append(self.right_guard.term)
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "function": self.function.name,
            "elements": [elem.to_dict() for elem in self.elements]
        }
        if self.left_guard:
            result["left_guard"] = self.left_guard.to_dict()
        if self.right_guard:
            result["right_guard"] = self.right_guard.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Aggregate':
        """Create from dictionary representation."""
        function = AggregateFunction[data["function"]]
        elements = [AggregateElement.from_dict(e) for e in data.get("elements", [])]
        left_guard = None
        right_guard = None
        if "left_guard" in data and data["left_guard"]:
            left_guard = Guard.from_dict(data["left_guard"])
        if "right_guard" in data and data["right_guard"]:
            right_guard = Guard.from_dict(data["right_guard"])
        return cls(
            function=function,
            elements=elements,
            left_guard=left_guard,
            right_guard=right_guard
        )
    
    def __str__(self) -> str:
        """Return string representation."""
        func_str = self.function.to_clingo()
        elements_str = "; ".join(str(e) for e in self.elements)
        agg_str = f"{func_str} {{ {elements_str} }}"
        
        parts = []
        if self.left_guard:
            parts.append(f"{self.left_guard.term} {self.left_guard.op.value}")
        parts.append(agg_str)
        if self.right_guard:
            parts.append(f"{self.right_guard.op.value} {self.right_guard.term}")
        
        return " ".join(parts)


@dataclass
class Comparison:
    """A comparison expression in rule body.
    
    Attributes:
        left: Left operand
        op: Comparison operator
        right: Right operand
    
    Examples:
        - X < Y
        - Cost <= Budget
        - N = 5
    """
    left: Term
    op: ComparisonOp
    right: Term
    
    def get_variables(self) -> List[Term]:
        """Get all variables in this comparison."""
        result = []
        if self.left.is_variable:
            result.append(self.left)
        if self.right.is_variable:
            result.append(self.right)
        result.extend(self.left.get_variables())
        result.extend(self.right.get_variables())
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "left": self.left.to_dict(),
            "op": self.op.value,
            "right": self.right.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Comparison':
        """Create from dictionary representation."""
        left = Term.from_dict(data["left"])
        op = ComparisonOp(data["op"])
        right = Term.from_dict(data["right"])
        return cls(left=left, op=op, right=right)
    
    def __str__(self) -> str:
        return f"{self.left} {self.op.value} {self.right}"


# Factory functions
def count(elements: List[AggregateElement] = None) -> Aggregate:
    """Create a #count aggregate."""
    return Aggregate(AggregateFunction.COUNT, elements or [])


def sum_agg(elements: List[AggregateElement] = None) -> Aggregate:
    """Create a #sum aggregate."""
    return Aggregate(AggregateFunction.SUM, elements or [])


def min_agg(elements: List[AggregateElement] = None) -> Aggregate:
    """Create a #min aggregate."""
    return Aggregate(AggregateFunction.MIN, elements or [])


def max_agg(elements: List[AggregateElement] = None) -> Aggregate:
    """Create a #max aggregate."""
    return Aggregate(AggregateFunction.MAX, elements or [])


def agg_element(terms: List[Union[Term, str]], condition: List[Literal]) -> AggregateElement:
    """Create an aggregate element."""
    term_list = []
    for t in terms:
        if isinstance(t, str):
            term_list.append(Term(t))
        else:
            term_list.append(t)
    return AggregateElement(terms=term_list, condition=condition)


def compare(left: Union[Term, str], op: ComparisonOp, right: Union[Term, str]) -> Comparison:
    """Create a comparison."""
    l = left if isinstance(left, Term) else Term(left)
    r = right if isinstance(right, Term) else Term(right)
    return Comparison(left=l, op=op, right=r)
