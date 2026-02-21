"""ASP Atom and Literal definitions.

This module defines Term, Atom, and Literal classes for representing
ASP predicates and their arguments.

Part of Phase 7D: Answer Set Programming
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from .asp_ir import NegationType, is_variable, validate_identifier


@dataclass
class Term:
    """A term in ASP (constant, variable, or function).
    
    Attributes:
        name: Term name (constant, variable name, or function symbol)
        args: Arguments for function terms (None for simple terms)
    
    Examples:
        - Constant: Term("red")
        - Variable: Term("X")
        - Integer: Term("42")
        - Function: Term("f", [Term("X"), Term("Y")])
    """
    name: str
    args: Optional[List['Term']] = None
    
    def __post_init__(self):
        """Validate term structure."""
        if not self.name:
            raise ValueError("Term name cannot be empty")
    
    @property
    def is_variable(self) -> bool:
        """Check if this is a variable (starts with uppercase)."""
        return is_variable(self.name)
    
    @property
    def is_constant(self) -> bool:
        """Check if this is a constant."""
        if self.args is not None:
            return False
        return not self.is_variable
    
    @property
    def is_function(self) -> bool:
        """Check if this is a function term."""
        return self.args is not None
    
    @property
    def is_integer(self) -> bool:
        """Check if this is an integer constant."""
        try:
            int(self.name)
            return True
        except ValueError:
            return False
    
    @property
    def arity(self) -> int:
        """Return arity of term (0 for simple terms)."""
        return len(self.args) if self.args else 0
    
    def get_variables(self) -> List['Term']:
        """Get all variables in this term."""
        result = []
        if self.is_variable:
            result.append(self)
        if self.args:
            for arg in self.args:
                result.extend(arg.get_variables())
        return result
    
    def substitute(self, mapping: Dict[str, 'Term']) -> 'Term':
        """Return term with variables substituted."""
        if self.is_variable and self.name in mapping:
            return mapping[self.name]
        if self.args is not None:
            new_args = [arg.substitute(mapping) for arg in self.args]
            return Term(self.name, new_args)
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {"name": self.name}
        if self.args is not None:
            result["args"] = [arg.to_dict() for arg in self.args]
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Term':
        """Create from dictionary representation."""
        args = None
        if "args" in data and data["args"] is not None:
            args = [cls.from_dict(arg) for arg in data["args"]]
        return cls(name=data["name"], args=args)
    
    def __str__(self) -> str:
        """Return string representation."""
        if self.args is None:
            return self.name
        args_str = ", ".join(str(arg) for arg in self.args)
        return f"{self.name}({args_str})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Term):
            return False
        return self.name == other.name and self.args == other.args
    
    def __hash__(self) -> int:
        args_tuple = tuple(self.args) if self.args else ()
        return hash((self.name, args_tuple))


@dataclass
class Atom:
    """A predicate atom in ASP.
    
    Attributes:
        predicate: Predicate name (e.g., "color", "edge", "path")
        terms: Arguments to the predicate
    
    Examples:
        - edge(X, Y)
        - color(node1, red)
        - connected(a, b, 5)
    """
    predicate: str
    terms: List[Term] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate atom structure."""
        if not self.predicate:
            raise ValueError("Predicate name cannot be empty")
        if not validate_identifier(self.predicate):
            raise ValueError(f"Invalid predicate name: {self.predicate}")
    
    @property
    def arity(self) -> int:
        """Return the arity (number of arguments)."""
        return len(self.terms)
    
    @property
    def signature(self) -> str:
        """Return predicate/arity signature."""
        return f"{self.predicate}/{self.arity}"
    
    def get_variables(self) -> List[Term]:
        """Get all variables in this atom."""
        result = []
        for term in self.terms:
            result.extend(term.get_variables())
        return result
    
    def is_ground(self) -> bool:
        """Check if atom contains no variables."""
        return len(self.get_variables()) == 0
    
    def substitute(self, mapping: Dict[str, Term]) -> 'Atom':
        """Return atom with variables substituted."""
        new_terms = [term.substitute(mapping) for term in self.terms]
        return Atom(self.predicate, new_terms)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "predicate": self.predicate,
            "terms": [term.to_dict() for term in self.terms]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Atom':
        """Create from dictionary representation."""
        terms = [Term.from_dict(t) for t in data.get("terms", [])]
        return cls(predicate=data["predicate"], terms=terms)
    
    def __str__(self) -> str:
        """Return string representation."""
        if not self.terms:
            return self.predicate
        terms_str = ", ".join(str(t) for t in self.terms)
        return f"{self.predicate}({terms_str})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Atom):
            return False
        return self.predicate == other.predicate and self.terms == other.terms
    
    def __hash__(self) -> int:
        return hash((self.predicate, tuple(self.terms)))


@dataclass
class Literal:
    """A literal in ASP (atom with optional negation).
    
    Attributes:
        atom: The underlying atom
        negation: Type of negation applied (NONE, DEFAULT, CLASSICAL)
    
    Examples:
        - Positive: edge(X, Y)
        - Default negation: not edge(X, Y)
        - Classical negation: -edge(X, Y)
    """
    atom: Atom
    negation: NegationType = NegationType.NONE
    
    @property
    def is_positive(self) -> bool:
        """Check if literal is positive (no negation)."""
        return self.negation == NegationType.NONE
    
    @property
    def is_negative(self) -> bool:
        """Check if literal has any form of negation."""
        return self.negation != NegationType.NONE
    
    @property
    def is_default_negated(self) -> bool:
        """Check if literal has default negation (not)."""
        return self.negation == NegationType.DEFAULT
    
    @property
    def is_classically_negated(self) -> bool:
        """Check if literal has classical negation (-)."""
        return self.negation == NegationType.CLASSICAL
    
    def negate(self, negation_type: NegationType = NegationType.DEFAULT) -> 'Literal':
        """Return negated version of this literal."""
        if self.negation == NegationType.NONE:
            return Literal(self.atom, negation_type)
        elif self.negation == negation_type:
            # Double negation cancels out
            return Literal(self.atom, NegationType.NONE)
        else:
            # Can't combine different negation types simply
            raise ValueError("Cannot combine different negation types")
    
    def get_variables(self) -> List[Term]:
        """Get all variables in this literal."""
        return self.atom.get_variables()
    
    def substitute(self, mapping: Dict[str, Term]) -> 'Literal':
        """Return literal with variables substituted."""
        return Literal(self.atom.substitute(mapping), self.negation)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "atom": self.atom.to_dict(),
            "negation": self.negation.name
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Literal':
        """Create from dictionary representation."""
        atom = Atom.from_dict(data["atom"])
        negation = NegationType[data.get("negation", "NONE")]
        return cls(atom=atom, negation=negation)
    
    def __str__(self) -> str:
        """Return string representation."""
        if self.negation == NegationType.NONE:
            return str(self.atom)
        elif self.negation == NegationType.DEFAULT:
            return f"not {self.atom}"
        else:  # CLASSICAL
            return f"-{self.atom}"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Literal):
            return False
        return self.atom == other.atom and self.negation == other.negation
    
    def __hash__(self) -> int:
        return hash((self.atom, self.negation))


# Factory functions for convenience
def term(name: str, *args) -> Term:
    """Create a term with optional arguments."""
    if args:
        return Term(name, list(args))
    return Term(name)


def var(name: str) -> Term:
    """Create a variable term (name should start with uppercase)."""
    if not name[0].isupper():
        name = name.upper()
    return Term(name)


def const(name: str) -> Term:
    """Create a constant term."""
    return Term(name)


def atom(predicate: str, *terms: Union[Term, str]) -> Atom:
    """Create an atom with the given predicate and terms."""
    term_list = []
    for t in terms:
        if isinstance(t, str):
            term_list.append(Term(t))
        else:
            term_list.append(t)
    return Atom(predicate, term_list)


def pos(a: Atom) -> Literal:
    """Create a positive literal from an atom."""
    return Literal(a, NegationType.NONE)


def neg(a: Atom) -> Literal:
    """Create a default-negated literal from an atom."""
    return Literal(a, NegationType.DEFAULT)


def classical_neg(a: Atom) -> Literal:
    """Create a classically-negated literal from an atom."""
    return Literal(a, NegationType.CLASSICAL)
