"""ASP IR Core Types and Enums.

This module defines the core enumeration types used throughout the
Answer Set Programming IR representation.

Part of Phase 7D: Answer Set Programming
"""

from enum import Enum, auto
from typing import Dict, Any


class RuleType(Enum):
    """Types of ASP rules."""
    NORMAL = auto()       # head :- body.
    CHOICE = auto()       # {head} :- body.
    CONSTRAINT = auto()   # :- body.
    DISJUNCTIVE = auto()  # head1 | head2 :- body.
    WEAK = auto()         # :~ body. [weight@priority]


class AggregateFunction(Enum):
    """ASP aggregate functions."""
    COUNT = auto()    # #count
    SUM = auto()      # #sum
    MIN = auto()      # #min
    MAX = auto()      # #max
    SUM_PLUS = auto() # #sum+ (positive only)
    
    def to_clingo(self) -> str:
        """Convert to Clingo syntax."""
        mapping = {
            AggregateFunction.COUNT: "#count",
            AggregateFunction.SUM: "#sum",
            AggregateFunction.MIN: "#min",
            AggregateFunction.MAX: "#max",
            AggregateFunction.SUM_PLUS: "#sum+",
        }
        return mapping.get(self, "#count")
    
    def to_dlv(self) -> str:
        """Convert to DLV syntax."""
        # DLV uses same syntax for basic aggregates
        return self.to_clingo()


class ComparisonOp(Enum):
    """Comparison operators."""
    EQ = "="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    
    def __str__(self) -> str:
        return self.value


class NegationType(Enum):
    """Types of negation in ASP."""
    NONE = auto()         # No negation (positive)
    DEFAULT = auto()      # Default negation (not)
    CLASSICAL = auto()    # Classical negation (-)


class ASPDialect(Enum):
    """ASP solver dialects."""
    CLINGO = auto()  # Clingo/gringo syntax
    DLV = auto()     # DLV syntax
    ASPC = auto()    # ASP-Core-2 standard


# Constants
DEFAULT_PRIORITY = 0
DEFAULT_WEIGHT = 1
MAX_ARITY = 100


def validate_identifier(name: str) -> bool:
    """Check if name is a valid ASP identifier."""
    if not name:
        return False
    # First char: letter or underscore
    if not (name[0].isalpha() or name[0] == '_'):
        return False
    # Rest: alphanumeric or underscore
    return all(c.isalnum() or c == '_' for c in name)


def is_variable(name: str) -> bool:
    """Check if name represents a variable (starts with uppercase)."""
    return bool(name) and name[0].isupper()


def is_constant(name: str) -> bool:
    """Check if name represents a constant (starts with lowercase)."""
    return bool(name) and (name[0].islower() or name[0].isdigit())


def canonical_name(name: str) -> str:
    """Return canonical form of a name."""
    return name.strip()
