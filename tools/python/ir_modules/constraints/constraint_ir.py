"""Core Constraint IR data structures for Constraint Programming.

This module defines the fundamental types and enumerations
for constraint programming in STUNIR.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Any
import json


class VariableType(Enum):
    """Types of decision variables."""
    INT = auto()           # Integer variable
    FLOAT = auto()         # Float variable
    BOOL = auto()          # Boolean variable
    SET = auto()           # Set variable
    ARRAY = auto()         # Array of variables


class DomainType(Enum):
    """Types of domains."""
    RANGE = auto()         # Integer/float range (lb..ub)
    SET = auto()           # Explicit set of values {1, 3, 5}
    BOOL = auto()          # Boolean domain {true, false}
    UNBOUNDED = auto()     # Unbounded domain


class ConstraintType(Enum):
    """Types of constraints."""
    # Arithmetic constraints
    EQ = auto()            # Equal (=)
    NE = auto()            # Not equal (!=)
    LT = auto()            # Less than (<)
    LE = auto()            # Less than or equal (<=)
    GT = auto()            # Greater than (>)
    GE = auto()            # Greater than or equal (>=)
    
    # Logical constraints
    AND = auto()           # Conjunction
    OR = auto()            # Disjunction
    NOT = auto()           # Negation
    IMPLIES = auto()       # Implication (->)
    IFF = auto()           # If and only if (<->)
    
    # Reification
    REIFY = auto()         # Constraint as boolean
    
    # Global constraints
    ALLDIFFERENT = auto()  # All different values
    CUMULATIVE = auto()    # Cumulative scheduling
    ELEMENT = auto()       # Array element access
    TABLE = auto()         # Table constraint
    REGULAR = auto()       # Regular language
    CIRCUIT = auto()       # Hamiltonian circuit
    COUNT = auto()         # Counting constraint
    BIN_PACKING = auto()   # Bin packing
    GLOBAL_CARDINALITY = auto()  # GCC
    SUM = auto()           # Sum constraint


class ObjectiveType(Enum):
    """Types of objectives."""
    MINIMIZE = auto()      # Minimize objective
    MAXIMIZE = auto()      # Maximize objective
    SATISFY = auto()       # Just find a solution


class SearchStrategy(Enum):
    """Search strategies for variable selection."""
    INPUT_ORDER = auto()       # Use input order
    FIRST_FAIL = auto()        # Smallest domain first
    ANTI_FIRST_FAIL = auto()   # Largest domain first
    SMALLEST = auto()          # Smallest value first
    LARGEST = auto()           # Largest value first
    OCCURRENCE = auto()        # Most constrained first
    MOST_CONSTRAINED = auto()  # Most constrained variable
    MAX_REGRET = auto()        # Maximum regret
    DOM_W_DEG = auto()         # Domain/weighted degree


class ValueChoice(Enum):
    """Value choice heuristics."""
    INDOMAIN_MIN = auto()      # Try smallest value first
    INDOMAIN_MAX = auto()      # Try largest value first
    INDOMAIN_MEDIAN = auto()   # Try median value first
    INDOMAIN_RANDOM = auto()   # Random value
    INDOMAIN_SPLIT = auto()    # Binary split
    INDOMAIN_REVERSE_SPLIT = auto()  # Reverse binary split


@dataclass
class ConstraintEmitterResult:
    """Result of constraint model emission.
    
    Attributes:
        code: Generated code string
        manifest: Build manifest dictionary
        warnings: List of warnings during emission
    """
    code: str
    manifest: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    
    def write_to_file(self, path: str) -> None:
        """Write code to file."""
        with open(path, 'w') as f:
            f.write(self.code)
    
    def write_manifest(self, path: str) -> None:
        """Write manifest to file."""
        with open(path, 'w') as f:
            json.dump(self.manifest, f, indent=2, sort_keys=True)


# Exception classes
class ConstraintModelError(Exception):
    """Base error for constraint model issues."""
    pass


class InvalidVariableError(ConstraintModelError):
    """Invalid variable definition."""
    pass


class InvalidDomainError(ConstraintModelError):
    """Invalid domain specification."""
    pass


class InvalidConstraintError(ConstraintModelError):
    """Invalid constraint definition."""
    pass


class UnsupportedConstraintError(ConstraintModelError):
    """Constraint not supported by target."""
    pass
