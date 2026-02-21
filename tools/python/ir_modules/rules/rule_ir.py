"""Core Rule IR data structures for Expert Systems.

This module defines the fundamental types and enumerations
for rule-based reasoning in STUNIR.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Any, Tuple, Union


class PatternType(Enum):
    """Types of pattern elements."""
    LITERAL = auto()       # Exact match
    VARIABLE = auto()      # Bindable variable (e.g., ?x)
    WILDCARD = auto()      # Match anything (e.g., ?)
    MULTIFIELD = auto()    # Match multiple values (e.g., $?)
    CONSTRAINT = auto()    # Constrained match (e.g., ?x&:(> ?x 10))


class ConditionType(Enum):
    """Types of rule conditions."""
    PATTERN = auto()       # Pattern match against facts
    TEST = auto()          # Predicate test
    AND = auto()           # Conjunction of conditions
    OR = auto()            # Disjunction of conditions
    NOT = auto()           # Negated condition
    EXISTS = auto()        # Existential quantifier
    FORALL = auto()        # Universal quantifier


class ActionType(Enum):
    """Types of rule actions."""
    ASSERT = auto()        # Assert new fact
    RETRACT = auto()       # Retract existing fact
    MODIFY = auto()        # Modify existing fact
    BIND = auto()          # Bind variable
    CALL = auto()          # Call function
    PRINTOUT = auto()      # Print output
    HALT = auto()          # Halt execution


class ConflictResolutionStrategy(Enum):
    """Strategies for conflict resolution."""
    SALIENCE = auto()      # Higher salience first
    RECENCY = auto()       # Most recently matched first
    SPECIFICITY = auto()   # Most specific rule first
    LEX = auto()           # Lexicographic order (CLIPS default)
    MEA = auto()           # Means-ends analysis
    RANDOM = auto()        # Random selection


@dataclass
class EmitterResult:
    """Result of expert system emission."""
    code: str
    manifest: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    
    def write_to_file(self, path: str) -> None:
        """Write code to file."""
        with open(path, 'w') as f:
            f.write(self.code)
    
    def write_manifest(self, path: str) -> None:
        """Write manifest to file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.manifest, f, indent=2, sort_keys=True)


@dataclass
class FunctionDef:
    """User-defined function for expert systems."""
    name: str
    parameters: List[str]
    body: str  # Expression body
    return_type: Optional[str] = None
    documentation: Optional[str] = None
