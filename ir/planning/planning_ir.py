"""Core Planning IR data structures for automated planning.

This module defines fundamental types and enumerations
for automated planning in STUNIR (PDDL support).
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Any
import json


class PDDLRequirement(Enum):
    """PDDL requirement flags.
    
    These correspond to PDDL 2.1+ requirement keywords.
    """
    STRIPS = auto()                    # Basic STRIPS planning
    TYPING = auto()                    # Typed objects
    NEGATIVE_PRECONDITIONS = auto()    # Negated atoms in preconditions
    DISJUNCTIVE_PRECONDITIONS = auto() # OR in preconditions
    EQUALITY = auto()                  # = predicate
    EXISTENTIAL_PRECONDITIONS = auto() # exists quantifier
    UNIVERSAL_PRECONDITIONS = auto()   # forall quantifier
    QUANTIFIED_PRECONDITIONS = auto()  # Both quantifiers
    CONDITIONAL_EFFECTS = auto()       # when effects
    FLUENTS = auto()                   # Numeric fluents
    NUMERIC_FLUENTS = auto()           # Numeric fluents alias
    OBJECT_FLUENTS = auto()            # Object-valued fluents
    ADL = auto()                       # Action Description Language
    DURATIVE_ACTIONS = auto()          # Actions with duration
    DURATION_INEQUALITIES = auto()     # Duration comparisons
    CONTINUOUS_EFFECTS = auto()        # Continuous numeric change
    DERIVED_PREDICATES = auto()        # Axioms / derived predicates
    TIMED_INITIAL_LITERALS = auto()    # Timed initial literals
    PREFERENCES = auto()               # Soft goals
    CONSTRAINTS = auto()               # State trajectory constraints
    ACTION_COSTS = auto()              # Action costs


class FormulaType(Enum):
    """Types of logical formulas."""
    ATOM = auto()          # Predicate application (on ?x ?y)
    NOT = auto()           # Negation (not ...)
    AND = auto()           # Conjunction (and ...)
    OR = auto()            # Disjunction (or ...)
    IMPLY = auto()         # Implication (imply p q)
    EXISTS = auto()        # Existential (exists (?x - type) ...)
    FORALL = auto()        # Universal (forall (?x - type) ...)
    WHEN = auto()          # Conditional effect (when cond effect)
    EQUALS = auto()        # Equality (= ?x ?y)


class EffectType(Enum):
    """Types of effects."""
    POSITIVE = auto()      # Add predicate (p ?x)
    NEGATIVE = auto()      # Remove predicate (not (p ?x))
    CONDITIONAL = auto()   # Conditional effect (when ...)
    FORALL = auto()        # Universal effect (forall ...)
    ASSIGN = auto()        # Numeric assignment (assign ...)
    INCREASE = auto()      # Numeric increase (increase ...)
    DECREASE = auto()      # Numeric decrease (decrease ...)
    SCALE_UP = auto()      # Numeric scale up
    SCALE_DOWN = auto()    # Numeric scale down
    COMPOUND = auto()      # Multiple effects (and ...)


@dataclass
class PlanningEmitterResult:
    """Result of planning model emission.
    
    Attributes:
        domain_code: Generated domain PDDL code
        problem_code: Generated problem PDDL code (if applicable)
        manifest: Build manifest dictionary
        warnings: List of warnings during emission
    """
    domain_code: str
    problem_code: str = ""
    manifest: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    def write_domain(self, path: str) -> None:
        """Write domain to file."""
        with open(path, 'w') as f:
            f.write(self.domain_code)
    
    def write_problem(self, path: str) -> None:
        """Write problem to file."""
        if self.problem_code:
            with open(path, 'w') as f:
                f.write(self.problem_code)
    
    def write_manifest(self, path: str) -> None:
        """Write manifest to file."""
        with open(path, 'w') as f:
            json.dump(self.manifest, f, indent=2, sort_keys=True)


# Exception classes
class PlanningError(Exception):
    """Base error for planning issues."""
    pass


class InvalidActionError(PlanningError):
    """Invalid action definition."""
    pass


class InvalidPredicateError(PlanningError):
    """Invalid predicate definition."""
    pass


class InvalidDomainError(PlanningError):
    """Invalid domain specification."""
    pass


class InvalidProblemError(PlanningError):
    """Invalid problem specification."""
    pass


class UnsupportedFeatureError(PlanningError):
    """Feature not supported by target."""
    pass
