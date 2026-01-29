"""Problem definition for Planning IR.

This module defines the Problem class for PDDL planning problems.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from .planning_ir import PDDLRequirement, InvalidProblemError
from .predicate import Atom
from .action import Formula
from .domain import ObjectDef


@dataclass
class InitialState:
    """Initial state specification.
    
    Represents the initial state in a PDDL problem.
    
    Attributes:
        facts: Initial ground atoms (positive facts)
        numeric_values: Initial function values {function_str: value}
        timed_literals: Timed initial literals [(time, atom)]
    """
    facts: List[Atom] = field(default_factory=list)
    numeric_values: Dict[str, float] = field(default_factory=dict)
    timed_literals: List[Tuple[float, Atom]] = field(default_factory=list)
    
    def add_fact(self, predicate: str, *args: str) -> 'InitialState':
        """Add a fact to initial state (fluent interface)."""
        self.facts.append(Atom(predicate, list(args)))
        return self
    
    def add_numeric(self, function: str, value: float) -> 'InitialState':
        """Add a numeric value (fluent interface)."""
        self.numeric_values[function] = value
        return self
    
    def add_timed_literal(self, time: float, predicate: str, *args: str) -> 'InitialState':
        """Add a timed literal (fluent interface)."""
        self.timed_literals.append((time, Atom(predicate, list(args))))
        return self


@dataclass
class Metric:
    """Optimization metric.
    
    Attributes:
        direction: "minimize" or "maximize"
        expression: Expression to optimize (e.g., "total-cost")
    """
    direction: str = "minimize"
    expression: str = "total-cost"
    
    def __post_init__(self):
        """Validate metric."""
        if self.direction not in ("minimize", "maximize"):
            raise ValueError(f"Invalid metric direction: {self.direction}")


@dataclass
class Problem:
    """PDDL problem definition.
    
    Represents a complete PDDL problem with objects, initial state,
    and goals.
    
    Attributes:
        name: Problem name
        domain_name: Reference to domain
        requirements: Problem-specific requirements
        objects: Object definitions
        init: Initial state
        goal: Goal formula
        metric: Optimization metric
    """
    name: str
    domain_name: str
    requirements: List[PDDLRequirement] = field(default_factory=list)
    objects: List[ObjectDef] = field(default_factory=list)
    init: InitialState = field(default_factory=InitialState)
    goal: Optional[Formula] = None
    metric: Optional[Metric] = None
    
    def __post_init__(self):
        """Validate problem."""
        if not self.name:
            raise ValueError("Problem name cannot be empty")
        if not self.domain_name:
            raise ValueError("Domain name cannot be empty")
    
    def add_object(self, name: str, obj_type: str = "object") -> 'Problem':
        """Add an object (fluent interface)."""
        self.objects.append(ObjectDef(name, obj_type))
        return self
    
    def add_objects(self, names: List[str], obj_type: str = "object") -> 'Problem':
        """Add multiple objects of the same type (fluent interface)."""
        for name in names:
            self.objects.append(ObjectDef(name, obj_type))
        return self
    
    def set_goal(self, goal: Formula) -> 'Problem':
        """Set the goal formula (fluent interface)."""
        self.goal = goal
        return self
    
    def set_metric(self, direction: str, expression: str) -> 'Problem':
        """Set the optimization metric (fluent interface)."""
        self.metric = Metric(direction, expression)
        return self
    
    def validate(self, domain=None) -> List[str]:
        """Validate problem and return list of errors.
        
        Args:
            domain: Optional Domain to validate against
        """
        errors = []
        
        # Check problem name
        if not self.name:
            errors.append("Problem name is empty")
        
        # Check domain reference
        if not self.domain_name:
            errors.append("Domain name is empty")
        
        # Check for duplicate object names
        obj_names = [o.name for o in self.objects]
        if len(obj_names) != len(set(obj_names)):
            errors.append("Duplicate object names")
        
        # Check goal is specified
        if self.goal is None:
            errors.append("Goal is not specified")
        
        # Validate against domain if provided
        if domain is not None:
            # Check domain name matches
            if self.domain_name != domain.name:
                errors.append(f"Domain name mismatch: expected '{domain.name}', got '{self.domain_name}'")
            
            # Check object types exist
            known_types = {t.name for t in domain.types} | {"object"}
            for obj in self.objects:
                if obj.obj_type not in known_types:
                    errors.append(f"Unknown type '{obj.obj_type}' for object '{obj.name}'")
            
            # Check predicates in init exist
            known_preds = {p.name for p in domain.predicates}
            for fact in self.init.facts:
                if fact.predicate not in known_preds:
                    errors.append(f"Unknown predicate '{fact.predicate}' in initial state")
        
        return errors
    
    def get_objects_by_type(self, obj_type: str) -> List[ObjectDef]:
        """Get all objects of a specific type."""
        return [o for o in self.objects if o.obj_type == obj_type]
