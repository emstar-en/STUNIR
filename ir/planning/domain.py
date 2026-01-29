"""Domain definition for Planning IR.

This module defines the Domain class for PDDL planning domains.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from .planning_ir import PDDLRequirement, InvalidDomainError
from .predicate import TypeDef, Predicate, Function
from .action import Action, Formula


@dataclass
class ObjectDef:
    """Object definition (for constants or problem objects).
    
    Attributes:
        name: Object name (e.g., "loc1", "truck1")
        obj_type: Object type (e.g., "location", "vehicle")
    """
    name: str
    obj_type: str = "object"
    
    def __post_init__(self):
        """Validate object definition."""
        if not self.name:
            raise ValueError("Object name cannot be empty")


@dataclass
class DerivedPredicate:
    """Derived predicate (axiom) definition.
    
    Attributes:
        predicate: The derived predicate
        condition: Condition formula for derivation
    """
    predicate: Predicate
    condition: Formula


@dataclass
class Domain:
    """PDDL domain definition.
    
    Represents a complete PDDL domain with types, predicates,
    functions, and actions.
    
    Attributes:
        name: Domain name
        requirements: PDDL requirements
        types: Type definitions
        constants: Domain-level constants
        predicates: Predicate definitions
        functions: Function definitions
        actions: Action schemas
        derived_predicates: Derived predicates (axioms)
    """
    name: str
    requirements: List[PDDLRequirement] = field(default_factory=list)
    types: List[TypeDef] = field(default_factory=list)
    constants: List[ObjectDef] = field(default_factory=list)
    predicates: List[Predicate] = field(default_factory=list)
    functions: List[Function] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    derived_predicates: List[DerivedPredicate] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate domain."""
        if not self.name:
            raise ValueError("Domain name cannot be empty")
    
    def add_type(self, name: str, parent: str = "object") -> 'Domain':
        """Add a type definition (fluent interface)."""
        self.types.append(TypeDef(name, parent))
        return self
    
    def add_predicate(self, predicate: Predicate) -> 'Domain':
        """Add a predicate definition (fluent interface)."""
        self.predicates.append(predicate)
        return self
    
    def add_function(self, function: Function) -> 'Domain':
        """Add a function definition (fluent interface)."""
        self.functions.append(function)
        return self
    
    def add_action(self, action: Action) -> 'Domain':
        """Add an action schema (fluent interface)."""
        self.actions.append(action)
        return self
    
    def add_constant(self, name: str, obj_type: str = "object") -> 'Domain':
        """Add a constant (fluent interface)."""
        self.constants.append(ObjectDef(name, obj_type))
        return self
    
    def add_requirement(self, req: PDDLRequirement) -> 'Domain':
        """Add a requirement (fluent interface)."""
        if req not in self.requirements:
            self.requirements.append(req)
        return self
    
    def validate(self) -> List[str]:
        """Validate domain and return list of errors."""
        errors = []
        
        # Check domain name
        if not self.name:
            errors.append("Domain name is empty")
        
        # Check for duplicate type names
        type_names = [t.name for t in self.types]
        if len(type_names) != len(set(type_names)):
            errors.append("Duplicate type names")
        
        # Check for duplicate predicate names
        pred_names = [p.name for p in self.predicates]
        if len(pred_names) != len(set(pred_names)):
            errors.append("Duplicate predicate names")
        
        # Check for duplicate action names
        action_names = [a.name for a in self.actions]
        if len(action_names) != len(set(action_names)):
            errors.append("Duplicate action names")
        
        # Check type references in predicates
        known_types = set(type_names) | {"object"}
        for pred in self.predicates:
            for param in pred.parameters:
                if param.param_type not in known_types:
                    errors.append(f"Unknown type '{param.param_type}' in predicate '{pred.name}'")
        
        # Check type references in actions
        for action in self.actions:
            for param in action.parameters:
                if param.param_type not in known_types:
                    errors.append(f"Unknown type '{param.param_type}' in action '{action.name}'")
        
        # Check requirements consistency
        if self.types and PDDLRequirement.TYPING not in self.requirements:
            errors.append("Types defined but :typing requirement not specified")
        
        if self.functions and PDDLRequirement.NUMERIC_FLUENTS not in self.requirements \
           and PDDLRequirement.FLUENTS not in self.requirements:
            errors.append("Functions defined but :fluents/:numeric-fluents requirement not specified")
        
        return errors
    
    def infer_requirements(self) -> List[PDDLRequirement]:
        """Infer required PDDL requirements from domain content."""
        reqs = set()
        
        # STRIPS is usually always needed
        reqs.add(PDDLRequirement.STRIPS)
        
        # Check if typing is needed
        if self.types:
            reqs.add(PDDLRequirement.TYPING)
        
        # Check if numeric fluents are needed
        if self.functions:
            reqs.add(PDDLRequirement.NUMERIC_FLUENTS)
        
        # Check if derived predicates are needed
        if self.derived_predicates:
            reqs.add(PDDLRequirement.DERIVED_PREDICATES)
        
        # Check actions for additional requirements
        for action in self.actions:
            if action.cost is not None:
                reqs.add(PDDLRequirement.ACTION_COSTS)
            if action.duration is not None:
                reqs.add(PDDLRequirement.DURATIVE_ACTIONS)
            
            # Check preconditions and effects for requirements
            self._check_formula_requirements(action.precondition, reqs)
            self._check_effect_requirements(action.effect, reqs)
        
        return list(reqs)
    
    def _check_formula_requirements(self, formula: Optional[Formula], reqs: set) -> None:
        """Check formula for additional requirements."""
        if formula is None:
            return
        
        from .planning_ir import FormulaType
        
        if formula.formula_type == FormulaType.OR:
            reqs.add(PDDLRequirement.DISJUNCTIVE_PRECONDITIONS)
        elif formula.formula_type == FormulaType.NOT:
            if formula.children and formula.children[0].formula_type == FormulaType.ATOM:
                reqs.add(PDDLRequirement.NEGATIVE_PRECONDITIONS)
        elif formula.formula_type == FormulaType.EXISTS:
            reqs.add(PDDLRequirement.EXISTENTIAL_PRECONDITIONS)
        elif formula.formula_type == FormulaType.FORALL:
            reqs.add(PDDLRequirement.UNIVERSAL_PRECONDITIONS)
        elif formula.formula_type == FormulaType.EQUALS:
            reqs.add(PDDLRequirement.EQUALITY)
        
        # Recurse into children
        for child in formula.children:
            self._check_formula_requirements(child, reqs)
    
    def _check_effect_requirements(self, effect, reqs: set) -> None:
        """Check effect for additional requirements."""
        if effect is None:
            return
        
        from .planning_ir import EffectType
        
        if effect.effect_type == EffectType.CONDITIONAL:
            reqs.add(PDDLRequirement.CONDITIONAL_EFFECTS)
        elif effect.effect_type in (EffectType.INCREASE, EffectType.DECREASE, 
                                    EffectType.ASSIGN, EffectType.SCALE_UP, 
                                    EffectType.SCALE_DOWN):
            reqs.add(PDDLRequirement.NUMERIC_FLUENTS)
        
        # Recurse into children
        for child in effect.children:
            self._check_effect_requirements(child, reqs)
