"""Action and formula definitions for Planning IR.

This module defines formulas, effects, and actions used in PDDL planning.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union

from .planning_ir import FormulaType, EffectType
from .predicate import Parameter, Atom, FunctionApplication


@dataclass
class Formula:
    """Logical formula (preconditions, goals, effects).
    
    Represents a logical formula in PDDL using a recursive structure.
    
    Attributes:
        formula_type: Type of formula (ATOM, AND, OR, NOT, etc.)
        atom: Atom (for ATOM type)
        children: Child formulas (for compound types)
        variables: Typed variables (for quantified formulas)
    """
    formula_type: FormulaType
    atom: Optional[Atom] = None
    children: List['Formula'] = field(default_factory=list)
    variables: List[Parameter] = field(default_factory=list)
    
    # Comparison operators for equality formulas
    left_term: Optional[str] = None
    right_term: Optional[str] = None
    
    @classmethod
    def make_atom(cls, predicate: str, *args: str) -> 'Formula':
        """Create an atomic formula.
        
        Args:
            predicate: Predicate name
            *args: Arguments to the predicate
            
        Returns:
            Formula with ATOM type
        """
        return cls(
            formula_type=FormulaType.ATOM,
            atom=Atom(predicate=predicate, arguments=list(args))
        )
    
    @classmethod
    def make_and(cls, *children: 'Formula') -> 'Formula':
        """Create a conjunction.
        
        Args:
            *children: Child formulas
            
        Returns:
            Formula with AND type
        """
        child_list = list(children)
        if len(child_list) == 1:
            return child_list[0]
        return cls(formula_type=FormulaType.AND, children=child_list)
    
    @classmethod
    def make_or(cls, *children: 'Formula') -> 'Formula':
        """Create a disjunction.
        
        Args:
            *children: Child formulas
            
        Returns:
            Formula with OR type
        """
        child_list = list(children)
        if len(child_list) == 1:
            return child_list[0]
        return cls(formula_type=FormulaType.OR, children=child_list)
    
    @classmethod
    def make_not(cls, child: 'Formula') -> 'Formula':
        """Create a negation.
        
        Args:
            child: Formula to negate
            
        Returns:
            Formula with NOT type
        """
        return cls(formula_type=FormulaType.NOT, children=[child])
    
    @classmethod
    def make_imply(cls, antecedent: 'Formula', consequent: 'Formula') -> 'Formula':
        """Create an implication.
        
        Args:
            antecedent: If-part
            consequent: Then-part
            
        Returns:
            Formula with IMPLY type
        """
        return cls(formula_type=FormulaType.IMPLY, children=[antecedent, consequent])
    
    @classmethod
    def make_exists(cls, variables: List[Parameter], body: 'Formula') -> 'Formula':
        """Create an existential formula.
        
        Args:
            variables: Quantified variables
            body: Body formula
            
        Returns:
            Formula with EXISTS type
        """
        return cls(formula_type=FormulaType.EXISTS, variables=variables, children=[body])
    
    @classmethod
    def make_forall(cls, variables: List[Parameter], body: 'Formula') -> 'Formula':
        """Create a universal formula.
        
        Args:
            variables: Quantified variables
            body: Body formula
            
        Returns:
            Formula with FORALL type
        """
        return cls(formula_type=FormulaType.FORALL, variables=variables, children=[body])
    
    @classmethod
    def make_equals(cls, left: str, right: str) -> 'Formula':
        """Create an equality formula.
        
        Args:
            left: Left term
            right: Right term
            
        Returns:
            Formula with EQUALS type
        """
        return cls(formula_type=FormulaType.EQUALS, left_term=left, right_term=right)
    
    @classmethod 
    def make_when(cls, condition: 'Formula', effect: 'Formula') -> 'Formula':
        """Create a conditional effect formula.
        
        Args:
            condition: Condition
            effect: Effect when condition holds
            
        Returns:
            Formula with WHEN type
        """
        return cls(formula_type=FormulaType.WHEN, children=[condition, effect])
    
    def is_atom(self) -> bool:
        """Check if this is an atomic formula."""
        return self.formula_type == FormulaType.ATOM
    
    def is_compound(self) -> bool:
        """Check if this is a compound formula."""
        return self.formula_type in (FormulaType.AND, FormulaType.OR, 
                                     FormulaType.NOT, FormulaType.IMPLY)
    
    def is_quantified(self) -> bool:
        """Check if this is a quantified formula."""
        return self.formula_type in (FormulaType.EXISTS, FormulaType.FORALL)


@dataclass
class Effect:
    """Action effect.
    
    Represents an effect of a PDDL action.
    
    Attributes:
        effect_type: Type of effect
        formula: Effect formula (positive/negative atom)
        condition: Condition for conditional effects
        variables: Variables for universal effects
        function_app: Function application for numeric effects
        value: Value for numeric effects
    """
    effect_type: EffectType
    formula: Optional[Formula] = None
    condition: Optional[Formula] = None
    variables: List[Parameter] = field(default_factory=list)
    children: List['Effect'] = field(default_factory=list)
    function_app: Optional[FunctionApplication] = None
    value: Optional[Union[float, str]] = None
    
    @classmethod
    def make_positive(cls, predicate: str, *args: str) -> 'Effect':
        """Create a positive effect (add fact)."""
        return cls(
            effect_type=EffectType.POSITIVE,
            formula=Formula.make_atom(predicate, *args)
        )
    
    @classmethod
    def make_negative(cls, predicate: str, *args: str) -> 'Effect':
        """Create a negative effect (delete fact)."""
        return cls(
            effect_type=EffectType.NEGATIVE,
            formula=Formula.make_atom(predicate, *args)
        )
    
    @classmethod
    def make_compound(cls, *effects: 'Effect') -> 'Effect':
        """Create a compound effect (and ...)."""
        return cls(effect_type=EffectType.COMPOUND, children=list(effects))
    
    @classmethod
    def make_conditional(cls, condition: Formula, effect: 'Effect') -> 'Effect':
        """Create a conditional effect (when ...)."""
        return cls(
            effect_type=EffectType.CONDITIONAL,
            condition=condition,
            children=[effect]
        )
    
    @classmethod
    def make_forall(cls, variables: List[Parameter], effect: 'Effect') -> 'Effect':
        """Create a universal effect (forall ...)."""
        return cls(
            effect_type=EffectType.FORALL,
            variables=variables,
            children=[effect]
        )
    
    @classmethod
    def make_increase(cls, func: str, args: List[str], value: Union[float, str]) -> 'Effect':
        """Create an increase effect."""
        return cls(
            effect_type=EffectType.INCREASE,
            function_app=FunctionApplication(func, args),
            value=value
        )
    
    @classmethod
    def make_decrease(cls, func: str, args: List[str], value: Union[float, str]) -> 'Effect':
        """Create a decrease effect."""
        return cls(
            effect_type=EffectType.DECREASE,
            function_app=FunctionApplication(func, args),
            value=value
        )
    
    @classmethod
    def make_assign(cls, func: str, args: List[str], value: Union[float, str]) -> 'Effect':
        """Create an assignment effect."""
        return cls(
            effect_type=EffectType.ASSIGN,
            function_app=FunctionApplication(func, args),
            value=value
        )


@dataclass
class Action:
    """Action schema definition.
    
    Represents an action schema in PDDL domain.
    
    Attributes:
        name: Action name (e.g., "move", "pickup")
        parameters: List of typed parameters
        precondition: Precondition formula
        effect: Effect specification
        duration: Duration expression (for durative actions)
        cost: Action cost (for action-costs requirement)
    """
    name: str
    parameters: List[Parameter] = field(default_factory=list)
    precondition: Optional[Formula] = None
    effect: Optional[Effect] = None
    duration: Optional[str] = None
    cost: Optional[float] = None
    
    def __post_init__(self):
        """Validate action."""
        if not self.name:
            raise ValueError("Action name cannot be empty")
    
    def get_parameter_names(self) -> List[str]:
        """Get list of parameter names."""
        return [p.name for p in self.parameters]
    
    def get_parameter_types(self) -> List[str]:
        """Get list of parameter types."""
        return [p.param_type for p in self.parameters]
