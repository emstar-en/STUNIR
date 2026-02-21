"""Rule definitions for rule-based systems.

This module provides Rule, Condition, Action, and RuleBase classes
for representing expert system knowledge bases.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Any, Tuple, Union
from .rule_ir import ConditionType, ActionType, FunctionDef
from .pattern import AnyPatternElement
from .fact import Fact, FactTemplate


# Condition Classes
@dataclass
class Condition:
    """Base condition class for rule LHS."""
    condition_type: ConditionType


@dataclass
class PatternCondition(Condition):
    """Pattern matching condition.
    
    Attributes:
        template_name: Template name to match (None for ordered facts)
        patterns: List of (slot_name, pattern) for template facts
        ordered_patterns: List of patterns for ordered facts
        binding_name: Optional variable to bind the matched fact
    """
    template_name: Optional[str]
    patterns: List[Tuple[str, AnyPatternElement]]  # (slot_name, pattern) for templates
    ordered_patterns: List[AnyPatternElement] = field(default_factory=list)  # For ordered facts
    binding_name: Optional[str] = None  # Variable to bind matched fact
    condition_type: ConditionType = field(default=ConditionType.PATTERN, init=False)


@dataclass
class TestCondition(Condition):
    """Predicate test condition.
    
    Attributes:
        expression: Test expression string (e.g., "(> ?x 10)")
    """
    expression: str  # Test expression
    condition_type: ConditionType = field(default=ConditionType.TEST, init=False)


@dataclass
class CompositeCondition(Condition):
    """Composite condition (AND/OR/NOT).
    
    Attributes:
        operator: The logical operator (AND, OR, NOT)
        children: List of child conditions
    """
    operator: ConditionType  # AND, OR, NOT
    children: List[Condition] = field(default_factory=list)
    condition_type: ConditionType = field(init=False)
    
    def __post_init__(self):
        self.condition_type = self.operator


# Action Classes
@dataclass
class Action:
    """Base action class for rule RHS."""
    action_type: ActionType


@dataclass
class AssertAction(Action):
    """Assert a new fact.
    
    Attributes:
        fact_template: Template name (None for ordered facts)
        slot_values: Slot values for template facts
        ordered_values: Values for ordered facts
    """
    fact_template: Optional[str]  # Template name or None for ordered
    slot_values: Dict[str, Any] = field(default_factory=dict)
    ordered_values: List[Any] = field(default_factory=list)
    action_type: ActionType = field(default=ActionType.ASSERT, init=False)


@dataclass
class RetractAction(Action):
    """Retract a fact.
    
    Attributes:
        fact_reference: Variable reference to bound fact
    """
    fact_reference: str  # Variable reference to bound fact
    action_type: ActionType = field(default=ActionType.RETRACT, init=False)


@dataclass
class ModifyAction(Action):
    """Modify a fact's slots.
    
    Attributes:
        fact_reference: Variable reference to bound fact
        modifications: Dict of slot_name -> new_value
    """
    fact_reference: str
    modifications: Dict[str, Any]  # slot_name -> new_value
    action_type: ActionType = field(default=ActionType.MODIFY, init=False)


@dataclass
class BindAction(Action):
    """Bind a variable.
    
    Attributes:
        variable: Variable name to bind
        expression: Expression to evaluate
    """
    variable: str
    expression: str
    action_type: ActionType = field(default=ActionType.BIND, init=False)


@dataclass
class CallAction(Action):
    """Call a function.
    
    Attributes:
        function_name: Name of function to call
        arguments: List of arguments
    """
    function_name: str
    arguments: List[Any] = field(default_factory=list)
    action_type: ActionType = field(default=ActionType.CALL, init=False)


@dataclass
class PrintoutAction(Action):
    """Print output.
    
    Attributes:
        router: Output router (t = stdout)
        items: List of items to print
    """
    router: str = "t"  # Output router (t = stdout)
    items: List[Any] = field(default_factory=list)
    action_type: ActionType = field(default=ActionType.PRINTOUT, init=False)


@dataclass
class HaltAction(Action):
    """Halt execution."""
    action_type: ActionType = field(default=ActionType.HALT, init=False)


# Rule Definition
@dataclass
class Rule:
    """A production rule (IF-THEN).
    
    Attributes:
        name: Unique rule name
        conditions: List of conditions (LHS/IF part)
        actions: List of actions (RHS/THEN part)
        salience: Priority (higher = more priority)
        auto_focus: Whether rule should auto-focus
        module: Optional module name
        documentation: Optional documentation string
    """
    name: str
    conditions: List[Condition]  # IF part (LHS)
    actions: List[Action]  # THEN part (RHS)
    salience: int = 0  # Priority (higher = more priority)
    auto_focus: bool = False
    module: Optional[str] = None
    documentation: Optional[str] = None
    
    def get_bound_variables(self) -> Set[str]:
        """Get all variables bound in conditions.
        
        Returns:
            Set of variable names bound in the rule's conditions
        """
        variables: Set[str] = set()
        for cond in self.conditions:
            if isinstance(cond, PatternCondition):
                if cond.binding_name:
                    variables.add(cond.binding_name)
                for _, pattern in cond.patterns:
                    from .pattern import VariablePattern
                    if isinstance(pattern, VariablePattern):
                        variables.add(pattern.name)
                for pattern in cond.ordered_patterns:
                    from .pattern import VariablePattern
                    if isinstance(pattern, VariablePattern):
                        variables.add(pattern.name)
        return variables
    
    def get_used_variables(self) -> Set[str]:
        """Get all variables used in actions.
        
        Returns:
            Set of variable names used in the rule's actions
        """
        variables: Set[str] = set()
        for action in self.actions:
            if isinstance(action, AssertAction):
                for v in action.slot_values.values():
                    if isinstance(v, str) and v.startswith('?'):
                        variables.add(v[1:])
                for v in action.ordered_values:
                    if isinstance(v, str) and v.startswith('?'):
                        variables.add(v[1:])
            elif isinstance(action, RetractAction):
                variables.add(action.fact_reference)
            elif isinstance(action, ModifyAction):
                variables.add(action.fact_reference)
                for v in action.modifications.values():
                    if isinstance(v, str) and v.startswith('?'):
                        variables.add(v[1:])
        return variables


# RuleBase (Knowledge Base)
@dataclass
class RuleBase:
    """Collection of rules and templates forming a knowledge base.
    
    Attributes:
        name: Name of the rule base
        rules: List of rules
        templates: List of fact templates
        initial_facts: List of initial facts
        functions: Dict of user-defined functions
        globals: Dict of global variables
    """
    name: str
    rules: List[Rule] = field(default_factory=list)
    templates: List[FactTemplate] = field(default_factory=list)
    initial_facts: List[Fact] = field(default_factory=list)
    functions: Dict[str, FunctionDef] = field(default_factory=dict)
    globals: Dict[str, Any] = field(default_factory=dict)
    
    def add_rule(self, rule: Rule) -> None:
        """Add a rule to the rule base."""
        self.rules.append(rule)
    
    def add_template(self, template: FactTemplate) -> None:
        """Add a template to the rule base."""
        self.templates.append(template)
    
    def add_initial_fact(self, fact: Fact) -> None:
        """Add an initial fact to the rule base."""
        self.initial_facts.append(fact)
    
    def add_function(self, func: FunctionDef) -> None:
        """Add a function to the rule base."""
        self.functions[func.name] = func
    
    def get_rule_by_name(self, name: str) -> Optional[Rule]:
        """Get a rule by name."""
        for rule in self.rules:
            if rule.name == name:
                return rule
        return None
    
    def get_template_by_name(self, name: str) -> Optional[FactTemplate]:
        """Get a template by name."""
        for template in self.templates:
            if template.name == name:
                return template
        return None
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about the rule base."""
        return {
            'rules': len(self.rules),
            'templates': len(self.templates),
            'initial_facts': len(self.initial_facts),
            'functions': len(self.functions),
            'globals': len(self.globals)
        }
