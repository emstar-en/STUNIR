"""Forward chaining inference engine.

This module provides the ForwardChainingEngine class implementing
a forward chaining inference engine with basic Rete algorithm concepts.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
import bisect

from .rule_ir import ConflictResolutionStrategy
from .rule import Rule, RuleBase, Condition, PatternCondition, TestCondition, CompositeCondition
from .rule import Action, AssertAction, RetractAction, ModifyAction, BindAction, CallAction, PrintoutAction, HaltAction
from .rule import ConditionType
from .fact import Fact
from .working_memory import WorkingMemory
from .pattern import PatternMatcher


@dataclass(order=True)
class Activation:
    """An activation in the agenda (rule ready to fire).
    
    Attributes:
        salience: Rule priority (negated for max-heap behavior)
        timestamp: When the activation was created
        rule: The rule to fire
        bindings: Variable bindings for this activation
        matched_facts: IDs of facts that matched
    """
    salience: int = field(compare=True)  # Negated for priority queue
    timestamp: int = field(compare=True, default=0)
    rule: Rule = field(compare=False, default=None)
    bindings: Dict[str, Any] = field(compare=False, default_factory=dict)
    matched_facts: List[int] = field(compare=False, default_factory=list)
    
    def __post_init__(self):
        # Negate salience for max-heap behavior (higher salience = higher priority)
        self.salience = -self.salience if self.rule else 0


class Agenda:
    """Priority queue of rule activations.
    
    The agenda manages activations (rules that are ready to fire)
    and supports various conflict resolution strategies.
    """
    
    def __init__(self, strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.SALIENCE):
        """Initialize the agenda.
        
        Args:
            strategy: Conflict resolution strategy to use
        """
        self._activations: List[Activation] = []
        self._strategy = strategy
        self._timestamp = 0
    
    def add_activation(self, rule: Rule, bindings: Dict[str, Any], matched_facts: List[int]) -> None:
        """Add an activation to the agenda.
        
        Args:
            rule: The rule that matched
            bindings: Variable bindings
            matched_facts: IDs of matched facts
        """
        # Check for duplicate activation
        for existing in self._activations:
            if (existing.rule.name == rule.name and 
                set(existing.matched_facts) == set(matched_facts)):
                return  # Already activated
        
        activation = Activation(
            salience=rule.salience,
            timestamp=self._timestamp,
            rule=rule,
            bindings=bindings,
            matched_facts=matched_facts
        )
        self._timestamp += 1
        
        # Insert in sorted order
        bisect.insort(self._activations, activation)
    
    def remove_activation(self, rule_name: str, matched_facts: List[int]) -> bool:
        """Remove activations matching a rule and facts.
        
        Args:
            rule_name: Name of the rule
            matched_facts: IDs of matched facts
            
        Returns:
            True if any activations were removed
        """
        original_len = len(self._activations)
        matched_set = set(matched_facts)
        self._activations = [
            a for a in self._activations
            if not (a.rule.name == rule_name and matched_set.intersection(set(a.matched_facts)))
        ]
        return len(self._activations) < original_len
    
    def get_next(self) -> Optional[Activation]:
        """Get the next activation to fire.
        
        Returns:
            The highest priority activation, or None if empty
        """
        if not self._activations:
            return None
        return self._activations.pop(0)
    
    def peek(self) -> Optional[Activation]:
        """Peek at the next activation without removing it.
        
        Returns:
            The highest priority activation, or None if empty
        """
        if not self._activations:
            return None
        return self._activations[0]
    
    def clear(self) -> None:
        """Clear all activations."""
        self._activations.clear()
    
    def __len__(self) -> int:
        return len(self._activations)
    
    def __bool__(self) -> bool:
        return len(self._activations) > 0


class ForwardChainingEngine:
    """Forward chaining inference engine with basic Rete algorithm concepts.
    
    This engine implements forward chaining inference where rules are
    fired based on matching facts in working memory.
    """
    
    def __init__(self, rulebase: RuleBase, strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.SALIENCE):
        """Initialize the engine.
        
        Args:
            rulebase: The knowledge base
            strategy: Conflict resolution strategy
        """
        self.rulebase = rulebase
        self.working_memory = WorkingMemory()
        self.agenda = Agenda(strategy)
        self.pattern_matcher = PatternMatcher()
        self._fired_rules: List[Tuple[str, Dict[str, Any]]] = []
        self._halt = False
        self._output: List[str] = []
        
        # Initialize templates
        for template in rulebase.templates:
            self.working_memory.register_template(template)
        
        # Listen for working memory changes
        self.working_memory.add_listener(self._on_fact_change)
    
    def reset(self) -> None:
        """Reset the engine to initial state."""
        self.working_memory.clear()
        self.agenda.clear()
        self._fired_rules.clear()
        self._halt = False
        self._output.clear()
        
        # Re-register templates
        for template in self.rulebase.templates:
            self.working_memory.register_template(template)
        
        # Assert initial facts
        for fact in self.rulebase.initial_facts:
            self.working_memory.assert_fact(Fact(
                template_name=fact.template_name,
                slots=dict(fact.slots),
                values=fact.values
            ))
    
    def run(self, max_iterations: int = 1000) -> int:
        """Run the inference engine.
        
        Args:
            max_iterations: Maximum number of rule firings
            
        Returns:
            Number of rules fired
        """
        iterations = 0
        
        while iterations < max_iterations and not self._halt:
            activation = self.agenda.get_next()
            if activation is None:
                break
            
            self._fire_rule(activation)
            iterations += 1
        
        return iterations
    
    def step(self) -> bool:
        """Execute a single inference step.
        
        Returns:
            True if a rule was fired, False if agenda is empty
        """
        if self._halt:
            return False
        
        activation = self.agenda.get_next()
        if activation is None:
            return False
        
        self._fire_rule(activation)
        return True
    
    def _fire_rule(self, activation: Activation) -> None:
        """Fire a rule activation.
        
        Args:
            activation: The activation to fire
        """
        rule = activation.rule
        bindings = dict(activation.bindings)
        
        # Record the firing
        self._fired_rules.append((rule.name, dict(bindings)))
        
        # Execute actions
        for action in rule.actions:
            if self._halt:
                break
            self._execute_action(action, bindings)
    
    def _execute_action(self, action: Action, bindings: Dict[str, Any]) -> None:
        """Execute a rule action.
        
        Args:
            action: The action to execute
            bindings: Current variable bindings
        """
        
        if isinstance(action, AssertAction):
            if action.fact_template:
                # Template fact
                slots = {}
                for slot, value in action.slot_values.items():
                    slots[slot] = self._resolve_value(value, bindings)
                fact = Fact(template_name=action.fact_template, slots=slots)
            else:
                # Ordered fact
                values = tuple(self._resolve_value(v, bindings) for v in action.ordered_values)
                fact = Fact(values=values)
            self.working_memory.assert_fact(fact)
        
        elif isinstance(action, RetractAction):
            fact = bindings.get(action.fact_reference)
            if fact and isinstance(fact, Fact) and fact.fact_id is not None:
                self.working_memory.retract_fact(fact.fact_id)
        
        elif isinstance(action, ModifyAction):
            fact = bindings.get(action.fact_reference)
            if fact and isinstance(fact, Fact) and fact.fact_id is not None:
                resolved_mods = {
                    k: self._resolve_value(v, bindings)
                    for k, v in action.modifications.items()
                }
                self.working_memory.modify_fact(fact.fact_id, resolved_mods)
        
        elif isinstance(action, BindAction):
            value = self._evaluate_expression(action.expression, bindings)
            bindings[action.variable] = value
        
        elif isinstance(action, CallAction):
            # Execute function call
            args = [self._resolve_value(a, bindings) for a in action.arguments]
            if action.function_name in self.rulebase.functions:
                # User-defined function
                func = self.rulebase.functions[action.function_name]
                # Simple evaluation - bind parameters and evaluate body
                func_bindings = dict(zip(func.parameters, args))
                self._evaluate_expression(func.body, func_bindings)
        
        elif isinstance(action, PrintoutAction):
            output = []
            for item in action.items:
                if isinstance(item, str):
                    if item.startswith("?"):
                        output.append(str(bindings.get(item[1:], item)))
                    elif item == "crlf":
                        output.append("\n")
                    else:
                        output.append(item)
                else:
                    output.append(str(item))
            self._output.append("".join(output))
        
        elif isinstance(action, HaltAction):
            self._halt = True
    
    def _on_fact_change(self, event: str, fact: Fact) -> None:
        """Handle working memory changes.
        
        Args:
            event: Event type (assert, retract, modify)
            fact: The affected fact
        """
        if event == "assert":
            self._update_activations_for_assert(fact)
        elif event == "retract":
            self._update_activations_for_retract(fact)
        elif event == "modify":
            # Treat modify as retract + assert
            self._update_activations_for_retract(fact)
            self._update_activations_for_assert(fact)
    
    def _update_activations_for_assert(self, fact: Fact) -> None:
        """Update agenda when a fact is asserted.
        
        Args:
            fact: The asserted fact
        """
        for rule in self.rulebase.rules:
            # Try to match rule conditions
            matches = self._find_rule_matches(rule, fact)
            for bindings, matched_facts in matches:
                self.agenda.add_activation(rule, bindings, matched_facts)
    
    def _update_activations_for_retract(self, fact: Fact) -> None:
        """Update agenda when a fact is retracted.
        
        Args:
            fact: The retracted fact
        """
        # Remove activations involving this fact
        if fact.fact_id is not None:
            for rule in self.rulebase.rules:
                self.agenda.remove_activation(rule.name, [fact.fact_id])
    
    def _find_rule_matches(
        self,
        rule: Rule,
        trigger_fact: Fact
    ) -> List[Tuple[Dict[str, Any], List[int]]]:
        """Find all matches for a rule given a triggering fact.
        
        Args:
            rule: The rule to match
            trigger_fact: The fact that triggered matching
            
        Returns:
            List of (bindings, matched_fact_ids) tuples
        """
        matches = []
        
        # Try each condition as potential trigger
        for i, condition in enumerate(rule.conditions):
            if not isinstance(condition, PatternCondition):
                continue
            
            # Try to match trigger fact against this condition
            initial_bindings = self.pattern_matcher.match_pattern(
                condition, trigger_fact, {}
            )
            
            if initial_bindings is None:
                continue
            
            # Try to satisfy remaining conditions
            remaining = rule.conditions[:i] + rule.conditions[i+1:]
            all_matches = self._satisfy_conditions(remaining, initial_bindings)
            
            for bindings in all_matches:
                matched_facts = [trigger_fact.fact_id] if trigger_fact.fact_id else []
                # Collect other matched facts from bindings
                for var, val in bindings.items():
                    if isinstance(val, Fact) and val.fact_id is not None:
                        if val.fact_id not in matched_facts:
                            matched_facts.append(val.fact_id)
                matches.append((bindings, matched_facts))
        
        return matches
    
    def _satisfy_conditions(
        self,
        conditions: List[Condition],
        bindings: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find all bindings that satisfy a list of conditions.
        
        Args:
            conditions: List of conditions to satisfy
            bindings: Current variable bindings
            
        Returns:
            List of binding dicts that satisfy all conditions
        """
        if not conditions:
            return [bindings]
        
        condition = conditions[0]
        remaining = conditions[1:]
        all_matches = []
        
        if isinstance(condition, PatternCondition):
            # Get facts that might match
            if condition.template_name:
                candidates = list(self.working_memory.get_facts_by_template(condition.template_name))
            else:
                candidates = list(self.working_memory.get_ordered_facts())
            
            for fact in candidates:
                new_bindings = self.pattern_matcher.match_pattern(condition, fact, bindings)
                if new_bindings is not None:
                    all_matches.extend(self._satisfy_conditions(remaining, new_bindings))
        
        elif isinstance(condition, TestCondition):
            if self._evaluate_test(condition.expression, bindings):
                all_matches.extend(self._satisfy_conditions(remaining, bindings))
        
        elif isinstance(condition, CompositeCondition):
            if condition.condition_type == ConditionType.NOT:
                # NOT succeeds if no child pattern matches any fact in working memory
                has_match = False
                for child in condition.children:
                    if isinstance(child, PatternCondition):
                        # Get facts that might match
                        if child.template_name:
                            candidates = list(self.working_memory.get_facts_by_template(child.template_name))
                        else:
                            candidates = list(self.working_memory.get_ordered_facts())
                        for fact in candidates:
                            if self.pattern_matcher.match_pattern(child, fact, bindings) is not None:
                                has_match = True
                                break
                    if has_match:
                        break
                if not has_match:
                    all_matches.extend(self._satisfy_conditions(remaining, bindings))
            elif condition.condition_type == ConditionType.AND:
                # AND succeeds if all children succeed
                child_matches = self._satisfy_conditions(list(condition.children), bindings)
                for child_bindings in child_matches:
                    all_matches.extend(self._satisfy_conditions(remaining, child_bindings))
            elif condition.condition_type == ConditionType.OR:
                # OR succeeds if any child succeeds
                for child in condition.children:
                    child_matches = self._satisfy_conditions([child], bindings)
                    for child_bindings in child_matches:
                        all_matches.extend(self._satisfy_conditions(remaining, child_bindings))
        
        return all_matches
    
    def _resolve_value(self, value: Any, bindings: Dict[str, Any]) -> Any:
        """Resolve a value, substituting variable references.
        
        Args:
            value: The value to resolve
            bindings: Current variable bindings
            
        Returns:
            Resolved value
        """
        if isinstance(value, str) and value.startswith("?"):
            var_name = value[1:]
            return bindings.get(var_name, value)
        return value
    
    def _evaluate_expression(self, expr: str, bindings: Dict[str, Any]) -> Any:
        """Evaluate an expression with bindings.
        
        Args:
            expr: Expression string
            bindings: Current variable bindings
            
        Returns:
            Evaluated result
        """
        # Simple expression evaluation
        resolved = expr
        for var, val in bindings.items():
            resolved = resolved.replace(f"?{var}", repr(val))
        try:
            allowed_builtins = {
                'abs': abs, 'min': min, 'max': max,
                'len': len, 'str': str, 'int': int, 'float': float,
                'True': True, 'False': False, 'None': None
            }
            return eval(resolved, {"__builtins__": allowed_builtins}, {})
        except Exception:
            return None
    
    def _evaluate_test(self, expr: str, bindings: Dict[str, Any]) -> bool:
        """Evaluate a test condition.
        
        Args:
            expr: Test expression
            bindings: Current variable bindings
            
        Returns:
            True if test passes
        """
        result = self._evaluate_expression(expr, bindings)
        return bool(result)
    
    def get_fired_rules(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Get list of fired rules with their bindings.
        
        Returns:
            List of (rule_name, bindings) tuples
        """
        return list(self._fired_rules)
    
    def get_output(self) -> List[str]:
        """Get printed output from rule firings.
        
        Returns:
            List of output strings
        """
        return list(self._output)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics.
        
        Returns:
            Dict of statistics
        """
        return {
            'facts_count': len(self.working_memory),
            'rules_count': len(self.rulebase.rules),
            'activations_count': len(self.agenda),
            'fired_count': len(self._fired_rules),
            'halted': self._halt
        }
