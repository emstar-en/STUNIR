"""ASP Program representation.

This module defines the ASPProgram class and supporting types for representing
complete Answer Set Programs with rules, show statements, and optimization.

Part of Phase 7D: Answer Set Programming
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
import json
import hashlib

from .asp_ir import RuleType, AggregateFunction, DEFAULT_PRIORITY, DEFAULT_WEIGHT
from .atom import Term, Atom, Literal, pos, neg
from .aggregate import Aggregate, AggregateElement, Comparison
from .rule import (
    Rule, BodyElement, HeadElement, ChoiceElement,
    normal_rule, fact, constraint, choice_rule, disjunctive_rule, weak_constraint
)


@dataclass
class ShowStatement:
    """A #show statement to control output.
    
    Attributes:
        predicate: Predicate name to show
        arity: Number of arguments
        positive: True to show, False to hide
    
    Examples:
        - #show color/2.     (show color with arity 2)
        - #show -edge/2.     (hide edge with arity 2)
    """
    predicate: str
    arity: int
    positive: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "predicate": self.predicate,
            "arity": self.arity,
            "positive": self.positive
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ShowStatement':
        """Create from dictionary representation."""
        return cls(
            predicate=data["predicate"],
            arity=data["arity"],
            positive=data.get("positive", True)
        )
    
    def __str__(self) -> str:
        """Return string representation."""
        sign = "" if self.positive else "-"
        return f"#show {sign}{self.predicate}/{self.arity}."


@dataclass
class OptimizeStatement:
    """An optimization statement (#minimize or #maximize).
    
    Attributes:
        minimize: True for minimize, False for maximize
        elements: Elements to optimize
        priority: Priority level
    
    Examples:
        - #minimize { W@0,X : cost(X,W) }.
        - #maximize { V@1,I : value(I,V) }.
    """
    minimize: bool
    elements: List[AggregateElement] = field(default_factory=list)
    priority: int = 0
    
    def add_element(self, terms: List[Term], condition: List[Literal]) -> 'OptimizeStatement':
        """Add an element to optimize."""
        self.elements.append(AggregateElement(terms=terms, condition=condition))
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "minimize": self.minimize,
            "elements": [elem.to_dict() for elem in self.elements],
            "priority": self.priority
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizeStatement':
        """Create from dictionary representation."""
        elements = [AggregateElement.from_dict(e) for e in data.get("elements", [])]
        return cls(
            minimize=data["minimize"],
            elements=elements,
            priority=data.get("priority", 0)
        )
    
    def __str__(self) -> str:
        """Return string representation (Clingo syntax)."""
        directive = "#minimize" if self.minimize else "#maximize"
        elements_str = "; ".join(str(e) for e in self.elements)
        return f"{directive} {{ {elements_str} }}."


@dataclass
class ConstantDef:
    """A constant definition (#const).
    
    Attributes:
        name: Constant name
        value: Constant value
    
    Examples:
        - #const n = 10.
        - #const max_colors = 3.
    """
    name: str
    value: Union[int, str, Term]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        if isinstance(self.value, Term):
            return {"name": self.name, "value": self.value.to_dict(), "type": "term"}
        return {"name": self.name, "value": self.value}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConstantDef':
        """Create from dictionary representation."""
        if data.get("type") == "term":
            value = Term.from_dict(data["value"])
        else:
            value = data["value"]
        return cls(name=data["name"], value=value)
    
    def __str__(self) -> str:
        """Return string representation."""
        return f"#const {self.name} = {self.value}."


@dataclass
class ASPProgram:
    """A complete ASP program.
    
    Attributes:
        name: Program name/module identifier
        rules: List of rules in the program
        show_statements: Show statements for output control
        optimize_statements: Optimization statements
        constants: Constant definitions
        comments: Program comments (list of strings)
        metadata: Additional metadata
    """
    name: str = "main"
    rules: List[Rule] = field(default_factory=list)
    show_statements: List[ShowStatement] = field(default_factory=list)
    optimize_statements: List[OptimizeStatement] = field(default_factory=list)
    constants: List[ConstantDef] = field(default_factory=list)
    comments: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Builder methods
    def add_rule(self, rule: Rule) -> 'ASPProgram':
        """Add a rule to the program."""
        self.rules.append(rule)
        return self
    
    def add_fact(self, atom: Atom) -> 'ASPProgram':
        """Add a fact (rule with no body)."""
        self.rules.append(fact(atom))
        return self
    
    def add_normal_rule(self, head: Atom, body: List[Literal]) -> 'ASPProgram':
        """Add a normal rule."""
        self.rules.append(normal_rule(head, body))
        return self
    
    def add_constraint(self, body: List[Literal]) -> 'ASPProgram':
        """Add a constraint (rule with no head)."""
        self.rules.append(constraint(body))
        return self
    
    def add_choice_rule(
        self,
        elements: List[Union[Atom, ChoiceElement]],
        body: List[Literal] = None,
        lower: int = None,
        upper: int = None
    ) -> 'ASPProgram':
        """Add a choice rule."""
        self.rules.append(choice_rule(elements, body, lower, upper))
        return self
    
    def add_disjunctive_rule(self, heads: List[Atom], body: List[Literal] = None) -> 'ASPProgram':
        """Add a disjunctive rule."""
        self.rules.append(disjunctive_rule(heads, body))
        return self
    
    def add_weak_constraint(
        self,
        body: List[Literal],
        weight: int = 1,
        priority: int = 0,
        terms: List[Term] = None
    ) -> 'ASPProgram':
        """Add a weak constraint."""
        self.rules.append(weak_constraint(body, weight, priority, terms))
        return self
    
    def add_show(self, predicate: str, arity: int, positive: bool = True) -> 'ASPProgram':
        """Add a show statement."""
        self.show_statements.append(ShowStatement(predicate, arity, positive))
        return self
    
    def add_minimize(self, elements: List[AggregateElement], priority: int = 0) -> 'ASPProgram':
        """Add a minimize statement."""
        self.optimize_statements.append(OptimizeStatement(minimize=True, elements=elements, priority=priority))
        return self
    
    def add_maximize(self, elements: List[AggregateElement], priority: int = 0) -> 'ASPProgram':
        """Add a maximize statement."""
        self.optimize_statements.append(OptimizeStatement(minimize=False, elements=elements, priority=priority))
        return self
    
    def add_constant(self, name: str, value: Union[int, str, Term]) -> 'ASPProgram':
        """Add a constant definition."""
        self.constants.append(ConstantDef(name, value))
        return self
    
    def add_comment(self, comment: str) -> 'ASPProgram':
        """Add a comment."""
        self.comments.append(comment)
        return self
    
    # Query methods
    def get_predicates(self) -> List[str]:
        """Get all predicate names used in the program."""
        predicates = set()
        for rule in self.rules:
            for elem in rule.head:
                predicates.add(elem.atom.predicate)
            for elem in rule.choice_elements:
                predicates.add(elem.atom.predicate)
            for elem in rule.body:
                if elem.literal:
                    predicates.add(elem.literal.atom.predicate)
        return sorted(predicates)
    
    def get_facts(self) -> List[Rule]:
        """Get all facts in the program."""
        return [r for r in self.rules if r.is_fact]
    
    def get_constraints(self) -> List[Rule]:
        """Get all constraints in the program."""
        return [r for r in self.rules if r.is_constraint]
    
    def get_choice_rules(self) -> List[Rule]:
        """Get all choice rules in the program."""
        return [r for r in self.rules if r.is_choice]
    
    def get_rules_for_predicate(self, predicate: str) -> List[Rule]:
        """Get all rules with the given predicate in the head."""
        result = []
        for rule in self.rules:
            for elem in rule.head:
                if elem.atom.predicate == predicate:
                    result.append(rule)
                    break
            else:
                for elem in rule.choice_elements:
                    if elem.atom.predicate == predicate:
                        result.append(rule)
                        break
        return result
    
    # Serialization
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "rules": [rule.to_dict() for rule in self.rules],
            "show_statements": [stmt.to_dict() for stmt in self.show_statements],
            "optimize_statements": [stmt.to_dict() for stmt in self.optimize_statements],
            "constants": [const.to_dict() for const in self.constants],
            "comments": self.comments,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ASPProgram':
        """Create from dictionary representation."""
        rules = [Rule.from_dict(r) for r in data.get("rules", [])]
        show_statements = [ShowStatement.from_dict(s) for s in data.get("show_statements", [])]
        optimize_statements = [OptimizeStatement.from_dict(o) for o in data.get("optimize_statements", [])]
        constants = [ConstantDef.from_dict(c) for c in data.get("constants", [])]
        return cls(
            name=data.get("name", "main"),
            rules=rules,
            show_statements=show_statements,
            optimize_statements=optimize_statements,
            constants=constants,
            comments=data.get("comments", []),
            metadata=data.get("metadata", {})
        )
    
    def to_json(self, pretty: bool = False) -> str:
        """Convert to JSON string."""
        indent = 2 if pretty else None
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ASPProgram':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def compute_hash(self) -> str:
        """Compute SHA256 hash of canonical representation."""
        canonical = self.to_json(pretty=False)
        return hashlib.sha256(canonical.encode()).hexdigest()
    
    def __str__(self) -> str:
        """Return string representation (Clingo syntax)."""
        lines = []
        
        # Header comment
        lines.append(f"% ASP Program: {self.name}")
        lines.append(f"% Generated by STUNIR ASP IR")
        lines.append("")
        
        # User comments
        for comment in self.comments:
            lines.append(f"% {comment}")
        if self.comments:
            lines.append("")
        
        # Constants
        for const in self.constants:
            lines.append(str(const))
        if self.constants:
            lines.append("")
        
        # Rules (grouped by type)
        facts = self.get_facts()
        if facts:
            lines.append("% Facts")
            for rule in facts:
                lines.append(str(rule))
            lines.append("")
        
        normal_rules = [r for r in self.rules if r.rule_type == RuleType.NORMAL and not r.is_fact]
        if normal_rules:
            lines.append("% Rules")
            for rule in normal_rules:
                lines.append(str(rule))
            lines.append("")
        
        choice_rules = self.get_choice_rules()
        if choice_rules:
            lines.append("% Choice Rules")
            for rule in choice_rules:
                lines.append(str(rule))
            lines.append("")
        
        disjunctive_rules = [r for r in self.rules if r.is_disjunctive]
        if disjunctive_rules:
            lines.append("% Disjunctive Rules")
            for rule in disjunctive_rules:
                lines.append(str(rule))
            lines.append("")
        
        constraints = self.get_constraints()
        if constraints:
            lines.append("% Constraints")
            for rule in constraints:
                lines.append(str(rule))
            lines.append("")
        
        weak_constraints = [r for r in self.rules if r.is_weak]
        if weak_constraints:
            lines.append("% Weak Constraints")
            for rule in weak_constraints:
                lines.append(str(rule))
            lines.append("")
        
        # Optimization
        if self.optimize_statements:
            lines.append("% Optimization")
            for stmt in self.optimize_statements:
                lines.append(str(stmt))
            lines.append("")
        
        # Show statements
        if self.show_statements:
            lines.append("% Output")
            for stmt in self.show_statements:
                lines.append(str(stmt))
            lines.append("")
        
        return "\n".join(lines)


# Convenience function to create program builder
def program(name: str = "main") -> ASPProgram:
    """Create a new ASP program builder."""
    return ASPProgram(name=name)
