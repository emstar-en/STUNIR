"""ASP Rule definitions.

This module defines the Rule class and supporting types for representing
ASP rules of various types: normal, choice, constraint, disjunctive, and weak.

Part of Phase 7D: Answer Set Programming
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from .asp_ir import RuleType, DEFAULT_PRIORITY, DEFAULT_WEIGHT
from .atom import Term, Atom, Literal
from .aggregate import Aggregate, Comparison


@dataclass
class BodyElement:
    """An element in a rule body.
    
    A body element can be one of:
    - A literal (atom with optional negation)
    - An aggregate expression
    - A comparison expression
    
    Attributes:
        literal: A literal (atom with negation)
        aggregate: An aggregate expression
        comparison: A comparison expression (term op term)
    """
    literal: Optional[Literal] = None
    aggregate: Optional[Aggregate] = None
    comparison: Optional[Comparison] = None
    
    def __post_init__(self):
        """Validate exactly one type is set."""
        count = sum([
            self.literal is not None,
            self.aggregate is not None,
            self.comparison is not None
        ])
        if count == 0:
            raise ValueError("BodyElement must have literal, aggregate, or comparison")
        if count > 1:
            raise ValueError("BodyElement must have exactly one of literal, aggregate, or comparison")
    
    @property
    def element_type(self) -> str:
        """Return the type of this body element."""
        if self.literal:
            return "literal"
        elif self.aggregate:
            return "aggregate"
        else:
            return "comparison"
    
    def get_variables(self) -> List[Term]:
        """Get all variables in this body element."""
        if self.literal:
            return self.literal.get_variables()
        elif self.aggregate:
            return self.aggregate.get_variables()
        else:
            return self.comparison.get_variables()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        if self.literal:
            return {"type": "literal", "literal": self.literal.to_dict()}
        elif self.aggregate:
            return {"type": "aggregate", "aggregate": self.aggregate.to_dict()}
        else:
            return {"type": "comparison", "comparison": self.comparison.to_dict()}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BodyElement':
        """Create from dictionary representation."""
        elem_type = data.get("type", "literal")
        if elem_type == "literal" or "literal" in data:
            return cls(literal=Literal.from_dict(data["literal"]))
        elif elem_type == "aggregate" or "aggregate" in data:
            return cls(aggregate=Aggregate.from_dict(data["aggregate"]))
        else:
            return cls(comparison=Comparison.from_dict(data["comparison"]))
    
    def __str__(self) -> str:
        """Return string representation."""
        if self.literal:
            return str(self.literal)
        elif self.aggregate:
            return str(self.aggregate)
        else:
            return str(self.comparison)


@dataclass
class HeadElement:
    """An element in a rule head.
    
    Attributes:
        atom: The head atom
        is_choice: Whether this is a choice element (for choice rules)
    """
    atom: Atom
    is_choice: bool = False
    
    def get_variables(self) -> List[Term]:
        """Get all variables in this head element."""
        return self.atom.get_variables()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "atom": self.atom.to_dict(),
            "is_choice": self.is_choice
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HeadElement':
        """Create from dictionary representation."""
        atom = Atom.from_dict(data["atom"])
        is_choice = data.get("is_choice", False)
        return cls(atom=atom, is_choice=is_choice)
    
    def __str__(self) -> str:
        return str(self.atom)


@dataclass
class ChoiceElement:
    """An element in a choice rule head.
    
    Attributes:
        atom: The choice atom
        condition: Optional condition literals
    """
    atom: Atom
    condition: List[Literal] = field(default_factory=list)
    
    def get_variables(self) -> List[Term]:
        """Get all variables in this choice element."""
        result = self.atom.get_variables()
        for lit in self.condition:
            result.extend(lit.get_variables())
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "atom": self.atom.to_dict(),
            "condition": [lit.to_dict() for lit in self.condition]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChoiceElement':
        """Create from dictionary representation."""
        atom = Atom.from_dict(data["atom"])
        condition = [Literal.from_dict(lit) for lit in data.get("condition", [])]
        return cls(atom=atom, condition=condition)
    
    def __str__(self) -> str:
        if self.condition:
            cond_str = ", ".join(str(lit) for lit in self.condition)
            return f"{self.atom} : {cond_str}"
        return str(self.atom)


@dataclass
class Rule:
    """An ASP rule.
    
    Supports multiple rule types:
    - NORMAL: head :- body.
    - CHOICE: {head} :- body. or L {head} U :- body.
    - CONSTRAINT: :- body.
    - DISJUNCTIVE: head1 | head2 :- body.
    - WEAK: :~ body. [weight@priority, terms]
    
    Attributes:
        rule_type: Type of rule (normal, choice, constraint, etc.)
        head: Head elements (empty for constraints)
        body: Body elements
        choice_elements: Elements for choice rules with conditions
        choice_lower: Lower bound for choice rules
        choice_upper: Upper bound for choice rules
        weight: Weight for weak constraints
        priority: Priority level for weak constraints
        terms: Terms for weak constraints
        comment: Optional comment for this rule
    """
    rule_type: RuleType
    head: List[HeadElement] = field(default_factory=list)
    body: List[BodyElement] = field(default_factory=list)
    choice_elements: List[ChoiceElement] = field(default_factory=list)
    choice_lower: Optional[int] = None
    choice_upper: Optional[int] = None
    weight: Optional[int] = None
    priority: Optional[int] = None
    terms: List[Term] = field(default_factory=list)
    comment: Optional[str] = None
    
    @property
    def is_fact(self) -> bool:
        """Check if this is a fact (no body)."""
        return self.rule_type == RuleType.NORMAL and len(self.body) == 0
    
    @property
    def is_constraint(self) -> bool:
        """Check if this is a constraint (no head)."""
        return self.rule_type == RuleType.CONSTRAINT
    
    @property
    def is_choice(self) -> bool:
        """Check if this is a choice rule."""
        return self.rule_type == RuleType.CHOICE
    
    @property
    def is_disjunctive(self) -> bool:
        """Check if this is a disjunctive rule."""
        return self.rule_type == RuleType.DISJUNCTIVE
    
    @property
    def is_weak(self) -> bool:
        """Check if this is a weak constraint."""
        return self.rule_type == RuleType.WEAK
    
    def get_head_variables(self) -> List[Term]:
        """Get all variables in the head."""
        result = []
        for elem in self.head:
            result.extend(elem.get_variables())
        for elem in self.choice_elements:
            result.extend(elem.get_variables())
        return result
    
    def get_body_variables(self) -> List[Term]:
        """Get all variables in the body."""
        result = []
        for elem in self.body:
            result.extend(elem.get_variables())
        return result
    
    def get_all_variables(self) -> List[Term]:
        """Get all variables in the rule."""
        result = self.get_head_variables()
        result.extend(self.get_body_variables())
        result.extend([t for t in self.terms if t.is_variable])
        return result
    
    def get_head_atoms(self) -> List[Atom]:
        """Get all atoms in the head."""
        result = [elem.atom for elem in self.head]
        result.extend([elem.atom for elem in self.choice_elements])
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "rule_type": self.rule_type.name,
            "head": [elem.to_dict() for elem in self.head],
            "body": [elem.to_dict() for elem in self.body]
        }
        if self.choice_elements:
            result["choice_elements"] = [elem.to_dict() for elem in self.choice_elements]
        if self.choice_lower is not None:
            result["choice_lower"] = self.choice_lower
        if self.choice_upper is not None:
            result["choice_upper"] = self.choice_upper
        if self.weight is not None:
            result["weight"] = self.weight
        if self.priority is not None:
            result["priority"] = self.priority
        if self.terms:
            result["terms"] = [t.to_dict() for t in self.terms]
        if self.comment:
            result["comment"] = self.comment
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Rule':
        """Create from dictionary representation."""
        rule_type = RuleType[data["rule_type"]]
        head = [HeadElement.from_dict(e) for e in data.get("head", [])]
        body = [BodyElement.from_dict(e) for e in data.get("body", [])]
        choice_elements = [ChoiceElement.from_dict(e) for e in data.get("choice_elements", [])]
        terms = [Term.from_dict(t) for t in data.get("terms", [])]
        return cls(
            rule_type=rule_type,
            head=head,
            body=body,
            choice_elements=choice_elements,
            choice_lower=data.get("choice_lower"),
            choice_upper=data.get("choice_upper"),
            weight=data.get("weight"),
            priority=data.get("priority"),
            terms=terms,
            comment=data.get("comment")
        )
    
    def __str__(self) -> str:
        """Return string representation (Clingo syntax)."""
        if self.rule_type == RuleType.NORMAL:
            return self._str_normal()
        elif self.rule_type == RuleType.CHOICE:
            return self._str_choice()
        elif self.rule_type == RuleType.CONSTRAINT:
            return self._str_constraint()
        elif self.rule_type == RuleType.DISJUNCTIVE:
            return self._str_disjunctive()
        elif self.rule_type == RuleType.WEAK:
            return self._str_weak()
        return "% Unknown rule type"
    
    def _str_normal(self) -> str:
        """String for normal rule."""
        head_str = "; ".join(str(e.atom) for e in self.head) if self.head else ""
        if not self.body:
            return f"{head_str}."
        body_str = ", ".join(str(e) for e in self.body)
        return f"{head_str} :- {body_str}."
    
    def _str_choice(self) -> str:
        """String for choice rule."""
        if self.choice_elements:
            elements_str = "; ".join(str(e) for e in self.choice_elements)
        else:
            elements_str = "; ".join(str(e.atom) for e in self.head)
        
        # Build choice head with bounds
        if self.choice_lower is not None and self.choice_upper is not None:
            head_str = f"{self.choice_lower} {{ {elements_str} }} {self.choice_upper}"
        elif self.choice_lower is not None:
            head_str = f"{self.choice_lower} {{ {elements_str} }}"
        elif self.choice_upper is not None:
            head_str = f"{{ {elements_str} }} {self.choice_upper}"
        else:
            head_str = f"{{ {elements_str} }}"
        
        if not self.body:
            return f"{head_str}."
        body_str = ", ".join(str(e) for e in self.body)
        return f"{head_str} :- {body_str}."
    
    def _str_constraint(self) -> str:
        """String for constraint."""
        body_str = ", ".join(str(e) for e in self.body)
        return f":- {body_str}."
    
    def _str_disjunctive(self) -> str:
        """String for disjunctive rule."""
        head_str = " | ".join(str(e.atom) for e in self.head)
        if not self.body:
            return f"{head_str}."
        body_str = ", ".join(str(e) for e in self.body)
        return f"{head_str} :- {body_str}."
    
    def _str_weak(self) -> str:
        """String for weak constraint (Clingo syntax)."""
        body_str = ", ".join(str(e) for e in self.body)
        weight = self.weight if self.weight is not None else DEFAULT_WEIGHT
        priority = self.priority if self.priority is not None else DEFAULT_PRIORITY
        if self.terms:
            terms_str = ", ".join(str(t) for t in self.terms)
            return f":~ {body_str}. [{weight}@{priority}, {terms_str}]"
        return f":~ {body_str}. [{weight}@{priority}]"


# Factory functions for creating rules
def normal_rule(head: Union[Atom, List[Atom]], body: List[Union[Literal, BodyElement]] = None) -> Rule:
    """Create a normal rule."""
    if isinstance(head, Atom):
        head_elems = [HeadElement(head)]
    else:
        head_elems = [HeadElement(a) for a in head]
    
    body_elems = []
    for b in (body or []):
        if isinstance(b, Literal):
            body_elems.append(BodyElement(literal=b))
        elif isinstance(b, BodyElement):
            body_elems.append(b)
    
    return Rule(rule_type=RuleType.NORMAL, head=head_elems, body=body_elems)


def fact(atom: Atom) -> Rule:
    """Create a fact (rule with no body)."""
    return Rule(rule_type=RuleType.NORMAL, head=[HeadElement(atom)])


def constraint(body: List[Union[Literal, BodyElement]]) -> Rule:
    """Create a constraint (rule with no head)."""
    body_elems = []
    for b in body:
        if isinstance(b, Literal):
            body_elems.append(BodyElement(literal=b))
        elif isinstance(b, BodyElement):
            body_elems.append(b)
    return Rule(rule_type=RuleType.CONSTRAINT, body=body_elems)


def choice_rule(
    elements: List[Union[Atom, ChoiceElement]],
    body: List[Union[Literal, BodyElement]] = None,
    lower: int = None,
    upper: int = None
) -> Rule:
    """Create a choice rule."""
    choice_elems = []
    for e in elements:
        if isinstance(e, Atom):
            choice_elems.append(ChoiceElement(e))
        else:
            choice_elems.append(e)
    
    body_elems = []
    for b in (body or []):
        if isinstance(b, Literal):
            body_elems.append(BodyElement(literal=b))
        elif isinstance(b, BodyElement):
            body_elems.append(b)
    
    return Rule(
        rule_type=RuleType.CHOICE,
        choice_elements=choice_elems,
        body=body_elems,
        choice_lower=lower,
        choice_upper=upper
    )


def disjunctive_rule(heads: List[Atom], body: List[Union[Literal, BodyElement]] = None) -> Rule:
    """Create a disjunctive rule."""
    head_elems = [HeadElement(a) for a in heads]
    body_elems = []
    for b in (body or []):
        if isinstance(b, Literal):
            body_elems.append(BodyElement(literal=b))
        elif isinstance(b, BodyElement):
            body_elems.append(b)
    return Rule(rule_type=RuleType.DISJUNCTIVE, head=head_elems, body=body_elems)


def weak_constraint(
    body: List[Union[Literal, BodyElement]],
    weight: int = 1,
    priority: int = 0,
    terms: List[Term] = None
) -> Rule:
    """Create a weak constraint."""
    body_elems = []
    for b in body:
        if isinstance(b, Literal):
            body_elems.append(BodyElement(literal=b))
        elif isinstance(b, BodyElement):
            body_elems.append(b)
    return Rule(
        rule_type=RuleType.WEAK,
        body=body_elems,
        weight=weight,
        priority=priority,
        terms=terms or []
    )
