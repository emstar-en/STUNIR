#!/usr/bin/env python3
"""STUNIR Logic IR Extensions.

This module provides logic programming constructs for STUNIR IR,
including terms, predicates, clauses, unification algorithm, and
backtracking support needed by Prolog-family emitters.

Part of Phase 5C-1: Logic Programming Foundation.
"""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Set
from abc import ABC, abstractmethod


class TermKind(Enum):
    """Term kinds in logic IR."""
    VARIABLE = "variable"
    ATOM = "atom"
    COMPOUND = "compound"
    LIST = "list_term"
    ANONYMOUS = "anonymous"
    NUMBER = "number"
    STRING = "string_term"


class GoalKind(Enum):
    """Goal kinds in logic IR."""
    CALL = "call"
    CUT = "cut"
    NEGATION = "negation"
    UNIFICATION = "unification"
    IF_THEN_ELSE = "if_then_else"
    CONJUNCTION = "conjunction"
    DISJUNCTION = "disjunction"


# All logic IR kinds
LOGIC_KINDS = {
    'variable', 'atom', 'compound', 'list_term', 'anonymous', 'string_term',
    'fact', 'rule', 'query', 'cut', 'negation', 'if_then_else',
    'findall', 'bagof', 'setof', 'assert', 'retract', 'unification',
    'conjunction', 'disjunction'
}


class UnificationError(Exception):
    """Raised when unification fails."""
    pass


class Term(ABC):
    """Base class for logic terms."""
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert to IR dictionary."""
        pass
    
    @abstractmethod
    def get_variables(self) -> List['Variable']:
        """Get all variables in this term."""
        pass
    
    @abstractmethod
    def apply_substitution(self, subst: 'Substitution') -> 'Term':
        """Apply substitution to get new term."""
        pass
    
    @abstractmethod
    def __eq__(self, other) -> bool:
        """Check equality."""
        pass
    
    @abstractmethod
    def __hash__(self) -> int:
        """Hash for use in sets/dicts."""
        pass


@dataclass
class Variable(Term):
    """Logic variable (starts with uppercase or _).
    
    In Prolog, variables are placeholders that can be bound to values
    through unification. Names starting with _ are anonymous variables.
    """
    name: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to IR dictionary representation."""
        return {'kind': 'variable', 'name': self.name}
    
    def get_variables(self) -> List['Variable']:
        """Return self as the only variable."""
        return [self]
    
    def apply_substitution(self, subst: 'Substitution') -> Term:
        """Apply substitution - returns bound value or self."""
        return subst.get(self, self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Variable':
        """Create Variable from IR dictionary."""
        return cls(name=data['name'])
    
    def is_anonymous(self) -> bool:
        """Check if this is an anonymous variable (_)."""
        return self.name.startswith('_')
    
    def __hash__(self) -> int:
        return hash(('variable', self.name))
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Variable) and self.name == other.name
    
    def __repr__(self) -> str:
        return f"Variable({self.name!r})"
    
    def __str__(self) -> str:
        return self.name


@dataclass
class Atom(Term):
    """Prolog atom (symbolic constant).
    
    Atoms are the basic named constants in Prolog, like 'hello', 'foo'.
    """
    value: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to IR dictionary representation."""
        return {'kind': 'atom', 'value': self.value}
    
    def get_variables(self) -> List[Variable]:
        """Atoms have no variables."""
        return []
    
    def apply_substitution(self, subst: 'Substitution') -> 'Atom':
        """Atoms are unchanged by substitution."""
        return self
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Atom':
        """Create Atom from IR dictionary."""
        return cls(value=data['value'])
    
    def __hash__(self) -> int:
        return hash(('atom', self.value))
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Atom) and self.value == other.value
    
    def __repr__(self) -> str:
        return f"Atom({self.value!r})"
    
    def __str__(self) -> str:
        return self.value


@dataclass
class Number(Term):
    """Numeric term (integer or float)."""
    value: Union[int, float]
    
    def to_dict(self) -> Union[int, float]:
        """Convert to IR representation (just the value)."""
        return self.value
    
    def get_variables(self) -> List[Variable]:
        """Numbers have no variables."""
        return []
    
    def apply_substitution(self, subst: 'Substitution') -> 'Number':
        """Numbers are unchanged by substitution."""
        return self
    
    @classmethod
    def from_dict(cls, data: Union[int, float, Dict]) -> 'Number':
        """Create Number from IR."""
        if isinstance(data, (int, float)):
            return cls(value=data)
        return cls(value=data.get('value', 0))
    
    def __hash__(self) -> int:
        return hash(('number', self.value))
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Number) and self.value == other.value
    
    def __repr__(self) -> str:
        return f"Number({self.value!r})"
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass
class StringTerm(Term):
    """String literal term."""
    value: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to IR dictionary representation."""
        return {'kind': 'string_term', 'value': self.value}
    
    def get_variables(self) -> List[Variable]:
        """Strings have no variables."""
        return []
    
    def apply_substitution(self, subst: 'Substitution') -> 'StringTerm':
        """Strings are unchanged by substitution."""
        return self
    
    def __hash__(self) -> int:
        return hash(('string', self.value))
    
    def __eq__(self, other) -> bool:
        return isinstance(other, StringTerm) and self.value == other.value
    
    def __repr__(self) -> str:
        return f"StringTerm({self.value!r})"
    
    def __str__(self) -> str:
        return f'"{self.value}"'


@dataclass
class Compound(Term):
    """Compound term: functor(arg1, arg2, ...).
    
    The primary structured data type in Prolog.
    """
    functor: str
    args: List[Term] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to IR dictionary representation."""
        return {
            'kind': 'compound',
            'functor': self.functor,
            'args': [_term_to_dict(arg) for arg in self.args]
        }
    
    def get_variables(self) -> List[Variable]:
        """Get all variables in arguments."""
        variables = []
        for arg in self.args:
            if hasattr(arg, 'get_variables'):
                variables.extend(arg.get_variables())
        return variables
    
    def apply_substitution(self, subst: 'Substitution') -> 'Compound':
        """Apply substitution to all arguments."""
        return Compound(
            self.functor,
            [_apply_subst(arg, subst) for arg in self.args]
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Compound':
        """Create Compound from IR dictionary."""
        return cls(
            functor=data['functor'],
            args=[term_from_dict(a) for a in data.get('args', [])]
        )
    
    @property
    def arity(self) -> int:
        """Return number of arguments."""
        return len(self.args)
    
    def __hash__(self) -> int:
        return hash(('compound', self.functor, tuple(hash(a) for a in self.args)))
    
    def __eq__(self, other) -> bool:
        return (isinstance(other, Compound) and 
                self.functor == other.functor and 
                self.args == other.args)
    
    def __repr__(self) -> str:
        args_str = ', '.join(repr(a) for a in self.args)
        return f"Compound({self.functor!r}, [{args_str}])"
    
    def __str__(self) -> str:
        if not self.args:
            return self.functor
        args_str = ', '.join(str(a) for a in self.args)
        return f"{self.functor}({args_str})"


@dataclass
class ListTerm(Term):
    """Prolog list: [H|T] or [a,b,c].
    
    Lists in Prolog can be proper (ending in []) or improper
    (ending in a variable for head/tail patterns).
    """
    elements: List[Term] = field(default_factory=list)
    tail: Optional[Term] = None  # None = proper list [], Variable = improper [H|T]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to IR dictionary representation."""
        result = {
            'kind': 'list_term',
            'elements': [_term_to_dict(el) for el in self.elements]
        }
        if self.tail is not None:
            result['tail'] = _term_to_dict(self.tail)
        return result
    
    def get_variables(self) -> List[Variable]:
        """Get all variables in elements and tail."""
        variables = []
        for el in self.elements:
            if hasattr(el, 'get_variables'):
                variables.extend(el.get_variables())
        if self.tail and hasattr(self.tail, 'get_variables'):
            variables.extend(self.tail.get_variables())
        return variables
    
    def apply_substitution(self, subst: 'Substitution') -> 'ListTerm':
        """Apply substitution to elements and tail."""
        new_elements = [_apply_subst(el, subst) for el in self.elements]
        new_tail = _apply_subst(self.tail, subst) if self.tail else None
        return ListTerm(new_elements, new_tail)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ListTerm':
        """Create ListTerm from IR dictionary."""
        elements = [term_from_dict(el) for el in data.get('elements', [])]
        tail = term_from_dict(data['tail']) if 'tail' in data else None
        return cls(elements=elements, tail=tail)
    
    def is_proper(self) -> bool:
        """Check if this is a proper list (ends in [])."""
        return self.tail is None
    
    def __hash__(self) -> int:
        tail_hash = hash(self.tail) if self.tail else 0
        return hash(('list', tuple(hash(e) for e in self.elements), tail_hash))
    
    def __eq__(self, other) -> bool:
        return (isinstance(other, ListTerm) and 
                self.elements == other.elements and 
                self.tail == other.tail)
    
    def __repr__(self) -> str:
        if self.tail:
            return f"ListTerm({self.elements!r}, tail={self.tail!r})"
        return f"ListTerm({self.elements!r})"
    
    def __str__(self) -> str:
        if not self.elements and self.tail is None:
            return '[]'
        els = ', '.join(str(el) for el in self.elements)
        if self.tail:
            return f'[{els}|{self.tail}]'
        return f'[{els}]'


@dataclass
class Anonymous(Term):
    """Anonymous variable (_).
    
    Anonymous variables are unique - each occurrence is independent.
    """
    _counter: int = field(default=0, compare=False)
    
    def __post_init__(self):
        Anonymous._counter += 1
        self._id = Anonymous._counter
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to IR dictionary representation."""
        return {'kind': 'anonymous'}
    
    def get_variables(self) -> List[Variable]:
        """Anonymous variables don't track as named variables."""
        return []
    
    def apply_substitution(self, subst: 'Substitution') -> 'Anonymous':
        """Anonymous variables don't bind."""
        return self
    
    def __hash__(self) -> int:
        return hash(('anonymous', self._id))
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Anonymous) and self._id == other._id
    
    def __repr__(self) -> str:
        return "Anonymous()"
    
    def __str__(self) -> str:
        return '_'


# Helper functions for term conversion
def _term_to_dict(term: Any) -> Any:
    """Convert term to dictionary representation."""
    if hasattr(term, 'to_dict'):
        return term.to_dict()
    return term


def _apply_subst(term: Any, subst: 'Substitution') -> Any:
    """Apply substitution to term if possible."""
    if hasattr(term, 'apply_substitution'):
        return term.apply_substitution(subst)
    return term


def term_from_dict(data: Any) -> Term:
    """Create a Term from IR dictionary representation."""
    if isinstance(data, (int, float)):
        return Number(data)
    if isinstance(data, str):
        return Atom(data)
    if not isinstance(data, dict):
        return Atom(str(data))
    
    kind = data.get('kind', '')
    
    if kind == 'variable':
        return Variable.from_dict(data)
    elif kind == 'atom':
        return Atom.from_dict(data)
    elif kind == 'compound':
        return Compound.from_dict(data)
    elif kind == 'list_term':
        return ListTerm.from_dict(data)
    elif kind == 'anonymous':
        return Anonymous()
    elif kind == 'string_term':
        return StringTerm(data.get('value', ''))
    elif kind == 'number':
        return Number(data.get('value', 0))
    else:
        # Try to interpret as atom
        if 'value' in data:
            return Atom(str(data['value']))
        return Atom(str(data))


# Substitution class
class Substitution:
    """Substitution mapping variables to terms.
    
    A substitution is the result of unification, mapping
    variables to their bound values.
    """
    
    def __init__(self, bindings: Optional[Dict[Variable, Term]] = None):
        """Initialize with optional bindings dictionary."""
        self.bindings: Dict[Variable, Term] = bindings or {}
    
    def get(self, var: Variable, default: Term = None) -> Term:
        """Get binding for variable, or default if not bound."""
        return self.bindings.get(var, default)
    
    def bind(self, var: Variable, term: Term) -> 'Substitution':
        """Create new substitution with additional binding."""
        new_bindings = dict(self.bindings)
        new_bindings[var] = term
        return Substitution(new_bindings)
    
    def compose(self, other: 'Substitution') -> 'Substitution':
        """Compose two substitutions (self ∘ other).
        
        The composition applies other to all terms in self,
        then adds any bindings from other not in self.
        """
        new_bindings = {}
        for var, term in self.bindings.items():
            new_bindings[var] = _apply_subst(term, other)
        for var, term in other.bindings.items():
            if var not in new_bindings:
                new_bindings[var] = term
        return Substitution(new_bindings)
    
    def domain(self) -> Set[Variable]:
        """Return the set of bound variables."""
        return set(self.bindings.keys())
    
    def is_empty(self) -> bool:
        """Check if substitution is empty."""
        return len(self.bindings) == 0
    
    def __contains__(self, var: Variable) -> bool:
        return var in self.bindings
    
    def __len__(self) -> int:
        return len(self.bindings)
    
    def __repr__(self) -> str:
        if not self.bindings:
            return "Substitution({})"
        items = ', '.join(f"{v.name}={t}" for v, t in self.bindings.items())
        return f"Substitution({{{items}}})"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Substitution) and self.bindings == other.bindings


# Unification Algorithm
def _occurs_in(var: Variable, term: Term) -> bool:
    """Check if variable occurs in term (for occurs check).
    
    This prevents the creation of infinite terms like X = f(X).
    """
    if isinstance(term, Variable):
        return var == term
    if isinstance(term, Compound):
        return any(_occurs_in(var, arg) for arg in term.args)
    if isinstance(term, ListTerm):
        in_elements = any(_occurs_in(var, el) for el in term.elements)
        in_tail = term.tail and _occurs_in(var, term.tail)
        return in_elements or in_tail
    return False


def unify(term1: Term, term2: Term, 
          subst: Optional[Substitution] = None,
          occurs_check: bool = True) -> Substitution:
    """Unify two terms, returning the most general unifier (MGU).
    
    Args:
        term1: First term to unify
        term2: Second term to unify
        subst: Initial substitution (default: empty)
        occurs_check: Whether to perform occurs check (default: True)
    
    Returns:
        Substitution representing the MGU
    
    Raises:
        UnificationError: If terms cannot be unified
    
    This implements Robinson's unification algorithm:
    1. Apply current substitution to both terms
    2. If terms are equal, return current substitution
    3. If either is a variable, bind it (checking occurs)
    4. If both are compounds, unify functors and args recursively
    5. If both are lists, unify elements and tails
    """
    if subst is None:
        subst = Substitution()
    
    # Apply current substitution
    t1 = _apply_subst(term1, subst)
    t2 = _apply_subst(term2, subst)
    
    # Same term - already unified
    if t1 == t2:
        return subst
    
    # Variable cases
    if isinstance(t1, Variable):
        if occurs_check and _occurs_in(t1, t2):
            raise UnificationError(f"Occurs check failed: {t1} in {t2}")
        return subst.bind(t1, t2)
    
    if isinstance(t2, Variable):
        if occurs_check and _occurs_in(t2, t1):
            raise UnificationError(f"Occurs check failed: {t2} in {t1}")
        return subst.bind(t2, t1)
    
    # Anonymous variable cases - always succeed without binding
    if isinstance(t1, Anonymous) or isinstance(t2, Anonymous):
        return subst
    
    # Compound unification
    if isinstance(t1, Compound) and isinstance(t2, Compound):
        if t1.functor != t2.functor:
            raise UnificationError(f"Functor mismatch: {t1.functor} vs {t2.functor}")
        if t1.arity != t2.arity:
            raise UnificationError(f"Arity mismatch: {t1.functor}/{t1.arity} vs {t2.functor}/{t2.arity}")
        
        for a1, a2 in zip(t1.args, t2.args):
            subst = unify(a1, a2, subst, occurs_check)
        return subst
    
    # List unification
    if isinstance(t1, ListTerm) and isinstance(t2, ListTerm):
        return _unify_lists(t1, t2, subst, occurs_check)
    
    # Atom equality
    if isinstance(t1, Atom) and isinstance(t2, Atom):
        if t1.value != t2.value:
            raise UnificationError(f"Atom mismatch: {t1.value} vs {t2.value}")
        return subst
    
    # Number equality
    if isinstance(t1, Number) and isinstance(t2, Number):
        if t1.value != t2.value:
            raise UnificationError(f"Number mismatch: {t1.value} vs {t2.value}")
        return subst
    
    # String equality
    if isinstance(t1, StringTerm) and isinstance(t2, StringTerm):
        if t1.value != t2.value:
            raise UnificationError(f"String mismatch: {t1.value} vs {t2.value}")
        return subst
    
    raise UnificationError(f"Cannot unify {type(t1).__name__} with {type(t2).__name__}")


def _unify_lists(list1: ListTerm, list2: ListTerm, 
                 subst: Substitution, occurs_check: bool) -> Substitution:
    """Unify two list terms.
    
    Handles both proper lists and [H|T] patterns.
    """
    # Both empty
    if not list1.elements and not list2.elements:
        if list1.tail is None and list2.tail is None:
            return subst
        if list1.tail is not None and list2.tail is not None:
            return unify(list1.tail, list2.tail, subst, occurs_check)
        if list1.tail is not None:
            return unify(list1.tail, ListTerm([]), subst, occurs_check)
        return unify(list2.tail, ListTerm([]), subst, occurs_check)
    
    # Pattern matching [H|T] with [a,b,c]
    if len(list1.elements) == 1 and list1.tail is not None and len(list2.elements) > 0:
        # list1 is [H|T], list2 is [a, b, c, ...]
        h1 = list1.elements[0]
        t1 = list1.tail
        h2 = list2.elements[0]
        rest2 = ListTerm(list2.elements[1:], list2.tail)
        
        subst = unify(h1, h2, subst, occurs_check)
        return unify(t1, rest2, subst, occurs_check)
    
    if len(list2.elements) == 1 and list2.tail is not None and len(list1.elements) > 0:
        # list2 is [H|T], list1 is [a, b, c, ...]
        h2 = list2.elements[0]
        t2 = list2.tail
        h1 = list1.elements[0]
        rest1 = ListTerm(list1.elements[1:], list1.tail)
        
        subst = unify(h1, h2, subst, occurs_check)
        return unify(rest1, t2, subst, occurs_check)
    
    # Same length - unify element by element
    if len(list1.elements) == len(list2.elements):
        for e1, e2 in zip(list1.elements, list2.elements):
            subst = unify(e1, e2, subst, occurs_check)
        # Unify tails
        if list1.tail is None and list2.tail is None:
            return subst
        if list1.tail is not None and list2.tail is not None:
            return unify(list1.tail, list2.tail, subst, occurs_check)
        if list1.tail is not None:
            return unify(list1.tail, ListTerm([]), subst, occurs_check)
        return unify(list2.tail, ListTerm([]), subst, occurs_check)
    
    # Different lengths without tail patterns
    raise UnificationError(f"List length mismatch: {len(list1.elements)} vs {len(list2.elements)}")


# Clause types
@dataclass
class Goal:
    """A goal to be proven.
    
    Goals are the building blocks of rule bodies and queries.
    """
    kind: GoalKind
    term: Optional[Term] = None
    goals: Optional[List['Goal']] = None
    left: Optional[Term] = None  # For unification goals
    right: Optional[Term] = None  # For unification goals
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to IR dictionary representation."""
        result = {'kind': self.kind.value}
        if self.term:
            result['term'] = _term_to_dict(self.term)
        if self.goals:
            result['goals'] = [g.to_dict() for g in self.goals]
        if self.left:
            result['left'] = _term_to_dict(self.left)
        if self.right:
            result['right'] = _term_to_dict(self.right)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Goal':
        """Create Goal from IR dictionary."""
        kind_str = data.get('kind', 'call')
        try:
            kind = GoalKind(kind_str)
        except ValueError:
            kind = GoalKind.CALL
        
        term = term_from_dict(data['term']) if 'term' in data else None
        goals = [cls.from_dict(g) for g in data.get('goals', [])] if 'goals' in data else None
        left = term_from_dict(data['left']) if 'left' in data else None
        right = term_from_dict(data['right']) if 'right' in data else None
        
        return cls(kind=kind, term=term, goals=goals, left=left, right=right)
    
    @classmethod
    def call(cls, term: Term) -> 'Goal':
        """Create a call goal."""
        return cls(kind=GoalKind.CALL, term=term)
    
    @classmethod
    def cut(cls) -> 'Goal':
        """Create a cut goal."""
        return cls(kind=GoalKind.CUT)
    
    @classmethod
    def negation(cls, goal: 'Goal') -> 'Goal':
        """Create a negation-as-failure goal."""
        return cls(kind=GoalKind.NEGATION, goals=[goal])
    
    @classmethod
    def unification(cls, left: Term, right: Term) -> 'Goal':
        """Create a unification goal (X = Y)."""
        return cls(kind=GoalKind.UNIFICATION, left=left, right=right)


@dataclass
class Fact:
    """A fact (ground clause without body).
    
    Facts are unconditionally true predicates.
    """
    predicate: str
    args: List[Term] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to IR dictionary representation."""
        return {
            'kind': 'fact',
            'predicate': self.predicate,
            'args': [_term_to_dict(arg) for arg in self.args]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Fact':
        """Create Fact from IR dictionary."""
        return cls(
            predicate=data['predicate'],
            args=[term_from_dict(a) for a in data.get('args', [])]
        )
    
    @property
    def arity(self) -> int:
        """Return number of arguments."""
        return len(self.args)
    
    def get_variables(self) -> List[Variable]:
        """Get all variables in arguments."""
        variables = []
        for arg in self.args:
            if hasattr(arg, 'get_variables'):
                variables.extend(arg.get_variables())
        return variables
    
    def __str__(self) -> str:
        if not self.args:
            return f"{self.predicate}."
        args_str = ', '.join(str(a) for a in self.args)
        return f"{self.predicate}({args_str})."


@dataclass
class Rule:
    """A rule (clause with head and body goals).
    
    Rules define conditional predicates: head :- body.
    """
    head: Compound
    body: List[Goal] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to IR dictionary representation."""
        return {
            'kind': 'rule',
            'head': self.head.to_dict(),
            'body': [goal.to_dict() for goal in self.body]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Rule':
        """Create Rule from IR dictionary."""
        head_data = data.get('head', {})
        head = Compound.from_dict(head_data) if isinstance(head_data, dict) else Compound(str(head_data), [])
        body = [_parse_goal(g) for g in data.get('body', [])]
        return cls(head=head, body=body)
    
    @property
    def predicate(self) -> str:
        """Return the predicate name."""
        return self.head.functor
    
    @property
    def arity(self) -> int:
        """Return the predicate arity."""
        return self.head.arity
    
    def get_variables(self) -> List[Variable]:
        """Get all variables in head and body."""
        variables = self.head.get_variables()
        for goal in self.body:
            if goal.term and hasattr(goal.term, 'get_variables'):
                variables.extend(goal.term.get_variables())
        return variables
    
    def __str__(self) -> str:
        head_str = str(self.head)
        if not self.body:
            return f"{head_str}."
        body_str = ', '.join(str(g.term) if g.term else g.kind.value for g in self.body)
        return f"{head_str} :- {body_str}."


def _parse_goal(data: Dict[str, Any]) -> Goal:
    """Parse a goal from IR dictionary."""
    kind = data.get('kind', 'call')
    
    if kind == 'cut':
        return Goal.cut()
    elif kind == 'negation':
        inner = _parse_goal(data.get('goal', data.get('goals', [{}])[0]))
        return Goal.negation(inner)
    elif kind == 'unification':
        left = term_from_dict(data.get('left', {}))
        right = term_from_dict(data.get('right', {}))
        return Goal.unification(left, right)
    elif kind == 'compound':
        term = Compound.from_dict(data)
        return Goal.call(term)
    else:
        # Default: treat as call
        if 'functor' in data:
            term = Compound.from_dict(data)
        else:
            term = term_from_dict(data)
        return Goal.call(term)


@dataclass
class Query:
    """A query (list of goals to prove).
    
    Queries are the questions we ask the Prolog system.
    """
    goals: List[Goal] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to IR dictionary representation."""
        return {
            'kind': 'query',
            'goals': [goal.to_dict() for goal in self.goals]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Query':
        """Create Query from IR dictionary."""
        goals = [_parse_goal(g) for g in data.get('goals', [])]
        return cls(goals=goals)
    
    def __str__(self) -> str:
        goals_str = ', '.join(str(g.term) if g.term else g.kind.value for g in self.goals)
        return f"?- {goals_str}."


@dataclass
class Predicate:
    """A predicate definition (collection of clauses).
    
    Groups all clauses with the same functor/arity.
    """
    name: str
    arity: int
    clauses: List[Union[Fact, Rule]] = field(default_factory=list)
    is_dynamic: bool = False
    is_multifile: bool = False
    is_discontiguous: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to IR dictionary representation."""
        return {
            'kind': 'predicate',
            'name': self.name,
            'arity': self.arity,
            'clauses': [c.to_dict() for c in self.clauses],
            'is_dynamic': self.is_dynamic,
            'is_multifile': self.is_multifile,
            'is_discontiguous': self.is_discontiguous
        }
    
    def add_clause(self, clause: Union[Fact, Rule]) -> None:
        """Add a clause to the predicate."""
        self.clauses.append(clause)
    
    @property
    def indicator(self) -> str:
        """Return predicate indicator (name/arity)."""
        return f"{self.name}/{self.arity}"
    
    def __str__(self) -> str:
        return f"Predicate({self.indicator}, {len(self.clauses)} clauses)"


# LogicIRExtension class
class LogicIRExtension:
    """Processes logic programming IR extensions.
    
    This class handles validation and extraction of logic programming
    constructs from STUNIR IR.
    """
    
    def __init__(self):
        """Initialize the extension."""
        self._schema = None
    
    def _load_schema(self) -> Optional[Dict[str, Any]]:
        """Load the logic IR schema."""
        schema_path = Path(__file__).parent.parent.parent / 'schemas' / 'logic_ir.json'
        if schema_path.exists():
            with open(schema_path) as f:
                return json.load(f)
        return None
    
    def validate(self, ir: Dict[str, Any]) -> bool:
        """Validate IR against logic schema.
        
        Args:
            ir: IR dictionary to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If IR is invalid
        """
        # Basic structure validation
        if not isinstance(ir, dict):
            raise ValueError("IR must be a dictionary")
        
        # Validate clauses
        for clause in ir.get('clauses', []):
            self._validate_clause(clause)
        
        # Validate queries
        for query in ir.get('queries', []):
            self._validate_query(query)
        
        return True
    
    def _validate_clause(self, clause: Dict[str, Any]) -> None:
        """Validate a clause."""
        kind = clause.get('kind')
        if kind not in ('fact', 'rule'):
            if 'functor' in clause or 'predicate' in clause:
                return  # Compound term treated as goal
            raise ValueError(f"Invalid clause kind: {kind}")
        
        if kind == 'fact':
            if 'predicate' not in clause:
                raise ValueError("Fact must have 'predicate' field")
        elif kind == 'rule':
            if 'head' not in clause:
                raise ValueError("Rule must have 'head' field")
    
    def _validate_query(self, query: Dict[str, Any]) -> None:
        """Validate a query."""
        if query.get('kind') != 'query':
            raise ValueError(f"Expected query, got: {query.get('kind')}")
    
    def has_logic_features(self, ir: Dict[str, Any]) -> bool:
        """Check if IR uses logic programming features.
        
        Args:
            ir: IR dictionary to check
            
        Returns:
            True if IR contains logic programming constructs
        """
        return self._contains_kinds(ir, LOGIC_KINDS)
    
    def _contains_kinds(self, data: Any, kinds: Set[str]) -> bool:
        """Recursively check if data contains any of the specified kinds."""
        if isinstance(data, dict):
            if data.get('kind') in kinds:
                return True
            return any(self._contains_kinds(v, kinds) for v in data.values())
        if isinstance(data, list):
            return any(self._contains_kinds(item, kinds) for item in data)
        return False
    
    def extract_predicates(self, ir: Dict[str, Any]) -> Dict[Tuple[str, int], Predicate]:
        """Extract predicate definitions from IR.
        
        Args:
            ir: IR dictionary
            
        Returns:
            Dictionary mapping (name, arity) to Predicate objects
        """
        predicates: Dict[Tuple[str, int], Predicate] = {}
        
        for item in ir.get('clauses', []):
            kind = item.get('kind')
            
            if kind == 'fact':
                fact = Fact.from_dict(item)
                key = (fact.predicate, fact.arity)
                if key not in predicates:
                    predicates[key] = Predicate(fact.predicate, fact.arity)
                predicates[key].add_clause(fact)
                
            elif kind == 'rule':
                rule = Rule.from_dict(item)
                key = (rule.predicate, rule.arity)
                if key not in predicates:
                    predicates[key] = Predicate(rule.predicate, rule.arity)
                predicates[key].add_clause(rule)
            
            elif 'functor' in item:
                # Compound term as clause
                name = item.get('functor')
                args = item.get('args', [])
                arity = len(args)
                key = (name, arity)
                if key not in predicates:
                    predicates[key] = Predicate(name, arity)
                fact = Fact(name, [term_from_dict(a) for a in args])
                predicates[key].add_clause(fact)
        
        # Mark dynamic predicates
        for dyn in ir.get('dynamic', []):
            pred = dyn.get('predicate')
            arity = dyn.get('arity', 0)
            key = (pred, arity)
            if key in predicates:
                predicates[key].is_dynamic = True
            else:
                predicates[key] = Predicate(pred, arity, is_dynamic=True)
        
        return predicates
    
    def extract_queries(self, ir: Dict[str, Any]) -> List[Query]:
        """Extract queries from IR.
        
        Args:
            ir: IR dictionary
            
        Returns:
            List of Query objects
        """
        queries = []
        for item in ir.get('queries', []):
            queries.append(Query.from_dict(item))
        return queries
    
    def extract_dcg_rules(self, ir: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract DCG rules from IR.
        
        Args:
            ir: IR dictionary
            
        Returns:
            List of DCG rule dictionaries
        """
        return ir.get('dcg_rules', [])


# Export all public symbols
__all__ = [
    # Enums
    'TermKind', 'GoalKind', 'LOGIC_KINDS',
    # Exceptions
    'UnificationError',
    # Term classes
    'Term', 'Variable', 'Atom', 'Number', 'StringTerm', 
    'Compound', 'ListTerm', 'Anonymous',
    # Clause classes
    'Goal', 'Fact', 'Rule', 'Query', 'Predicate',
    # Substitution and unification
    'Substitution', 'unify',
    # Conversion functions
    'term_from_dict',
    # Extension class
    'LogicIRExtension',
]
