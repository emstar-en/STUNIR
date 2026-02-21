#!/usr/bin/env python3
"""STUNIR Actor Model IR - Core Classes.

This module provides the core IR constructs for actor-based concurrency,
including processes, expressions, and base classes for message passing.

The actor model is centered around lightweight processes that communicate
via asynchronous message passing, as implemented in BEAM VM languages
(Erlang, Elixir).

Usage:
    from ir.actor import (
        ActorModule, ActorNode,
        Expr, LiteralExpr, VarExpr, AtomExpr,
        TupleExpr, ListExpr, MapExpr, CallExpr,
        FunctionDef, FunctionClause
    )
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum


class ActorNodeKind(Enum):
    """Kinds of actor model nodes."""
    PROCESS = 'process'
    MESSAGE = 'message'
    RECEIVE = 'receive'
    SEND = 'send'
    SPAWN = 'spawn'
    LINK = 'link'
    MONITOR = 'monitor'
    BEHAVIOR = 'behavior'
    SUPERVISOR = 'supervisor'
    APPLICATION = 'application'


@dataclass
class ActorNode:
    """Base class for all actor model IR nodes."""
    kind: str = 'actor_node'
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {'kind': self.kind}
        for key, value in self.__dict__.items():
            if key != 'kind':
                if hasattr(value, 'to_dict'):
                    result[key] = value.to_dict()
                elif isinstance(value, list):
                    result[key] = [
                        v.to_dict() if hasattr(v, 'to_dict') else v
                        for v in value
                    ]
                elif isinstance(value, Enum):
                    result[key] = value.value
                else:
                    result[key] = value
        return result


@dataclass
class ActorModule(ActorNode):
    """Top-level module for actor-based code.
    
    Attributes:
        name: Module name
        exports: List of exported function signatures (e.g., 'start/0')
        imports: List of imported modules
        attributes: Module attributes (e.g., -vsn, -author)
        processes: Process definitions
        functions: Function definitions
        behaviors: OTP behavior implementations
    """
    name: str = ''
    exports: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    processes: List['ProcessDef'] = field(default_factory=list)
    functions: List['FunctionDef'] = field(default_factory=list)
    behaviors: List['BehaviorImpl'] = field(default_factory=list)
    kind: str = 'actor_module'


# =============================================================================
# Expression Types
# =============================================================================

@dataclass
class Expr(ActorNode):
    """Base expression class."""
    kind: str = 'expr'


@dataclass
class LiteralExpr(Expr):
    """Literal value expression.
    
    Attributes:
        value: The literal value
        literal_type: Type hint ('atom', 'integer', 'float', 'string', 'binary')
    """
    value: Any = None
    literal_type: str = 'auto'
    kind: str = 'literal'


@dataclass
class VarExpr(Expr):
    """Variable reference expression."""
    name: str = ''
    kind: str = 'var'


@dataclass
class AtomExpr(Expr):
    """Atom literal expression.
    
    In Erlang: atom or 'quoted atom'
    In Elixir: :atom or :"quoted atom"
    """
    value: str = ''
    kind: str = 'atom'


@dataclass
class TupleExpr(Expr):
    """Tuple expression {A, B, C}."""
    elements: List[Expr] = field(default_factory=list)
    kind: str = 'tuple'


@dataclass
class ListExpr(Expr):
    """List expression [A, B, C] or [H|T].
    
    Attributes:
        elements: List elements
        tail: Optional tail for cons notation [H|T]
    """
    elements: List[Expr] = field(default_factory=list)
    tail: Optional[Expr] = None
    kind: str = 'list'


@dataclass
class MapExpr(Expr):
    """Map expression #{key => value} or %{key => value}."""
    pairs: List[tuple] = field(default_factory=list)  # [(key_expr, value_expr), ...]
    kind: str = 'map'


@dataclass
class BinaryExpr(Expr):
    """Binary expression <<1, 2, 3>>."""
    segments: List['BinarySegment'] = field(default_factory=list)
    kind: str = 'binary'


@dataclass
class BinarySegment(ActorNode):
    """Segment of binary expression/pattern.
    
    Attributes:
        value: The value or pattern
        size: Size in bits (optional)
        type_spec: Type specifier ('integer', 'float', 'binary', etc.)
        unit: Unit size multiplier
    """
    value: Any = None
    size: Optional[int] = None
    type_spec: str = 'integer'
    unit: int = 1
    kind: str = 'binary_segment'


@dataclass
class CallExpr(Expr):
    """Function call expression.
    
    Attributes:
        module: Optional module name (for Module:function calls)
        function: Function name
        args: Function arguments
    """
    module: Optional[str] = None
    function: str = ''
    args: List[Expr] = field(default_factory=list)
    kind: str = 'call'


@dataclass
class CaseExpr(Expr):
    """Case expression."""
    expr: Expr = None
    clauses: List['CaseClause'] = field(default_factory=list)
    kind: str = 'case'


@dataclass
class CaseClause(ActorNode):
    """Clause in case expression."""
    pattern: 'Pattern' = None
    guards: List['Guard'] = field(default_factory=list)
    body: List[Expr] = field(default_factory=list)
    kind: str = 'case_clause'


@dataclass
class IfExpr(Expr):
    """If expression (guards-based in Erlang)."""
    clauses: List[tuple] = field(default_factory=list)  # [(guards, body), ...]
    kind: str = 'if'


@dataclass
class FunExpr(Expr):
    """Anonymous function expression."""
    clauses: List['FunClause'] = field(default_factory=list)
    kind: str = 'fun'


@dataclass
class FunClause(ActorNode):
    """Clause of anonymous function."""
    args: List['Pattern'] = field(default_factory=list)
    guards: List['Guard'] = field(default_factory=list)
    body: List[Expr] = field(default_factory=list)
    kind: str = 'fun_clause'


@dataclass
class ListCompExpr(Expr):
    """List comprehension expression.
    
    Erlang: [Expr || Qualifiers]
    Elixir: for qualifiers, do: expr
    """
    expr: Expr = None
    qualifiers: List['Qualifier'] = field(default_factory=list)
    kind: str = 'list_comp'


@dataclass
class Qualifier(ActorNode):
    """Base class for list comprehension qualifiers."""
    kind: str = 'qualifier'


@dataclass
class Generator(Qualifier):
    """Generator qualifier (Pattern <- List)."""
    pattern: 'Pattern' = None
    list_expr: Expr = None
    kind: str = 'generator'


@dataclass
class Filter(Qualifier):
    """Filter qualifier (boolean expression)."""
    condition: Expr = None
    kind: str = 'filter'


@dataclass
class TryExpr(Expr):
    """Try-catch-after expression."""
    body: List[Expr] = field(default_factory=list)
    catch_clauses: List['CatchClause'] = field(default_factory=list)
    after_body: List[Expr] = field(default_factory=list)
    kind: str = 'try'


@dataclass
class CatchClause(ActorNode):
    """Clause in try-catch.
    
    Attributes:
        exception_type: 'throw', 'error', or 'exit'
        pattern: Pattern to match exception
        stacktrace: Optional variable for stacktrace
        guards: Guard expressions
        body: Handler body
    """
    exception_type: str = 'throw'
    pattern: 'Pattern' = None
    stacktrace: Optional[str] = None
    guards: List['Guard'] = field(default_factory=list)
    body: List[Expr] = field(default_factory=list)
    kind: str = 'catch_clause'


# =============================================================================
# Guard Expressions
# =============================================================================

@dataclass
class Guard(ActorNode):
    """Guard expression for pattern matching.
    
    Guards are comma-separated (AND) within a group,
    and semicolon-separated (OR) between groups.
    """
    conditions: List[Expr] = field(default_factory=list)
    kind: str = 'guard'


# =============================================================================
# Function Definitions
# =============================================================================

@dataclass
class FunctionClause(ActorNode):
    """A clause of a function definition.
    
    Erlang functions can have multiple clauses with pattern matching.
    """
    args: List['Pattern'] = field(default_factory=list)
    guards: List[Guard] = field(default_factory=list)
    body: List[Expr] = field(default_factory=list)
    kind: str = 'function_clause'


@dataclass
class FunctionDef(ActorNode):
    """Function definition with multiple clauses.
    
    Attributes:
        name: Function name
        arity: Number of arguments
        clauses: Function clauses (for pattern matching)
        spec: Optional type specification
        exported: Whether function is exported
    """
    name: str = ''
    arity: int = 0
    clauses: List[FunctionClause] = field(default_factory=list)
    spec: Optional[str] = None
    exported: bool = False
    kind: str = 'function_def'


# Forward reference placeholder for Pattern (defined in message.py)
Pattern = 'Pattern'
ProcessDef = 'ProcessDef'
BehaviorImpl = 'BehaviorImpl'
