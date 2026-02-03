#!/usr/bin/env python3
"""STUNIR Actor Model IR - Message Passing.

This module provides IR constructs for message passing between processes,
including send, receive, and pattern matching.

Usage:
    from ir.actor import (
        SendExpr, ReceiveExpr, ReceiveClause,
        Pattern, VarPattern, TuplePattern, ListPattern,
        WildcardPattern, LiteralPattern, MapPattern,
        BinaryPattern, Guard
    )
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from ir.actor.actor_ir import ActorNode, Expr, Guard, BinarySegment


# =============================================================================
# Pattern Matching
# =============================================================================

@dataclass
class Pattern(ActorNode):
    """Base class for patterns in receive and function clauses."""
    kind: str = 'pattern'


@dataclass
class WildcardPattern(Pattern):
    """Wildcard pattern that matches anything (_)."""
    kind: str = 'wildcard_pattern'


@dataclass
class VarPattern(Pattern):
    """Variable pattern that binds a value.
    
    In Erlang: Variable (capitalized)
    In Elixir: variable (lowercase)
    """
    name: str = ''
    kind: str = 'var_pattern'


@dataclass
class LiteralPattern(Pattern):
    """Literal value pattern (atoms, numbers, strings)."""
    value: Any = None
    literal_type: str = 'auto'
    kind: str = 'literal_pattern'


@dataclass
class AtomPattern(Pattern):
    """Atom pattern."""
    value: str = ''
    kind: str = 'atom_pattern'


@dataclass
class TuplePattern(Pattern):
    """Tuple pattern {A, B, C}.
    
    Common for message patterns like {atom, Data}.
    """
    elements: List[Pattern] = field(default_factory=list)
    kind: str = 'tuple_pattern'


@dataclass
class ListPattern(Pattern):
    """List pattern [A, B, C] or [H|T].
    
    Attributes:
        elements: Fixed elements
        tail: Optional tail variable for [H|T] syntax
    """
    elements: List[Pattern] = field(default_factory=list)
    tail: Optional[Pattern] = None
    kind: str = 'list_pattern'


@dataclass
class MapPattern(Pattern):
    """Map pattern #{key := Value}.
    
    Note: Map patterns use := for matching existing keys.
    """
    pairs: List[tuple] = field(default_factory=list)  # [(key, pattern), ...]
    kind: str = 'map_pattern'


@dataclass
class BinaryPattern(Pattern):
    """Binary pattern <<A:8, B:16>>.
    
    Used for parsing binary protocols, network packets, etc.
    """
    segments: List[BinarySegment] = field(default_factory=list)
    kind: str = 'binary_pattern'


@dataclass
class RecordPattern(Pattern):
    """Record pattern #record{field = Value}.
    
    Erlang records are compile-time tuples with named fields.
    """
    record_name: str = ''
    fields: Dict[str, Pattern] = field(default_factory=dict)
    kind: str = 'record_pattern'


@dataclass
class PinPattern(Pattern):
    """Pin operator pattern (Elixir only).
    
    ^variable matches the existing value of variable.
    """
    variable: str = ''
    kind: str = 'pin_pattern'


# =============================================================================
# Message Passing
# =============================================================================

@dataclass
class SendExpr(ActorNode):
    """Send message to process.
    
    Erlang: Pid ! Message
    Elixir: send(pid, message)
    
    Attributes:
        target: PID or registered name to send to
        message: Message to send (any term)
    """
    target: Expr = None
    message: Expr = None
    kind: str = 'send'


@dataclass
class ReceiveClause(ActorNode):
    """A clause in a receive block.
    
    Attributes:
        pattern: Pattern to match incoming message
        guards: Optional guard expressions
        body: Expressions to execute when pattern matches
    """
    pattern: Pattern = None
    guards: List[Guard] = field(default_factory=list)
    body: List[Expr] = field(default_factory=list)
    kind: str = 'receive_clause'


@dataclass
class ReceiveExpr(ActorNode):
    """Receive expression for message handling.
    
    Blocks until a message matching one of the clauses arrives,
    or timeout expires.
    
    Erlang:
        receive
            Pattern1 -> Body1;
            Pattern2 -> Body2
        after Timeout ->
            TimeoutBody
        end
    
    Elixir:
        receive do
            pattern1 -> body1
            pattern2 -> body2
        after
            timeout -> timeout_body
        end
    
    Attributes:
        clauses: Message handling clauses
        timeout: Optional timeout expression (milliseconds)
        timeout_body: Expressions to execute on timeout
    """
    clauses: List[ReceiveClause] = field(default_factory=list)
    timeout: Optional[Expr] = None
    timeout_body: List[Expr] = field(default_factory=list)
    kind: str = 'receive'


@dataclass
class FlushExpr(ActorNode):
    """Flush all messages from mailbox."""
    kind: str = 'flush'


@dataclass
class SelectiveReceiveExpr(ActorNode):
    """Selective receive with reference.
    
    Used for request-response patterns where we only want
    to receive messages with a specific reference.
    """
    reference: Expr = None  # Reference to match
    clauses: List[ReceiveClause] = field(default_factory=list)
    timeout: Optional[Expr] = None
    timeout_body: List[Expr] = field(default_factory=list)
    kind: str = 'selective_receive'
