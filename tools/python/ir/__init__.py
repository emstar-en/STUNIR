#!/usr/bin/env python3
"""STUNIR IR Tools Package.

Provides IR processing, control flow analysis, symbolic and logic extensions.
"""

from .symbolic_ir import (
    SymbolicExprKind,
    SymbolicStmtKind,
    SYMBOLIC_KINDS,
    Symbol,
    Atom,
    SList,
    Cons,
    Quote,
    Quasiquote,
    Unquote,
    UnquoteSplicing,
    Lambda,
    Macro,
    SymbolicIRExtension,
    sexpr,
    sym,
    quote,
    quasiquote,
    unquote,
    unquote_splicing
)

from .logic_ir import (
    # Enums
    TermKind,
    GoalKind,
    LOGIC_KINDS,
    # Exceptions
    UnificationError,
    # Term classes (use LogicAtom to avoid conflict with symbolic Atom)
    Variable,
    Atom as LogicAtom,
    Number,
    StringTerm,
    Compound,
    ListTerm,
    Anonymous,
    # Clause classes
    Goal,
    Fact,
    Rule,
    Query,
    Predicate,
    # Substitution and unification
    Substitution,
    unify,
    # Conversion
    term_from_dict,
    # Extension
    LogicIRExtension,
)

__all__ = [
    # Symbolic IR Enums
    'SymbolicExprKind',
    'SymbolicStmtKind',
    'SYMBOLIC_KINDS',
    # Symbolic IR Data classes
    'Symbol',
    'Atom',
    'SList',
    'Cons',
    'Quote',
    'Quasiquote',
    'Unquote',
    'UnquoteSplicing',
    'Lambda',
    'Macro',
    # Symbolic IR Processor
    'SymbolicIRExtension',
    # Symbolic IR Builders
    'sexpr',
    'sym',
    'quote',
    'quasiquote',
    'unquote',
    'unquote_splicing',
    
    # Logic IR Enums
    'TermKind',
    'GoalKind',
    'LOGIC_KINDS',
    # Logic IR Exceptions
    'UnificationError',
    # Logic IR Term classes
    'Variable',
    'LogicAtom',
    'Number',
    'StringTerm',
    'Compound',
    'ListTerm',
    'Anonymous',
    # Logic IR Clause classes
    'Goal',
    'Fact',
    'Rule',
    'Query',
    'Predicate',
    # Logic IR Substitution and unification
    'Substitution',
    'unify',
    # Logic IR Conversion
    'term_from_dict',
    # Logic IR Extension
    'LogicIRExtension',
]
