#!/usr/bin/env python3
"""STUNIR Functional IR Package.

This package provides IR constructs for functional programming,
including expressions, patterns, algebraic data types, and type inference.

Core Components:
    - functional_ir: Core expression types
    - pattern: Pattern matching definitions
    - adt: Algebraic data types
    - type_system: Type inference

Usage:
    from ir.functional import (
        # Expressions
        Expr, LiteralExpr, VarExpr, AppExpr, LambdaExpr,
        LetExpr, IfExpr, CaseExpr, ListExpr, TupleExpr,
        BinaryOpExpr, UnaryOpExpr, DoExpr,
        
        # Types
        TypeExpr, TypeVar, TypeCon, FunctionType,
        
        # Patterns
        Pattern, WildcardPattern, VarPattern, ConstructorPattern,
        
        # ADTs
        DataType, DataConstructor, TypeParameter,
        FunctionDef, Module,
        
        # Type System
        TypeInference, TypeEnvironment
    )
"""

# Core expressions and types
from ir.functional.functional_ir import (
    FunctionalNode,
    # Type expressions
    TypeExpr,
    TypeVar,
    TypeCon,
    FunctionType,
    TupleType,
    ListType,
    # Expressions
    Expr,
    LiteralExpr,
    VarExpr,
    AppExpr,
    LambdaExpr,
    LetExpr,
    IfExpr,
    CaseBranch,
    CaseExpr,
    ListGenerator,
    ListExpr,
    TupleExpr,
    BinaryOpExpr,
    UnaryOpExpr,
    # Do notation
    DoStatement,
    DoBindStatement,
    DoLetStatement,
    DoExprStatement,
    DoExpr,
    # Utilities
    make_app,
    make_lambda,
    make_let_chain,
)

# Pattern matching
from ir.functional.pattern import (
    Pattern,
    WildcardPattern,
    VarPattern,
    LiteralPattern,
    ConstructorPattern,
    TuplePattern,
    ListPattern,
    AsPattern,
    RecordPattern,
    OrPattern,
    get_pattern_variables,
    is_exhaustive,
    simplify_pattern,
)

# Algebraic data types
from ir.functional.adt import (
    TypeParameter,
    DataConstructor,
    DataType,
    TypeAlias,
    NewType,
    RecordField,
    RecordType,
    MethodSignature,
    TypeClass,
    TypeClassInstance,
    Import,
    FunctionClause,
    FunctionDef,
    Module,
)

# Type system
from ir.functional.type_system import (
    TypeError,
    TypeEnvironment,
    TypeInference,
    free_type_vars,
    type_to_string,
)

__all__ = [
    # Base
    'FunctionalNode',
    # Type expressions
    'TypeExpr', 'TypeVar', 'TypeCon', 'FunctionType', 'TupleType', 'ListType',
    # Expressions
    'Expr', 'LiteralExpr', 'VarExpr', 'AppExpr', 'LambdaExpr',
    'LetExpr', 'IfExpr', 'CaseBranch', 'CaseExpr', 'ListGenerator',
    'ListExpr', 'TupleExpr', 'BinaryOpExpr', 'UnaryOpExpr',
    # Do notation
    'DoStatement', 'DoBindStatement', 'DoLetStatement', 'DoExprStatement', 'DoExpr',
    # Patterns
    'Pattern', 'WildcardPattern', 'VarPattern', 'LiteralPattern',
    'ConstructorPattern', 'TuplePattern', 'ListPattern', 'AsPattern',
    'RecordPattern', 'OrPattern',
    # ADTs
    'TypeParameter', 'DataConstructor', 'DataType', 'TypeAlias', 'NewType',
    'RecordField', 'RecordType',
    # Type classes
    'MethodSignature', 'TypeClass', 'TypeClassInstance',
    # Module
    'Import', 'FunctionClause', 'FunctionDef', 'Module',
    # Type system
    'TypeError', 'TypeEnvironment', 'TypeInference',
    # Utilities
    'make_app', 'make_lambda', 'make_let_chain',
    'get_pattern_variables', 'is_exhaustive', 'simplify_pattern',
    'free_type_vars', 'type_to_string',
]
