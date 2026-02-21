#!/usr/bin/env python3
"""STUNIR Systems IR - Package exports.

This package provides IR nodes for systems programming languages
including Ada (with SPARK formal verification) and D.

Usage:
    from ir.systems import Package, Subprogram, TaskType, Contract
    from ir.systems import AdaEmitter, DEmitter  # When available
"""

# Core IR nodes
from ir.systems.systems_ir import (
    # Enums
    Visibility,
    Mode,
    SafetyLevel,
    # Base
    SystemsNode,
    TypeRef,
    # Expressions
    Expr,
    Literal,
    VarExpr,
    BinaryOp,
    UnaryOp,
    CallExpr,
    MemberAccess,
    IndexExpr,
    IfExpr,
    CastExpr,
    RangeExpr,
    AggregateExpr,
    AttributeExpr,
    QualifiedExpr,
    # Statements
    Statement,
    Assignment,
    IfStatement,
    ElsifPart,
    CaseStatement,
    CaseAlternative,
    WhileLoop,
    ForLoop,
    BasicLoop,
    ExitStatement,
    ReturnStatement,
    NullStatement,
    BlockStatement,
    RaiseStatement,
    CallStatement,
    # Exception handling
    ExceptionHandler,
    TryStatement,
    # Declarations
    Declaration,
    VariableDecl,
    ConstantDecl,
    ComponentDecl,
    Discriminant,
    # Parameters and Subprograms
    Parameter,
    Subprogram,
    # Imports and Packages
    Import,
    Package,
    ExceptionDecl,
)

# Memory management
from ir.systems.memory import (
    AccessType,
    AccessTypeDecl,
    Allocator,
    Deallocate,
    AddressOf,
    Dereference,
    StoragePoolDecl,
    UsePool,
    UncheckedConversion,
    UncheckedDeallocation,
    ScopedRef,
    SliceExpr,
)

# Type system
from ir.systems.types import (
    TypeDecl,
    RangeConstraint,
    DigitsConstraint,
    DeltaConstraint,
    IndexConstraint,
    DiscriminantConstraint,
    SubtypeDecl,
    DerivedTypeDecl,
    RecordExtension,
    RecordType,
    VariantPart,
    Variant,
    NullRecord,
    ArrayType,
    EnumType,
    EnumLiteral,
    IntegerType,
    ModularType,
    FloatingPointType,
    FixedPointType,
    DecimalType,
    InterfaceType,
    TypeAlias,
    ClassType,
    IncompleteType,
    PrivateTypeDecl,
)

# Concurrency
from ir.systems.concurrency import (
    Entry,
    TaskType,
    SingleTask,
    ProtectedType,
    ProtectedBody,
    EntryBody,
    SingleProtected,
    AcceptStatement,
    EntryCallStatement,
    RequeueStatement,
    SelectAlternative,
    AcceptAlternative,
    DelayAlternative,
    TerminateAlternative,
    SelectStatement,
    ConditionalEntryCall,
    TimedEntryCall,
    AsynchronousSelect,
    AbortStatement,
    DelayStatement,
    SharedVariable,
    SynchronizedBlock,
    AtomicOp,
)

# Verification (SPARK)
from ir.systems.verification import (
    Contract,
    ContractCase,
    TypeInvariant,
    SubtypePredicate,
    GlobalSpec,
    DependsSpec,
    InitializesSpec,
    AbstractState,
    RefinedState,
    LoopInvariant,
    LoopVariant,
    VariantExpr,
    GhostCode,
    GhostVariable,
    GhostFunction,
    AssertPragma,
    AssumePragma,
    CheckPragma,
    SparkMode,
    QuantifiedExpr,
    OldExpr,
    ResultExpr,
    LoopEntryExpr,
    DContractIn,
    DContractOut,
    DInvariant,
)

__all__ = [
    # Enums
    'Visibility', 'Mode', 'SafetyLevel',
    # Base
    'SystemsNode', 'TypeRef',
    # Expressions  
    'Expr', 'Literal', 'VarExpr', 'BinaryOp', 'UnaryOp', 'CallExpr',
    'MemberAccess', 'IndexExpr', 'IfExpr', 'CastExpr', 'RangeExpr',
    'AggregateExpr', 'AttributeExpr', 'QualifiedExpr',
    # Statements
    'Statement', 'Assignment', 'IfStatement', 'ElsifPart', 'CaseStatement',
    'CaseAlternative', 'WhileLoop', 'ForLoop', 'BasicLoop', 'ExitStatement',
    'ReturnStatement', 'NullStatement', 'BlockStatement', 'RaiseStatement',
    'CallStatement',
    # Exception handling
    'ExceptionHandler', 'TryStatement',
    # Declarations
    'Declaration', 'VariableDecl', 'ConstantDecl', 'ComponentDecl', 'Discriminant',
    # Parameters and Subprograms
    'Parameter', 'Subprogram',
    # Imports and Packages
    'Import', 'Package', 'ExceptionDecl',
    # Memory
    'AccessType', 'AccessTypeDecl', 'Allocator', 'Deallocate', 'AddressOf',
    'Dereference', 'StoragePoolDecl', 'UsePool', 'UncheckedConversion',
    'UncheckedDeallocation', 'ScopedRef', 'SliceExpr',
    # Types
    'TypeDecl', 'RangeConstraint', 'DigitsConstraint', 'DeltaConstraint',
    'IndexConstraint', 'DiscriminantConstraint', 'SubtypeDecl', 'DerivedTypeDecl',
    'RecordExtension', 'RecordType', 'VariantPart', 'Variant', 'NullRecord',
    'ArrayType', 'EnumType', 'EnumLiteral', 'IntegerType', 'ModularType',
    'FloatingPointType', 'FixedPointType', 'DecimalType', 'InterfaceType',
    'TypeAlias', 'ClassType', 'IncompleteType', 'PrivateTypeDecl',
    # Concurrency
    'Entry', 'TaskType', 'SingleTask', 'ProtectedType', 'ProtectedBody',
    'EntryBody', 'SingleProtected', 'AcceptStatement', 'EntryCallStatement',
    'RequeueStatement', 'SelectAlternative', 'AcceptAlternative',
    'DelayAlternative', 'TerminateAlternative', 'SelectStatement',
    'ConditionalEntryCall', 'TimedEntryCall', 'AsynchronousSelect',
    'AbortStatement', 'DelayStatement', 'SharedVariable', 'SynchronizedBlock',
    'AtomicOp',
    # Verification
    'Contract', 'ContractCase', 'TypeInvariant', 'SubtypePredicate',
    'GlobalSpec', 'DependsSpec', 'InitializesSpec', 'AbstractState',
    'RefinedState', 'LoopInvariant', 'LoopVariant', 'VariantExpr',
    'GhostCode', 'GhostVariable', 'GhostFunction', 'AssertPragma',
    'AssumePragma', 'CheckPragma', 'SparkMode', 'QuantifiedExpr',
    'OldExpr', 'ResultExpr', 'LoopEntryExpr', 'DContractIn', 'DContractOut',
    'DInvariant',
]
