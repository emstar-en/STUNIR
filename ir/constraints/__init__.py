"""Constraint Programming IR for STUNIR.

This package provides intermediate representation classes for
constraint satisfaction and optimization problems.

Example:
    from ir.constraints import (
        ConstraintModel, Variable, Domain, 
        Objective, VariableType, ConstraintType
    )
    
    # Create N-Queens model
    model = ConstraintModel("nqueens")
    n = 8
    for i in range(1, n + 1):
        model.add_int_variable(f"q{i}", 1, n)
    # Add alldifferent constraint...
    model.set_objective(Objective.satisfy())
"""

# Core types and enums
from .constraint_ir import (
    VariableType,
    DomainType,
    ConstraintType,
    ObjectiveType,
    SearchStrategy,
    ValueChoice,
    ConstraintEmitterResult,
    ConstraintModelError,
    InvalidVariableError,
    InvalidDomainError,
    InvalidConstraintError,
    UnsupportedConstraintError,
)

# Variable definitions
from .variable import (
    Variable,
    ArrayVariable,
    IndexSet,
    Parameter,
)

# Domain definitions
from .domain import Domain

# Constraint definitions
from .constraint import (
    Expression,
    VariableRef,
    Literal,
    ArrayAccess,
    BinaryOp,
    UnaryOp,
    FunctionCall,
    SetLiteral,
    Comprehension,
    Constraint,
    RelationalConstraint,
    LogicalConstraint,
    GlobalConstraint,
    # Helper functions
    eq,
    ne,
    lt,
    le,
    gt,
    ge,
    alldifferent,
    conjunction,
    disjunction,
    negation,
    implies,
)

# Objective and model
from .objective import (
    Objective,
    SearchAnnotation,
    ConstraintModel,
)

__all__ = [
    # Enums
    'VariableType',
    'DomainType',
    'ConstraintType',
    'ObjectiveType',
    'SearchStrategy',
    'ValueChoice',
    # Results and errors
    'ConstraintEmitterResult',
    'ConstraintModelError',
    'InvalidVariableError',
    'InvalidDomainError',
    'InvalidConstraintError',
    'UnsupportedConstraintError',
    # Variables
    'Variable',
    'ArrayVariable',
    'IndexSet',
    'Parameter',
    # Domains
    'Domain',
    # Expressions
    'Expression',
    'VariableRef',
    'Literal',
    'ArrayAccess',
    'BinaryOp',
    'UnaryOp',
    'FunctionCall',
    'SetLiteral',
    'Comprehension',
    # Constraints
    'Constraint',
    'RelationalConstraint',
    'LogicalConstraint',
    'GlobalConstraint',
    # Constraint helpers
    'eq',
    'ne',
    'lt',
    'le',
    'gt',
    'ge',
    'alldifferent',
    'conjunction',
    'disjunction',
    'negation',
    'implies',
    # Objective and model
    'Objective',
    'SearchAnnotation',
    'ConstraintModel',
]
