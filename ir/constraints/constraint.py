"""Constraint definitions for Constraint Programming.

This module defines constraint expressions and constraint types
for constraint satisfaction problems.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from abc import ABC

from .constraint_ir import ConstraintType


# Expression classes
@dataclass
class Expression(ABC):
    """Base class for constraint expressions."""
    kind: str = field(init=False)


@dataclass
class VariableRef(Expression):
    """Reference to a decision variable.
    
    Attributes:
        name: Variable name
    """
    name: str
    
    def __post_init__(self):
        self.kind = 'var'
    
    def __str__(self) -> str:
        return self.name


@dataclass
class Literal(Expression):
    """Literal value in expression.
    
    Attributes:
        value: The literal value
    """
    value: Union[int, float, bool, str]
    
    def __post_init__(self):
        self.kind = 'literal'
    
    def __str__(self) -> str:
        if isinstance(self.value, bool):
            return 'true' if self.value else 'false'
        return str(self.value)


@dataclass
class ArrayAccess(Expression):
    """Array element access.
    
    Attributes:
        array: Array name
        indices: List of index expressions
    """
    array: str
    indices: List[Expression]
    
    def __post_init__(self):
        self.kind = 'array_access'
    
    def __str__(self) -> str:
        idx_str = ", ".join(str(i) for i in self.indices)
        return f"{self.array}[{idx_str}]"


@dataclass
class BinaryOp(Expression):
    """Binary operation.
    
    Attributes:
        op: Operator ('+', '-', '*', '/', 'div', 'mod', 'min', 'max')
        left: Left operand
        right: Right operand
    """
    op: str
    left: Expression
    right: Expression
    
    def __post_init__(self):
        self.kind = 'binary_op'
    
    def __str__(self) -> str:
        if self.op in ('min', 'max', 'div', 'mod'):
            return f"{self.op}({self.left}, {self.right})"
        return f"({self.left} {self.op} {self.right})"


@dataclass
class UnaryOp(Expression):
    """Unary operation.
    
    Attributes:
        op: Operator ('-', 'abs', 'not')
        operand: Operand
    """
    op: str
    operand: Expression
    
    def __post_init__(self):
        self.kind = 'unary_op'
    
    def __str__(self) -> str:
        if self.op == '-':
            return f"-{self.operand}"
        return f"{self.op}({self.operand})"


@dataclass
class FunctionCall(Expression):
    """Function call in expression.
    
    Attributes:
        name: Function name
        args: Function arguments
    """
    name: str
    args: List[Expression]
    
    def __post_init__(self):
        self.kind = 'call'
    
    def __str__(self) -> str:
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.name}({args_str})"


@dataclass
class SetLiteral(Expression):
    """Set literal expression.
    
    Attributes:
        elements: Elements of the set
    """
    elements: List[Expression]
    
    def __post_init__(self):
        self.kind = 'set_literal'
    
    def __str__(self) -> str:
        elems_str = ", ".join(str(e) for e in self.elements)
        return "{" + elems_str + "}"


@dataclass
class Comprehension(Expression):
    """List/Set comprehension.
    
    Attributes:
        expr: Expression to evaluate for each iteration
        generators: List of (var, range) generators
        condition: Optional filter condition
    """
    expr: Expression
    generators: List[tuple]  # [(var_name, lower, upper), ...]
    condition: Optional[Expression] = None
    
    def __post_init__(self):
        self.kind = 'comprehension'
    
    def __str__(self) -> str:
        gen_strs = [f"{v} in {lb}..{ub}" for v, lb, ub in self.generators]
        gen_str = ", ".join(gen_strs)
        if self.condition:
            return f"[{self.expr} | {gen_str} where {self.condition}]"
        return f"[{self.expr} | {gen_str}]"


# Constraint classes
@dataclass
class Constraint(ABC):
    """Base class for constraints.
    
    Attributes:
        constraint_type: Type of constraint
        name: Optional constraint name
        reified: If True, this constraint can be treated as a boolean
    """
    constraint_type: ConstraintType
    name: Optional[str] = None
    reified: bool = False


@dataclass
class RelationalConstraint(Constraint):
    """Relational constraint (=, !=, <, <=, >, >=).
    
    Attributes:
        left: Left-hand side expression
        right: Right-hand side expression
    """
    left: Expression = field(default=None)
    right: Expression = field(default=None)
    
    def __str__(self) -> str:
        op_map = {
            ConstraintType.EQ: '=',
            ConstraintType.NE: '!=',
            ConstraintType.LT: '<',
            ConstraintType.LE: '<=',
            ConstraintType.GT: '>',
            ConstraintType.GE: '>=',
        }
        op = op_map.get(self.constraint_type, '?')
        return f"{self.left} {op} {self.right}"


@dataclass
class LogicalConstraint(Constraint):
    """Logical constraint (and, or, not, implies, iff).
    
    Attributes:
        operands: List of sub-constraints
    """
    operands: List[Constraint] = field(default_factory=list)
    
    def __str__(self) -> str:
        op_map = {
            ConstraintType.AND: ' /\\ ',
            ConstraintType.OR: ' \\/ ',
            ConstraintType.IMPLIES: ' -> ',
            ConstraintType.IFF: ' <-> ',
        }
        if self.constraint_type == ConstraintType.NOT:
            return f"not({self.operands[0]})"
        op = op_map.get(self.constraint_type, ' ? ')
        operands_str = op.join(f"({o})" for o in self.operands)
        return operands_str


@dataclass
class GlobalConstraint(Constraint):
    """Global constraint (alldifferent, cumulative, etc.).
    
    Attributes:
        args: Arguments to the global constraint
        params: Additional parameters (e.g., for cumulative)
    """
    args: List[Expression] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        name_map = {
            ConstraintType.ALLDIFFERENT: 'alldifferent',
            ConstraintType.CUMULATIVE: 'cumulative',
            ConstraintType.ELEMENT: 'element',
            ConstraintType.TABLE: 'table',
            ConstraintType.REGULAR: 'regular',
            ConstraintType.CIRCUIT: 'circuit',
            ConstraintType.COUNT: 'count',
            ConstraintType.BIN_PACKING: 'bin_packing',
            ConstraintType.GLOBAL_CARDINALITY: 'global_cardinality',
            ConstraintType.SUM: 'sum',
        }
        name = name_map.get(self.constraint_type, 'global')
        args_str = ", ".join(str(a) for a in self.args)
        if self.params:
            params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
            return f"{name}({args_str}, {params_str})"
        return f"{name}({args_str})"


# Helper functions for creating constraints
def eq(left: Expression, right: Expression, name: str = None) -> RelationalConstraint:
    """Create equality constraint."""
    return RelationalConstraint(ConstraintType.EQ, name=name, left=left, right=right)


def ne(left: Expression, right: Expression, name: str = None) -> RelationalConstraint:
    """Create not-equal constraint."""
    return RelationalConstraint(ConstraintType.NE, name=name, left=left, right=right)


def lt(left: Expression, right: Expression, name: str = None) -> RelationalConstraint:
    """Create less-than constraint."""
    return RelationalConstraint(ConstraintType.LT, name=name, left=left, right=right)


def le(left: Expression, right: Expression, name: str = None) -> RelationalConstraint:
    """Create less-than-or-equal constraint."""
    return RelationalConstraint(ConstraintType.LE, name=name, left=left, right=right)


def gt(left: Expression, right: Expression, name: str = None) -> RelationalConstraint:
    """Create greater-than constraint."""
    return RelationalConstraint(ConstraintType.GT, name=name, left=left, right=right)


def ge(left: Expression, right: Expression, name: str = None) -> RelationalConstraint:
    """Create greater-than-or-equal constraint."""
    return RelationalConstraint(ConstraintType.GE, name=name, left=left, right=right)


def alldifferent(args: List[Expression], name: str = None) -> GlobalConstraint:
    """Create alldifferent constraint."""
    return GlobalConstraint(ConstraintType.ALLDIFFERENT, name=name, args=args)


def conjunction(constraints: List[Constraint], name: str = None) -> LogicalConstraint:
    """Create conjunction (AND) of constraints."""
    return LogicalConstraint(ConstraintType.AND, name=name, operands=constraints)


def disjunction(constraints: List[Constraint], name: str = None) -> LogicalConstraint:
    """Create disjunction (OR) of constraints."""
    return LogicalConstraint(ConstraintType.OR, name=name, operands=constraints)


def negation(constraint: Constraint, name: str = None) -> LogicalConstraint:
    """Create negation of constraint."""
    return LogicalConstraint(ConstraintType.NOT, name=name, operands=[constraint])


def implies(left: Constraint, right: Constraint, name: str = None) -> LogicalConstraint:
    """Create implication constraint."""
    return LogicalConstraint(ConstraintType.IMPLIES, name=name, operands=[left, right])
