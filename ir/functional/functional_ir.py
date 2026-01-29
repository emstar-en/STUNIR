#!/usr/bin/env python3
"""STUNIR Functional IR - Core functional programming constructs.

This module defines the core IR nodes for functional programming,
including expressions, types, and higher-order constructs.

Usage:
    from ir.functional.functional_ir import Expr, LiteralExpr, LambdaExpr
    
    # Create a lambda expression
    lam = LambdaExpr(param='x', body=VarExpr(name='x'))
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from abc import ABC


@dataclass
class FunctionalNode(ABC):
    """Base class for all functional IR nodes."""
    kind: str = 'node'
    
    def to_dict(self) -> dict:
        """Convert node to dictionary representation."""
        result = {'kind': self.kind}
        for key, value in self.__dict__.items():
            if key != 'kind' and value is not None:
                if isinstance(value, FunctionalNode):
                    result[key] = value.to_dict()
                elif isinstance(value, list):
                    result[key] = [
                        v.to_dict() if isinstance(v, FunctionalNode) else v
                        for v in value
                    ]
                else:
                    result[key] = value
        return result


# =============================================================================
# Type Expressions
# =============================================================================

@dataclass
class TypeExpr(FunctionalNode):
    """Base class for type expressions."""
    kind: str = 'type_expr'


@dataclass
class TypeVar(TypeExpr):
    """Type variable (e.g., 'a', 'b').
    
    Used for polymorphic types in both Haskell and OCaml.
    """
    name: str = ''
    kind: str = 'type_var'
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, TypeVar):
            return self.name == other.name
        return False


@dataclass
class TypeCon(TypeExpr):
    """Type constructor (e.g., Int, Bool, List a).
    
    Represents named types with optional type arguments.
    """
    name: str = ''
    args: List['TypeExpr'] = field(default_factory=list)
    kind: str = 'type_con'


@dataclass
class FunctionType(TypeExpr):
    """Function type (a -> b).
    
    Represents the type of functions.
    """
    param_type: 'TypeExpr' = None
    return_type: 'TypeExpr' = None
    kind: str = 'function_type'


@dataclass
class TupleType(TypeExpr):
    """Tuple type (a, b, c)."""
    elements: List['TypeExpr'] = field(default_factory=list)
    kind: str = 'tuple_type'


@dataclass
class ListType(TypeExpr):
    """List type [a]."""
    element_type: 'TypeExpr' = None
    kind: str = 'list_type'


# =============================================================================
# Expressions
# =============================================================================

@dataclass
class Expr(FunctionalNode):
    """Base class for expressions."""
    type_annotation: Optional[TypeExpr] = None
    kind: str = 'expr'


@dataclass
class LiteralExpr(Expr):
    """Literal value expression.
    
    Supported literal types: int, float, bool, string, char
    """
    value: Any = None
    literal_type: str = 'int'  # 'int', 'float', 'bool', 'string', 'char'
    kind: str = 'literal'


@dataclass
class VarExpr(Expr):
    """Variable reference expression."""
    name: str = ''
    kind: str = 'var'


@dataclass
class AppExpr(Expr):
    """Function application expression.
    
    Represents applying a function to an argument.
    """
    func: 'Expr' = None
    arg: 'Expr' = None
    kind: str = 'app'


@dataclass
class LambdaExpr(Expr):
    """Lambda abstraction expression.
    
    Anonymous function definition.
    """
    param: str = ''
    param_type: Optional[TypeExpr] = None
    body: 'Expr' = None
    kind: str = 'lambda'


@dataclass
class LetExpr(Expr):
    """Let binding expression.
    
    Binds a name to a value within a body expression.
    """
    name: str = ''
    value: 'Expr' = None
    body: 'Expr' = None
    is_recursive: bool = False
    kind: str = 'let'


@dataclass
class IfExpr(Expr):
    """Conditional expression."""
    condition: 'Expr' = None
    then_branch: 'Expr' = None
    else_branch: 'Expr' = None
    kind: str = 'if'


@dataclass
class CaseBranch:
    """A single branch in a case expression.
    
    Attributes:
        pattern: Pattern to match against
        guard: Optional guard condition
        body: Expression to evaluate if pattern matches
    """
    pattern: 'Pattern' = None
    guard: Optional['Expr'] = None
    body: 'Expr' = None
    
    def to_dict(self) -> dict:
        result = {}
        if self.pattern:
            result['pattern'] = self.pattern.to_dict()
        if self.guard:
            result['guard'] = self.guard.to_dict()
        if self.body:
            result['body'] = self.body.to_dict()
        return result


@dataclass
class CaseExpr(Expr):
    """Pattern matching expression.
    
    Case analysis on a scrutinee value.
    """
    scrutinee: 'Expr' = None
    branches: List[CaseBranch] = field(default_factory=list)
    kind: str = 'case'


@dataclass
class ListGenerator:
    """Generator for list comprehensions.
    
    Attributes:
        pattern: Pattern to bind
        source: Source list expression
        conditions: Filter conditions
    """
    pattern: 'Pattern' = None
    source: 'Expr' = None
    conditions: List['Expr'] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        result = {}
        if self.pattern:
            result['pattern'] = self.pattern.to_dict()
        if self.source:
            result['source'] = self.source.to_dict()
        if self.conditions:
            result['conditions'] = [c.to_dict() for c in self.conditions]
        return result


@dataclass
class ListExpr(Expr):
    """List literal or comprehension.
    
    Can represent either a literal list [1, 2, 3] or a
    list comprehension [x*x | x <- xs, x > 0].
    """
    elements: List['Expr'] = field(default_factory=list)
    is_comprehension: bool = False
    generators: List[ListGenerator] = field(default_factory=list)
    kind: str = 'list'


@dataclass
class TupleExpr(Expr):
    """Tuple expression."""
    elements: List['Expr'] = field(default_factory=list)
    kind: str = 'tuple'


@dataclass
class BinaryOpExpr(Expr):
    """Binary operation expression.
    
    Supported operators: +, -, *, /, %, ==, /=, <, >, <=, >=, &&, ||, ++, :
    """
    op: str = ''
    left: 'Expr' = None
    right: 'Expr' = None
    kind: str = 'binary_op'


@dataclass
class UnaryOpExpr(Expr):
    """Unary operation expression.
    
    Supported operators: -, not
    """
    op: str = ''
    operand: 'Expr' = None
    kind: str = 'unary_op'


# =============================================================================
# Do Notation (Monadic, Haskell-specific)
# =============================================================================

@dataclass
class DoStatement(FunctionalNode):
    """Base class for do notation statements."""
    kind: str = 'do_statement'


@dataclass
class DoBindStatement(DoStatement):
    """Bind statement (x <- action)."""
    pattern: 'Pattern' = None
    action: 'Expr' = None
    kind: str = 'do_bind'


@dataclass
class DoLetStatement(DoStatement):
    """Let statement in do notation."""
    name: str = ''
    value: 'Expr' = None
    kind: str = 'do_let'


@dataclass
class DoExprStatement(DoStatement):
    """Expression statement in do notation."""
    expr: 'Expr' = None
    kind: str = 'do_expr'


@dataclass
class DoExpr(Expr):
    """Do notation expression (monadic)."""
    statements: List[DoStatement] = field(default_factory=list)
    kind: str = 'do'


# =============================================================================
# Pattern Import (from pattern.py)
# =============================================================================

# Import Pattern class for type hints - actual implementation in pattern.py
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ir.functional.pattern import Pattern


# =============================================================================
# Utility Functions
# =============================================================================

def make_app(func: Expr, *args: Expr) -> Expr:
    """Create a curried function application.
    
    Args:
        func: Function to apply
        *args: Arguments to apply
        
    Returns:
        Nested AppExpr for curried application
    """
    result = func
    for arg in args:
        result = AppExpr(func=result, arg=arg)
    return result


def make_lambda(params: List[str], body: Expr) -> Expr:
    """Create a curried lambda expression.
    
    Args:
        params: Parameter names
        body: Lambda body
        
    Returns:
        Nested LambdaExpr for curried function
    """
    result = body
    for param in reversed(params):
        result = LambdaExpr(param=param, body=result)
    return result


def make_let_chain(bindings: List[tuple], body: Expr) -> Expr:
    """Create a chain of let bindings.
    
    Args:
        bindings: List of (name, value) tuples
        body: Final body expression
        
    Returns:
        Nested LetExpr chain
    """
    result = body
    for name, value in reversed(bindings):
        result = LetExpr(name=name, value=value, body=result)
    return result
