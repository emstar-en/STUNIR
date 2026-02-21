#!/usr/bin/env python3
"""STUNIR Expression Analysis Module.

Provides expression analysis including operator precedence, expression
simplification, constant folding, and common subexpression elimination.

This module is part of the STUNIR code generation enhancement suite.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from enum import Enum, auto
import operator


class OperatorPrecedence(Enum):
    """Operator precedence levels (higher = binds tighter)."""
    COMMA = 1
    ASSIGN = 2
    TERNARY = 3
    LOGICAL_OR = 4
    LOGICAL_AND = 5
    BITWISE_OR = 6
    BITWISE_XOR = 7
    BITWISE_AND = 8
    EQUALITY = 9
    RELATIONAL = 10
    SHIFT = 11
    ADDITIVE = 12
    MULTIPLICATIVE = 13
    UNARY = 14
    POSTFIX = 15
    PRIMARY = 16


class OperatorAssociativity(Enum):
    """Operator associativity."""
    LEFT = auto()
    RIGHT = auto()
    NONE = auto()


@dataclass
class OperatorInfo:
    """Information about an operator."""
    symbol: str
    precedence: OperatorPrecedence
    associativity: OperatorAssociativity
    arity: int  # 1 for unary, 2 for binary, 3 for ternary
    is_comparison: bool = False
    is_logical: bool = False
    is_bitwise: bool = False
    is_assignment: bool = False
    
    @property
    def precedence_value(self) -> int:
        return self.precedence.value


# Operator registry
OPERATORS: Dict[str, OperatorInfo] = {
    # Assignment operators
    '=': OperatorInfo('=', OperatorPrecedence.ASSIGN, OperatorAssociativity.RIGHT, 2, is_assignment=True),
    '+=': OperatorInfo('+=', OperatorPrecedence.ASSIGN, OperatorAssociativity.RIGHT, 2, is_assignment=True),
    '-=': OperatorInfo('-=', OperatorPrecedence.ASSIGN, OperatorAssociativity.RIGHT, 2, is_assignment=True),
    '*=': OperatorInfo('*=', OperatorPrecedence.ASSIGN, OperatorAssociativity.RIGHT, 2, is_assignment=True),
    '/=': OperatorInfo('/=', OperatorPrecedence.ASSIGN, OperatorAssociativity.RIGHT, 2, is_assignment=True),
    '%=': OperatorInfo('%=', OperatorPrecedence.ASSIGN, OperatorAssociativity.RIGHT, 2, is_assignment=True),
    '&=': OperatorInfo('&=', OperatorPrecedence.ASSIGN, OperatorAssociativity.RIGHT, 2, is_assignment=True),
    '|=': OperatorInfo('|=', OperatorPrecedence.ASSIGN, OperatorAssociativity.RIGHT, 2, is_assignment=True),
    '^=': OperatorInfo('^=', OperatorPrecedence.ASSIGN, OperatorAssociativity.RIGHT, 2, is_assignment=True),
    '<<=': OperatorInfo('<<=', OperatorPrecedence.ASSIGN, OperatorAssociativity.RIGHT, 2, is_assignment=True),
    '>>=': OperatorInfo('>>=', OperatorPrecedence.ASSIGN, OperatorAssociativity.RIGHT, 2, is_assignment=True),
    
    # Logical operators
    '||': OperatorInfo('||', OperatorPrecedence.LOGICAL_OR, OperatorAssociativity.LEFT, 2, is_logical=True),
    'or': OperatorInfo('or', OperatorPrecedence.LOGICAL_OR, OperatorAssociativity.LEFT, 2, is_logical=True),
    '&&': OperatorInfo('&&', OperatorPrecedence.LOGICAL_AND, OperatorAssociativity.LEFT, 2, is_logical=True),
    'and': OperatorInfo('and', OperatorPrecedence.LOGICAL_AND, OperatorAssociativity.LEFT, 2, is_logical=True),
    
    # Bitwise operators
    '|': OperatorInfo('|', OperatorPrecedence.BITWISE_OR, OperatorAssociativity.LEFT, 2, is_bitwise=True),
    '^': OperatorInfo('^', OperatorPrecedence.BITWISE_XOR, OperatorAssociativity.LEFT, 2, is_bitwise=True),
    '&': OperatorInfo('&', OperatorPrecedence.BITWISE_AND, OperatorAssociativity.LEFT, 2, is_bitwise=True),
    
    # Equality operators
    '==': OperatorInfo('==', OperatorPrecedence.EQUALITY, OperatorAssociativity.LEFT, 2, is_comparison=True),
    '!=': OperatorInfo('!=', OperatorPrecedence.EQUALITY, OperatorAssociativity.LEFT, 2, is_comparison=True),
    
    # Relational operators
    '<': OperatorInfo('<', OperatorPrecedence.RELATIONAL, OperatorAssociativity.LEFT, 2, is_comparison=True),
    '>': OperatorInfo('>', OperatorPrecedence.RELATIONAL, OperatorAssociativity.LEFT, 2, is_comparison=True),
    '<=': OperatorInfo('<=', OperatorPrecedence.RELATIONAL, OperatorAssociativity.LEFT, 2, is_comparison=True),
    '>=': OperatorInfo('>=', OperatorPrecedence.RELATIONAL, OperatorAssociativity.LEFT, 2, is_comparison=True),
    
    # Shift operators
    '<<': OperatorInfo('<<', OperatorPrecedence.SHIFT, OperatorAssociativity.LEFT, 2, is_bitwise=True),
    '>>': OperatorInfo('>>', OperatorPrecedence.SHIFT, OperatorAssociativity.LEFT, 2, is_bitwise=True),
    
    # Additive operators
    '+': OperatorInfo('+', OperatorPrecedence.ADDITIVE, OperatorAssociativity.LEFT, 2),
    '-': OperatorInfo('-', OperatorPrecedence.ADDITIVE, OperatorAssociativity.LEFT, 2),
    
    # Multiplicative operators
    '*': OperatorInfo('*', OperatorPrecedence.MULTIPLICATIVE, OperatorAssociativity.LEFT, 2),
    '/': OperatorInfo('/', OperatorPrecedence.MULTIPLICATIVE, OperatorAssociativity.LEFT, 2),
    '%': OperatorInfo('%', OperatorPrecedence.MULTIPLICATIVE, OperatorAssociativity.LEFT, 2),
    
    # Unary operators (using prefix notation)
    'u-': OperatorInfo('-', OperatorPrecedence.UNARY, OperatorAssociativity.RIGHT, 1),
    'u+': OperatorInfo('+', OperatorPrecedence.UNARY, OperatorAssociativity.RIGHT, 1),
    '!': OperatorInfo('!', OperatorPrecedence.UNARY, OperatorAssociativity.RIGHT, 1, is_logical=True),
    'not': OperatorInfo('not', OperatorPrecedence.UNARY, OperatorAssociativity.RIGHT, 1, is_logical=True),
    '~': OperatorInfo('~', OperatorPrecedence.UNARY, OperatorAssociativity.RIGHT, 1, is_bitwise=True),
    'u*': OperatorInfo('*', OperatorPrecedence.UNARY, OperatorAssociativity.RIGHT, 1),  # Dereference
    'u&': OperatorInfo('&', OperatorPrecedence.UNARY, OperatorAssociativity.RIGHT, 1),  # Address-of
    '++': OperatorInfo('++', OperatorPrecedence.UNARY, OperatorAssociativity.RIGHT, 1),
    '--': OperatorInfo('--', OperatorPrecedence.UNARY, OperatorAssociativity.RIGHT, 1),
}


@dataclass
class Expression:
    """Base class for expressions."""
    pass


@dataclass
class LiteralExpr(Expression):
    """Literal value expression."""
    value: Any
    type: str = 'auto'
    
    def is_constant(self) -> bool:
        return True
    
    def evaluate(self) -> Any:
        return self.value


@dataclass
class VariableExpr(Expression):
    """Variable reference expression."""
    name: str
    
    def is_constant(self) -> bool:
        return False


@dataclass
class BinaryExpr(Expression):
    """Binary operator expression."""
    op: str
    left: Expression
    right: Expression
    
    def is_constant(self) -> bool:
        return self.left.is_constant() and self.right.is_constant()


@dataclass
class UnaryExpr(Expression):
    """Unary operator expression."""
    op: str
    operand: Expression
    prefix: bool = True
    
    def is_constant(self) -> bool:
        return self.operand.is_constant()


@dataclass
class CallExpr(Expression):
    """Function call expression."""
    func: str
    args: List[Expression] = field(default_factory=list)
    
    def is_constant(self) -> bool:
        return False


@dataclass
class IndexExpr(Expression):
    """Array/pointer index expression."""
    base: Expression
    index: Expression
    
    def is_constant(self) -> bool:
        return False


@dataclass
class MemberExpr(Expression):
    """Member access expression."""
    base: Expression
    member: str
    is_pointer: bool = False  # -> vs .
    
    def is_constant(self) -> bool:
        return False


@dataclass
class TernaryExpr(Expression):
    """Ternary conditional expression."""
    condition: Expression
    then_expr: Expression
    else_expr: Expression
    
    def is_constant(self) -> bool:
        return (self.condition.is_constant() and 
                self.then_expr.is_constant() and 
                self.else_expr.is_constant())


@dataclass
class CastExpr(Expression):
    """Type cast expression."""
    expr: Expression
    target_type: str
    
    def is_constant(self) -> bool:
        return self.expr.is_constant()


class ExpressionParser:
    """Parses expression dictionaries into Expression trees."""
    
    def parse(self, expr: Any) -> Expression:
        """Parse an expression from dict or value."""
        if expr is None:
            return LiteralExpr(value=None, type='void')
        
        if isinstance(expr, dict):
            return self._parse_dict(expr)
        elif isinstance(expr, (int, float)):
            return LiteralExpr(value=expr, type='number')
        elif isinstance(expr, bool):
            return LiteralExpr(value=expr, type='bool')
        elif isinstance(expr, str):
            # Could be variable name or string literal
            if expr.startswith('"') or expr.startswith("'"):
                return LiteralExpr(value=expr[1:-1], type='string')
            return VariableExpr(name=expr)
        
        return LiteralExpr(value=expr)
    
    def _parse_dict(self, expr: Dict) -> Expression:
        """Parse a dictionary expression."""
        expr_type = expr.get('type', 'literal')
        
        if expr_type == 'literal':
            return LiteralExpr(
                value=expr.get('value'),
                type=expr.get('lit_type', 'auto')
            )
        elif expr_type == 'var':
            return VariableExpr(name=expr.get('name', ''))
        elif expr_type == 'binary':
            return BinaryExpr(
                op=expr.get('op', '+'),
                left=self.parse(expr.get('left')),
                right=self.parse(expr.get('right'))
            )
        elif expr_type == 'unary':
            return UnaryExpr(
                op=expr.get('op', '-'),
                operand=self.parse(expr.get('operand')),
                prefix=expr.get('prefix', True)
            )
        elif expr_type == 'call':
            return CallExpr(
                func=expr.get('func', ''),
                args=[self.parse(a) for a in expr.get('args', [])]
            )
        elif expr_type == 'index':
            return IndexExpr(
                base=self.parse(expr.get('base')),
                index=self.parse(expr.get('index'))
            )
        elif expr_type == 'member':
            return MemberExpr(
                base=self.parse(expr.get('base')),
                member=expr.get('member', ''),
                is_pointer=expr.get('is_pointer', False)
            )
        elif expr_type == 'ternary':
            return TernaryExpr(
                condition=self.parse(expr.get('cond')),
                then_expr=self.parse(expr.get('then')),
                else_expr=self.parse(expr.get('else'))
            )
        elif expr_type == 'cast':
            return CastExpr(
                expr=self.parse(expr.get('value')),
                target_type=expr.get('target_type', 'i32')
            )
        
        # Fallback: try 'value' field
        if 'value' in expr:
            return self.parse(expr['value'])
        
        return LiteralExpr(value=None)


class ConstantFolder:
    """Performs constant folding on expressions."""
    
    # Operators that can be evaluated at compile time
    BINARY_OPS: Dict[str, callable] = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': lambda a, b: a // b if isinstance(a, int) and isinstance(b, int) else a / b,
        '%': operator.mod,
        '**': operator.pow,
        '&': operator.and_,
        '|': operator.or_,
        '^': operator.xor,
        '<<': operator.lshift,
        '>>': operator.rshift,
        '==': operator.eq,
        '!=': operator.ne,
        '<': operator.lt,
        '>': operator.gt,
        '<=': operator.le,
        '>=': operator.ge,
        '&&': lambda a, b: a and b,
        'and': lambda a, b: a and b,
        '||': lambda a, b: a or b,
        'or': lambda a, b: a or b,
    }
    
    UNARY_OPS: Dict[str, callable] = {
        '-': operator.neg,
        '+': operator.pos,
        '!': operator.not_,
        'not': operator.not_,
        '~': operator.invert,
    }
    
    def fold(self, expr: Expression) -> Expression:
        """Fold constants in an expression."""
        if isinstance(expr, LiteralExpr):
            return expr
        
        if isinstance(expr, BinaryExpr):
            return self._fold_binary(expr)
        
        if isinstance(expr, UnaryExpr):
            return self._fold_unary(expr)
        
        if isinstance(expr, TernaryExpr):
            return self._fold_ternary(expr)
        
        if isinstance(expr, CastExpr):
            return self._fold_cast(expr)
        
        return expr
    
    def _fold_binary(self, expr: BinaryExpr) -> Expression:
        """Fold a binary expression."""
        left = self.fold(expr.left)
        right = self.fold(expr.right)
        
        # Both operands are constant
        if isinstance(left, LiteralExpr) and isinstance(right, LiteralExpr):
            op_func = self.BINARY_OPS.get(expr.op)
            if op_func:
                try:
                    result = op_func(left.value, right.value)
                    return LiteralExpr(value=result)
                except (ZeroDivisionError, OverflowError, TypeError):
                    pass
        
        # Identity optimizations
        if isinstance(right, LiteralExpr):
            if expr.op == '+' and right.value == 0:
                return left
            if expr.op == '-' and right.value == 0:
                return left
            if expr.op == '*' and right.value == 1:
                return left
            if expr.op == '*' and right.value == 0:
                return LiteralExpr(value=0)
            if expr.op == '/' and right.value == 1:
                return left
            if expr.op == '&&' and right.value == True:
                return left
            if expr.op == '||' and right.value == False:
                return left
            if expr.op == '&&' and right.value == False:
                return LiteralExpr(value=False)
            if expr.op == '||' and right.value == True:
                return LiteralExpr(value=True)
        
        if isinstance(left, LiteralExpr):
            if expr.op == '+' and left.value == 0:
                return right
            if expr.op == '*' and left.value == 1:
                return right
            if expr.op == '*' and left.value == 0:
                return LiteralExpr(value=0)
            if expr.op == '&&' and left.value == True:
                return right
            if expr.op == '||' and left.value == False:
                return right
            if expr.op == '&&' and left.value == False:
                return LiteralExpr(value=False)
            if expr.op == '||' and left.value == True:
                return LiteralExpr(value=True)
        
        # x - x = 0, x ^ x = 0
        if isinstance(left, VariableExpr) and isinstance(right, VariableExpr):
            if left.name == right.name:
                if expr.op == '-':
                    return LiteralExpr(value=0)
                if expr.op == '^':
                    return LiteralExpr(value=0)
                if expr.op == '==':
                    return LiteralExpr(value=True)
                if expr.op == '!=':
                    return LiteralExpr(value=False)
        
        return BinaryExpr(op=expr.op, left=left, right=right)
    
    def _fold_unary(self, expr: UnaryExpr) -> Expression:
        """Fold a unary expression."""
        operand = self.fold(expr.operand)
        
        if isinstance(operand, LiteralExpr):
            op_func = self.UNARY_OPS.get(expr.op)
            if op_func:
                try:
                    result = op_func(operand.value)
                    return LiteralExpr(value=result)
                except (TypeError, ValueError):
                    pass
        
        # Double negation
        if expr.op == '-' and isinstance(operand, UnaryExpr) and operand.op == '-':
            return operand.operand
        
        # Double not
        if expr.op in ('!', 'not') and isinstance(operand, UnaryExpr) and operand.op in ('!', 'not'):
            return operand.operand
        
        return UnaryExpr(op=expr.op, operand=operand, prefix=expr.prefix)
    
    def _fold_ternary(self, expr: TernaryExpr) -> Expression:
        """Fold a ternary expression."""
        condition = self.fold(expr.condition)
        then_expr = self.fold(expr.then_expr)
        else_expr = self.fold(expr.else_expr)
        
        # Constant condition
        if isinstance(condition, LiteralExpr):
            if condition.value:
                return then_expr
            else:
                return else_expr
        
        # Same then/else
        if self._exprs_equal(then_expr, else_expr):
            return then_expr
        
        return TernaryExpr(condition=condition, then_expr=then_expr, else_expr=else_expr)
    
    def _fold_cast(self, expr: CastExpr) -> Expression:
        """Fold a cast expression."""
        inner = self.fold(expr.expr)
        
        if isinstance(inner, LiteralExpr):
            # Perform cast at compile time
            if expr.target_type in ('i32', 'int'):
                return LiteralExpr(value=int(inner.value), type='i32')
            elif expr.target_type in ('i64', 'long'):
                return LiteralExpr(value=int(inner.value), type='i64')
            elif expr.target_type in ('f32', 'float'):
                return LiteralExpr(value=float(inner.value), type='f32')
            elif expr.target_type in ('f64', 'double'):
                return LiteralExpr(value=float(inner.value), type='f64')
            elif expr.target_type == 'bool':
                return LiteralExpr(value=bool(inner.value), type='bool')
        
        return CastExpr(expr=inner, target_type=expr.target_type)
    
    def _exprs_equal(self, a: Expression, b: Expression) -> bool:
        """Check if two expressions are structurally equal."""
        if type(a) != type(b):
            return False
        
        if isinstance(a, LiteralExpr):
            return a.value == b.value
        if isinstance(a, VariableExpr):
            return a.name == b.name
        if isinstance(a, BinaryExpr):
            return (a.op == b.op and 
                    self._exprs_equal(a.left, b.left) and 
                    self._exprs_equal(a.right, b.right))
        if isinstance(a, UnaryExpr):
            return a.op == b.op and self._exprs_equal(a.operand, b.operand)
        
        return False


class CommonSubexpressionEliminator:
    """Performs common subexpression elimination."""
    
    def __init__(self):
        self.subexpressions: Dict[str, Tuple[Expression, str]] = {}  # hash -> (expr, temp_name)
        self._temp_counter = 0
    
    def eliminate(self, exprs: List[Expression]) -> Tuple[List[Expression], Dict[str, Expression]]:
        """Eliminate common subexpressions from a list of expressions.
        
        Returns transformed expressions and a dict of extracted subexpressions.
        """
        self.subexpressions.clear()
        self._temp_counter = 0
        
        # First pass: count subexpressions
        counts: Dict[str, int] = {}
        for expr in exprs:
            self._count_subexprs(expr, counts)
        
        # Second pass: extract common subexpressions
        extracted: Dict[str, Expression] = {}
        result_exprs = []
        
        for expr in exprs:
            transformed = self._transform(expr, counts, extracted)
            result_exprs.append(transformed)
        
        return result_exprs, extracted
    
    def _count_subexprs(self, expr: Expression, counts: Dict[str, int]) -> None:
        """Count occurrences of subexpressions."""
        if isinstance(expr, (LiteralExpr, VariableExpr)):
            return
        
        expr_hash = self._hash_expr(expr)
        counts[expr_hash] = counts.get(expr_hash, 0) + 1
        
        if isinstance(expr, BinaryExpr):
            self._count_subexprs(expr.left, counts)
            self._count_subexprs(expr.right, counts)
        elif isinstance(expr, UnaryExpr):
            self._count_subexprs(expr.operand, counts)
        elif isinstance(expr, CallExpr):
            for arg in expr.args:
                self._count_subexprs(arg, counts)
        elif isinstance(expr, IndexExpr):
            self._count_subexprs(expr.base, counts)
            self._count_subexprs(expr.index, counts)
        elif isinstance(expr, MemberExpr):
            self._count_subexprs(expr.base, counts)
        elif isinstance(expr, TernaryExpr):
            self._count_subexprs(expr.condition, counts)
            self._count_subexprs(expr.then_expr, counts)
            self._count_subexprs(expr.else_expr, counts)
        elif isinstance(expr, CastExpr):
            self._count_subexprs(expr.expr, counts)
    
    def _transform(self, expr: Expression, counts: Dict[str, int],
                  extracted: Dict[str, Expression]) -> Expression:
        """Transform expression, extracting common subexpressions."""
        if isinstance(expr, (LiteralExpr, VariableExpr)):
            return expr
        
        expr_hash = self._hash_expr(expr)
        
        # If this subexpression occurs more than once, extract it
        if counts.get(expr_hash, 0) > 1 and not isinstance(expr, (LiteralExpr, VariableExpr)):
            if expr_hash not in extracted:
                # First occurrence: extract
                temp_name = f'_cse_{self._temp_counter}'
                self._temp_counter += 1
                extracted[temp_name] = expr
            else:
                # Already extracted, find temp name
                for name, e in extracted.items():
                    if self._hash_expr(e) == expr_hash:
                        temp_name = name
                        break
            
            return VariableExpr(name=temp_name)
        
        # Transform children
        if isinstance(expr, BinaryExpr):
            return BinaryExpr(
                op=expr.op,
                left=self._transform(expr.left, counts, extracted),
                right=self._transform(expr.right, counts, extracted)
            )
        elif isinstance(expr, UnaryExpr):
            return UnaryExpr(
                op=expr.op,
                operand=self._transform(expr.operand, counts, extracted),
                prefix=expr.prefix
            )
        elif isinstance(expr, CallExpr):
            return CallExpr(
                func=expr.func,
                args=[self._transform(a, counts, extracted) for a in expr.args]
            )
        elif isinstance(expr, IndexExpr):
            return IndexExpr(
                base=self._transform(expr.base, counts, extracted),
                index=self._transform(expr.index, counts, extracted)
            )
        elif isinstance(expr, MemberExpr):
            return MemberExpr(
                base=self._transform(expr.base, counts, extracted),
                member=expr.member,
                is_pointer=expr.is_pointer
            )
        elif isinstance(expr, TernaryExpr):
            return TernaryExpr(
                condition=self._transform(expr.condition, counts, extracted),
                then_expr=self._transform(expr.then_expr, counts, extracted),
                else_expr=self._transform(expr.else_expr, counts, extracted)
            )
        elif isinstance(expr, CastExpr):
            return CastExpr(
                expr=self._transform(expr.expr, counts, extracted),
                target_type=expr.target_type
            )
        
        return expr
    
    def _hash_expr(self, expr: Expression) -> str:
        """Create a hash string for an expression."""
        if isinstance(expr, LiteralExpr):
            return f"lit:{expr.value}:{expr.type}"
        elif isinstance(expr, VariableExpr):
            return f"var:{expr.name}"
        elif isinstance(expr, BinaryExpr):
            left_hash = self._hash_expr(expr.left)
            right_hash = self._hash_expr(expr.right)
            return f"bin:{expr.op}:{left_hash}:{right_hash}"
        elif isinstance(expr, UnaryExpr):
            operand_hash = self._hash_expr(expr.operand)
            return f"un:{expr.op}:{operand_hash}"
        elif isinstance(expr, CallExpr):
            args_hash = ','.join(self._hash_expr(a) for a in expr.args)
            return f"call:{expr.func}:[{args_hash}]"
        elif isinstance(expr, IndexExpr):
            base_hash = self._hash_expr(expr.base)
            index_hash = self._hash_expr(expr.index)
            return f"idx:{base_hash}[{index_hash}]"
        elif isinstance(expr, MemberExpr):
            base_hash = self._hash_expr(expr.base)
            return f"mem:{base_hash}.{expr.member}"
        elif isinstance(expr, TernaryExpr):
            cond_hash = self._hash_expr(expr.condition)
            then_hash = self._hash_expr(expr.then_expr)
            else_hash = self._hash_expr(expr.else_expr)
            return f"tern:{cond_hash}?{then_hash}:{else_hash}"
        elif isinstance(expr, CastExpr):
            inner_hash = self._hash_expr(expr.expr)
            return f"cast:{inner_hash}:{expr.target_type}"
        
        return f"unknown:{id(expr)}"


class ExpressionEmitter:
    """Emits expressions for different target languages."""
    
    def __init__(self, target: str = 'c'):
        self.target = target
    
    def emit(self, expr: Expression) -> str:
        """Emit expression as target language code."""
        if isinstance(expr, LiteralExpr):
            return self._emit_literal(expr)
        elif isinstance(expr, VariableExpr):
            return expr.name
        elif isinstance(expr, BinaryExpr):
            return self._emit_binary(expr)
        elif isinstance(expr, UnaryExpr):
            return self._emit_unary(expr)
        elif isinstance(expr, CallExpr):
            return self._emit_call(expr)
        elif isinstance(expr, IndexExpr):
            return self._emit_index(expr)
        elif isinstance(expr, MemberExpr):
            return self._emit_member(expr)
        elif isinstance(expr, TernaryExpr):
            return self._emit_ternary(expr)
        elif isinstance(expr, CastExpr):
            return self._emit_cast(expr)
        
        return str(expr)
    
    def _emit_literal(self, expr: LiteralExpr) -> str:
        """Emit a literal value."""
        value = expr.value
        
        if value is None:
            if self.target == 'python':
                return 'None'
            elif self.target == 'rust':
                return '()'
            elif self.target == 'haskell':
                return '()'
            return 'NULL'
        
        if isinstance(value, bool):
            if self.target in ('python', 'haskell'):
                return 'True' if value else 'False'
            return 'true' if value else 'false'
        
        if isinstance(value, str):
            return f'"{value}"'
        
        if isinstance(value, float):
            s = str(value)
            if '.' not in s and 'e' not in s.lower():
                s += '.0'
            return s
        
        return str(value)
    
    def _emit_binary(self, expr: BinaryExpr) -> str:
        """Emit a binary expression."""
        left = self.emit(expr.left)
        right = self.emit(expr.right)
        op = expr.op
        
        # Translate operators for different targets
        if self.target == 'python':
            if op == '&&':
                op = 'and'
            elif op == '||':
                op = 'or'
        elif self.target == 'haskell':
            if op == '&&':
                op = '&&'  # Haskell uses same
            elif op == '||':
                op = '||'
            elif op == '!=':
                op = '/='
        
        # Check if we need parentheses
        left_needs_parens = self._needs_parens(expr.left, expr, 'left')
        right_needs_parens = self._needs_parens(expr.right, expr, 'right')
        
        if left_needs_parens:
            left = f'({left})'
        if right_needs_parens:
            right = f'({right})'
        
        return f'{left} {op} {right}'
    
    def _emit_unary(self, expr: UnaryExpr) -> str:
        """Emit a unary expression."""
        operand = self.emit(expr.operand)
        op = expr.op
        
        if self.target == 'python' and op == '!':
            op = 'not '
        elif self.target == 'haskell' and op == '!':
            op = 'not '
        
        needs_parens = isinstance(expr.operand, BinaryExpr)
        if needs_parens:
            operand = f'({operand})'
        
        if expr.prefix:
            return f'{op}{operand}'
        else:
            return f'{operand}{op}'
    
    def _emit_call(self, expr: CallExpr) -> str:
        """Emit a function call."""
        args = ', '.join(self.emit(a) for a in expr.args)
        
        if self.target == 'haskell':
            # Haskell uses space-separated arguments
            args = ' '.join(self.emit(a) for a in expr.args)
            return f'{expr.func} {args}' if args else expr.func
        
        return f'{expr.func}({args})'
    
    def _emit_index(self, expr: IndexExpr) -> str:
        """Emit an index expression."""
        base = self.emit(expr.base)
        index = self.emit(expr.index)
        
        if self.target == 'haskell':
            return f'{base} !! {index}'
        
        return f'{base}[{index}]'
    
    def _emit_member(self, expr: MemberExpr) -> str:
        """Emit a member access expression."""
        base = self.emit(expr.base)
        
        if self.target == 'haskell':
            return f'{expr.member} {base}'
        
        op = '->' if expr.is_pointer else '.'
        return f'{base}{op}{expr.member}'
    
    def _emit_ternary(self, expr: TernaryExpr) -> str:
        """Emit a ternary expression."""
        cond = self.emit(expr.condition)
        then_e = self.emit(expr.then_expr)
        else_e = self.emit(expr.else_expr)
        
        if self.target == 'python':
            return f'{then_e} if {cond} else {else_e}'
        elif self.target == 'rust':
            return f'if {cond} {{ {then_e} }} else {{ {else_e} }}'
        elif self.target == 'haskell':
            return f'if {cond} then {then_e} else {else_e}'
        
        return f'{cond} ? {then_e} : {else_e}'
    
    def _emit_cast(self, expr: CastExpr) -> str:
        """Emit a type cast expression."""
        inner = self.emit(expr.expr)
        target = expr.target_type
        
        if self.target == 'python':
            type_map = {'i32': 'int', 'i64': 'int', 'f32': 'float', 'f64': 'float', 
                       'bool': 'bool', 'string': 'str'}
            py_type = type_map.get(target, target)
            return f'{py_type}({inner})'
        elif self.target == 'rust':
            return f'{inner} as {target}'
        elif self.target == 'haskell':
            return f'fromIntegral {inner}'  # Simplified
        
        return f'({target}){inner}'
    
    def _needs_parens(self, child: Expression, parent: BinaryExpr, side: str) -> bool:
        """Check if child expression needs parentheses."""
        if not isinstance(child, BinaryExpr):
            return False
        
        parent_info = OPERATORS.get(parent.op)
        child_info = OPERATORS.get(child.op)
        
        if not parent_info or not child_info:
            return True
        
        # Lower precedence needs parens
        if child_info.precedence_value < parent_info.precedence_value:
            return True
        
        # Same precedence: check associativity
        if child_info.precedence_value == parent_info.precedence_value:
            if side == 'right' and child_info.associativity == OperatorAssociativity.LEFT:
                return True
            if side == 'left' and child_info.associativity == OperatorAssociativity.RIGHT:
                return True
        
        return False


__all__ = [
    'OperatorPrecedence', 'OperatorAssociativity', 'OperatorInfo', 'OPERATORS',
    'Expression', 'LiteralExpr', 'VariableExpr', 'BinaryExpr', 'UnaryExpr',
    'CallExpr', 'IndexExpr', 'MemberExpr', 'TernaryExpr', 'CastExpr',
    'ExpressionParser', 'ConstantFolder', 'CommonSubexpressionEliminator',
    'ExpressionEmitter'
]
