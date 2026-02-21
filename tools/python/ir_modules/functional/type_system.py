#!/usr/bin/env python3
"""STUNIR Functional IR - Type System and Inference.

This module provides type checking and basic Hindley-Milner style
type inference for functional IR.

Usage:
    from ir.functional.type_system import TypeInference, TypeEnvironment
    
    inference = TypeInference()
    env = TypeEnvironment()
    
    expr_type = inference.infer(expr, env)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Set
from ir.functional.functional_ir import (
    TypeExpr, TypeVar, TypeCon, FunctionType, TupleType, ListType,
    Expr, LiteralExpr, VarExpr, AppExpr, LambdaExpr, LetExpr, IfExpr,
    CaseExpr, ListExpr, TupleExpr, BinaryOpExpr, UnaryOpExpr
)


class TypeError(Exception):
    """Type error during inference or checking."""
    pass


class TypeEnvironment:
    """Type environment for type checking/inference.
    
    Maps variable names to their types.
    """
    
    def __init__(self, parent: 'TypeEnvironment' = None):
        self.bindings: Dict[str, TypeExpr] = {}
        self.type_constructors: Dict[str, 'DataType'] = {}
        self.parent: Optional['TypeEnvironment'] = parent
    
    def extend(self, name: str, type_expr: TypeExpr) -> 'TypeEnvironment':
        """Create new environment with additional binding.
        
        Args:
            name: Variable name
            type_expr: Type to bind
            
        Returns:
            New environment with binding added
        """
        new_env = TypeEnvironment(parent=self)
        new_env.bindings[name] = type_expr
        new_env.type_constructors = self.type_constructors
        return new_env
    
    def extend_many(self, bindings: Dict[str, TypeExpr]) -> 'TypeEnvironment':
        """Create new environment with multiple bindings.
        
        Args:
            bindings: Dictionary of name -> type bindings
            
        Returns:
            New environment with bindings added
        """
        new_env = TypeEnvironment(parent=self)
        new_env.bindings.update(bindings)
        new_env.type_constructors = self.type_constructors
        return new_env
    
    def lookup(self, name: str) -> Optional[TypeExpr]:
        """Look up type of variable.
        
        Args:
            name: Variable name
            
        Returns:
            Type if found, None otherwise
        """
        if name in self.bindings:
            return self.bindings[name]
        if self.parent:
            return self.parent.lookup(name)
        return None
    
    def contains(self, name: str) -> bool:
        """Check if variable is in environment."""
        return self.lookup(name) is not None


class TypeInference:
    """Basic Hindley-Milner type inference.
    
    Implements unification-based type inference for functional expressions.
    """
    
    def __init__(self):
        self.substitution: Dict[str, TypeExpr] = {}
        self.counter = 0
    
    def reset(self):
        """Reset inference state."""
        self.substitution = {}
        self.counter = 0
    
    def fresh_type_var(self, prefix: str = 't') -> TypeVar:
        """Generate fresh type variable.
        
        Args:
            prefix: Prefix for variable name
            
        Returns:
            New unique type variable
        """
        self.counter += 1
        return TypeVar(name=f'{prefix}{self.counter}')
    
    def unify(self, t1: TypeExpr, t2: TypeExpr) -> bool:
        """Unify two types.
        
        Args:
            t1: First type
            t2: Second type
            
        Returns:
            True if unification succeeds
        """
        t1 = self.apply_substitution(t1)
        t2 = self.apply_substitution(t2)
        
        # Both are same type variable
        if isinstance(t1, TypeVar) and isinstance(t2, TypeVar):
            if t1.name == t2.name:
                return True
        
        # t1 is type variable - bind it
        if isinstance(t1, TypeVar):
            return self._bind(t1.name, t2)
        
        # t2 is type variable - bind it
        if isinstance(t2, TypeVar):
            return self._bind(t2.name, t1)
        
        # Both are type constructors
        if isinstance(t1, TypeCon) and isinstance(t2, TypeCon):
            if t1.name != t2.name or len(t1.args) != len(t2.args):
                return False
            return all(self.unify(a1, a2) for a1, a2 in zip(t1.args, t2.args))
        
        # Both are function types
        if isinstance(t1, FunctionType) and isinstance(t2, FunctionType):
            return (self.unify(t1.param_type, t2.param_type) and
                    self.unify(t1.return_type, t2.return_type))
        
        # Both are tuple types
        if isinstance(t1, TupleType) and isinstance(t2, TupleType):
            if len(t1.elements) != len(t2.elements):
                return False
            return all(self.unify(e1, e2) for e1, e2 in zip(t1.elements, t2.elements))
        
        # Both are list types
        if isinstance(t1, ListType) and isinstance(t2, ListType):
            return self.unify(t1.element_type, t2.element_type)
        
        return False
    
    def _bind(self, name: str, type_expr: TypeExpr) -> bool:
        """Bind type variable to type.
        
        Args:
            name: Type variable name
            type_expr: Type to bind
            
        Returns:
            True if binding succeeds
        """
        # Don't bind to itself
        if isinstance(type_expr, TypeVar) and type_expr.name == name:
            return True
        
        # Occurs check - prevent infinite types
        if self._occurs_check(name, type_expr):
            return False
        
        self.substitution[name] = type_expr
        return True
    
    def _occurs_check(self, name: str, type_expr: TypeExpr) -> bool:
        """Check if type variable occurs in type.
        
        Prevents construction of infinite types like t = t -> t.
        
        Args:
            name: Type variable name
            type_expr: Type to check
            
        Returns:
            True if variable occurs in type
        """
        type_expr = self.apply_substitution(type_expr)
        
        if isinstance(type_expr, TypeVar):
            return type_expr.name == name
        if isinstance(type_expr, TypeCon):
            return any(self._occurs_check(name, arg) for arg in type_expr.args)
        if isinstance(type_expr, FunctionType):
            return (self._occurs_check(name, type_expr.param_type) or
                    self._occurs_check(name, type_expr.return_type))
        if isinstance(type_expr, TupleType):
            return any(self._occurs_check(name, e) for e in type_expr.elements)
        if isinstance(type_expr, ListType):
            return self._occurs_check(name, type_expr.element_type)
        return False
    
    def apply_substitution(self, type_expr: TypeExpr) -> TypeExpr:
        """Apply current substitution to type.
        
        Args:
            type_expr: Type to apply substitution to
            
        Returns:
            Type with substitutions applied
        """
        if type_expr is None:
            return None
            
        if isinstance(type_expr, TypeVar):
            if type_expr.name in self.substitution:
                return self.apply_substitution(self.substitution[type_expr.name])
            return type_expr
        
        if isinstance(type_expr, TypeCon):
            return TypeCon(
                name=type_expr.name,
                args=[self.apply_substitution(arg) for arg in type_expr.args]
            )
        
        if isinstance(type_expr, FunctionType):
            return FunctionType(
                param_type=self.apply_substitution(type_expr.param_type),
                return_type=self.apply_substitution(type_expr.return_type)
            )
        
        if isinstance(type_expr, TupleType):
            return TupleType(
                elements=[self.apply_substitution(e) for e in type_expr.elements]
            )
        
        if isinstance(type_expr, ListType):
            return ListType(
                element_type=self.apply_substitution(type_expr.element_type)
            )
        
        return type_expr
    
    def infer(self, expr: Expr, env: TypeEnvironment) -> TypeExpr:
        """Infer type of expression.
        
        Args:
            expr: Expression to infer type for
            env: Type environment
            
        Returns:
            Inferred type
            
        Raises:
            TypeError: If type inference fails
        """
        if isinstance(expr, LiteralExpr):
            return self._infer_literal(expr)
        
        elif isinstance(expr, VarExpr):
            t = env.lookup(expr.name)
            if t is None:
                raise TypeError(f"Unbound variable: {expr.name}")
            return self._instantiate(t)
        
        elif isinstance(expr, LambdaExpr):
            param_type = expr.param_type if expr.param_type else self.fresh_type_var('a')
            new_env = env.extend(expr.param, param_type)
            body_type = self.infer(expr.body, new_env)
            return FunctionType(param_type=param_type, return_type=body_type)
        
        elif isinstance(expr, AppExpr):
            func_type = self.infer(expr.func, env)
            arg_type = self.infer(expr.arg, env) if expr.arg else TypeCon(name='Unit')
            result_type = self.fresh_type_var('r')
            expected_func_type = FunctionType(param_type=arg_type, return_type=result_type)
            if not self.unify(func_type, expected_func_type):
                raise TypeError(f"Type mismatch in application: expected function type")
            return self.apply_substitution(result_type)
        
        elif isinstance(expr, IfExpr):
            cond_type = self.infer(expr.condition, env)
            if not self.unify(cond_type, TypeCon(name='Bool')):
                raise TypeError("Condition must be Bool")
            then_type = self.infer(expr.then_branch, env)
            else_type = self.infer(expr.else_branch, env)
            if not self.unify(then_type, else_type):
                raise TypeError("If branches must have same type")
            return self.apply_substitution(then_type)
        
        elif isinstance(expr, LetExpr):
            if expr.is_recursive:
                # For recursive let, add binding before inferring value
                var_type = self.fresh_type_var('rec')
                rec_env = env.extend(expr.name, var_type)
                value_type = self.infer(expr.value, rec_env)
                if not self.unify(var_type, value_type):
                    raise TypeError("Recursive binding type mismatch")
                body_type = self.infer(expr.body, rec_env)
            else:
                value_type = self.infer(expr.value, env)
                new_env = env.extend(expr.name, value_type)
                body_type = self.infer(expr.body, new_env)
            return body_type
        
        elif isinstance(expr, ListExpr):
            if not expr.elements:
                elem_type = self.fresh_type_var('elem')
            else:
                elem_type = self.infer(expr.elements[0], env)
                for elem in expr.elements[1:]:
                    t = self.infer(elem, env)
                    if not self.unify(elem_type, t):
                        raise TypeError("List elements must have same type")
            return ListType(element_type=self.apply_substitution(elem_type))
        
        elif isinstance(expr, TupleExpr):
            elem_types = [self.infer(e, env) for e in expr.elements]
            return TupleType(elements=elem_types)
        
        elif isinstance(expr, BinaryOpExpr):
            return self._infer_binary_op(expr, env)
        
        elif isinstance(expr, UnaryOpExpr):
            return self._infer_unary_op(expr, env)
        
        elif isinstance(expr, CaseExpr):
            # Simplified case inference
            scrutinee_type = self.infer(expr.scrutinee, env)
            if not expr.branches:
                return self.fresh_type_var('case')
            result_type = self.fresh_type_var('case_result')
            for branch in expr.branches:
                branch_type = self.infer(branch.body, env)
                if not self.unify(result_type, branch_type):
                    raise TypeError("Case branches must have same type")
            return self.apply_substitution(result_type)
        
        else:
            return self.fresh_type_var('unknown')
    
    def _infer_literal(self, expr: LiteralExpr) -> TypeExpr:
        """Infer type of literal.
        
        Args:
            expr: Literal expression
            
        Returns:
            Type of literal
        """
        type_map = {
            'int': TypeCon(name='Int'),
            'float': TypeCon(name='Float'),
            'bool': TypeCon(name='Bool'),
            'string': TypeCon(name='String'),
            'char': TypeCon(name='Char'),
        }
        return type_map.get(expr.literal_type, TypeCon(name='Int'))
    
    def _infer_binary_op(self, expr: BinaryOpExpr, env: TypeEnvironment) -> TypeExpr:
        """Infer type of binary operation.
        
        Args:
            expr: Binary operation expression
            env: Type environment
            
        Returns:
            Result type
        """
        left_type = self.infer(expr.left, env)
        right_type = self.infer(expr.right, env)
        
        # Numeric operations
        if expr.op in ('+', '-', '*', '/', '%'):
            if not self.unify(left_type, right_type):
                raise TypeError(f"Type mismatch in {expr.op}")
            return self.apply_substitution(left_type)
        
        # Comparison operations
        elif expr.op in ('==', '/=', '<', '>', '<=', '>='):
            if not self.unify(left_type, right_type):
                raise TypeError(f"Type mismatch in {expr.op}")
            return TypeCon(name='Bool')
        
        # Boolean operations
        elif expr.op in ('&&', '||'):
            if not self.unify(left_type, TypeCon(name='Bool')):
                raise TypeError(f"Left operand of {expr.op} must be Bool")
            if not self.unify(right_type, TypeCon(name='Bool')):
                raise TypeError(f"Right operand of {expr.op} must be Bool")
            return TypeCon(name='Bool')
        
        # List operations
        elif expr.op == '++':
            if not self.unify(left_type, right_type):
                raise TypeError("List concatenation types must match")
            return self.apply_substitution(left_type)
        
        elif expr.op == ':':
            # Cons operation: a : [a] -> [a]
            elem_type = left_type
            if not self.unify(right_type, ListType(element_type=elem_type)):
                raise TypeError("Cons type mismatch")
            return self.apply_substitution(right_type)
        
        else:
            return self.fresh_type_var('binop')
    
    def _infer_unary_op(self, expr: UnaryOpExpr, env: TypeEnvironment) -> TypeExpr:
        """Infer type of unary operation.
        
        Args:
            expr: Unary operation expression
            env: Type environment
            
        Returns:
            Result type
        """
        operand_type = self.infer(expr.operand, env)
        
        if expr.op == '-':
            return operand_type  # Numeric negation
        elif expr.op == 'not':
            if not self.unify(operand_type, TypeCon(name='Bool')):
                raise TypeError("not requires Bool operand")
            return TypeCon(name='Bool')
        else:
            return operand_type
    
    def _instantiate(self, type_expr: TypeExpr) -> TypeExpr:
        """Instantiate a type scheme with fresh variables.
        
        Args:
            type_expr: Type to instantiate
            
        Returns:
            Instantiated type
        """
        # For now, just return the type as-is
        # Full polymorphism would require tracking quantified variables
        return type_expr


# =============================================================================
# Type Utilities
# =============================================================================

def free_type_vars(type_expr: TypeExpr) -> Set[str]:
    """Get free type variables in a type.
    
    Args:
        type_expr: Type to analyze
        
    Returns:
        Set of free type variable names
    """
    if type_expr is None:
        return set()
    
    if isinstance(type_expr, TypeVar):
        return {type_expr.name}
    elif isinstance(type_expr, TypeCon):
        result = set()
        for arg in type_expr.args:
            result |= free_type_vars(arg)
        return result
    elif isinstance(type_expr, FunctionType):
        return free_type_vars(type_expr.param_type) | free_type_vars(type_expr.return_type)
    elif isinstance(type_expr, TupleType):
        result = set()
        for elem in type_expr.elements:
            result |= free_type_vars(elem)
        return result
    elif isinstance(type_expr, ListType):
        return free_type_vars(type_expr.element_type)
    return set()


def type_to_string(type_expr: TypeExpr) -> str:
    """Convert type to string representation.
    
    Args:
        type_expr: Type to convert
        
    Returns:
        String representation
    """
    if type_expr is None:
        return '?'
    
    if isinstance(type_expr, TypeVar):
        return type_expr.name
    elif isinstance(type_expr, TypeCon):
        if type_expr.args:
            args = ' '.join(type_to_string(a) for a in type_expr.args)
            return f"({type_expr.name} {args})"
        return type_expr.name
    elif isinstance(type_expr, FunctionType):
        param = type_to_string(type_expr.param_type)
        ret = type_to_string(type_expr.return_type)
        return f"({param} -> {ret})"
    elif isinstance(type_expr, TupleType):
        elems = ', '.join(type_to_string(e) for e in type_expr.elements)
        return f"({elems})"
    elif isinstance(type_expr, ListType):
        elem = type_to_string(type_expr.element_type)
        return f"[{elem}]"
    return str(type_expr)
