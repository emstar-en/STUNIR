#!/usr/bin/env python3
"""STUNIR Type Inference Module.

Provides type inference, type checking, and type coercion rules
for STUNIR code generation.

This module is part of the STUNIR code generation enhancement suite.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum, auto

from .type_system import (
    STUNIRType, TypeKind,
    VoidType, UnitType, BoolType, IntType, FloatType, CharType, StringType,
    PointerType, ReferenceType, ArrayType, SliceType,
    StructType, StructField, UnionType,
    EnumType, TaggedUnionType, TaggedVariant,
    FunctionType, ClosureType,
    GenericType, TypeVar, OpaqueType, RecursiveType,
    OptionalType, ResultType, TupleType,
    TypeRegistry, parse_type,
    I32, I64, F32, F64, BOOL, VOID, UNIT, STRING, CHAR
)


class TypeErrorKind(Enum):
    """Kinds of type errors."""
    TYPE_MISMATCH = auto()
    UNDEFINED_VARIABLE = auto()
    UNDEFINED_FUNCTION = auto()
    UNDEFINED_TYPE = auto()
    WRONG_ARGUMENT_COUNT = auto()
    INVALID_OPERATION = auto()
    CANNOT_INFER = auto()
    CIRCULAR_REFERENCE = auto()
    AMBIGUOUS_TYPE = auto()
    INCOMPATIBLE_TYPES = auto()


@dataclass
class TypeError:
    """Represents a type error."""
    kind: TypeErrorKind
    message: str
    location: Optional[str] = None
    expected: Optional[STUNIRType] = None
    actual: Optional[STUNIRType] = None
    
    def __str__(self) -> str:
        loc = f" at {self.location}" if self.location else ""
        details = []
        if self.expected:
            details.append(f"expected: {self.expected}")
        if self.actual:
            details.append(f"actual: {self.actual}")
        detail_str = f" ({', '.join(details)})" if details else ""
        return f"TypeError{loc}: {self.message}{detail_str}"


@dataclass
class TypeConstraint:
    """Represents a type constraint for inference."""
    left: Union[STUNIRType, str]  # Type or type variable name
    right: Union[STUNIRType, str]
    source: str = ""  # Description of where constraint came from


class TypeScope:
    """Manages type bindings in a scope."""
    
    def __init__(self, parent: Optional[TypeScope] = None):
        self.parent = parent
        self.bindings: Dict[str, STUNIRType] = {}
        self.functions: Dict[str, FunctionType] = {}
        self.type_defs: Dict[str, STUNIRType] = {}
    
    def bind(self, name: str, typ: STUNIRType) -> None:
        """Bind a variable to a type."""
        self.bindings[name] = typ
    
    def lookup(self, name: str) -> Optional[STUNIRType]:
        """Look up a variable's type."""
        if name in self.bindings:
            return self.bindings[name]
        if self.parent:
            return self.parent.lookup(name)
        return None
    
    def bind_function(self, name: str, typ: FunctionType) -> None:
        """Bind a function to its type."""
        self.functions[name] = typ
    
    def lookup_function(self, name: str) -> Optional[FunctionType]:
        """Look up a function's type."""
        if name in self.functions:
            return self.functions[name]
        if self.parent:
            return self.parent.lookup_function(name)
        return None
    
    def define_type(self, name: str, typ: STUNIRType) -> None:
        """Define a named type."""
        self.type_defs[name] = typ
    
    def resolve_type(self, name: str) -> Optional[STUNIRType]:
        """Resolve a type name."""
        if name in self.type_defs:
            return self.type_defs[name]
        if self.parent:
            return self.parent.resolve_type(name)
        return None
    
    def child_scope(self) -> TypeScope:
        """Create a child scope."""
        return TypeScope(parent=self)


class TypeInferenceEngine:
    """Type inference engine using Hindley-Milner style inference."""
    
    def __init__(self, registry: Optional[TypeRegistry] = None):
        self.registry = registry or TypeRegistry()
        self.errors: List[TypeError] = []
        self.substitutions: Dict[str, STUNIRType] = {}
        self.constraints: List[TypeConstraint] = []
        self._fresh_counter = 0
    
    def fresh_type_var(self, prefix: str = "T") -> TypeVar:
        """Generate a fresh type variable."""
        name = f"${prefix}{self._fresh_counter}"
        self._fresh_counter += 1
        return TypeVar(name)
    
    def infer_expression(self, expr: Any, scope: TypeScope) -> STUNIRType:
        """Infer the type of an expression."""
        if isinstance(expr, dict):
            return self._infer_dict_expr(expr, scope)
        elif isinstance(expr, str):
            return self._infer_string_expr(expr, scope)
        elif isinstance(expr, bool):
            return BOOL
        elif isinstance(expr, int):
            # Determine int size based on value
            if -128 <= expr <= 127:
                return IntType(bits=8, signed=True)
            elif -32768 <= expr <= 32767:
                return IntType(bits=16, signed=True)
            elif -2147483648 <= expr <= 2147483647:
                return I32
            else:
                return I64
        elif isinstance(expr, float):
            return F64
        elif expr is None:
            return VOID
        
        return self.fresh_type_var()
    
    def _infer_dict_expr(self, expr: Dict, scope: TypeScope) -> STUNIRType:
        """Infer type from a dictionary expression."""
        expr_type = expr.get('type', 'unknown')
        
        if expr_type == 'literal':
            return self._infer_literal(expr, scope)
        elif expr_type == 'var':
            return self._infer_variable(expr, scope)
        elif expr_type == 'binary':
            return self._infer_binary(expr, scope)
        elif expr_type == 'unary':
            return self._infer_unary(expr, scope)
        elif expr_type == 'call':
            return self._infer_call(expr, scope)
        elif expr_type == 'index':
            return self._infer_index(expr, scope)
        elif expr_type == 'member':
            return self._infer_member(expr, scope)
        elif expr_type == 'cast':
            return self._infer_cast(expr, scope)
        elif expr_type == 'if':
            return self._infer_conditional(expr, scope)
        elif expr_type == 'lambda':
            return self._infer_lambda(expr, scope)
        elif expr_type == 'array':
            return self._infer_array_literal(expr, scope)
        elif expr_type == 'tuple':
            return self._infer_tuple_literal(expr, scope)
        elif expr_type == 'struct':
            return self._infer_struct_literal(expr, scope)
        elif expr_type in ('var_decl', 'let'):
            return self._infer_var_decl(expr, scope)
        elif expr_type == 'return':
            return self.infer_expression(expr.get('value'), scope)
        elif expr_type == 'assign':
            return self._infer_assign(expr, scope)
        
        # Try to infer from 'value' field
        if 'value' in expr:
            return self.infer_expression(expr['value'], scope)
        
        return self.fresh_type_var()
    
    def _infer_literal(self, expr: Dict, scope: TypeScope) -> STUNIRType:
        """Infer type of a literal expression."""
        value = expr.get('value')
        lit_type = expr.get('lit_type', 'auto')
        
        if lit_type != 'auto':
            return parse_type(lit_type, self.registry)
        
        return self.infer_expression(value, scope)
    
    def _infer_variable(self, expr: Dict, scope: TypeScope) -> STUNIRType:
        """Infer type of a variable reference."""
        name = expr.get('name', '')
        typ = scope.lookup(name)
        
        if typ is None:
            self.errors.append(TypeError(
                kind=TypeErrorKind.UNDEFINED_VARIABLE,
                message=f"Undefined variable: {name}",
                location=expr.get('location')
            ))
            return self.fresh_type_var()
        
        return typ
    
    def _infer_string_expr(self, expr: str, scope: TypeScope) -> STUNIRType:
        """Infer type from a string expression (variable name or literal)."""
        # Check if it's a variable
        typ = scope.lookup(expr)
        if typ:
            return typ
        
        # It's a string literal
        return STRING
    
    def _infer_binary(self, expr: Dict, scope: TypeScope) -> STUNIRType:
        """Infer type of a binary expression."""
        op = expr.get('op', '+')
        left = self.infer_expression(expr.get('left'), scope)
        right = self.infer_expression(expr.get('right'), scope)
        
        # Arithmetic operators
        if op in ('+', '-', '*', '/', '%'):
            return self._unify_numeric(left, right, expr)
        
        # Comparison operators
        if op in ('==', '!=', '<', '>', '<=', '>='):
            self._add_constraint(left, right, f"comparison {op}")
            return BOOL
        
        # Logical operators
        if op in ('&&', '||', 'and', 'or'):
            self._add_constraint(left, BOOL, f"logical {op} left")
            self._add_constraint(right, BOOL, f"logical {op} right")
            return BOOL
        
        # Bitwise operators
        if op in ('&', '|', '^', '<<', '>>'):
            return self._unify_numeric(left, right, expr)
        
        # String concatenation
        if op == '++':
            return STRING
        
        return self.fresh_type_var()
    
    def _infer_unary(self, expr: Dict, scope: TypeScope) -> STUNIRType:
        """Infer type of a unary expression."""
        op = expr.get('op', '-')
        operand = self.infer_expression(expr.get('operand'), scope)
        
        if op == '-':
            return operand
        elif op in ('!', 'not'):
            self._add_constraint(operand, BOOL, "logical not")
            return BOOL
        elif op == '~':  # Bitwise not
            return operand
        elif op == '*':  # Dereference
            if isinstance(operand, PointerType):
                return operand.pointee
            elif isinstance(operand, ReferenceType):
                return operand.referent
            self.errors.append(TypeError(
                kind=TypeErrorKind.INVALID_OPERATION,
                message="Cannot dereference non-pointer type",
                actual=operand
            ))
            return self.fresh_type_var()
        elif op == '&':  # Address-of
            return PointerType(operand)
        
        return operand
    
    def _infer_call(self, expr: Dict, scope: TypeScope) -> STUNIRType:
        """Infer type of a function call."""
        func_name = expr.get('func', '')
        args = expr.get('args', [])
        
        func_type = scope.lookup_function(func_name)
        if func_type is None:
            # Try as a variable (function pointer)
            var_type = scope.lookup(func_name)
            if isinstance(var_type, FunctionType):
                func_type = var_type
            elif isinstance(var_type, ClosureType):
                func_type = FunctionType(var_type.params, var_type.returns)
            else:
                self.errors.append(TypeError(
                    kind=TypeErrorKind.UNDEFINED_FUNCTION,
                    message=f"Undefined function: {func_name}",
                    location=expr.get('location')
                ))
                return self.fresh_type_var()
        
        # Check argument count
        expected_count = len(func_type.params)
        actual_count = len(args)
        
        if not func_type.variadic and actual_count != expected_count:
            self.errors.append(TypeError(
                kind=TypeErrorKind.WRONG_ARGUMENT_COUNT,
                message=f"Function {func_name} expects {expected_count} arguments, got {actual_count}"
            ))
        
        # Type check arguments
        for i, (arg, param_type) in enumerate(zip(args, func_type.params)):
            arg_type = self.infer_expression(arg, scope)
            if not self.is_compatible(arg_type, param_type):
                self.errors.append(TypeError(
                    kind=TypeErrorKind.TYPE_MISMATCH,
                    message=f"Argument {i+1} type mismatch in call to {func_name}",
                    expected=param_type,
                    actual=arg_type
                ))
        
        return func_type.returns
    
    def _infer_index(self, expr: Dict, scope: TypeScope) -> STUNIRType:
        """Infer type of an index expression."""
        base = self.infer_expression(expr.get('base'), scope)
        index = self.infer_expression(expr.get('index'), scope)
        
        # Check index is integer
        if not self._is_integer(index):
            self.errors.append(TypeError(
                kind=TypeErrorKind.INVALID_OPERATION,
                message="Array index must be an integer",
                actual=index
            ))
        
        # Get element type
        if isinstance(base, ArrayType):
            return base.element
        elif isinstance(base, SliceType):
            return base.element
        elif isinstance(base, PointerType):
            return base.pointee
        elif isinstance(base, StringType):
            return CHAR
        
        self.errors.append(TypeError(
            kind=TypeErrorKind.INVALID_OPERATION,
            message="Cannot index into non-array type",
            actual=base
        ))
        return self.fresh_type_var()
    
    def _infer_member(self, expr: Dict, scope: TypeScope) -> STUNIRType:
        """Infer type of a member access expression."""
        base = self.infer_expression(expr.get('base'), scope)
        member = expr.get('member', '')
        
        # Handle references/pointers
        while isinstance(base, (ReferenceType, PointerType)):
            if isinstance(base, ReferenceType):
                base = base.referent
            else:
                base = base.pointee
        
        if isinstance(base, StructType):
            field = base.get_field(member)
            if field:
                return field.type
            self.errors.append(TypeError(
                kind=TypeErrorKind.UNDEFINED_VARIABLE,
                message=f"No field '{member}' in struct {base.name}"
            ))
        elif isinstance(base, TupleType):
            # Tuple indexing (tuple.0, tuple.1, etc.)
            try:
                idx = int(member)
                if 0 <= idx < len(base.elements):
                    return base.elements[idx]
            except ValueError:
                pass
            self.errors.append(TypeError(
                kind=TypeErrorKind.INVALID_OPERATION,
                message=f"Invalid tuple index: {member}"
            ))
        else:
            self.errors.append(TypeError(
                kind=TypeErrorKind.INVALID_OPERATION,
                message=f"Cannot access member of non-struct type",
                actual=base
            ))
        
        return self.fresh_type_var()
    
    def _infer_cast(self, expr: Dict, scope: TypeScope) -> STUNIRType:
        """Infer type of a type cast expression."""
        target_type_str = expr.get('target_type', 'i32')
        target_type = parse_type(target_type_str, self.registry)
        source = self.infer_expression(expr.get('value'), scope)
        
        # Check if cast is valid
        if not self._is_castable(source, target_type):
            self.errors.append(TypeError(
                kind=TypeErrorKind.INVALID_OPERATION,
                message="Invalid type cast",
                expected=target_type,
                actual=source
            ))
        
        return target_type
    
    def _infer_conditional(self, expr: Dict, scope: TypeScope) -> STUNIRType:
        """Infer type of a conditional expression (ternary)."""
        cond = self.infer_expression(expr.get('cond'), scope)
        then_expr = self.infer_expression(expr.get('then'), scope)
        else_expr = self.infer_expression(expr.get('else'), scope)
        
        self._add_constraint(cond, BOOL, "if condition")
        
        # Then and else must have compatible types
        if not self.is_compatible(then_expr, else_expr):
            self.errors.append(TypeError(
                kind=TypeErrorKind.TYPE_MISMATCH,
                message="Then and else branches have incompatible types",
                expected=then_expr,
                actual=else_expr
            ))
        
        return then_expr
    
    def _infer_lambda(self, expr: Dict, scope: TypeScope) -> STUNIRType:
        """Infer type of a lambda expression."""
        params = expr.get('params', [])
        body = expr.get('body')
        
        # Create new scope for lambda body
        lambda_scope = scope.child_scope()
        param_types = []
        
        for param in params:
            if isinstance(param, dict):
                param_name = param.get('name', '')
                param_type_str = param.get('type')
                if param_type_str:
                    param_type = parse_type(param_type_str, self.registry)
                else:
                    param_type = self.fresh_type_var()
            else:
                param_name = str(param)
                param_type = self.fresh_type_var()
            
            lambda_scope.bind(param_name, param_type)
            param_types.append(param_type)
        
        # Infer body type
        return_type = self.infer_expression(body, lambda_scope)
        
        return ClosureType(params=param_types, returns=return_type)
    
    def _infer_array_literal(self, expr: Dict, scope: TypeScope) -> STUNIRType:
        """Infer type of an array literal."""
        elements = expr.get('elements', [])
        
        if not elements:
            # Empty array - need element type hint
            elem_type_str = expr.get('element_type')
            if elem_type_str:
                elem_type = parse_type(elem_type_str, self.registry)
            else:
                elem_type = self.fresh_type_var()
            return ArrayType(elem_type, size=0)
        
        # Infer type from first element
        first_type = self.infer_expression(elements[0], scope)
        
        # Check all elements have compatible types
        for i, elem in enumerate(elements[1:], 2):
            elem_type = self.infer_expression(elem, scope)
            if not self.is_compatible(first_type, elem_type):
                self.errors.append(TypeError(
                    kind=TypeErrorKind.TYPE_MISMATCH,
                    message=f"Array element {i} has incompatible type",
                    expected=first_type,
                    actual=elem_type
                ))
        
        return ArrayType(first_type, size=len(elements))
    
    def _infer_tuple_literal(self, expr: Dict, scope: TypeScope) -> STUNIRType:
        """Infer type of a tuple literal."""
        elements = expr.get('elements', [])
        element_types = [self.infer_expression(e, scope) for e in elements]
        return TupleType(element_types)
    
    def _infer_struct_literal(self, expr: Dict, scope: TypeScope) -> STUNIRType:
        """Infer type of a struct literal."""
        struct_name = expr.get('name', '')
        fields = expr.get('fields', {})
        
        struct_type = scope.resolve_type(struct_name)
        if not isinstance(struct_type, StructType):
            self.errors.append(TypeError(
                kind=TypeErrorKind.UNDEFINED_TYPE,
                message=f"Unknown struct type: {struct_name}"
            ))
            return self.fresh_type_var()
        
        # Check field types
        for field_name, field_value in fields.items():
            field_def = struct_type.get_field(field_name)
            if field_def:
                value_type = self.infer_expression(field_value, scope)
                if not self.is_compatible(value_type, field_def.type):
                    self.errors.append(TypeError(
                        kind=TypeErrorKind.TYPE_MISMATCH,
                        message=f"Field '{field_name}' type mismatch",
                        expected=field_def.type,
                        actual=value_type
                    ))
            else:
                self.errors.append(TypeError(
                    kind=TypeErrorKind.UNDEFINED_VARIABLE,
                    message=f"Unknown field '{field_name}' in struct {struct_name}"
                ))
        
        return struct_type
    
    def _infer_var_decl(self, expr: Dict, scope: TypeScope) -> STUNIRType:
        """Infer type of a variable declaration."""
        var_name = expr.get('var_name', expr.get('name', ''))
        var_type_str = expr.get('var_type', expr.get('type'))
        init = expr.get('init', expr.get('value'))
        
        if var_type_str:
            declared_type = parse_type(var_type_str, self.registry)
        else:
            declared_type = None
        
        if init is not None:
            init_type = self.infer_expression(init, scope)
            if declared_type:
                if not self.is_compatible(init_type, declared_type):
                    self.errors.append(TypeError(
                        kind=TypeErrorKind.TYPE_MISMATCH,
                        message=f"Variable '{var_name}' initialization type mismatch",
                        expected=declared_type,
                        actual=init_type
                    ))
                final_type = declared_type
            else:
                final_type = init_type
        elif declared_type:
            final_type = declared_type
        else:
            final_type = self.fresh_type_var()
        
        scope.bind(var_name, final_type)
        return final_type
    
    def _infer_assign(self, expr: Dict, scope: TypeScope) -> STUNIRType:
        """Infer type of an assignment expression."""
        target = expr.get('target', '')
        value = expr.get('value')
        
        target_type = scope.lookup(target)
        value_type = self.infer_expression(value, scope)
        
        if target_type is None:
            self.errors.append(TypeError(
                kind=TypeErrorKind.UNDEFINED_VARIABLE,
                message=f"Undefined variable: {target}"
            ))
            return value_type
        
        if not self.is_compatible(value_type, target_type):
            self.errors.append(TypeError(
                kind=TypeErrorKind.TYPE_MISMATCH,
                message=f"Assignment to '{target}' type mismatch",
                expected=target_type,
                actual=value_type
            ))
        
        return target_type
    
    def _add_constraint(self, left: Union[STUNIRType, str], 
                       right: Union[STUNIRType, str],
                       source: str = "") -> None:
        """Add a type constraint."""
        self.constraints.append(TypeConstraint(left, right, source))
    
    def _unify_numeric(self, left: STUNIRType, right: STUNIRType, 
                      expr: Dict) -> STUNIRType:
        """Unify two numeric types."""
        # Float + anything = float
        if isinstance(left, FloatType) or isinstance(right, FloatType):
            if isinstance(left, FloatType) and isinstance(right, FloatType):
                return FloatType(bits=max(left.bits, right.bits))
            return left if isinstance(left, FloatType) else right
        
        # Integer promotion
        if isinstance(left, IntType) and isinstance(right, IntType):
            # Larger size wins
            bits = max(left.bits, right.bits)
            # If either is signed, result is signed
            signed = left.signed or right.signed
            return IntType(bits=bits, signed=signed)
        
        if isinstance(left, IntType):
            return left
        if isinstance(right, IntType):
            return right
        
        # Fallback
        return I32
    
    def _is_integer(self, typ: STUNIRType) -> bool:
        """Check if type is an integer type."""
        return isinstance(typ, IntType)
    
    def _is_castable(self, source: STUNIRType, target: STUNIRType) -> bool:
        """Check if source can be cast to target."""
        # Numeric casts are always allowed
        if source.kind in (TypeKind.INT, TypeKind.FLOAT):
            if target.kind in (TypeKind.INT, TypeKind.FLOAT):
                return True
        
        # Pointer casts
        if source.kind == TypeKind.POINTER and target.kind == TypeKind.POINTER:
            return True
        
        # Reference to pointer
        if source.kind == TypeKind.REFERENCE and target.kind == TypeKind.POINTER:
            return True
        
        # Same types
        if type(source) == type(target):
            return True
        
        return False
    
    def is_compatible(self, actual: STUNIRType, expected: STUNIRType) -> bool:
        """Check if actual type is compatible with expected type."""
        # Type variables are compatible with anything
        if isinstance(actual, TypeVar) or isinstance(expected, TypeVar):
            return True
        
        # Same type kind
        if actual.kind != expected.kind:
            # Check for implicit conversions
            return self._can_coerce(actual, expected)
        
        # Check structural compatibility
        if isinstance(actual, IntType) and isinstance(expected, IntType):
            # Allow smaller to larger
            return actual.bits <= expected.bits or actual.signed == expected.signed
        
        if isinstance(actual, FloatType) and isinstance(expected, FloatType):
            return actual.bits <= expected.bits
        
        if isinstance(actual, ArrayType) and isinstance(expected, ArrayType):
            return self.is_compatible(actual.element, expected.element)
        
        if isinstance(actual, PointerType) and isinstance(expected, PointerType):
            return self.is_compatible(actual.pointee, expected.pointee)
        
        if isinstance(actual, FunctionType) and isinstance(expected, FunctionType):
            if len(actual.params) != len(expected.params):
                return False
            for ap, ep in zip(actual.params, expected.params):
                if not self.is_compatible(ap, ep):
                    return False
            return self.is_compatible(actual.returns, expected.returns)
        
        if isinstance(actual, GenericType) and isinstance(expected, GenericType):
            if actual.base != expected.base:
                return False
            if len(actual.args) != len(expected.args):
                return False
            return all(self.is_compatible(a, e) 
                      for a, e in zip(actual.args, expected.args))
        
        # For other types, require exact match
        return type(actual) == type(expected)
    
    def _can_coerce(self, source: STUNIRType, target: STUNIRType) -> bool:
        """Check if source can be implicitly coerced to target."""
        # Int to float
        if isinstance(source, IntType) and isinstance(target, FloatType):
            return True
        
        # Smaller int to larger int
        if isinstance(source, IntType) and isinstance(target, IntType):
            if source.bits < target.bits:
                return True
        
        # Array to slice
        if isinstance(source, ArrayType) and isinstance(target, SliceType):
            return self.is_compatible(source.element, target.element)
        
        # String to &str
        if isinstance(source, StringType) and isinstance(target, StringType):
            return source.owned and not target.owned
        
        # Reference coercions
        if isinstance(target, ReferenceType):
            return self.is_compatible(source, target.referent)
        
        return False
    
    def solve_constraints(self) -> Dict[str, STUNIRType]:
        """Solve accumulated type constraints using unification."""
        for constraint in self.constraints:
            self._unify(constraint.left, constraint.right, constraint.source)
        return self.substitutions
    
    def _unify(self, left: Union[STUNIRType, str], 
              right: Union[STUNIRType, str],
              source: str = "") -> None:
        """Unify two types."""
        # Resolve type variables
        if isinstance(left, str):
            left = self.substitutions.get(left, TypeVar(left))
        if isinstance(right, str):
            right = self.substitutions.get(right, TypeVar(right))
        
        # Apply existing substitutions
        left = self._apply_substitution(left)
        right = self._apply_substitution(right)
        
        # Same type
        if left == right:
            return
        
        # Type variable unification
        if isinstance(left, TypeVar):
            self.substitutions[left.name] = right
            return
        if isinstance(right, TypeVar):
            self.substitutions[right.name] = left
            return
        
        # Structural unification
        if left.kind != right.kind:
            self.errors.append(TypeError(
                kind=TypeErrorKind.INCOMPATIBLE_TYPES,
                message=f"Cannot unify types ({source})",
                expected=right,
                actual=left
            ))
            return
        
        # Unify components
        if isinstance(left, ArrayType) and isinstance(right, ArrayType):
            self._unify(left.element, right.element, f"{source} array element")
        elif isinstance(left, PointerType) and isinstance(right, PointerType):
            self._unify(left.pointee, right.pointee, f"{source} pointer")
        elif isinstance(left, FunctionType) and isinstance(right, FunctionType):
            for i, (lp, rp) in enumerate(zip(left.params, right.params)):
                self._unify(lp, rp, f"{source} param {i}")
            self._unify(left.returns, right.returns, f"{source} return")
    
    def _apply_substitution(self, typ: STUNIRType) -> STUNIRType:
        """Apply current substitutions to a type."""
        if isinstance(typ, TypeVar):
            if typ.name in self.substitutions:
                return self._apply_substitution(self.substitutions[typ.name])
        return typ
    
    def infer_function(self, func: Dict, scope: TypeScope) -> FunctionType:
        """Infer the type of a function definition."""
        name = func.get('name', '')
        params = func.get('params', [])
        returns_str = func.get('returns', 'void')
        body = func.get('body', [])
        
        # Parse return type
        return_type = parse_type(returns_str, self.registry)
        
        # Create function scope
        func_scope = scope.child_scope()
        param_types = []
        
        for param in params:
            if isinstance(param, dict):
                param_name = param.get('name', '')
                param_type_str = param.get('type', 'i32')
                param_type = parse_type(param_type_str, self.registry)
            else:
                param_name = str(param)
                param_type = self.fresh_type_var()
            
            func_scope.bind(param_name, param_type)
            param_types.append(param_type)
        
        # Infer body types and check return
        for stmt in body:
            stmt_type = self.infer_expression(stmt, func_scope)
            if isinstance(stmt, dict) and stmt.get('type') == 'return':
                if not self.is_compatible(stmt_type, return_type):
                    self.errors.append(TypeError(
                        kind=TypeErrorKind.TYPE_MISMATCH,
                        message=f"Return type mismatch in function {name}",
                        expected=return_type,
                        actual=stmt_type
                    ))
        
        func_type = FunctionType(params=param_types, returns=return_type)
        scope.bind_function(name, func_type)
        
        return func_type


class TypeChecker:
    """Type checker for STUNIR IR."""
    
    def __init__(self, registry: Optional[TypeRegistry] = None):
        self.inference = TypeInferenceEngine(registry)
        self.scope = TypeScope()
    
    def check_ir(self, ir_data: Dict) -> List[TypeError]:
        """Type check IR data."""
        self.inference.errors.clear()
        
        # Register type definitions
        for type_def in ir_data.get('ir_types', []):
            self._register_type(type_def)
        
        # Check functions
        for func in ir_data.get('ir_functions', []):
            self.inference.infer_function(func, self.scope)
        
        # Solve constraints
        self.inference.solve_constraints()
        
        return self.inference.errors
    
    def _register_type(self, type_def: Dict) -> None:
        """Register a type definition."""
        name = type_def.get('name', '')
        if not name:
            return
        
        kind = type_def.get('kind', 'alias')
        
        if kind == 'struct':
            fields = []
            for f in type_def.get('fields', []):
                field_type = parse_type(f.get('type', 'i32'), self.inference.registry)
                fields.append(StructField(
                    name=f.get('name', ''),
                    type=field_type
                ))
            struct_type = StructType(name=name, fields=fields)
            self.scope.define_type(name, struct_type)
            self.inference.registry.register(name, struct_type)
        
        elif kind == 'enum':
            variants = [
                EnumVariant(name=v.get('name', ''), value=v.get('value'))
                for v in type_def.get('variants', [])
            ]
            enum_type = EnumType(name=name, variants=variants)
            self.scope.define_type(name, enum_type)
            self.inference.registry.register(name, enum_type)
        
        elif kind == 'alias':
            base_type = parse_type(type_def.get('base', 'i32'), self.inference.registry)
            self.scope.define_type(name, base_type)
            self.inference.registry.register(name, base_type)


__all__ = [
    'TypeErrorKind', 'TypeError', 'TypeConstraint',
    'TypeScope', 'TypeInferenceEngine', 'TypeChecker'
]
