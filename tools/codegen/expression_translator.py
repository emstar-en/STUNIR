#!/usr/bin/env python3
"""STUNIR Expression Translator Module.

This module provides the base class for translating IR expressions to target
language code. Each language-specific generator extends this class to provide
language-specific implementations.

Part of Phase 3 (Advanced Code Generation) of STUNIR Enhancement Integration.

Supported Expression Types:
- Literals (literal) - numbers, strings, booleans
- Variables (var) - variable references
- Binary operations (binary) - arithmetic, comparison, logical
- Unary operations (unary) - negation, not
- Function calls (call) - function/method invocations
- Index/subscript (index) - array/list indexing
- Member access (member) - field/property access
- Type cast (cast) - type conversion
- Ternary (ternary) - conditional expression
- Array literal (array) - array/list literals
- Struct literal (struct) - struct/object initialization
- Nested calls (chain) - method chaining
- Lambda expressions (lambda) - anonymous functions
- Multi-dimensional array (multi_array) - nested arrays

Usage:
    from tools.codegen.expression_translator import ExpressionTranslator
    
    class PythonExpressionTranslator(ExpressionTranslator):
        def translate_binary_op(self, left, op, right, **kwargs):
            return f"({left} {op} {right})"
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from tools.integration import EnhancementContext

logger = logging.getLogger(__name__)


# Operator categories for reference
ARITHMETIC_OPS = {'+', '-', '*', '/', '%', '//', '**', '>>>', '<<', '>>', '&', '|', '^'}
COMPARISON_OPS = {'==', '!=', '<', '<=', '>', '>=', '===', '!=='}
LOGICAL_OPS = {'and', 'or', '&&', '||', 'not', '!'}


class ExpressionTranslator(ABC):
    """Base class for expression translation to target languages.
    
    This class defines the interface for translating IR expressions to
    target language code. Subclasses must implement all abstract methods
    to provide language-specific translations.
    
    Attributes:
        enhancement_context: Optional EnhancementContext for type info.
        operator_map: Mapping of IR operators to target language operators.
    """
    
    # Override in subclasses
    TARGET: str = 'generic'
    
    # Default operator mappings (override in subclasses as needed)
    LOGICAL_AND: str = 'and'
    LOGICAL_OR: str = 'or'
    LOGICAL_NOT: str = 'not'
    
    def __init__(
        self,
        enhancement_context: Optional['EnhancementContext'] = None
    ):
        """Initialize the expression translator.
        
        Args:
            enhancement_context: Optional EnhancementContext for type info.
        """
        self.enhancement_context = enhancement_context
        self.operator_map = self._build_operator_map()
    
    def _build_operator_map(self) -> Dict[str, str]:
        """Build the operator mapping for this language.
        
        Returns:
            Dictionary mapping IR operators to target language operators.
        """
        return {
            # Arithmetic
            '+': '+',
            '-': '-',
            '*': '*',
            '/': '/',
            '%': '%',
            '//': '//',
            '**': '**',
            
            # Comparison
            '==': '==',
            '!=': '!=',
            '<': '<',
            '<=': '<=',
            '>': '>',
            '>=': '>=',
            
            # Logical
            'and': self.LOGICAL_AND,
            '&&': self.LOGICAL_AND,
            'or': self.LOGICAL_OR,
            '||': self.LOGICAL_OR,
            'not': self.LOGICAL_NOT,
            '!': self.LOGICAL_NOT,
            
            # Bitwise
            '&': '&',
            '|': '|',
            '^': '^',
            '<<': '<<',
            '>>': '>>',
            '~': '~',
        }
    
    def translate_expression(self, expr: Any) -> str:
        """Translate an expression from IR to target language.
        
        Args:
            expr: Expression IR (dict, literal value, or string).
            
        Returns:
            Target language expression string.
        """
        if expr is None:
            return self.translate_literal(None, 'void')
        
        if isinstance(expr, dict):
            return self._translate_expr_dict(expr)
        
        if isinstance(expr, bool):
            return self.translate_literal(expr, 'bool')
        
        if isinstance(expr, int):
            return self.translate_literal(expr, 'i32')
        
        if isinstance(expr, float):
            return self.translate_literal(expr, 'f64')
        
        if isinstance(expr, str):
            # Could be a variable name or string literal
            if expr.startswith('"') or expr.startswith("'"):
                return expr  # Already a string literal
            if expr.isidentifier():
                return self.translate_variable(expr)
            # Treat as string literal
            return self.translate_literal(expr, 'string')
        
        if isinstance(expr, list):
            return self._translate_list_literal(expr)
        
        return str(expr)
    
    def _translate_expr_dict(self, expr: Dict[str, Any]) -> str:
        """Translate an expression dictionary.
        
        Args:
            expr: Expression IR dictionary.
            
        Returns:
            Target language expression string.
        """
        expr_type = expr.get('type', '')
        
        dispatch = {
            'literal': self._translate_literal_expr,
            'var': self._translate_var_expr,
            'binary': self._translate_binary_expr,
            'unary': self._translate_unary_expr,
            'call': self._translate_call_expr,
            'index': self._translate_index_expr,
            'member': self._translate_member_expr,
            'cast': self._translate_cast_expr,
            'ternary': self._translate_ternary_expr,
            'conditional': self._translate_ternary_expr,  # alias
            'array': self._translate_array_expr,
            'struct': self._translate_struct_expr,
            'object': self._translate_struct_expr,  # alias for JS-like syntax
            'chain': self._translate_chain_expr,
            'method_chain': self._translate_chain_expr,  # alias
            'lambda': self._translate_lambda_expr,
            'arrow': self._translate_lambda_expr,  # alias for JS arrow functions
            'multi_array': self._translate_multi_array_expr,
            'nested_array': self._translate_multi_array_expr,  # alias
            'sizeof': self._translate_sizeof_expr,
            'typeof': self._translate_typeof_expr,
            'new': self._translate_new_expr,
            'ref': self._translate_ref_expr,
            'deref': self._translate_deref_expr,
            'address_of': self._translate_ref_expr,  # alias
            'spread': self._translate_spread_expr,
            'await': self._translate_await_expr,
        }
        
        handler = dispatch.get(expr_type)
        if handler is None:
            logger.warning(f"Unknown expression type: {expr_type}")
            return self._translate_unknown_expr(expr)
        
        return handler(expr)
    
    # -------------------------------------------------------------------------
    # Internal dispatch methods
    # -------------------------------------------------------------------------
    
    def _translate_literal_expr(self, expr: Dict[str, Any]) -> str:
        """Translate a literal expression."""
        value = expr.get('value')
        lit_type = expr.get('lit_type', self._infer_type(value))
        return self.translate_literal(value, lit_type)
    
    def _translate_var_expr(self, expr: Dict[str, Any]) -> str:
        """Translate a variable expression."""
        name = expr.get('name', 'x')
        
        # Get type info from enhancement context
        var_type = None
        if self.enhancement_context:
            var_info = self.enhancement_context.lookup_variable(name)
            if var_info and hasattr(var_info, 'type'):
                var_type = str(var_info.type)
        
        return self.translate_variable(name, var_type=var_type)
    
    def _translate_binary_expr(self, expr: Dict[str, Any]) -> str:
        """Translate a binary expression."""
        left_expr = expr.get('left')
        right_expr = expr.get('right')
        op = expr.get('op', '+')
        
        left = self.translate_expression(left_expr)
        right = self.translate_expression(right_expr)
        
        # Map operator
        mapped_op = self.operator_map.get(op, op)
        
        return self.translate_binary_op(left, mapped_op, right)
    
    def _translate_unary_expr(self, expr: Dict[str, Any]) -> str:
        """Translate a unary expression."""
        operand_expr = expr.get('operand')
        op = expr.get('op', '-')
        
        operand = self.translate_expression(operand_expr)
        
        # Map operator
        mapped_op = self.operator_map.get(op, op)
        
        return self.translate_unary_op(mapped_op, operand)
    
    def _translate_call_expr(self, expr: Dict[str, Any]) -> str:
        """Translate a function call expression."""
        func_name = expr.get('func', expr.get('name', 'unknown'))
        args = expr.get('args', [])
        receiver = expr.get('receiver')
        
        # Translate arguments
        translated_args = [self.translate_expression(arg) for arg in args]
        
        # Translate receiver if present
        receiver_code = None
        if receiver:
            receiver_code = self.translate_expression(receiver)
        
        return self.translate_function_call(func_name, translated_args, receiver_code)
    
    def _translate_index_expr(self, expr: Dict[str, Any]) -> str:
        """Translate an index/subscript expression."""
        base = self.translate_expression(expr.get('base'))
        index = self.translate_expression(expr.get('index'))
        return self.translate_index(base, index)
    
    def _translate_member_expr(self, expr: Dict[str, Any]) -> str:
        """Translate a member access expression."""
        base = self.translate_expression(expr.get('base'))
        member = expr.get('member', 'field')
        return self.translate_member_access(base, member)
    
    def _translate_cast_expr(self, expr: Dict[str, Any]) -> str:
        """Translate a type cast expression."""
        value = self.translate_expression(expr.get('value'))
        target_type = expr.get('target_type', 'i32')
        return self.translate_cast(value, target_type)
    
    def _translate_ternary_expr(self, expr: Dict[str, Any]) -> str:
        """Translate a ternary/conditional expression."""
        cond = self.translate_expression(expr.get('condition'))
        then = self.translate_expression(expr.get('then'))
        else_val = self.translate_expression(expr.get('else'))
        return self.translate_ternary(cond, then, else_val)
    
    def _translate_array_expr(self, expr: Dict[str, Any]) -> str:
        """Translate an array literal expression."""
        elements = expr.get('elements', [])
        translated = [self.translate_expression(e) for e in elements]
        return self._translate_list_literal(translated)
    
    def _translate_struct_expr(self, expr: Dict[str, Any]) -> str:
        """Translate a struct literal expression."""
        struct_type = expr.get('struct_type', 'Anonymous')
        fields = expr.get('fields', {})
        translated_fields = {k: self.translate_expression(v) for k, v in fields.items()}
        return self.translate_struct_literal(struct_type, translated_fields)
    
    def _translate_chain_expr(self, expr: Dict[str, Any]) -> str:
        """Translate a method chain expression.
        
        Expected IR structure:
        {
            "type": "chain",
            "base": <expression>,
            "calls": [
                {"method": "name", "args": [...]},
                ...
            ]
        }
        """
        base = self.translate_expression(expr.get('base'))
        calls = expr.get('calls', [])
        
        result = base
        for call in calls:
            method_name = call.get('method', call.get('name', 'unknown'))
            args = [self.translate_expression(a) for a in call.get('args', [])]
            args_str = ', '.join(args)
            result = f"{result}.{method_name}({args_str})"
        
        return result
    
    def _translate_lambda_expr(self, expr: Dict[str, Any]) -> str:
        """Translate a lambda/anonymous function expression.
        
        Expected IR structure:
        {
            "type": "lambda",
            "params": [{"name": "x", "type": "i32"}, ...],
            "body": <expression or statements>,
            "return_type": "i32" (optional)
        }
        """
        params = expr.get('params', [])
        body = expr.get('body')
        return_type = expr.get('return_type')
        
        return self.translate_lambda(params, body, return_type)
    
    def _translate_multi_array_expr(self, expr: Dict[str, Any]) -> str:
        """Translate a multi-dimensional array expression.
        
        Expected IR structure:
        {
            "type": "multi_array",
            "elements": [[...], [...], ...],
            "dimensions": [rows, cols, ...]
        }
        """
        elements = expr.get('elements', [])
        return self._translate_nested_list(elements)
    
    def _translate_nested_list(self, elements: List[Any]) -> str:
        """Recursively translate nested list structures."""
        if not elements:
            return '[]'
        
        if isinstance(elements[0], list):
            # Nested array
            inner = [self._translate_nested_list(e) for e in elements]
            return f"[{', '.join(inner)}]"
        else:
            # Leaf array
            translated = [self.translate_expression(e) for e in elements]
            return f"[{', '.join(translated)}]"
    
    def _translate_sizeof_expr(self, expr: Dict[str, Any]) -> str:
        """Translate sizeof expression (C/C++ style)."""
        target = expr.get('target')
        if isinstance(target, str):
            # Type name
            return self.translate_sizeof(target)
        else:
            # Expression
            target_code = self.translate_expression(target)
            return self.translate_sizeof(target_code)
    
    def _translate_typeof_expr(self, expr: Dict[str, Any]) -> str:
        """Translate typeof expression (JS/TS style)."""
        target = self.translate_expression(expr.get('target'))
        return self.translate_typeof(target)
    
    def _translate_new_expr(self, expr: Dict[str, Any]) -> str:
        """Translate new/instantiation expression.
        
        Expected IR structure:
        {
            "type": "new",
            "class": "ClassName",
            "args": [...]
        }
        """
        class_name = expr.get('class', expr.get('type_name', 'Object'))
        args = [self.translate_expression(a) for a in expr.get('args', [])]
        return self.translate_new(class_name, args)
    
    def _translate_ref_expr(self, expr: Dict[str, Any]) -> str:
        """Translate reference/address-of expression."""
        target = self.translate_expression(expr.get('target'))
        mutable = expr.get('mutable', False)
        return self.translate_reference(target, mutable)
    
    def _translate_deref_expr(self, expr: Dict[str, Any]) -> str:
        """Translate dereference expression."""
        target = self.translate_expression(expr.get('target'))
        return self.translate_dereference(target)
    
    def _translate_spread_expr(self, expr: Dict[str, Any]) -> str:
        """Translate spread operator expression (JS/Python style)."""
        target = self.translate_expression(expr.get('target'))
        return self.translate_spread(target)
    
    def _translate_await_expr(self, expr: Dict[str, Any]) -> str:
        """Translate await expression."""
        target = self.translate_expression(expr.get('target'))
        return self.translate_await(target)
    
    def _translate_unknown_expr(self, expr: Dict[str, Any]) -> str:
        """Handle unknown expression types."""
        return f"/* unknown expr: {expr.get('type', 'unknown')} */"
    
    def _translate_list_literal(self, elements: List[Any]) -> str:
        """Translate a list/array literal."""
        if isinstance(elements, list) and all(isinstance(e, str) for e in elements):
            # Already translated elements
            return f"[{', '.join(elements)}]"
        translated = [self.translate_expression(e) for e in elements]
        return f"[{', '.join(translated)}]"
    
    def _infer_type(self, value: Any) -> str:
        """Infer the IR type from a value.
        
        Args:
            value: Value to infer type from.
            
        Returns:
            IR type string.
        """
        if value is None:
            return 'void'
        if isinstance(value, bool):
            return 'bool'
        if isinstance(value, int):
            return 'i32'
        if isinstance(value, float):
            return 'f64'
        if isinstance(value, str):
            return 'string'
        if isinstance(value, list):
            return 'array'
        return 'unknown'
    
    # -------------------------------------------------------------------------
    # Abstract methods (must be implemented by subclasses)
    # -------------------------------------------------------------------------
    
    @abstractmethod
    def translate_literal(self, value: Any, lit_type: str) -> str:
        """Translate a literal value.
        
        Args:
            value: Literal value (int, float, str, bool, None).
            lit_type: IR type of the literal.
            
        Returns:
            Target language literal representation.
        """
        pass
    
    @abstractmethod
    def translate_variable(self, name: str, var_type: Optional[str] = None) -> str:
        """Translate a variable reference.
        
        Args:
            name: Variable name.
            var_type: Optional type information.
            
        Returns:
            Target language variable reference.
        """
        pass
    
    @abstractmethod
    def translate_binary_op(self, left: str, op: str, right: str) -> str:
        """Translate a binary operation.
        
        Args:
            left: Left operand (already translated).
            op: Operator string.
            right: Right operand (already translated).
            
        Returns:
            Target language binary expression.
        """
        pass
    
    @abstractmethod
    def translate_unary_op(self, op: str, operand: str) -> str:
        """Translate a unary operation.
        
        Args:
            op: Operator string.
            operand: Operand (already translated).
            
        Returns:
            Target language unary expression.
        """
        pass
    
    @abstractmethod
    def translate_function_call(
        self,
        func_name: str,
        args: List[str],
        receiver: Optional[str] = None
    ) -> str:
        """Translate a function call.
        
        Args:
            func_name: Function name.
            args: Translated argument list.
            receiver: Optional receiver for method calls.
            
        Returns:
            Target language function call expression.
        """
        pass
    
    # -------------------------------------------------------------------------
    # Virtual methods (can be overridden by subclasses)
    # -------------------------------------------------------------------------
    
    def translate_index(self, base: str, index: str) -> str:
        """Translate an index/subscript expression.
        
        Args:
            base: Base expression (already translated).
            index: Index expression (already translated).
            
        Returns:
            Target language index expression.
        """
        return f"{base}[{index}]"
    
    def translate_member_access(self, base: str, member: str) -> str:
        """Translate a member access expression.
        
        Args:
            base: Base expression (already translated).
            member: Member name.
            
        Returns:
            Target language member access expression.
        """
        return f"{base}.{member}"
    
    def translate_cast(self, value: str, target_type: str) -> str:
        """Translate a type cast expression.
        
        Args:
            value: Value expression (already translated).
            target_type: Target IR type.
            
        Returns:
            Target language cast expression.
        """
        mapped_type = self.map_type(target_type)
        return f"{mapped_type}({value})"
    
    def translate_ternary(self, condition: str, then_val: str, else_val: str) -> str:
        """Translate a ternary/conditional expression.
        
        Args:
            condition: Condition expression (already translated).
            then_val: Then value (already translated).
            else_val: Else value (already translated).
            
        Returns:
            Target language ternary expression.
        """
        return f"({then_val} if {condition} else {else_val})"
    
    def translate_struct_literal(
        self,
        struct_type: str,
        fields: Dict[str, str]
    ) -> str:
        """Translate a struct literal expression.
        
        Args:
            struct_type: Struct type name.
            fields: Dictionary of field names to translated values.
            
        Returns:
            Target language struct literal.
        """
        field_strs = [f"{k}: {v}" for k, v in fields.items()]
        return f"{{{', '.join(field_strs)}}}"
    
    def map_type(self, ir_type: str) -> str:
        """Map IR type to target language type.
        
        Args:
            ir_type: IR type string (e.g., 'i32', 'f64', 'bool', 'string').
            
        Returns:
            Target language type string.
        """
        # Default mapping, override in subclasses
        type_map = {
            'i8': 'int',
            'i16': 'int',
            'i32': 'int',
            'i64': 'int',
            'u8': 'int',
            'u16': 'int',
            'u32': 'int',
            'u64': 'int',
            'f32': 'float',
            'f64': 'float',
            'bool': 'bool',
            'string': 'str',
            'void': 'None',
            'unit': 'None',
        }
        return type_map.get(ir_type, ir_type)
    
    # -------------------------------------------------------------------------
    # Virtual methods for complex expressions (override in subclasses)
    # -------------------------------------------------------------------------
    
    def translate_lambda(
        self,
        params: List[Dict[str, Any]],
        body: Any,
        return_type: Optional[str] = None
    ) -> str:
        """Translate a lambda/anonymous function expression.
        
        Args:
            params: List of parameter dictionaries with 'name' and optional 'type'.
            body: Body expression or statement list.
            return_type: Optional return type.
            
        Returns:
            Target language lambda expression.
        """
        # Default Python-style lambda
        param_names = [p.get('name', 'x') if isinstance(p, dict) else p for p in params]
        params_str = ', '.join(param_names)
        
        if isinstance(body, dict):
            body_code = self.translate_expression(body)
        elif isinstance(body, list):
            body_code = '...'  # Statements not supported in Python lambda
        else:
            body_code = str(body)
        
        return f"lambda {params_str}: {body_code}"
    
    def translate_sizeof(self, target: str) -> str:
        """Translate sizeof expression.
        
        Args:
            target: Type name or expression.
            
        Returns:
            Target language sizeof expression.
        """
        # Default C-style
        return f"sizeof({target})"
    
    def translate_typeof(self, target: str) -> str:
        """Translate typeof expression.
        
        Args:
            target: Expression to get type of.
            
        Returns:
            Target language typeof expression.
        """
        # Default Python-style
        return f"type({target})"
    
    def translate_new(self, class_name: str, args: List[str]) -> str:
        """Translate new/instantiation expression.
        
        Args:
            class_name: Class/type name.
            args: Constructor arguments.
            
        Returns:
            Target language instantiation expression.
        """
        args_str = ', '.join(args)
        return f"{class_name}({args_str})"
    
    def translate_reference(self, target: str, mutable: bool = False) -> str:
        """Translate reference/address-of expression.
        
        Args:
            target: Target expression.
            mutable: Whether reference is mutable.
            
        Returns:
            Target language reference expression.
        """
        # Default C-style
        return f"&{target}"
    
    def translate_dereference(self, target: str) -> str:
        """Translate dereference expression.
        
        Args:
            target: Pointer expression.
            
        Returns:
            Target language dereference expression.
        """
        # Default C-style
        return f"*{target}"
    
    def translate_spread(self, target: str) -> str:
        """Translate spread operator expression.
        
        Args:
            target: Target to spread.
            
        Returns:
            Target language spread expression.
        """
        # Default Python-style
        return f"*{target}"
    
    def translate_await(self, target: str) -> str:
        """Translate await expression.
        
        Args:
            target: Awaitable expression.
            
        Returns:
            Target language await expression.
        """
        return f"await {target}"
    
    def needs_parentheses(self, outer_op: str, inner_op: str, is_right: bool = False) -> bool:
        """Determine if inner expression needs parentheses.
        
        Args:
            outer_op: Outer operator.
            inner_op: Inner operator.
            is_right: Whether inner expression is on right side.
            
        Returns:
            True if parentheses are needed.
        """
        precedence = {
            '**': 15,
            '*': 13, '/': 13, '%': 13, '//': 13,
            '+': 12, '-': 12,
            '<<': 11, '>>': 11,
            '&': 10,
            '^': 9,
            '|': 8,
            '==': 7, '!=': 7, '<': 7, '<=': 7, '>': 7, '>=': 7,
            'not': 6, '!': 6,
            'and': 5, '&&': 5,
            'or': 4, '||': 4,
        }
        
        outer_prec = precedence.get(outer_op, 0)
        inner_prec = precedence.get(inner_op, 0)
        
        if inner_prec < outer_prec:
            return True
        if inner_prec == outer_prec and is_right:
            # Right associativity matters for some operators
            return outer_op in ('**', '-', '/')
        return False
