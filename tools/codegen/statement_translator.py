#!/usr/bin/env python3
"""STUNIR Statement Translator Module.

This module provides the base class for translating IR statements to target
language code. Each language-specific generator extends this class to provide
language-specific implementations.

Part of Phase 3 (Advanced Code Generation) of STUNIR Enhancement Integration.

Supported Statement Types:
- Variable declarations (var_decl)
- Constant declarations (const_decl)
- Assignments (assign)
- Return statements (return)
- Expression statements (expr_stmt)
- Function calls (call)
- Control flow: if/else (if)
- Control flow: while loops (while)
- Control flow: for loops (for)
- Control flow: switch/match (switch)
- Control flow: do-while (do_while)
- Block statements (block)
- Break/continue statements

Usage:
    from tools.codegen.statement_translator import StatementTranslator
    
    class PythonStatementTranslator(StatementTranslator):
        def translate_variable_declaration(self, var_name, var_type, init_value, **kwargs):
            return f"{var_name}: {var_type} = {init_value}"
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from tools.integration import EnhancementContext

logger = logging.getLogger(__name__)


class StatementTranslator(ABC):
    """Base class for statement translation to target languages.
    
    This class defines the interface for translating IR statements to
    target language code. Subclasses must implement all abstract methods
    to provide language-specific translations.
    
    Attributes:
        enhancement_context: Optional EnhancementContext for type info.
        indent_size: Number of spaces per indentation level.
        indent_char: Character to use for indentation.
    """
    
    # Override in subclasses
    TARGET: str = 'generic'
    STATEMENT_TERMINATOR: str = ''  # e.g., ';' for C-like languages
    
    def __init__(
        self,
        enhancement_context: Optional['EnhancementContext'] = None,
        indent_size: int = 4,
        indent_char: str = ' '
    ):
        """Initialize the statement translator.
        
        Args:
            enhancement_context: Optional EnhancementContext for type info.
            indent_size: Number of spaces per indentation level.
            indent_char: Character to use for indentation.
        """
        self.enhancement_context = enhancement_context
        self.indent_size = indent_size
        self.indent_char = indent_char
        self._expression_translator = None  # Set by subclasses
    
    def set_expression_translator(self, expr_translator: Any) -> None:
        """Set the expression translator to use for nested expressions.
        
        Args:
            expr_translator: Expression translator instance.
        """
        self._expression_translator = expr_translator
    
    def get_indent(self, level: int = 0) -> str:
        """Get indentation string for given level.
        
        Args:
            level: Indentation level (0 = no indent).
            
        Returns:
            Indentation string.
        """
        return self.indent_char * (self.indent_size * level)
    
    def translate_statement(self, stmt: Dict[str, Any], indent: int = 0) -> str:
        """Translate a statement from IR to target language.
        
        Args:
            stmt: Statement IR dictionary.
            indent: Indentation level.
            
        Returns:
            Target language code string.
            
        Raises:
            ValueError: If statement type is unknown.
        """
        stmt_type = stmt.get('type', '')
        
        dispatch = {
            'var_decl': self._translate_var_decl,
            'const_decl': self._translate_const_decl,
            'assign': self._translate_assign,
            'return': self._translate_return,
            'expr_stmt': self._translate_expr_stmt,
            'call': self._translate_call,
            'if': self._translate_if,
            'while': self._translate_while,
            'for': self._translate_for,
            'for_each': self._translate_for_each,
            'for_range': self._translate_for_range,
            'do_while': self._translate_do_while,
            'switch': self._translate_switch,
            'match': self._translate_switch,  # alias for switch
            'block': self._translate_block,
            'break': self._translate_break,
            'continue': self._translate_continue,
            'loop': self._translate_loop,
        }
        
        handler = dispatch.get(stmt_type)
        if handler is None:
            logger.warning(f"Unknown statement type: {stmt_type}")
            return self._translate_unknown(stmt, indent)
        
        return handler(stmt, indent)
    
    def translate_statements(self, statements: List[Dict[str, Any]], indent: int = 0) -> str:
        """Translate multiple statements.
        
        Args:
            statements: List of statement IR dictionaries.
            indent: Indentation level.
            
        Returns:
            Target language code string with all statements.
        """
        if not statements:
            return self._empty_body(indent)
        
        lines = []
        for stmt in statements:
            translated = self.translate_statement(stmt, indent)
            if translated:
                lines.append(translated)
        
        return '\n'.join(lines) if lines else self._empty_body(indent)
    
    # -------------------------------------------------------------------------
    # Internal dispatch methods
    # -------------------------------------------------------------------------
    
    def _translate_var_decl(self, stmt: Dict[str, Any], indent: int) -> str:
        """Translate variable declaration."""
        var_name = stmt.get('var_name', 'x')
        var_type = stmt.get('var_type', 'i32')
        init = stmt.get('init')
        mutable = stmt.get('mutable', True)
        
        # Translate init value if present
        init_code = None
        if init is not None:
            init_code = self._translate_value(init)
        
        # Get type info from enhancement context if available
        if self.enhancement_context:
            var_info = self.enhancement_context.lookup_variable(var_name)
            if var_info and hasattr(var_info, 'type'):
                var_type = str(var_info.type)
        
        return self.translate_variable_declaration(
            var_name, var_type, init_code,
            mutable=mutable, indent=indent
        )
    
    def _translate_const_decl(self, stmt: Dict[str, Any], indent: int) -> str:
        """Translate constant declaration."""
        var_name = stmt.get('var_name', 'x')
        var_type = stmt.get('var_type', 'i32')
        init = stmt.get('init', stmt.get('value'))
        
        init_code = self._translate_value(init) if init is not None else 'None'
        
        return self.translate_variable_declaration(
            var_name, var_type, init_code,
            mutable=False, indent=indent
        )
    
    def _translate_assign(self, stmt: Dict[str, Any], indent: int) -> str:
        """Translate assignment statement."""
        target = stmt.get('target', 'x')
        value = stmt.get('value')
        
        # Handle compound assignment operators
        op = stmt.get('op')  # e.g., '+=', '-=', etc.
        
        value_code = self._translate_value(value)
        
        if op:
            return self.translate_compound_assignment(target, op, value_code, indent=indent)
        else:
            return self.translate_assignment(target, value_code, indent=indent)
    
    def _translate_return(self, stmt: Dict[str, Any], indent: int) -> str:
        """Translate return statement."""
        value = stmt.get('value')
        
        if value is None:
            return self.translate_return(None, indent=indent)
        
        value_code = self._translate_value(value)
        return self.translate_return(value_code, indent=indent)
    
    def _translate_expr_stmt(self, stmt: Dict[str, Any], indent: int) -> str:
        """Translate expression statement."""
        expr = stmt.get('expr', stmt.get('expression'))
        
        if expr is None:
            return ''
        
        expr_code = self._translate_value(expr)
        return self.translate_expression_statement(expr_code, indent=indent)
    
    def _translate_call(self, stmt: Dict[str, Any], indent: int) -> str:
        """Translate function call statement."""
        func_name = stmt.get('func', stmt.get('name', 'unknown'))
        args = stmt.get('args', [])
        receiver = stmt.get('receiver')
        
        args_code = [self._translate_value(arg) for arg in args]
        receiver_code = self._translate_value(receiver) if receiver else None
        
        call_expr = self._format_call(func_name, args_code, receiver_code)
        return self.translate_expression_statement(call_expr, indent=indent)
    
    def _translate_if(self, stmt: Dict[str, Any], indent: int) -> str:
        """Translate if statement.
        
        Expected IR structure:
        {
            "type": "if",
            "condition": <expression>,
            "then": [<statements>],
            "else": [<statements>] (optional),
            "elif": [{"condition": <expr>, "body": [<stmts>]}] (optional)
        }
        """
        condition = stmt.get('condition')
        then_block = stmt.get('then', stmt.get('body', []))
        else_block = stmt.get('else')
        elif_blocks = stmt.get('elif', [])
        
        cond_code = self._translate_value(condition)
        
        return self.translate_if_statement(
            cond_code, then_block, else_block, elif_blocks, indent
        )
    
    def _translate_while(self, stmt: Dict[str, Any], indent: int) -> str:
        """Translate while loop.
        
        Expected IR structure:
        {
            "type": "while",
            "condition": <expression>,
            "body": [<statements>]
        }
        """
        condition = stmt.get('condition')
        body = stmt.get('body', [])
        
        cond_code = self._translate_value(condition)
        
        return self.translate_while_loop(cond_code, body, indent)
    
    def _translate_for(self, stmt: Dict[str, Any], indent: int) -> str:
        """Translate C-style for loop.
        
        Expected IR structure:
        {
            "type": "for",
            "init": <statement or expression>,
            "condition": <expression>,
            "update": <expression>,
            "body": [<statements>]
        }
        """
        init = stmt.get('init')
        condition = stmt.get('condition')
        update = stmt.get('update', stmt.get('increment'))
        body = stmt.get('body', [])
        
        init_code = self._translate_for_init(init) if init else ''
        cond_code = self._translate_value(condition) if condition else ''
        update_code = self._translate_value(update) if update else ''
        
        return self.translate_for_loop(init_code, cond_code, update_code, body, indent)
    
    def _translate_for_each(self, stmt: Dict[str, Any], indent: int) -> str:
        """Translate for-each/for-in loop.
        
        Expected IR structure:
        {
            "type": "for_each",
            "var": "name" or {"name": "x", "type": "T"},
            "iterable": <expression>,
            "body": [<statements>]
        }
        """
        var = stmt.get('var', stmt.get('variable', 'item'))
        iterable = stmt.get('iterable', stmt.get('collection'))
        body = stmt.get('body', [])
        
        # Handle variable spec
        if isinstance(var, dict):
            var_name = var.get('name', 'item')
            var_type = var.get('type')
        else:
            var_name = var
            var_type = None
        
        iter_code = self._translate_value(iterable)
        
        return self.translate_for_each(var_name, var_type, iter_code, body, indent)
    
    def _translate_for_range(self, stmt: Dict[str, Any], indent: int) -> str:
        """Translate range-based for loop.
        
        Expected IR structure:
        {
            "type": "for_range",
            "var": "name",
            "start": <expression>,
            "end": <expression>,
            "step": <expression> (optional, default 1),
            "body": [<statements>]
        }
        """
        var = stmt.get('var', 'i')
        start = stmt.get('start', {'type': 'literal', 'value': 0})
        end = stmt.get('end')
        step = stmt.get('step')
        body = stmt.get('body', [])
        
        start_code = self._translate_value(start)
        end_code = self._translate_value(end)
        step_code = self._translate_value(step) if step else None
        
        return self.translate_for_range(var, start_code, end_code, step_code, body, indent)
    
    def _translate_do_while(self, stmt: Dict[str, Any], indent: int) -> str:
        """Translate do-while loop.
        
        Expected IR structure:
        {
            "type": "do_while",
            "condition": <expression>,
            "body": [<statements>]
        }
        """
        condition = stmt.get('condition')
        body = stmt.get('body', [])
        
        cond_code = self._translate_value(condition)
        
        return self.translate_do_while(cond_code, body, indent)
    
    def _translate_switch(self, stmt: Dict[str, Any], indent: int) -> str:
        """Translate switch/match statement.
        
        Expected IR structure:
        {
            "type": "switch" or "match",
            "value": <expression>,
            "cases": [
                {"value": <expr>, "body": [<stmts>], "fallthrough": bool},
                ...
            ],
            "default": [<statements>] (optional)
        }
        """
        value = stmt.get('value', stmt.get('expr'))
        cases = stmt.get('cases', [])
        default = stmt.get('default')
        
        value_code = self._translate_value(value)
        
        return self.translate_switch_statement(value_code, cases, default, indent)
    
    def _translate_block(self, stmt: Dict[str, Any], indent: int) -> str:
        """Translate a block of statements.
        
        Expected IR structure:
        {
            "type": "block",
            "statements": [<statements>]
        }
        """
        statements = stmt.get('statements', stmt.get('body', []))
        return self.translate_block(statements, indent)
    
    def _translate_loop(self, stmt: Dict[str, Any], indent: int) -> str:
        """Translate infinite loop (Rust-style loop).
        
        Expected IR structure:
        {
            "type": "loop",
            "body": [<statements>]
        }
        """
        body = stmt.get('body', [])
        return self.translate_infinite_loop(body, indent)
    
    def _translate_break(self, stmt: Dict[str, Any], indent: int) -> str:
        """Translate break statement."""
        ind = self.get_indent(indent)
        label = stmt.get('label')
        if label:
            return f"{ind}break {label}{self.STATEMENT_TERMINATOR}"
        return f"{ind}break{self.STATEMENT_TERMINATOR}"
    
    def _translate_continue(self, stmt: Dict[str, Any], indent: int) -> str:
        """Translate continue statement."""
        ind = self.get_indent(indent)
        label = stmt.get('label')
        if label:
            return f"{ind}continue {label}{self.STATEMENT_TERMINATOR}"
        return f"{ind}continue{self.STATEMENT_TERMINATOR}"
    
    def _translate_for_init(self, init: Any) -> str:
        """Translate for loop initialization."""
        if init is None:
            return ''
        if isinstance(init, dict):
            init_type = init.get('type')
            if init_type == 'var_decl':
                var_name = init.get('var_name', 'i')
                var_type = init.get('var_type', 'i32')
                init_val = init.get('init')
                init_code = self._translate_value(init_val) if init_val else '0'
                return f"{var_name} = {init_code}"
            return self._translate_value(init)
        return str(init)
    
    def _translate_unknown(self, stmt: Dict[str, Any], indent: int) -> str:
        """Handle unknown statement types."""
        ind = self.get_indent(indent)
        stmt_type = stmt.get('type', 'unknown')
        return f"{ind}# TODO: Unhandled statement type: {stmt_type}"
    
    def _translate_value(self, value: Any) -> str:
        """Translate a value (expression or literal).
        
        Args:
            value: Value to translate (dict, str, int, float, etc.)
            
        Returns:
            Target language code string.
        """
        if value is None:
            return self._translate_none()
        
        if isinstance(value, dict):
            # It's an expression IR
            if self._expression_translator:
                return self._expression_translator.translate_expression(value)
            else:
                return self._fallback_expr_translate(value)
        
        if isinstance(value, str):
            # Could be a variable name or a string literal
            if value.startswith('"') or value.startswith("'"):
                return value  # Already a string literal
            # Check if it looks like a variable name
            if value.isidentifier():
                return value  # Variable reference
            # Otherwise, treat as string literal
            return f'"{value}"'
        
        if isinstance(value, bool):
            return self._translate_bool(value)
        
        if isinstance(value, (int, float)):
            return str(value)
        
        if isinstance(value, list):
            return self._translate_list(value)
        
        return str(value)
    
    def _fallback_expr_translate(self, expr: Dict[str, Any]) -> str:
        """Fallback expression translation when no expression translator set."""
        expr_type = expr.get('type', '')
        
        if expr_type == 'literal':
            return self._translate_value(expr.get('value'))
        
        if expr_type == 'var':
            return expr.get('name', 'x')
        
        if expr_type == 'binary':
            left = self._translate_value(expr.get('left'))
            right = self._translate_value(expr.get('right'))
            op = expr.get('op', '+')
            return f"({left} {op} {right})"
        
        if expr_type == 'unary':
            operand = self._translate_value(expr.get('operand'))
            op = expr.get('op', '-')
            return f"({op}{operand})"
        
        if expr_type == 'call':
            func = expr.get('func', 'unknown')
            args = [self._translate_value(a) for a in expr.get('args', [])]
            return f"{func}({', '.join(args)})"
        
        return str(expr)
    
    def _format_call(self, func_name: str, args: List[str], receiver: Optional[str]) -> str:
        """Format a function/method call.
        
        Args:
            func_name: Function name.
            args: Translated argument list.
            receiver: Optional receiver for method calls.
            
        Returns:
            Call expression string.
        """
        args_str = ', '.join(args)
        if receiver:
            return f"{receiver}.{func_name}({args_str})"
        return f"{func_name}({args_str})"
    
    # -------------------------------------------------------------------------
    # Abstract methods (must be implemented by subclasses)
    # -------------------------------------------------------------------------
    
    @abstractmethod
    def translate_variable_declaration(
        self,
        var_name: str,
        var_type: str,
        init_value: Optional[str],
        mutable: bool = True,
        indent: int = 0
    ) -> str:
        """Translate a variable declaration.
        
        Args:
            var_name: Variable name.
            var_type: IR type string (e.g., 'i32', 'f64').
            init_value: Initial value code (already translated).
            mutable: Whether the variable is mutable.
            indent: Indentation level.
            
        Returns:
            Target language variable declaration.
        """
        pass
    
    @abstractmethod
    def translate_assignment(
        self,
        target: str,
        value: str,
        indent: int = 0
    ) -> str:
        """Translate a simple assignment.
        
        Args:
            target: Assignment target (variable name).
            value: Value code (already translated).
            indent: Indentation level.
            
        Returns:
            Target language assignment statement.
        """
        pass
    
    @abstractmethod
    def translate_compound_assignment(
        self,
        target: str,
        op: str,
        value: str,
        indent: int = 0
    ) -> str:
        """Translate a compound assignment (+=, -=, etc.).
        
        Args:
            target: Assignment target.
            op: Compound operator (e.g., '+=', '-=').
            value: Value code (already translated).
            indent: Indentation level.
            
        Returns:
            Target language compound assignment.
        """
        pass
    
    @abstractmethod
    def translate_return(
        self,
        value: Optional[str],
        indent: int = 0
    ) -> str:
        """Translate a return statement.
        
        Args:
            value: Return value code (already translated), or None for void return.
            indent: Indentation level.
            
        Returns:
            Target language return statement.
        """
        pass
    
    @abstractmethod
    def translate_expression_statement(
        self,
        expr: str,
        indent: int = 0
    ) -> str:
        """Translate an expression statement.
        
        Args:
            expr: Expression code (already translated).
            indent: Indentation level.
            
        Returns:
            Target language expression statement.
        """
        pass
    
    # -------------------------------------------------------------------------
    # Virtual methods for control flow (override in subclasses)
    # -------------------------------------------------------------------------
    
    def translate_if_statement(
        self,
        condition: str,
        then_block: List[Dict[str, Any]],
        else_block: Optional[List[Dict[str, Any]]],
        elif_blocks: List[Dict[str, Any]],
        indent: int = 0
    ) -> str:
        """Translate an if statement. Override in subclasses.
        
        Args:
            condition: Translated condition expression.
            then_block: List of then-branch statements.
            else_block: Optional list of else-branch statements.
            elif_blocks: List of elif branches with 'condition' and 'body'.
            indent: Indentation level.
            
        Returns:
            Target language if statement.
        """
        ind = self.get_indent(indent)
        lines = [f"{ind}if ({condition}) {{"]
        lines.append(self.translate_statements(then_block, indent + 1))
        
        for elif_branch in elif_blocks:
            elif_cond = self._translate_value(elif_branch.get('condition'))
            elif_body = elif_branch.get('body', [])
            lines.append(f"{ind}}} else if ({elif_cond}) {{")
            lines.append(self.translate_statements(elif_body, indent + 1))
        
        if else_block:
            lines.append(f"{ind}}} else {{")
            lines.append(self.translate_statements(else_block, indent + 1))
        
        lines.append(f"{ind}}}")
        return '\n'.join(lines)
    
    def translate_while_loop(
        self,
        condition: str,
        body: List[Dict[str, Any]],
        indent: int = 0
    ) -> str:
        """Translate a while loop. Override in subclasses.
        
        Args:
            condition: Translated condition expression.
            body: List of body statements.
            indent: Indentation level.
            
        Returns:
            Target language while loop.
        """
        ind = self.get_indent(indent)
        lines = [f"{ind}while ({condition}) {{"]
        lines.append(self.translate_statements(body, indent + 1))
        lines.append(f"{ind}}}")
        return '\n'.join(lines)
    
    def translate_for_loop(
        self,
        init: str,
        condition: str,
        update: str,
        body: List[Dict[str, Any]],
        indent: int = 0
    ) -> str:
        """Translate a C-style for loop. Override in subclasses.
        
        Args:
            init: Translated initialization expression.
            condition: Translated condition expression.
            update: Translated update expression.
            body: List of body statements.
            indent: Indentation level.
            
        Returns:
            Target language for loop.
        """
        ind = self.get_indent(indent)
        lines = [f"{ind}for ({init}; {condition}; {update}) {{"]
        lines.append(self.translate_statements(body, indent + 1))
        lines.append(f"{ind}}}")
        return '\n'.join(lines)
    
    def translate_for_each(
        self,
        var_name: str,
        var_type: Optional[str],
        iterable: str,
        body: List[Dict[str, Any]],
        indent: int = 0
    ) -> str:
        """Translate a for-each loop. Override in subclasses.
        
        Args:
            var_name: Loop variable name.
            var_type: Optional type for the loop variable.
            iterable: Translated iterable expression.
            body: List of body statements.
            indent: Indentation level.
            
        Returns:
            Target language for-each loop.
        """
        ind = self.get_indent(indent)
        lines = [f"{ind}for ({var_name} in {iterable}) {{"]
        lines.append(self.translate_statements(body, indent + 1))
        lines.append(f"{ind}}}")
        return '\n'.join(lines)
    
    def translate_for_range(
        self,
        var_name: str,
        start: str,
        end: str,
        step: Optional[str],
        body: List[Dict[str, Any]],
        indent: int = 0
    ) -> str:
        """Translate a range-based for loop. Override in subclasses.
        
        Args:
            var_name: Loop variable name.
            start: Translated start value.
            end: Translated end value.
            step: Optional translated step value.
            body: List of body statements.
            indent: Indentation level.
            
        Returns:
            Target language range-based loop.
        """
        # Default: convert to C-style for loop
        update = f"{var_name} += {step}" if step else f"{var_name}++"
        return self.translate_for_loop(
            f"{var_name} = {start}",
            f"{var_name} < {end}",
            update,
            body,
            indent
        )
    
    def translate_do_while(
        self,
        condition: str,
        body: List[Dict[str, Any]],
        indent: int = 0
    ) -> str:
        """Translate a do-while loop. Override in subclasses.
        
        Args:
            condition: Translated condition expression.
            body: List of body statements.
            indent: Indentation level.
            
        Returns:
            Target language do-while loop.
        """
        ind = self.get_indent(indent)
        lines = [f"{ind}do {{"]
        lines.append(self.translate_statements(body, indent + 1))
        lines.append(f"{ind}}} while ({condition}){self.STATEMENT_TERMINATOR}")
        return '\n'.join(lines)
    
    def translate_switch_statement(
        self,
        value: str,
        cases: List[Dict[str, Any]],
        default: Optional[List[Dict[str, Any]]],
        indent: int = 0
    ) -> str:
        """Translate a switch/match statement. Override in subclasses.
        
        Args:
            value: Translated switch value.
            cases: List of case branches with 'value' and 'body'.
            default: Optional default case statements.
            indent: Indentation level.
            
        Returns:
            Target language switch statement.
        """
        ind = self.get_indent(indent)
        case_ind = self.get_indent(indent + 1)
        
        lines = [f"{ind}switch ({value}) {{"]
        
        for case in cases:
            case_val = self._translate_value(case.get('value'))
            case_body = case.get('body', [])
            fallthrough = case.get('fallthrough', False)
            
            lines.append(f"{case_ind}case {case_val}:")
            lines.append(self.translate_statements(case_body, indent + 2))
            if not fallthrough:
                lines.append(f"{self.get_indent(indent + 2)}break{self.STATEMENT_TERMINATOR}")
        
        if default:
            lines.append(f"{case_ind}default:")
            lines.append(self.translate_statements(default, indent + 2))
        
        lines.append(f"{ind}}}")
        return '\n'.join(lines)
    
    def translate_block(
        self,
        statements: List[Dict[str, Any]],
        indent: int = 0
    ) -> str:
        """Translate a block of statements. Override in subclasses.
        
        Args:
            statements: List of statements in the block.
            indent: Indentation level.
            
        Returns:
            Target language block.
        """
        ind = self.get_indent(indent)
        lines = [f"{ind}{{"]
        lines.append(self.translate_statements(statements, indent + 1))
        lines.append(f"{ind}}}")
        return '\n'.join(lines)
    
    def translate_infinite_loop(
        self,
        body: List[Dict[str, Any]],
        indent: int = 0
    ) -> str:
        """Translate an infinite loop. Override in subclasses.
        
        Args:
            body: List of body statements.
            indent: Indentation level.
            
        Returns:
            Target language infinite loop.
        """
        # Default: while(true)
        return self.translate_while_loop("true", body, indent)
    
    # -------------------------------------------------------------------------
    # Virtual methods (can be overridden by subclasses)
    # -------------------------------------------------------------------------
    
    def _empty_body(self, indent: int = 0) -> str:
        """Generate an empty body placeholder.
        
        Args:
            indent: Indentation level.
            
        Returns:
            Target language empty body code.
        """
        ind = self.get_indent(indent)
        return f"{ind}pass"  # Python default, override in other languages
    
    def _translate_none(self) -> str:
        """Translate None/null value."""
        return "None"  # Override for target-specific null
    
    def _translate_bool(self, value: bool) -> str:
        """Translate boolean value."""
        return "True" if value else "False"
    
    def _translate_list(self, value: list) -> str:
        """Translate a list literal."""
        items = [self._translate_value(v) for v in value]
        return f"[{', '.join(items)}]"
    
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
