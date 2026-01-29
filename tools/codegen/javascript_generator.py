#!/usr/bin/env python3
"""STUNIR JavaScript Code Generator.

This module implements JavaScript-specific code generation by extending the
statement and expression translators.

Part of Phase 3 (Advanced Code Generation) of STUNIR Enhancement Integration.

Features:
- Modern ES6+ JavaScript generation
- Arrow functions and template literals
- Destructuring support
- JSDoc comments for type hints
- Full control flow support

Usage:
    from tools.codegen.javascript_generator import JavaScriptCodeGenerator
    
    generator = JavaScriptCodeGenerator(enhancement_context=ctx)
    code = generator.generate_function(func_ir)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .statement_translator import StatementTranslator
from .expression_translator import ExpressionTranslator

if TYPE_CHECKING:
    from tools.integration import EnhancementContext

logger = logging.getLogger(__name__)


class JavaScriptExpressionTranslator(ExpressionTranslator):
    """JavaScript-specific expression translator."""
    
    TARGET = 'javascript'
    LOGICAL_AND = '&&'
    LOGICAL_OR = '||'
    LOGICAL_NOT = '!'
    
    # JavaScript type mapping (for JSDoc)
    TYPE_MAP = {
        'i8': 'number',
        'i16': 'number',
        'i32': 'number',
        'i64': 'number',
        'u8': 'number',
        'u16': 'number',
        'u32': 'number',
        'u64': 'number',
        'f32': 'number',
        'f64': 'number',
        'bool': 'boolean',
        'string': 'string',
        'void': 'void',
        'unit': 'void',
        'char': 'string',
        'byte': 'number',
        'usize': 'number',
        'isize': 'number',
        'any': 'any',
        'object': 'Object',
        'array': 'Array',
    }
    
    def _build_operator_map(self) -> Dict[str, str]:
        """Build JavaScript-specific operator mappings."""
        base = super()._build_operator_map()
        base['//'] = 'Math.floor(/) '  # Note: JS doesn't have integer division
        base['**'] = '**'  # ES7 power operator
        base['and'] = '&&'
        base['&&'] = '&&'
        base['or'] = '||'
        base['||'] = '||'
        base['not'] = '!'
        base['!'] = '!'
        base['==='] = '==='
        base['!=='] = '!=='
        return base
    
    def translate_literal(self, value: Any, lit_type: str) -> str:
        """Translate a literal value to JavaScript."""
        if value is None:
            return 'null'
        
        if lit_type == 'void':
            return 'undefined'
        
        if lit_type == 'bool' or isinstance(value, bool):
            return 'true' if value else 'false'
        
        if lit_type == 'string' or isinstance(value, str):
            if isinstance(value, str):
                # Use template literals for strings with special chars
                if '\n' in value or '${' in value:
                    escaped = value.replace('`', '\\`').replace('${', '\\${')
                    return f'`{escaped}`'
                escaped = value.replace('\\', '\\\\').replace("'", "\\'")
                return f"'{escaped}'"
            return f"'{value}'"
        
        if isinstance(value, (int, float)):
            return str(value)
        
        return str(value)
    
    def translate_variable(self, name: str, var_type: Optional[str] = None) -> str:
        """Translate a variable reference to JavaScript."""
        return name
    
    def translate_binary_op(self, left: str, op: str, right: str) -> str:
        """Translate a binary operation to JavaScript."""
        # Handle integer division specially
        if 'Math.floor' in op:
            return f"Math.floor({left} / {right})"
        return f"({left} {op} {right})"
    
    def translate_unary_op(self, op: str, operand: str) -> str:
        """Translate a unary operation to JavaScript."""
        return f"({op}{operand})"
    
    def translate_function_call(
        self,
        func_name: str,
        args: List[str],
        receiver: Optional[str] = None
    ) -> str:
        """Translate a function call to JavaScript."""
        args_str = ', '.join(args)
        if receiver:
            return f"{receiver}.{func_name}({args_str})"
        return f"{func_name}({args_str})"
    
    def translate_ternary(self, condition: str, then_val: str, else_val: str) -> str:
        """Translate a ternary expression to JavaScript."""
        return f"({condition} ? {then_val} : {else_val})"
    
    def translate_cast(self, value: str, target_type: str) -> str:
        """Translate a type cast to JavaScript."""
        mapped = self.map_type(target_type)
        if mapped == 'number':
            return f"Number({value})"
        if mapped == 'string':
            return f"String({value})"
        if mapped == 'boolean':
            return f"Boolean({value})"
        return value
    
    def translate_struct_literal(
        self,
        struct_type: str,
        fields: Dict[str, str]
    ) -> str:
        """Translate a struct literal to JavaScript object."""
        field_strs = [f"{k}: {v}" for k, v in fields.items()]
        return f"{{ {', '.join(field_strs)} }}"
    
    def translate_lambda(
        self,
        params: List[Dict[str, Any]],
        body: Any,
        return_type: Optional[str] = None
    ) -> str:
        """Translate a lambda to JavaScript arrow function."""
        param_names = [p.get('name', 'x') if isinstance(p, dict) else p for p in params]
        params_str = ', '.join(param_names)
        
        if isinstance(body, dict):
            body_code = self.translate_expression(body)
            return f"({params_str}) => {body_code}"
        elif isinstance(body, list):
            return f"({params_str}) => {{ /* ... */ }}"
        else:
            return f"({params_str}) => {body}"
    
    def translate_new(self, class_name: str, args: List[str]) -> str:
        """Translate new expression to JavaScript."""
        args_str = ', '.join(args)
        return f"new {class_name}({args_str})"
    
    def translate_typeof(self, target: str) -> str:
        """Translate typeof expression to JavaScript."""
        return f"typeof {target}"
    
    def translate_spread(self, target: str) -> str:
        """Translate spread operator to JavaScript."""
        return f"...{target}"
    
    def translate_await(self, target: str) -> str:
        """Translate await expression to JavaScript."""
        return f"await {target}"
    
    def map_type(self, ir_type: str) -> str:
        """Map IR type to JavaScript type (for JSDoc)."""
        return self.TYPE_MAP.get(ir_type, ir_type)


class JavaScriptStatementTranslator(StatementTranslator):
    """JavaScript-specific statement translator."""
    
    TARGET = 'javascript'
    STATEMENT_TERMINATOR = ';'
    
    TYPE_MAP = JavaScriptExpressionTranslator.TYPE_MAP
    
    def __init__(
        self,
        enhancement_context: Optional['EnhancementContext'] = None,
        indent_size: int = 2,
        indent_char: str = ' ',
        use_const: bool = True,
        use_semicolons: bool = True
    ):
        """Initialize JavaScript statement translator."""
        super().__init__(enhancement_context, indent_size, indent_char)
        self.use_const = use_const
        self.use_semicolons = use_semicolons
        if not use_semicolons:
            self.STATEMENT_TERMINATOR = ''
    
    def translate_variable_declaration(
        self,
        var_name: str,
        var_type: str,
        init_value: Optional[str],
        mutable: bool = True,
        indent: int = 0
    ) -> str:
        """Translate a variable declaration to JavaScript."""
        ind = self.get_indent(indent)
        term = self.STATEMENT_TERMINATOR
        
        keyword = 'let' if mutable else 'const'
        if self.use_const and not mutable:
            keyword = 'const'
        
        if init_value is not None:
            return f"{ind}{keyword} {var_name} = {init_value}{term}"
        else:
            return f"{ind}{keyword} {var_name}{term}"
    
    def translate_assignment(
        self,
        target: str,
        value: str,
        indent: int = 0
    ) -> str:
        """Translate an assignment to JavaScript."""
        ind = self.get_indent(indent)
        term = self.STATEMENT_TERMINATOR
        return f"{ind}{target} = {value}{term}"
    
    def translate_compound_assignment(
        self,
        target: str,
        op: str,
        value: str,
        indent: int = 0
    ) -> str:
        """Translate a compound assignment to JavaScript."""
        ind = self.get_indent(indent)
        term = self.STATEMENT_TERMINATOR
        return f"{ind}{target} {op} {value}{term}"
    
    def translate_return(
        self,
        value: Optional[str],
        indent: int = 0
    ) -> str:
        """Translate a return statement to JavaScript."""
        ind = self.get_indent(indent)
        term = self.STATEMENT_TERMINATOR
        if value is None:
            return f"{ind}return{term}"
        return f"{ind}return {value}{term}"
    
    def translate_expression_statement(
        self,
        expr: str,
        indent: int = 0
    ) -> str:
        """Translate an expression statement to JavaScript."""
        ind = self.get_indent(indent)
        term = self.STATEMENT_TERMINATOR
        return f"{ind}{expr}{term}"
    
    def _empty_body(self, indent: int = 0) -> str:
        """Generate JavaScript empty body."""
        ind = self.get_indent(indent)
        return f"{ind}// empty"
    
    def _translate_none(self) -> str:
        """Translate None value for JavaScript."""
        return "null"
    
    def _translate_bool(self, value: bool) -> str:
        """Translate boolean value for JavaScript."""
        return "true" if value else "false"
    
    def _translate_list(self, value: list) -> str:
        """Translate a list literal for JavaScript."""
        items = [self._translate_value(v) for v in value]
        return f"[{', '.join(items)}]"
    
    def map_type(self, ir_type: str) -> str:
        """Map IR type to JavaScript type."""
        return self.TYPE_MAP.get(ir_type, ir_type)
    
    # -------------------------------------------------------------------------
    # Control flow methods
    # -------------------------------------------------------------------------
    
    def translate_if_statement(
        self,
        condition: str,
        then_block: List[Dict[str, Any]],
        else_block: Optional[List[Dict[str, Any]]],
        elif_blocks: List[Dict[str, Any]],
        indent: int = 0
    ) -> str:
        """Translate an if statement to JavaScript."""
        ind = self.get_indent(indent)
        lines = [f"{ind}if ({condition}) {{"]
        
        if then_block:
            lines.append(self.translate_statements(then_block, indent + 1))
        else:
            lines.append(f"{self.get_indent(indent + 1)}// empty")
        
        for elif_branch in elif_blocks:
            elif_cond = self._translate_value(elif_branch.get('condition'))
            elif_body = elif_branch.get('body', [])
            lines.append(f"{ind}}} else if ({elif_cond}) {{")
            if elif_body:
                lines.append(self.translate_statements(elif_body, indent + 1))
            else:
                lines.append(f"{self.get_indent(indent + 1)}// empty")
        
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
        """Translate a while loop to JavaScript."""
        ind = self.get_indent(indent)
        lines = [f"{ind}while ({condition}) {{"]
        
        if body:
            lines.append(self.translate_statements(body, indent + 1))
        else:
            lines.append(f"{self.get_indent(indent + 1)}// empty")
        
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
        """Translate a C-style for loop to JavaScript."""
        ind = self.get_indent(indent)
        lines = [f"{ind}for ({init}; {condition}; {update}) {{"]
        
        if body:
            lines.append(self.translate_statements(body, indent + 1))
        else:
            lines.append(f"{self.get_indent(indent + 1)}// empty")
        
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
        """Translate a for-each loop to JavaScript for...of."""
        ind = self.get_indent(indent)
        lines = [f"{ind}for (const {var_name} of {iterable}) {{"]
        
        if body:
            lines.append(self.translate_statements(body, indent + 1))
        else:
            lines.append(f"{self.get_indent(indent + 1)}// empty")
        
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
        """Translate a range-based for loop to JavaScript."""
        ind = self.get_indent(indent)
        
        if step:
            update = f"{var_name} += {step}"
        else:
            update = f"{var_name}++"
        
        lines = [f"{ind}for (let {var_name} = {start}; {var_name} < {end}; {update}) {{"]
        
        if body:
            lines.append(self.translate_statements(body, indent + 1))
        else:
            lines.append(f"{self.get_indent(indent + 1)}// empty")
        
        lines.append(f"{ind}}}")
        return '\n'.join(lines)
    
    def translate_switch_statement(
        self,
        value: str,
        cases: List[Dict[str, Any]],
        default: Optional[List[Dict[str, Any]]],
        indent: int = 0
    ) -> str:
        """Translate a switch statement to JavaScript."""
        ind = self.get_indent(indent)
        case_ind = self.get_indent(indent + 1)
        
        lines = [f"{ind}switch ({value}) {{"]
        
        for case in cases:
            case_val = self._translate_value(case.get('value'))
            case_body = case.get('body', [])
            fallthrough = case.get('fallthrough', False)
            
            lines.append(f"{case_ind}case {case_val}:")
            if case_body:
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
        """Translate a block of statements to JavaScript."""
        ind = self.get_indent(indent)
        lines = [f"{ind}{{"]
        if statements:
            lines.append(self.translate_statements(statements, indent + 1))
        else:
            lines.append(f"{self.get_indent(indent + 1)}// empty")
        lines.append(f"{ind}}}")
        return '\n'.join(lines)
    
    def translate_infinite_loop(
        self,
        body: List[Dict[str, Any]],
        indent: int = 0
    ) -> str:
        """Translate an infinite loop to JavaScript while(true)."""
        return self.translate_while_loop("true", body, indent)


class JavaScriptCodeGenerator:
    """Complete JavaScript code generator."""
    
    TARGET = 'javascript'
    FILE_EXTENSION = 'js'
    
    def __init__(
        self,
        enhancement_context: Optional['EnhancementContext'] = None,
        indent_size: int = 2,
        use_const: bool = True,
        use_semicolons: bool = True
    ):
        """Initialize the JavaScript code generator."""
        self.enhancement_context = enhancement_context
        self.indent_size = indent_size
        self.use_const = use_const
        self.use_semicolons = use_semicolons
        
        self.expr_translator = JavaScriptExpressionTranslator(enhancement_context)
        self.stmt_translator = JavaScriptStatementTranslator(
            enhancement_context,
            indent_size=indent_size,
            use_const=use_const,
            use_semicolons=use_semicolons
        )
        
        self.stmt_translator.set_expression_translator(self.expr_translator)
    
    def generate_function(self, func_ir: Dict[str, Any]) -> str:
        """Generate JavaScript function code from IR."""
        name = func_ir.get('name', 'unnamed')
        params = func_ir.get('params', [])
        return_type = func_ir.get('return_type', 'void')
        body = func_ir.get('body', [])
        is_async = func_ir.get('async', False)
        is_arrow = func_ir.get('arrow', False)
        docstring = func_ir.get('docstring', func_ir.get('description'))
        
        lines = []
        indent = ' ' * self.indent_size
        term = ';' if self.use_semicolons else ''
        
        # JSDoc comment
        if docstring or params or return_type != 'void':
            lines.append('/**')
            if docstring:
                lines.append(f' * {docstring}')
            for param in params:
                if isinstance(param, dict):
                    p_name = param.get('name', 'arg')
                    p_type = self.stmt_translator.map_type(param.get('type', 'any'))
                    lines.append(f' * @param {{{p_type}}} {p_name}')
            if return_type != 'void':
                ret_type = self.stmt_translator.map_type(return_type)
                lines.append(f' * @returns {{{ret_type}}}')
            lines.append(' */')
        
        # Function signature
        params_str = self._format_params(params)
        async_prefix = 'async ' if is_async else ''
        
        if is_arrow:
            lines.append(f"const {name} = {async_prefix}({params_str}) => {{")
        else:
            lines.append(f"{async_prefix}function {name}({params_str}) {{")
        
        # Body
        if body:
            body_code = self.stmt_translator.translate_statements(body, indent=1)
            lines.append(body_code)
        else:
            lines.append(f"{indent}// empty")
        
        lines.append(f"}}{term}")
        
        return '\n'.join(lines)
    
    def generate_module(self, module_ir: Dict[str, Any]) -> str:
        """Generate JavaScript module code from IR."""
        lines = []
        indent = ' ' * self.indent_size
        
        module_name = module_ir.get('name', module_ir.get('ir_module', 'module'))
        lines.append(f"/**")
        lines.append(f" * @fileoverview Module: {module_name}")
        lines.append(f" * Generated by STUNIR JavaScript Code Generator")
        lines.append(f" */")
        lines.append('')
        lines.append("'use strict';")
        lines.append('')
        
        # Imports
        imports = module_ir.get('imports', [])
        for imp in imports:
            if isinstance(imp, dict):
                mod = imp.get('module', '')
                items = imp.get('items', [])
                default = imp.get('default')
                if default:
                    lines.append(f"import {default} from '{mod}';")
                elif items:
                    items_str = ', '.join(items)
                    lines.append(f"import {{ {items_str} }} from '{mod}';")
                else:
                    lines.append(f"import '{mod}';")
            else:
                lines.append(f"import '{imp}';")
        
        if imports:
            lines.append('')
        
        # Functions
        functions = module_ir.get('functions', module_ir.get('ir_functions', []))
        for func in functions:
            lines.append(self.generate_function(func))
            lines.append('')
        
        # Exports
        exports = module_ir.get('exports', module_ir.get('ir_exports', []))
        if exports:
            export_names = [e.get('name', e) if isinstance(e, dict) else e for e in exports]
            lines.append(f"export {{ {', '.join(export_names)} }};")
        
        return '\n'.join(lines)
    
    def _format_params(self, params: List[Dict[str, Any]]) -> str:
        """Format function parameters."""
        formatted = []
        for param in params:
            if isinstance(param, dict):
                name = param.get('name', 'arg')
                default = param.get('default')
                if default is not None:
                    default_str = self.expr_translator.translate_expression(default)
                    formatted.append(f"{name} = {default_str}")
                else:
                    formatted.append(name)
            else:
                formatted.append(str(param))
        return ', '.join(formatted)
