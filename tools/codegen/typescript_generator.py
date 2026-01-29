#!/usr/bin/env python3
"""STUNIR TypeScript Code Generator.

This module implements TypeScript-specific code generation by extending the
statement and expression translators.

Part of Phase 3 (Advanced Code Generation) of STUNIR Enhancement Integration.

Features:
- Full TypeScript type system
- Interface generation
- Generic type support
- Type annotations on all declarations
- Modern ES6+ features

Usage:
    from tools.codegen.typescript_generator import TypeScriptCodeGenerator
    
    generator = TypeScriptCodeGenerator(enhancement_context=ctx)
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


class TypeScriptExpressionTranslator(ExpressionTranslator):
    """TypeScript-specific expression translator."""
    
    TARGET = 'typescript'
    LOGICAL_AND = '&&'
    LOGICAL_OR = '||'
    LOGICAL_NOT = '!'
    
    # TypeScript type mapping
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
        'unknown': 'unknown',
        'never': 'never',
        'object': 'object',
        'array': 'Array<any>',
    }
    
    def _build_operator_map(self) -> Dict[str, str]:
        """Build TypeScript-specific operator mappings."""
        base = super()._build_operator_map()
        base['//'] = 'Math.floor(/) '
        base['**'] = '**'
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
        """Translate a literal value to TypeScript."""
        if value is None:
            return 'null'
        
        if lit_type == 'void':
            return 'undefined'
        
        if lit_type == 'bool' or isinstance(value, bool):
            return 'true' if value else 'false'
        
        if lit_type == 'string' or isinstance(value, str):
            if isinstance(value, str):
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
        """Translate a variable reference to TypeScript."""
        return name
    
    def translate_binary_op(self, left: str, op: str, right: str) -> str:
        """Translate a binary operation to TypeScript."""
        if 'Math.floor' in op:
            return f"Math.floor({left} / {right})"
        return f"({left} {op} {right})"
    
    def translate_unary_op(self, op: str, operand: str) -> str:
        """Translate a unary operation to TypeScript."""
        return f"({op}{operand})"
    
    def translate_function_call(
        self,
        func_name: str,
        args: List[str],
        receiver: Optional[str] = None
    ) -> str:
        """Translate a function call to TypeScript."""
        args_str = ', '.join(args)
        if receiver:
            return f"{receiver}.{func_name}({args_str})"
        return f"{func_name}({args_str})"
    
    def translate_ternary(self, condition: str, then_val: str, else_val: str) -> str:
        """Translate a ternary expression to TypeScript."""
        return f"({condition} ? {then_val} : {else_val})"
    
    def translate_cast(self, value: str, target_type: str) -> str:
        """Translate a type cast to TypeScript (as expression)."""
        mapped = self.map_type(target_type)
        return f"({value} as {mapped})"
    
    def translate_struct_literal(
        self,
        struct_type: str,
        fields: Dict[str, str]
    ) -> str:
        """Translate a struct literal to TypeScript object."""
        field_strs = [f"{k}: {v}" for k, v in fields.items()]
        return f"{{ {', '.join(field_strs)} }}"
    
    def translate_lambda(
        self,
        params: List[Dict[str, Any]],
        body: Any,
        return_type: Optional[str] = None
    ) -> str:
        """Translate a lambda to TypeScript arrow function."""
        param_parts = []
        for p in params:
            if isinstance(p, dict):
                name = p.get('name', 'x')
                p_type = self.map_type(p.get('type', 'any'))
                param_parts.append(f"{name}: {p_type}")
            else:
                param_parts.append(str(p))
        params_str = ', '.join(param_parts)
        
        ret_annotation = ''
        if return_type:
            ret_annotation = f": {self.map_type(return_type)}"
        
        if isinstance(body, dict):
            body_code = self.translate_expression(body)
            return f"({params_str}){ret_annotation} => {body_code}"
        elif isinstance(body, list):
            return f"({params_str}){ret_annotation} => {{ /* ... */ }}"
        else:
            return f"({params_str}){ret_annotation} => {body}"
    
    def translate_new(self, class_name: str, args: List[str]) -> str:
        """Translate new expression to TypeScript."""
        args_str = ', '.join(args)
        return f"new {class_name}({args_str})"
    
    def translate_typeof(self, target: str) -> str:
        """Translate typeof expression to TypeScript."""
        return f"typeof {target}"
    
    def translate_spread(self, target: str) -> str:
        """Translate spread operator to TypeScript."""
        return f"...{target}"
    
    def translate_await(self, target: str) -> str:
        """Translate await expression to TypeScript."""
        return f"await {target}"
    
    def map_type(self, ir_type: str) -> str:
        """Map IR type to TypeScript type."""
        # Handle array types
        if ir_type.endswith('[]'):
            base = ir_type[:-2]
            mapped_base = self.TYPE_MAP.get(base, base)
            return f"{mapped_base}[]"
        # Handle generic types
        if '<' in ir_type:
            return ir_type  # Pass through complex types
        return self.TYPE_MAP.get(ir_type, ir_type)


class TypeScriptStatementTranslator(StatementTranslator):
    """TypeScript-specific statement translator."""
    
    TARGET = 'typescript'
    STATEMENT_TERMINATOR = ';'
    
    TYPE_MAP = TypeScriptExpressionTranslator.TYPE_MAP
    
    def __init__(
        self,
        enhancement_context: Optional['EnhancementContext'] = None,
        indent_size: int = 2,
        indent_char: str = ' ',
        use_const: bool = True,
        use_semicolons: bool = True,
        strict: bool = True
    ):
        """Initialize TypeScript statement translator."""
        super().__init__(enhancement_context, indent_size, indent_char)
        self.use_const = use_const
        self.use_semicolons = use_semicolons
        self.strict = strict
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
        """Translate a variable declaration to TypeScript."""
        ind = self.get_indent(indent)
        term = self.STATEMENT_TERMINATOR
        
        keyword = 'let' if mutable else 'const'
        mapped_type = self.map_type(var_type)
        
        if init_value is not None:
            return f"{ind}{keyword} {var_name}: {mapped_type} = {init_value}{term}"
        else:
            if mapped_type == 'any':
                return f"{ind}{keyword} {var_name}{term}"
            return f"{ind}{keyword} {var_name}: {mapped_type}{term}"
    
    def translate_assignment(
        self,
        target: str,
        value: str,
        indent: int = 0
    ) -> str:
        """Translate an assignment to TypeScript."""
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
        """Translate a compound assignment to TypeScript."""
        ind = self.get_indent(indent)
        term = self.STATEMENT_TERMINATOR
        return f"{ind}{target} {op} {value}{term}"
    
    def translate_return(
        self,
        value: Optional[str],
        indent: int = 0
    ) -> str:
        """Translate a return statement to TypeScript."""
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
        """Translate an expression statement to TypeScript."""
        ind = self.get_indent(indent)
        term = self.STATEMENT_TERMINATOR
        return f"{ind}{expr}{term}"
    
    def _empty_body(self, indent: int = 0) -> str:
        """Generate TypeScript empty body."""
        ind = self.get_indent(indent)
        return f"{ind}// empty"
    
    def _translate_none(self) -> str:
        """Translate None value for TypeScript."""
        return "null"
    
    def _translate_bool(self, value: bool) -> str:
        """Translate boolean value for TypeScript."""
        return "true" if value else "false"
    
    def _translate_list(self, value: list) -> str:
        """Translate a list literal for TypeScript."""
        items = [self._translate_value(v) for v in value]
        return f"[{', '.join(items)}]"
    
    def map_type(self, ir_type: str) -> str:
        """Map IR type to TypeScript type."""
        if ir_type.endswith('[]'):
            base = ir_type[:-2]
            mapped_base = self.TYPE_MAP.get(base, base)
            return f"{mapped_base}[]"
        if '<' in ir_type:
            return ir_type
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
        """Translate an if statement to TypeScript."""
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
        """Translate a while loop to TypeScript."""
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
        """Translate a C-style for loop to TypeScript."""
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
        """Translate a for-each loop to TypeScript for...of."""
        ind = self.get_indent(indent)
        
        if var_type:
            mapped_type = self.map_type(var_type)
            lines = [f"{ind}for (const {var_name}: {mapped_type} of {iterable}) {{"]
        else:
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
        """Translate a range-based for loop to TypeScript."""
        ind = self.get_indent(indent)
        
        if step:
            update = f"{var_name} += {step}"
        else:
            update = f"{var_name}++"
        
        lines = [f"{ind}for (let {var_name}: number = {start}; {var_name} < {end}; {update}) {{"]
        
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
        """Translate a switch statement to TypeScript."""
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
        """Translate a block of statements to TypeScript."""
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
        """Translate an infinite loop to TypeScript while(true)."""
        return self.translate_while_loop("true", body, indent)


class TypeScriptCodeGenerator:
    """Complete TypeScript code generator."""
    
    TARGET = 'typescript'
    FILE_EXTENSION = 'ts'
    
    def __init__(
        self,
        enhancement_context: Optional['EnhancementContext'] = None,
        indent_size: int = 2,
        use_const: bool = True,
        use_semicolons: bool = True,
        strict: bool = True
    ):
        """Initialize the TypeScript code generator."""
        self.enhancement_context = enhancement_context
        self.indent_size = indent_size
        self.use_const = use_const
        self.use_semicolons = use_semicolons
        self.strict = strict
        
        self.expr_translator = TypeScriptExpressionTranslator(enhancement_context)
        self.stmt_translator = TypeScriptStatementTranslator(
            enhancement_context,
            indent_size=indent_size,
            use_const=use_const,
            use_semicolons=use_semicolons,
            strict=strict
        )
        
        self.stmt_translator.set_expression_translator(self.expr_translator)
    
    def generate_function(self, func_ir: Dict[str, Any]) -> str:
        """Generate TypeScript function code from IR."""
        name = func_ir.get('name', 'unnamed')
        params = func_ir.get('params', [])
        return_type = func_ir.get('return_type', 'void')
        body = func_ir.get('body', [])
        is_async = func_ir.get('async', False)
        is_export = func_ir.get('export', True)
        docstring = func_ir.get('docstring', func_ir.get('description'))
        generics = func_ir.get('generics', [])
        
        lines = []
        indent = ' ' * self.indent_size
        term = ';' if self.use_semicolons else ''
        
        # JSDoc comment
        if docstring:
            lines.append('/**')
            lines.append(f' * {docstring}')
            lines.append(' */')
        
        # Function signature
        params_str = self._format_params(params)
        ret_type_str = self.stmt_translator.map_type(return_type)
        async_prefix = 'async ' if is_async else ''
        export_prefix = 'export ' if is_export else ''
        
        # Generic type parameters
        generic_str = ''
        if generics:
            generic_str = f"<{', '.join(generics)}>"
        
        lines.append(f"{export_prefix}{async_prefix}function {name}{generic_str}({params_str}): {ret_type_str} {{")
        
        # Body
        if body:
            body_code = self.stmt_translator.translate_statements(body, indent=1)
            lines.append(body_code)
        else:
            lines.append(f"{indent}// empty")
        
        lines.append(f"}}")
        
        return '\n'.join(lines)
    
    def generate_interface(self, type_def: Dict[str, Any]) -> str:
        """Generate TypeScript interface from type definition."""
        name = type_def.get('name', 'IType')
        fields = type_def.get('fields', [])
        extends = type_def.get('extends', [])
        is_export = type_def.get('export', True)
        
        lines = []
        indent = ' ' * self.indent_size
        
        export_prefix = 'export ' if is_export else ''
        extends_clause = ''
        if extends:
            extends_clause = f" extends {', '.join(extends)}"
        
        lines.append(f"{export_prefix}interface {name}{extends_clause} {{")
        
        for field in fields:
            if isinstance(field, dict):
                f_name = field.get('name', 'field')
                f_type = self.stmt_translator.map_type(field.get('type', 'any'))
                optional = '?' if field.get('optional', False) else ''
                readonly = 'readonly ' if field.get('readonly', False) else ''
                lines.append(f"{indent}{readonly}{f_name}{optional}: {f_type};")
            else:
                lines.append(f"{indent}{field}: any;")
        
        lines.append("}")
        
        return '\n'.join(lines)
    
    def generate_module(self, module_ir: Dict[str, Any]) -> str:
        """Generate TypeScript module code from IR."""
        lines = []
        indent = ' ' * self.indent_size
        
        module_name = module_ir.get('name', module_ir.get('ir_module', 'module'))
        lines.append(f"/**")
        lines.append(f" * @fileoverview Module: {module_name}")
        lines.append(f" * Generated by STUNIR TypeScript Code Generator")
        lines.append(f" */")
        lines.append('')
        
        # Imports
        imports = module_ir.get('imports', [])
        for imp in imports:
            if isinstance(imp, dict):
                mod = imp.get('module', '')
                items = imp.get('items', [])
                default = imp.get('default')
                type_only = 'type ' if imp.get('type_only', False) else ''
                if default and items:
                    items_str = ', '.join(items)
                    lines.append(f"import {type_only}{default}, {{ {items_str} }} from '{mod}';")
                elif default:
                    lines.append(f"import {type_only}{default} from '{mod}';")
                elif items:
                    items_str = ', '.join(items)
                    lines.append(f"import {type_only}{{ {items_str} }} from '{mod}';")
                else:
                    lines.append(f"import '{mod}';")
            else:
                lines.append(f"import '{imp}';")
        
        if imports:
            lines.append('')
        
        # Types/Interfaces
        types = module_ir.get('types', module_ir.get('ir_types', []))
        for type_def in types:
            lines.append(self.generate_interface(type_def))
            lines.append('')
        
        # Functions
        functions = module_ir.get('functions', module_ir.get('ir_functions', []))
        for func in functions:
            lines.append(self.generate_function(func))
            lines.append('')
        
        return '\n'.join(lines)
    
    def _format_params(self, params: List[Dict[str, Any]]) -> str:
        """Format function parameters with types."""
        formatted = []
        for param in params:
            if isinstance(param, dict):
                name = param.get('name', 'arg')
                p_type = self.stmt_translator.map_type(param.get('type', 'any'))
                optional = '?' if param.get('optional', False) else ''
                default = param.get('default')
                if default is not None:
                    default_str = self.expr_translator.translate_expression(default)
                    formatted.append(f"{name}{optional}: {p_type} = {default_str}")
                else:
                    formatted.append(f"{name}{optional}: {p_type}")
            else:
                formatted.append(str(param))
        return ', '.join(formatted)
