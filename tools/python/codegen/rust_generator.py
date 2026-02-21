#!/usr/bin/env python3
"""STUNIR Rust Code Generator.

This module implements Rust-specific code generation by extending the
statement and expression translators.

Part of Phase 3 (Advanced Code Generation) of STUNIR Enhancement Integration.

Features:
- Proper ownership and borrowing patterns
- Mutability handling (let mut vs let)
- Rust-specific type mappings
- Support for all statement types including control flow
- Support for all expression types including complex expressions
- Rust match expression support

Usage:
    from tools.codegen.rust_generator import RustCodeGenerator
    
    generator = RustCodeGenerator(enhancement_context=ctx)
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


class RustExpressionTranslator(ExpressionTranslator):
    """Rust-specific expression translator."""
    
    TARGET = 'rust'
    LOGICAL_AND = '&&'
    LOGICAL_OR = '||'
    LOGICAL_NOT = '!'
    
    # Rust type mapping
    TYPE_MAP = {
        'i8': 'i8',
        'i16': 'i16',
        'i32': 'i32',
        'i64': 'i64',
        'u8': 'u8',
        'u16': 'u16',
        'u32': 'u32',
        'u64': 'u64',
        'f32': 'f32',
        'f64': 'f64',
        'bool': 'bool',
        'string': 'String',
        'str': '&str',
        'void': '()',
        'unit': '()',
        'char': 'char',
        'byte': 'u8',
        'usize': 'usize',
        'isize': 'isize',
    }
    
    def _build_operator_map(self) -> Dict[str, str]:
        """Build Rust-specific operator mappings."""
        base = super()._build_operator_map()
        # Rust-specific overrides
        base['//'] = '/'  # Integer division (Rust uses / for both)
        base['**'] = '.pow'  # Power (method call in Rust)
        base['and'] = '&&'
        base['&&'] = '&&'
        base['or'] = '||'
        base['||'] = '||'
        base['not'] = '!'
        base['!'] = '!'
        return base
    
    def translate_literal(self, value: Any, lit_type: str) -> str:
        """Translate a literal value to Rust."""
        if value is None:
            return 'None'  # Option::None
        
        if lit_type == 'bool' or isinstance(value, bool):
            return 'true' if value else 'false'
        
        if lit_type == 'string' or isinstance(value, str):
            if isinstance(value, str):
                escaped = value.replace('\\', '\\\\').replace('"', '\\"')
                return f'"{escaped}".to_string()'
            return f'"{value}".to_string()'
        
        if lit_type == 'str':
            if isinstance(value, str):
                escaped = value.replace('\\', '\\\\').replace('"', '\\"')
                return f'"{escaped}"'
            return f'"{value}"'
        
        if lit_type == 'char':
            if isinstance(value, str) and len(value) == 1:
                return f"'{value}'"
            return f"'{value}'"
        
        if lit_type in ('f32', 'f64') or isinstance(value, float):
            suffix = '_f32' if lit_type == 'f32' else '_f64'
            return f"{float(value)}{suffix}"
        
        if lit_type in ('i8', 'i16', 'i32', 'i64', 'u8', 'u16', 'u32', 'u64'):
            return f"{int(value)}_{lit_type}"
        
        if isinstance(value, int):
            return str(value)
        
        return repr(value)
    
    def translate_variable(self, name: str, var_type: Optional[str] = None) -> str:
        """Translate a variable reference to Rust."""
        return name
    
    def translate_binary_op(self, left: str, op: str, right: str) -> str:
        """Translate a binary operation to Rust."""
        # Handle power operator specially (it's a method in Rust)
        if op == '.pow':
            return f"({left}).pow({right} as u32)"
        return f"({left} {op} {right})"
    
    def translate_unary_op(self, op: str, operand: str) -> str:
        """Translate a unary operation to Rust."""
        return f"({op}{operand})"
    
    def translate_function_call(
        self,
        func_name: str,
        args: List[str],
        receiver: Optional[str] = None
    ) -> str:
        """Translate a function call to Rust."""
        args_str = ', '.join(args)
        if receiver:
            return f"{receiver}.{func_name}({args_str})"
        return f"{func_name}({args_str})"
    
    def translate_ternary(self, condition: str, then_val: str, else_val: str) -> str:
        """Translate a ternary expression to Rust if-else expression."""
        return f"(if {condition} {{ {then_val} }} else {{ {else_val} }})"
    
    def translate_cast(self, value: str, target_type: str) -> str:
        """Translate a type cast to Rust."""
        mapped = self.map_type(target_type)
        return f"({value} as {mapped})"
    
    def translate_struct_literal(
        self,
        struct_type: str,
        fields: Dict[str, str]
    ) -> str:
        """Translate a struct literal to Rust."""
        field_strs = [f"{k}: {v}" for k, v in fields.items()]
        return f"{struct_type} {{ {', '.join(field_strs)} }}"
    
    def map_type(self, ir_type: str) -> str:
        """Map IR type to Rust type."""
        return self.TYPE_MAP.get(ir_type, ir_type)


class RustStatementTranslator(StatementTranslator):
    """Rust-specific statement translator."""
    
    TARGET = 'rust'
    STATEMENT_TERMINATOR = ';'
    
    # Rust type mapping (same as expression translator)
    TYPE_MAP = RustExpressionTranslator.TYPE_MAP
    
    def translate_variable_declaration(
        self,
        var_name: str,
        var_type: str,
        init_value: Optional[str],
        mutable: bool = True,
        indent: int = 0
    ) -> str:
        """Translate a variable declaration to Rust."""
        ind = self.get_indent(indent)
        mapped_type = self.map_type(var_type)
        
        mut_kw = 'mut ' if mutable else ''
        
        if init_value is not None:
            return f"{ind}let {mut_kw}{var_name}: {mapped_type} = {init_value};"
        else:
            # Rust requires initialization, use default or uninitialized
            default = self._default_value(mapped_type)
            return f"{ind}let {mut_kw}{var_name}: {mapped_type} = {default};"
    
    def translate_assignment(
        self,
        target: str,
        value: str,
        indent: int = 0
    ) -> str:
        """Translate an assignment to Rust."""
        ind = self.get_indent(indent)
        return f"{ind}{target} = {value};"
    
    def translate_compound_assignment(
        self,
        target: str,
        op: str,
        value: str,
        indent: int = 0
    ) -> str:
        """Translate a compound assignment to Rust."""
        ind = self.get_indent(indent)
        return f"{ind}{target} {op} {value};"
    
    def translate_return(
        self,
        value: Optional[str],
        indent: int = 0
    ) -> str:
        """Translate a return statement to Rust."""
        ind = self.get_indent(indent)
        if value is None:
            return f"{ind}return;"
        return f"{ind}return {value};"
    
    def translate_expression_statement(
        self,
        expr: str,
        indent: int = 0
    ) -> str:
        """Translate an expression statement to Rust."""
        ind = self.get_indent(indent)
        return f"{ind}{expr};"
    
    def _empty_body(self, indent: int = 0) -> str:
        """Generate Rust empty body (unit expression)."""
        ind = self.get_indent(indent)
        return f"{ind}()"
    
    def _translate_none(self) -> str:
        """Translate None value for Rust (Option::None)."""
        return "None"
    
    def _translate_bool(self, value: bool) -> str:
        """Translate boolean value for Rust."""
        return "true" if value else "false"
    
    def _translate_list(self, value: list) -> str:
        """Translate a list literal for Rust (Vec)."""
        items = [self._translate_value(v) for v in value]
        return f"vec![{', '.join(items)}]"
    
    def _default_value(self, rust_type: str) -> str:
        """Get default value for a Rust type."""
        defaults = {
            'i8': '0_i8',
            'i16': '0_i16',
            'i32': '0_i32',
            'i64': '0_i64',
            'u8': '0_u8',
            'u16': '0_u16',
            'u32': '0_u32',
            'u64': '0_u64',
            'f32': '0.0_f32',
            'f64': '0.0_f64',
            'bool': 'false',
            'String': 'String::new()',
            '&str': '""',
            'char': "' '",
            '()': '()',
        }
        return defaults.get(rust_type, 'Default::default()')
    
    def map_type(self, ir_type: str) -> str:
        """Map IR type to Rust type."""
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
        """Translate an if statement to Rust."""
        ind = self.get_indent(indent)
        lines = [f"{ind}if {condition} {{"]
        
        if then_block:
            lines.append(self.translate_statements(then_block, indent + 1))
        else:
            lines.append(f"{self.get_indent(indent + 1)}()")
        
        for elif_branch in elif_blocks:
            elif_cond = self._translate_value(elif_branch.get('condition'))
            elif_body = elif_branch.get('body', [])
            lines.append(f"{ind}}} else if {elif_cond} {{")
            if elif_body:
                lines.append(self.translate_statements(elif_body, indent + 1))
            else:
                lines.append(f"{self.get_indent(indent + 1)}()")
        
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
        """Translate a while loop to Rust."""
        ind = self.get_indent(indent)
        lines = [f"{ind}while {condition} {{"]
        
        if body:
            lines.append(self.translate_statements(body, indent + 1))
        else:
            lines.append(f"{self.get_indent(indent + 1)}()")
        
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
        """Translate a C-style for loop to Rust (using while)."""
        ind = self.get_indent(indent)
        lines = []
        
        # Rust doesn't have C-style for, convert to while with block
        lines.append(f"{ind}{{")
        if init:
            lines.append(f"{self.get_indent(indent + 1)}let mut {init};")
        
        lines.append(f"{self.get_indent(indent + 1)}while {condition} {{")
        
        if body:
            lines.append(self.translate_statements(body, indent + 2))
        
        if update:
            lines.append(f"{self.get_indent(indent + 2)}{update};")
        
        lines.append(f"{self.get_indent(indent + 1)}}}")
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
        """Translate a for-each loop to Rust."""
        ind = self.get_indent(indent)
        lines = [f"{ind}for {var_name} in {iterable} {{"]
        
        if body:
            lines.append(self.translate_statements(body, indent + 1))
        else:
            lines.append(f"{self.get_indent(indent + 1)}()")
        
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
        """Translate a range-based for loop to Rust."""
        ind = self.get_indent(indent)
        
        if step:
            range_expr = f"({start}..{end}).step_by({step} as usize)"
        else:
            range_expr = f"{start}..{end}"
        
        lines = [f"{ind}for {var_name} in {range_expr} {{"]
        
        if body:
            lines.append(self.translate_statements(body, indent + 1))
        else:
            lines.append(f"{self.get_indent(indent + 1)}()")
        
        lines.append(f"{ind}}}")
        return '\n'.join(lines)
    
    def translate_switch_statement(
        self,
        value: str,
        cases: List[Dict[str, Any]],
        default: Optional[List[Dict[str, Any]]],
        indent: int = 0
    ) -> str:
        """Translate a switch statement to Rust match."""
        ind = self.get_indent(indent)
        arm_ind = self.get_indent(indent + 1)
        
        lines = [f"{ind}match {value} {{"]
        
        for case in cases:
            case_val = self._translate_value(case.get('value'))
            case_body = case.get('body', [])
            
            if case_body:
                if len(case_body) == 1:
                    body_code = self.translate_statements(case_body, 0).strip()
                    lines.append(f"{arm_ind}{case_val} => {body_code}")
                else:
                    lines.append(f"{arm_ind}{case_val} => {{")
                    lines.append(self.translate_statements(case_body, indent + 2))
                    lines.append(f"{arm_ind}}}")
            else:
                lines.append(f"{arm_ind}{case_val} => ()")
        
        if default:
            if len(default) == 1:
                body_code = self.translate_statements(default, 0).strip()
                lines.append(f"{arm_ind}_ => {body_code}")
            else:
                lines.append(f"{arm_ind}_ => {{")
                lines.append(self.translate_statements(default, indent + 2))
                lines.append(f"{arm_ind}}}")
        else:
            lines.append(f"{arm_ind}_ => ()")
        
        lines.append(f"{ind}}}")
        return '\n'.join(lines)
    
    def translate_block(
        self,
        statements: List[Dict[str, Any]],
        indent: int = 0
    ) -> str:
        """Translate a block of statements to Rust."""
        ind = self.get_indent(indent)
        lines = [f"{ind}{{"]
        if statements:
            lines.append(self.translate_statements(statements, indent + 1))
        else:
            lines.append(f"{self.get_indent(indent + 1)}()")
        lines.append(f"{ind}}}")
        return '\n'.join(lines)
    
    def translate_infinite_loop(
        self,
        body: List[Dict[str, Any]],
        indent: int = 0
    ) -> str:
        """Translate an infinite loop to Rust loop."""
        ind = self.get_indent(indent)
        lines = [f"{ind}loop {{"]
        
        if body:
            lines.append(self.translate_statements(body, indent + 1))
        else:
            lines.append(f"{self.get_indent(indent + 1)}()")
        
        lines.append(f"{ind}}}")
        return '\n'.join(lines)


class RustCodeGenerator:
    """Complete Rust code generator combining statement and expression translators.
    
    This class provides a high-level interface for generating Rust code from
    STUNIR IR, including function definitions, struct definitions, and module
    structure.
    
    Attributes:
        enhancement_context: Optional EnhancementContext for type info.
        expr_translator: Rust expression translator.
        stmt_translator: Rust statement translator.
        indent_size: Number of spaces for indentation.
    """
    
    TARGET = 'rust'
    FILE_EXTENSION = 'rs'
    
    def __init__(
        self,
        enhancement_context: Optional['EnhancementContext'] = None,
        indent_size: int = 4
    ):
        """Initialize the Rust code generator.
        
        Args:
            enhancement_context: Optional EnhancementContext for type info.
            indent_size: Number of spaces for indentation.
        """
        self.enhancement_context = enhancement_context
        self.indent_size = indent_size
        
        # Create translators
        self.expr_translator = RustExpressionTranslator(enhancement_context)
        self.stmt_translator = RustStatementTranslator(
            enhancement_context,
            indent_size=indent_size
        )
        
        # Link translators
        self.stmt_translator.set_expression_translator(self.expr_translator)
    
    def generate_function(self, func_ir: Dict[str, Any]) -> str:
        """Generate Rust function code from IR.
        
        Args:
            func_ir: Function IR dictionary.
            
        Returns:
            Complete Rust function definition.
        """
        name = func_ir.get('name', 'unnamed')
        params = func_ir.get('params', [])
        return_type = func_ir.get('return_type', 'void')
        body = func_ir.get('body', [])
        visibility = func_ir.get('visibility', 'pub')
        docstring = func_ir.get('docstring', func_ir.get('description'))
        
        lines = []
        
        # Doc comment
        if docstring:
            lines.append(f"/// {docstring}")
        
        # Function signature
        vis_prefix = f"{visibility} " if visibility else ''
        params_str = self._format_params(params)
        ret_type_str = self.stmt_translator.map_type(return_type)
        
        if ret_type_str == '()':
            lines.append(f"{vis_prefix}fn {name}({params_str}) {{")
        else:
            lines.append(f"{vis_prefix}fn {name}({params_str}) -> {ret_type_str} {{")
        
        # Body
        if body:
            body_code = self.stmt_translator.translate_statements(body, indent=1)
            lines.append(body_code)
        else:
            lines.append(f"{' ' * self.indent_size}()")
        
        lines.append("}")
        
        return '\n'.join(lines)
    
    def generate_module(self, module_ir: Dict[str, Any]) -> str:
        """Generate Rust module code from IR.
        
        Args:
            module_ir: Module IR dictionary.
            
        Returns:
            Complete Rust module code.
        """
        lines = []
        
        # Module doc comment
        module_name = module_ir.get('name', module_ir.get('ir_module', 'module'))
        lines.append(f"//! Module: {module_name}")
        lines.append(f"//! Generated by STUNIR Rust Code Generator")
        lines.append('')
        
        # Imports/uses
        imports = module_ir.get('imports', [])
        if imports:
            for imp in imports:
                if isinstance(imp, dict):
                    mod = imp.get('module', '')
                    items = imp.get('items', [])
                    if items:
                        lines.append(f"use {mod}::{{{', '.join(items)}}};")
                    else:
                        lines.append(f"use {mod};")
                else:
                    lines.append(f"use {imp};")
            lines.append('')
        
        # Types/Structs
        types = module_ir.get('types', module_ir.get('ir_types', []))
        for type_def in types:
            lines.append(self._generate_struct(type_def))
            lines.append('')
        
        # Functions
        functions = module_ir.get('functions', module_ir.get('ir_functions', []))
        for func in functions:
            lines.append(self.generate_function(func))
            lines.append('')
        
        return '\n'.join(lines)
    
    def _format_params(self, params: List[Dict[str, Any]]) -> str:
        """Format function parameters with types.
        
        Args:
            params: List of parameter IR dictionaries.
            
        Returns:
            Formatted parameter string.
        """
        formatted = []
        for param in params:
            if isinstance(param, dict):
                name = param.get('name', 'arg')
                p_type = param.get('type', 'i32')
                mapped_type = self.stmt_translator.map_type(p_type)
                mutable = param.get('mutable', False)
                
                if mutable:
                    formatted.append(f"mut {name}: {mapped_type}")
                else:
                    formatted.append(f"{name}: {mapped_type}")
            else:
                formatted.append(str(param))
        
        return ', '.join(formatted)
    
    def _generate_struct(self, type_def: Dict[str, Any]) -> str:
        """Generate Rust struct from type definition.
        
        Args:
            type_def: Type definition IR dictionary.
            
        Returns:
            Rust struct definition string.
        """
        name = type_def.get('name', 'MyStruct')
        fields = type_def.get('fields', [])
        visibility = type_def.get('visibility', 'pub')
        derive = type_def.get('derive', ['Debug', 'Clone'])
        
        lines = []
        
        # Derive macros
        if derive:
            lines.append(f"#[derive({', '.join(derive)})]")
        
        # Struct definition
        vis_prefix = f"{visibility} " if visibility else ''
        lines.append(f"{vis_prefix}struct {name} {{")
        
        indent = ' ' * self.indent_size
        for field in fields:
            if isinstance(field, dict):
                f_name = field.get('name', 'field')
                f_type = self.stmt_translator.map_type(field.get('type', 'i32'))
                f_vis = field.get('visibility', 'pub')
                f_vis_prefix = f"{f_vis} " if f_vis else ''
                lines.append(f"{indent}{f_vis_prefix}{f_name}: {f_type},")
            else:
                lines.append(f"{indent}pub {field}: i32,")
        
        lines.append("}")
        
        return '\n'.join(lines)
