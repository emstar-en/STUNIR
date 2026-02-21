#!/usr/bin/env python3
"""STUNIR Go Code Generator.

This module implements Go-specific code generation by extending the
statement and expression translators.

Part of Phase 3 (Advanced Code Generation) of STUNIR Enhancement Integration.

Features:
- Idiomatic Go code generation
- Proper Go naming conventions (camelCase)
- Short variable declarations (:=)
- Support for all statement types including control flow
- Support for all expression types
- Go switch statement support

Usage:
    from tools.codegen.go_generator import GoCodeGenerator
    
    generator = GoCodeGenerator(enhancement_context=ctx)
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


class GoExpressionTranslator(ExpressionTranslator):
    """Go-specific expression translator."""
    
    TARGET = 'go'
    LOGICAL_AND = '&&'
    LOGICAL_OR = '||'
    LOGICAL_NOT = '!'
    
    # Go type mapping
    TYPE_MAP = {
        'i8': 'int8',
        'i16': 'int16',
        'i32': 'int32',
        'i64': 'int64',
        'u8': 'uint8',
        'u16': 'uint16',
        'u32': 'uint32',
        'u64': 'uint64',
        'f32': 'float32',
        'f64': 'float64',
        'bool': 'bool',
        'string': 'string',
        'void': '',
        'unit': '',
        'char': 'rune',
        'byte': 'byte',
        'usize': 'uint',
        'isize': 'int',
        'int': 'int',
    }
    
    def _build_operator_map(self) -> Dict[str, str]:
        """Build Go-specific operator mappings."""
        base = super()._build_operator_map()
        # Go-specific overrides
        base['//'] = '/'  # Go uses / for both int and float division
        base['**'] = '**'  # Note: Go doesn't have power operator, need math.Pow
        base['and'] = '&&'
        base['&&'] = '&&'
        base['or'] = '||'
        base['||'] = '||'
        base['not'] = '!'
        base['!'] = '!'
        return base
    
    def translate_literal(self, value: Any, lit_type: str) -> str:
        """Translate a literal value to Go."""
        if value is None:
            return 'nil'
        
        if lit_type == 'bool' or isinstance(value, bool):
            return 'true' if value else 'false'
        
        if lit_type == 'string' or isinstance(value, str):
            if isinstance(value, str):
                escaped = value.replace('\\', '\\\\').replace('"', '\\"')
                return f'"{escaped}"'
            return f'"{value}"'
        
        if lit_type == 'char' or lit_type == 'rune':
            if isinstance(value, str) and len(value) == 1:
                return f"'{value}'"
            return f"'{value}'"
        
        if isinstance(value, (int, float)):
            return str(value)
        
        return repr(value)
    
    def translate_variable(self, name: str, var_type: Optional[str] = None) -> str:
        """Translate a variable reference to Go."""
        return name
    
    def translate_binary_op(self, left: str, op: str, right: str) -> str:
        """Translate a binary operation to Go."""
        # Handle power operator specially (needs math.Pow in Go)
        if op == '**':
            return f"math.Pow(float64({left}), float64({right}))"
        return f"({left} {op} {right})"
    
    def translate_unary_op(self, op: str, operand: str) -> str:
        """Translate a unary operation to Go."""
        return f"({op}{operand})"
    
    def translate_function_call(
        self,
        func_name: str,
        args: List[str],
        receiver: Optional[str] = None
    ) -> str:
        """Translate a function call to Go."""
        args_str = ', '.join(args)
        if receiver:
            return f"{receiver}.{func_name}({args_str})"
        return f"{func_name}({args_str})"
    
    def translate_ternary(self, condition: str, then_val: str, else_val: str) -> str:
        """Translate a ternary expression to Go.
        
        Note: Go doesn't have a ternary operator, so we use a helper function pattern
        or inline if statement is not possible - generate a comment indicating this.
        """
        # Go doesn't support ternary, this will need to be handled at statement level
        return f"/* Go has no ternary: {condition} ? {then_val} : {else_val} */"
    
    def translate_cast(self, value: str, target_type: str) -> str:
        """Translate a type cast to Go."""
        mapped = self.map_type(target_type)
        if not mapped:
            return value
        return f"{mapped}({value})"
    
    def translate_struct_literal(
        self,
        struct_type: str,
        fields: Dict[str, str]
    ) -> str:
        """Translate a struct literal to Go."""
        field_strs = [f"{k}: {v}" for k, v in fields.items()]
        return f"{struct_type}{{{', '.join(field_strs)}}}"
    
    def map_type(self, ir_type: str) -> str:
        """Map IR type to Go type."""
        return self.TYPE_MAP.get(ir_type, ir_type)


class GoStatementTranslator(StatementTranslator):
    """Go-specific statement translator."""
    
    TARGET = 'go'
    STATEMENT_TERMINATOR = ''  # Go doesn't require semicolons (usually)
    
    # Go type mapping (same as expression translator)
    TYPE_MAP = GoExpressionTranslator.TYPE_MAP
    
    def __init__(
        self,
        enhancement_context: Optional['EnhancementContext'] = None,
        indent_size: int = 4,
        indent_char: str = '\t',  # Go convention uses tabs
        use_short_decl: bool = True
    ):
        """Initialize Go statement translator.
        
        Args:
            enhancement_context: Optional EnhancementContext for type info.
            indent_size: Number of indent units.
            indent_char: Character for indentation (tab by default for Go).
            use_short_decl: Whether to use := for short variable declarations.
        """
        super().__init__(enhancement_context, indent_size, indent_char)
        self.use_short_decl = use_short_decl
    
    def translate_variable_declaration(
        self,
        var_name: str,
        var_type: str,
        init_value: Optional[str],
        mutable: bool = True,
        indent: int = 0
    ) -> str:
        """Translate a variable declaration to Go."""
        ind = self.get_indent(indent)
        mapped_type = self.map_type(var_type)
        
        if not mutable:
            # Go const
            if init_value is not None:
                return f"{ind}const {var_name} {mapped_type} = {init_value}"
            return f"{ind}const {var_name} {mapped_type}"
        
        if init_value is not None:
            if self.use_short_decl:
                # Short declaration with :=
                return f"{ind}{var_name} := {init_value}"
            else:
                # Full var declaration
                if mapped_type:
                    return f"{ind}var {var_name} {mapped_type} = {init_value}"
                return f"{ind}var {var_name} = {init_value}"
        else:
            # Declaration without initialization
            if mapped_type:
                return f"{ind}var {var_name} {mapped_type}"
            return f"{ind}var {var_name}"
    
    def translate_assignment(
        self,
        target: str,
        value: str,
        indent: int = 0
    ) -> str:
        """Translate an assignment to Go."""
        ind = self.get_indent(indent)
        return f"{ind}{target} = {value}"
    
    def translate_compound_assignment(
        self,
        target: str,
        op: str,
        value: str,
        indent: int = 0
    ) -> str:
        """Translate a compound assignment to Go."""
        ind = self.get_indent(indent)
        return f"{ind}{target} {op} {value}"
    
    def translate_return(
        self,
        value: Optional[str],
        indent: int = 0
    ) -> str:
        """Translate a return statement to Go."""
        ind = self.get_indent(indent)
        if value is None:
            return f"{ind}return"
        return f"{ind}return {value}"
    
    def translate_expression_statement(
        self,
        expr: str,
        indent: int = 0
    ) -> str:
        """Translate an expression statement to Go."""
        ind = self.get_indent(indent)
        return f"{ind}{expr}"
    
    def _empty_body(self, indent: int = 0) -> str:
        """Generate Go empty body (nothing, or a comment)."""
        ind = self.get_indent(indent)
        return f"{ind}// empty"
    
    def _translate_none(self) -> str:
        """Translate None value for Go."""
        return "nil"
    
    def _translate_bool(self, value: bool) -> str:
        """Translate boolean value for Go."""
        return "true" if value else "false"
    
    def _translate_list(self, value: list) -> str:
        """Translate a list literal for Go (slice)."""
        items = [self._translate_value(v) for v in value]
        # Go needs type info for slices, use interface{} as fallback
        return f"[]interface{{}}{{{', '.join(items)}}}"
    
    def map_type(self, ir_type: str) -> str:
        """Map IR type to Go type."""
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
        """Translate an if statement to Go."""
        ind = self.get_indent(indent)
        lines = [f"{ind}if {condition} {{"]
        
        if then_block:
            lines.append(self.translate_statements(then_block, indent + 1))
        else:
            lines.append(f"{self.get_indent(indent + 1)}// empty")
        
        for elif_branch in elif_blocks:
            elif_cond = self._translate_value(elif_branch.get('condition'))
            elif_body = elif_branch.get('body', [])
            lines.append(f"{ind}}} else if {elif_cond} {{")
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
        """Translate a while loop to Go for loop."""
        # Go uses for for all loops
        ind = self.get_indent(indent)
        lines = [f"{ind}for {condition} {{"]
        
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
        """Translate a C-style for loop to Go."""
        ind = self.get_indent(indent)
        lines = [f"{ind}for {init}; {condition}; {update} {{"]
        
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
        """Translate a for-each loop to Go range loop."""
        ind = self.get_indent(indent)
        lines = [f"{ind}for _, {var_name} := range {iterable} {{"]
        
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
        """Translate a range-based for loop to Go."""
        ind = self.get_indent(indent)
        
        if step:
            update = f"{var_name} += {step}"
        else:
            update = f"{var_name}++"
        
        lines = [f"{ind}for {var_name} := {start}; {var_name} < {end}; {update} {{"]
        
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
        """Translate a switch statement to Go."""
        ind = self.get_indent(indent)
        case_ind = self.get_indent(indent + 1)
        
        lines = [f"{ind}switch {value} {{"]
        
        for case in cases:
            case_val = self._translate_value(case.get('value'))
            case_body = case.get('body', [])
            fallthrough = case.get('fallthrough', False)
            
            lines.append(f"{case_ind}case {case_val}:")
            if case_body:
                lines.append(self.translate_statements(case_body, indent + 2))
            if fallthrough:
                lines.append(f"{self.get_indent(indent + 2)}fallthrough")
        
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
        """Translate a block of statements to Go."""
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
        """Translate an infinite loop to Go for {}."""
        ind = self.get_indent(indent)
        lines = [f"{ind}for {{"]
        
        if body:
            lines.append(self.translate_statements(body, indent + 1))
        else:
            lines.append(f"{self.get_indent(indent + 1)}// empty")
        
        lines.append(f"{ind}}}")
        return '\n'.join(lines)


class GoCodeGenerator:
    """Complete Go code generator combining statement and expression translators.
    
    This class provides a high-level interface for generating Go code from
    STUNIR IR, including function definitions, struct definitions, and package
    structure.
    
    Attributes:
        enhancement_context: Optional EnhancementContext for type info.
        expr_translator: Go expression translator.
        stmt_translator: Go statement translator.
        indent_char: Character for indentation.
    """
    
    TARGET = 'go'
    FILE_EXTENSION = 'go'
    
    def __init__(
        self,
        enhancement_context: Optional['EnhancementContext'] = None,
        indent_char: str = '\t',
        use_short_decl: bool = True
    ):
        """Initialize the Go code generator.
        
        Args:
            enhancement_context: Optional EnhancementContext for type info.
            indent_char: Character for indentation (tab by default).
            use_short_decl: Whether to use := for short declarations.
        """
        self.enhancement_context = enhancement_context
        self.indent_char = indent_char
        
        # Create translators
        self.expr_translator = GoExpressionTranslator(enhancement_context)
        self.stmt_translator = GoStatementTranslator(
            enhancement_context,
            indent_size=1,
            indent_char=indent_char,
            use_short_decl=use_short_decl
        )
        
        # Link translators
        self.stmt_translator.set_expression_translator(self.expr_translator)
    
    def generate_function(self, func_ir: Dict[str, Any]) -> str:
        """Generate Go function code from IR.
        
        Args:
            func_ir: Function IR dictionary.
            
        Returns:
            Complete Go function definition.
        """
        name = func_ir.get('name', 'unnamed')
        params = func_ir.get('params', [])
        return_type = func_ir.get('return_type', 'void')
        body = func_ir.get('body', [])
        receiver = func_ir.get('receiver')  # For methods
        docstring = func_ir.get('docstring', func_ir.get('description'))
        
        lines = []
        
        # Doc comment
        if docstring:
            lines.append(f"// {name} {docstring}")
        
        # Function signature
        params_str = self._format_params(params)
        ret_type_str = self.stmt_translator.map_type(return_type)
        
        # Capitalize first letter for exported functions
        exported = name[0].isupper() if name else False
        
        # Method receiver
        recv_str = ''
        if receiver:
            recv_type = receiver.get('type', 'T')
            recv_name = receiver.get('name', 'r')
            recv_str = f"({recv_name} {recv_type}) "
        
        if ret_type_str:
            lines.append(f"func {recv_str}{name}({params_str}) {ret_type_str} {{")
        else:
            lines.append(f"func {recv_str}{name}({params_str}) {{")
        
        # Body
        if body:
            body_code = self.stmt_translator.translate_statements(body, indent=1)
            lines.append(body_code)
        else:
            lines.append(f"{self.indent_char}// empty")
        
        lines.append("}")
        
        return '\n'.join(lines)
    
    def generate_module(self, module_ir: Dict[str, Any]) -> str:
        """Generate Go package code from IR.
        
        Args:
            module_ir: Module IR dictionary.
            
        Returns:
            Complete Go package code.
        """
        lines = []
        
        # Package name
        package_name = module_ir.get('package', module_ir.get('name', 
                                     module_ir.get('ir_module', 'main')))
        # Sanitize package name for Go
        package_name = package_name.lower().replace('-', '_').replace('.', '_')
        
        lines.append(f"// Package {package_name} - Generated by STUNIR Go Code Generator")
        lines.append(f"package {package_name}")
        lines.append('')
        
        # Imports
        imports = module_ir.get('imports', [])
        if imports:
            lines.append('import (')
            for imp in imports:
                if isinstance(imp, dict):
                    mod = imp.get('module', '')
                    alias = imp.get('alias')
                    if alias:
                        lines.append(f'{self.indent_char}{alias} "{mod}"')
                    else:
                        lines.append(f'{self.indent_char}"{mod}"')
                else:
                    lines.append(f'{self.indent_char}"{imp}"')
            lines.append(')')
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
                p_type = param.get('type', 'interface{}')
                mapped_type = self.stmt_translator.map_type(p_type)
                formatted.append(f"{name} {mapped_type}")
            else:
                formatted.append(str(param))
        
        return ', '.join(formatted)
    
    def _generate_struct(self, type_def: Dict[str, Any]) -> str:
        """Generate Go struct from type definition.
        
        Args:
            type_def: Type definition IR dictionary.
            
        Returns:
            Go struct definition string.
        """
        name = type_def.get('name', 'MyStruct')
        fields = type_def.get('fields', [])
        
        # Capitalize first letter for exported struct
        if name and name[0].islower():
            name = name[0].upper() + name[1:]
        
        lines = [f"// {name} represents a data structure"]
        lines.append(f"type {name} struct {{")
        
        for field in fields:
            if isinstance(field, dict):
                f_name = field.get('name', 'field')
                # Capitalize field names for export
                if f_name and f_name[0].islower():
                    f_name = f_name[0].upper() + f_name[1:]
                f_type = self.stmt_translator.map_type(field.get('type', 'interface{}'))
                tag = field.get('tag', '')
                if tag:
                    lines.append(f'{self.indent_char}{f_name} {f_type} `{tag}`')
                else:
                    lines.append(f'{self.indent_char}{f_name} {f_type}')
            else:
                lines.append(f'{self.indent_char}{field} interface{{}}')
        
        lines.append("}")
        
        return '\n'.join(lines)
