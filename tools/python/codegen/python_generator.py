#!/usr/bin/env python3
"""STUNIR Python Code Generator.

This module implements Python-specific code generation by extending the
statement and expression translators.

Part of Phase 3 (Advanced Code Generation) of STUNIR Enhancement Integration.

Features:
- PEP 484 type hints
- Idiomatic Python code generation
- Support for all statement types including control flow
- Support for all expression types including complex expressions
- Python 3.10+ match statement support

Usage:
    from tools.codegen.python_generator import PythonCodeGenerator
    
    generator = PythonCodeGenerator(enhancement_context=ctx)
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


class PythonExpressionTranslator(ExpressionTranslator):
    """Python-specific expression translator."""
    
    TARGET = 'python'
    LOGICAL_AND = 'and'
    LOGICAL_OR = 'or'
    LOGICAL_NOT = 'not '
    
    # Python type mapping
    TYPE_MAP = {
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
        'char': 'str',
        'byte': 'int',
        'usize': 'int',
        'isize': 'int',
    }
    
    def _build_operator_map(self) -> Dict[str, str]:
        """Build Python-specific operator mappings."""
        base = super()._build_operator_map()
        # Python-specific overrides
        base['//'] = '//'  # Integer division
        base['**'] = '**'  # Power
        base['and'] = 'and'
        base['&&'] = 'and'
        base['or'] = 'or'
        base['||'] = 'or'
        base['not'] = 'not '
        base['!'] = 'not '
        return base
    
    def translate_literal(self, value: Any, lit_type: str) -> str:
        """Translate a literal value to Python."""
        if value is None:
            return 'None'
        
        if lit_type == 'bool' or isinstance(value, bool):
            return 'True' if value else 'False'
        
        if lit_type == 'string' or isinstance(value, str):
            if isinstance(value, str):
                # Escape special characters
                escaped = value.replace('\\', '\\\\').replace('"', '\\"')
                return f'"{escaped}"'
            return f'"{value}"'
        
        if lit_type in ('f32', 'f64') or isinstance(value, float):
            return str(float(value))
        
        if isinstance(value, (int, float)):
            return str(value)
        
        return repr(value)
    
    def translate_variable(self, name: str, var_type: Optional[str] = None) -> str:
        """Translate a variable reference to Python."""
        return name
    
    def translate_binary_op(self, left: str, op: str, right: str) -> str:
        """Translate a binary operation to Python."""
        # Handle spacing for 'not' operator (shouldn't appear in binary)
        op = op.strip()
        return f"({left} {op} {right})"
    
    def translate_unary_op(self, op: str, operand: str) -> str:
        """Translate a unary operation to Python."""
        if op in ('not', 'not '):
            return f"(not {operand})"
        return f"({op}{operand})"
    
    def translate_function_call(
        self,
        func_name: str,
        args: List[str],
        receiver: Optional[str] = None
    ) -> str:
        """Translate a function call to Python."""
        args_str = ', '.join(args)
        if receiver:
            return f"{receiver}.{func_name}({args_str})"
        return f"{func_name}({args_str})"
    
    def translate_ternary(self, condition: str, then_val: str, else_val: str) -> str:
        """Translate a ternary expression to Python."""
        return f"({then_val} if {condition} else {else_val})"
    
    def translate_cast(self, value: str, target_type: str) -> str:
        """Translate a type cast to Python."""
        mapped = self.map_type(target_type)
        if mapped == 'None':
            return value  # Can't cast to None
        return f"{mapped}({value})"
    
    def translate_struct_literal(
        self,
        struct_type: str,
        fields: Dict[str, str]
    ) -> str:
        """Translate a struct literal to Python dict or dataclass."""
        # Use dict syntax for Python
        field_strs = [f'"{k}": {v}' for k, v in fields.items()]
        return f"{{{', '.join(field_strs)}}}"
    
    def map_type(self, ir_type: str) -> str:
        """Map IR type to Python type."""
        # Handle pointer types
        if ir_type.endswith('*'):
            base_type = ir_type.rstrip('*').strip()
            mapped_base = self.TYPE_MAP.get(base_type, base_type)
            return f"Optional['{mapped_base}']"
        return self.TYPE_MAP.get(ir_type, ir_type)


class PythonStatementTranslator(StatementTranslator):
    """Python-specific statement translator."""
    
    TARGET = 'python'
    STATEMENT_TERMINATOR = ''  # Python doesn't use semicolons
    
    # Python type mapping (same as expression translator)
    TYPE_MAP = PythonExpressionTranslator.TYPE_MAP
    
    def translate_variable_declaration(
        self,
        var_name: str,
        var_type: str,
        init_value: Optional[str],
        mutable: bool = True,
        indent: int = 0
    ) -> str:
        """Translate a variable declaration to Python."""
        ind = self.get_indent(indent)
        mapped_type = self.map_type(var_type)
        
        if init_value is not None:
            # With type hint: var_name: type = value
            if not mutable:
                # Use Final for immutable (requires typing import)
                return f"{ind}{var_name}: Final[{mapped_type}] = {init_value}"
            return f"{ind}{var_name}: {mapped_type} = {init_value}"
        else:
            # Declaration without initialization
            if mapped_type == 'None':
                return f"{ind}{var_name} = None"
            return f"{ind}{var_name}: {mapped_type}"
    
    def translate_assignment(
        self,
        target: str,
        value: str,
        indent: int = 0
    ) -> str:
        """Translate an assignment to Python."""
        ind = self.get_indent(indent)
        return f"{ind}{target} = {value}"
    
    def translate_compound_assignment(
        self,
        target: str,
        op: str,
        value: str,
        indent: int = 0
    ) -> str:
        """Translate a compound assignment to Python."""
        ind = self.get_indent(indent)
        return f"{ind}{target} {op} {value}"
    
    def translate_return(
        self,
        value: Optional[str],
        indent: int = 0
    ) -> str:
        """Translate a return statement to Python."""
        ind = self.get_indent(indent)
        if value is None:
            return f"{ind}return"
        return f"{ind}return {value}"
    
    def translate_expression_statement(
        self,
        expr: str,
        indent: int = 0
    ) -> str:
        """Translate an expression statement to Python."""
        ind = self.get_indent(indent)
        return f"{ind}{expr}"
    
    def _empty_body(self, indent: int = 0) -> str:
        """Generate Python's pass statement for empty bodies."""
        ind = self.get_indent(indent)
        return f"{ind}pass"
    
    def _translate_none(self) -> str:
        """Translate None value for Python."""
        return "None"
    
    def _translate_bool(self, value: bool) -> str:
        """Translate boolean value for Python."""
        return "True" if value else "False"
    
    def _translate_list(self, value: list) -> str:
        """Translate a list literal for Python."""
        items = [self._translate_value(v) for v in value]
        return f"[{', '.join(items)}]"
    
    def map_type(self, ir_type: str) -> str:
        """Map IR type to Python type."""
        # Handle pointer types
        if ir_type.endswith('*'):
            base_type = ir_type.rstrip('*').strip()
            mapped_base = self.TYPE_MAP.get(base_type, base_type)
            return f"Optional['{mapped_base}']"
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
        """Translate an if statement to Python."""
        ind = self.get_indent(indent)
        lines = [f"{ind}if {condition}:"]
        
        if then_block:
            lines.append(self.translate_statements(then_block, indent + 1))
        else:
            lines.append(f"{self.get_indent(indent + 1)}pass")
        
        for elif_branch in elif_blocks:
            elif_cond = self._translate_value(elif_branch.get('condition'))
            elif_body = elif_branch.get('body', [])
            lines.append(f"{ind}elif {elif_cond}:")
            if elif_body:
                lines.append(self.translate_statements(elif_body, indent + 1))
            else:
                lines.append(f"{self.get_indent(indent + 1)}pass")
        
        if else_block:
            lines.append(f"{ind}else:")
            lines.append(self.translate_statements(else_block, indent + 1))
        
        return '\n'.join(lines)
    
    def translate_while_loop(
        self,
        condition: str,
        body: List[Dict[str, Any]],
        indent: int = 0
    ) -> str:
        """Translate a while loop to Python."""
        ind = self.get_indent(indent)
        lines = [f"{ind}while {condition}:"]
        
        if body:
            lines.append(self.translate_statements(body, indent + 1))
        else:
            lines.append(f"{self.get_indent(indent + 1)}pass")
        
        return '\n'.join(lines)
    
    def translate_for_loop(
        self,
        init: str,
        condition: str,
        update: str,
        body: List[Dict[str, Any]],
        indent: int = 0
    ) -> str:
        """Translate a C-style for loop to Python while loop."""
        ind = self.get_indent(indent)
        lines = []
        
        # Python doesn't have C-style for, convert to while
        if init:
            lines.append(f"{ind}{init}")
        
        lines.append(f"{ind}while {condition}:")
        
        if body:
            lines.append(self.translate_statements(body, indent + 1))
        
        if update:
            lines.append(f"{self.get_indent(indent + 1)}{update}")
        
        if not body and not update:
            lines.append(f"{self.get_indent(indent + 1)}pass")
        
        return '\n'.join(lines)
    
    def translate_for_each(
        self,
        var_name: str,
        var_type: Optional[str],
        iterable: str,
        body: List[Dict[str, Any]],
        indent: int = 0
    ) -> str:
        """Translate a for-each loop to Python."""
        ind = self.get_indent(indent)
        lines = [f"{ind}for {var_name} in {iterable}:"]
        
        if body:
            lines.append(self.translate_statements(body, indent + 1))
        else:
            lines.append(f"{self.get_indent(indent + 1)}pass")
        
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
        """Translate a range-based for loop to Python."""
        ind = self.get_indent(indent)
        
        if step:
            range_expr = f"range({start}, {end}, {step})"
        elif start == '0':
            range_expr = f"range({end})"
        else:
            range_expr = f"range({start}, {end})"
        
        lines = [f"{ind}for {var_name} in {range_expr}:"]
        
        if body:
            lines.append(self.translate_statements(body, indent + 1))
        else:
            lines.append(f"{self.get_indent(indent + 1)}pass")
        
        return '\n'.join(lines)
    
    def translate_switch_statement(
        self,
        value: str,
        cases: List[Dict[str, Any]],
        default: Optional[List[Dict[str, Any]]],
        indent: int = 0
    ) -> str:
        """Translate a switch statement to Python 3.10+ match statement."""
        ind = self.get_indent(indent)
        case_ind = self.get_indent(indent + 1)
        
        lines = [f"{ind}match {value}:"]
        
        for case in cases:
            case_val = self._translate_value(case.get('value'))
            case_body = case.get('body', [])
            
            lines.append(f"{case_ind}case {case_val}:")
            if case_body:
                lines.append(self.translate_statements(case_body, indent + 2))
            else:
                lines.append(f"{self.get_indent(indent + 2)}pass")
        
        if default:
            lines.append(f"{case_ind}case _:")
            lines.append(self.translate_statements(default, indent + 2))
        
        return '\n'.join(lines)
    
    def translate_block(
        self,
        statements: List[Dict[str, Any]],
        indent: int = 0
    ) -> str:
        """Translate a block of statements to Python (just statements, no braces)."""
        if statements:
            return self.translate_statements(statements, indent)
        return self._empty_body(indent)
    
    def translate_infinite_loop(
        self,
        body: List[Dict[str, Any]],
        indent: int = 0
    ) -> str:
        """Translate an infinite loop to Python while True."""
        return self.translate_while_loop("True", body, indent)


class PythonCodeGenerator:
    """Complete Python code generator combining statement and expression translators.
    
    This class provides a high-level interface for generating Python code from
    STUNIR IR, including function definitions, class definitions, and module
    structure.
    
    Attributes:
        enhancement_context: Optional EnhancementContext for type info.
        expr_translator: Python expression translator.
        stmt_translator: Python statement translator.
        indent_size: Number of spaces for indentation.
    """
    
    TARGET = 'python'
    FILE_EXTENSION = 'py'
    
    def __init__(
        self,
        enhancement_context: Optional['EnhancementContext'] = None,
        indent_size: int = 4
    ):
        """Initialize the Python code generator.
        
        Args:
            enhancement_context: Optional EnhancementContext for type info.
            indent_size: Number of spaces for indentation.
        """
        self.enhancement_context = enhancement_context
        self.indent_size = indent_size
        
        # Create translators
        self.expr_translator = PythonExpressionTranslator(enhancement_context)
        self.stmt_translator = PythonStatementTranslator(
            enhancement_context,
            indent_size=indent_size
        )
        
        # Link translators
        self.stmt_translator.set_expression_translator(self.expr_translator)
    
    def generate_function(self, func_ir: Dict[str, Any]) -> str:
        """Generate Python function code from IR.
        
        Args:
            func_ir: Function IR dictionary.
            
        Returns:
            Complete Python function definition.
        """
        name = func_ir.get('name', 'unnamed')
        params = func_ir.get('params', [])
        return_type = func_ir.get('return_type', 'void')
        body = func_ir.get('body', [])
        docstring = func_ir.get('docstring', func_ir.get('description'))
        
        lines = []
        
        # Function signature
        params_str = self._format_params(params)
        ret_type_str = self.stmt_translator.map_type(return_type)
        
        if ret_type_str == 'None':
            lines.append(f"def {name}({params_str}) -> None:")
        else:
            lines.append(f"def {name}({params_str}) -> {ret_type_str}:")
        
        # Docstring
        if docstring:
            indent = ' ' * self.indent_size
            lines.append(f'{indent}"""')
            lines.append(f'{indent}{docstring}')
            lines.append(f'{indent}"""')
        
        # Body
        if body:
            body_code = self.stmt_translator.translate_statements(body, indent=1)
            lines.append(body_code)
        else:
            lines.append(f"{' ' * self.indent_size}pass")
        
        return '\n'.join(lines)
    
    def generate_module(self, module_ir: Dict[str, Any]) -> str:
        """Generate Python module code from IR.
        
        Args:
            module_ir: Module IR dictionary.
            
        Returns:
            Complete Python module code.
        """
        lines = []
        
        # Module docstring
        module_name = module_ir.get('name', module_ir.get('ir_module', 'module'))
        lines.append(f'"""')
        lines.append(f'Module: {module_name}')
        lines.append(f'Generated by STUNIR Python Code Generator')
        lines.append(f'"""')
        lines.append('')
        
        # Imports
        imports = module_ir.get('imports', [])
        if imports:
            for imp in imports:
                if isinstance(imp, dict):
                    mod = imp.get('module', '')
                    items = imp.get('items', [])
                    if items:
                        lines.append(f"from {mod} import {', '.join(items)}")
                    else:
                        lines.append(f"import {mod}")
                else:
                    lines.append(f"import {imp}")
            lines.append('')
        
        # Add typing imports if needed
        lines.append("from typing import Any, Dict, List, Optional, Final")
        lines.append('')
        
        # Types/Classes
        types = module_ir.get('types', module_ir.get('ir_types', []))
        for type_def in types:
            lines.append(self._generate_type(type_def))
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
            lines.append(f"__all__ = {export_names!r}")
        
        return '\n'.join(lines)
    
    def _format_params(self, params: List[Dict[str, Any]]) -> str:
        """Format function parameters with type hints.
        
        Args:
            params: List of parameter IR dictionaries.
            
        Returns:
            Formatted parameter string.
        """
        formatted = []
        for param in params:
            if isinstance(param, dict):
                name = param.get('name', 'arg')
                p_type = param.get('type', 'Any')
                mapped_type = self.stmt_translator.map_type(p_type)
                default = param.get('default')
                
                if default is not None:
                    default_str = self.expr_translator.translate_expression(default)
                    formatted.append(f"{name}: {mapped_type} = {default_str}")
                else:
                    formatted.append(f"{name}: {mapped_type}")
            else:
                formatted.append(str(param))
        
        return ', '.join(formatted)
    
    def _generate_type(self, type_def: Dict[str, Any]) -> str:
        """Generate Python class from type definition.
        
        Args:
            type_def: Type definition IR dictionary.
            
        Returns:
            Python class definition string.
        """
        name = type_def.get('name', 'MyType')
        fields = type_def.get('fields', [])
        
        lines = [f"class {name}:"]
        indent = ' ' * self.indent_size
        
        if not fields:
            lines.append(f"{indent}pass")
            return '\n'.join(lines)
        
        # Generate __init__ method
        param_parts = ['self']
        for field in fields:
            if isinstance(field, dict):
                f_name = field.get('name', 'field')
                f_type = self.stmt_translator.map_type(field.get('type', 'Any'))
                param_parts.append(f"{f_name}: {f_type}")
            else:
                param_parts.append(str(field))
        
        lines.append(f"{indent}def __init__({', '.join(param_parts)}):")
        
        for field in fields:
            if isinstance(field, dict):
                f_name = field.get('name', 'field')
                lines.append(f"{indent}{indent}self.{f_name} = {f_name}")
            else:
                lines.append(f"{indent}{indent}self.{field} = {field}")
        
        return '\n'.join(lines)