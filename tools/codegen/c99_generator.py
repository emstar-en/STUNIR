#!/usr/bin/env python3
"""STUNIR C99 Code Generator.

This module implements C99-specific code generation by extending the
statement and expression translators.

Part of Phase 3 (Advanced Code Generation) of STUNIR Enhancement Integration.

Features:
- C99 standard compliance
- Fixed-width integer types (int32_t, etc.)
- Proper pointer handling
- Support for all statement types including control flow
- Support for all expression types
- Full switch/case/default support
- do-while loop support

Usage:
    from tools.codegen.c99_generator import C99CodeGenerator
    
    generator = C99CodeGenerator(enhancement_context=ctx)
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


class C99ExpressionTranslator(ExpressionTranslator):
    """C99-specific expression translator."""
    
    TARGET = 'c99'
    LOGICAL_AND = '&&'
    LOGICAL_OR = '||'
    LOGICAL_NOT = '!'
    
    # C99 type mapping
    TYPE_MAP = {
        'i8': 'int8_t',
        'i16': 'int16_t',
        'i32': 'int32_t',
        'i64': 'int64_t',
        'u8': 'uint8_t',
        'u16': 'uint16_t',
        'u32': 'uint32_t',
        'u64': 'uint64_t',
        'f32': 'float',
        'f64': 'double',
        'bool': 'bool',
        'string': 'char*',
        'void': 'void',
        'unit': 'void',
        'char': 'char',
        'byte': 'uint8_t',
        'usize': 'size_t',
        'isize': 'ssize_t',
        'int': 'int',
        'float': 'float',
        'double': 'double',
    }
    
    def _build_operator_map(self) -> Dict[str, str]:
        """Build C99-specific operator mappings."""
        base = super()._build_operator_map()
        # C99-specific overrides
        base['//'] = '/'  # C uses / for both (integer division for int operands)
        base['**'] = '**'  # Note: C doesn't have power operator, need pow()
        base['and'] = '&&'
        base['&&'] = '&&'
        base['or'] = '||'
        base['||'] = '||'
        base['not'] = '!'
        base['!'] = '!'
        return base
    
    def translate_literal(self, value: Any, lit_type: str) -> str:
        """Translate a literal value to C99."""
        if value is None:
            return 'NULL'
        
        if lit_type == 'bool' or isinstance(value, bool):
            return 'true' if value else 'false'
        
        if lit_type == 'string' or (isinstance(value, str) and lit_type not in ('char',)):
            if isinstance(value, str):
                escaped = value.replace('\\', '\\\\').replace('"', '\\"')
                return f'"{escaped}"'
            return f'"{value}"'
        
        if lit_type == 'char':
            if isinstance(value, str) and len(value) == 1:
                if value == "'":
                    return "'\\''"
                if value == '\\':
                    return "'\\\\'"
                return f"'{value}'"
            return f"'{value}'"
        
        if lit_type == 'f32':
            return f"{float(value)}f"
        
        if lit_type == 'f64' or isinstance(value, float):
            return str(float(value))
        
        if lit_type in ('i64', 'u64'):
            suffix = 'LL' if lit_type == 'i64' else 'ULL'
            return f"{int(value)}{suffix}"
        
        if lit_type in ('u8', 'u16', 'u32'):
            return f"{int(value)}U"
        
        if isinstance(value, int):
            return str(value)
        
        return str(value)
    
    def translate_variable(self, name: str, var_type: Optional[str] = None) -> str:
        """Translate a variable reference to C99."""
        return name
    
    def translate_binary_op(self, left: str, op: str, right: str) -> str:
        """Translate a binary operation to C99."""
        # Handle power operator specially (needs pow() in C)
        if op == '**':
            return f"pow({left}, {right})"
        return f"({left} {op} {right})"
    
    def translate_unary_op(self, op: str, operand: str) -> str:
        """Translate a unary operation to C99."""
        return f"({op}{operand})"
    
    def translate_function_call(
        self,
        func_name: str,
        args: List[str],
        receiver: Optional[str] = None
    ) -> str:
        """Translate a function call to C99."""
        args_str = ', '.join(args)
        if receiver:
            # C doesn't have methods, treat as first argument
            return f"{func_name}({receiver}, {args_str})" if args_str else f"{func_name}({receiver})"
        return f"{func_name}({args_str})"
    
    def translate_ternary(self, condition: str, then_val: str, else_val: str) -> str:
        """Translate a ternary expression to C99."""
        return f"(({condition}) ? ({then_val}) : ({else_val}))"
    
    def translate_cast(self, value: str, target_type: str) -> str:
        """Translate a type cast to C99."""
        mapped = self.map_type(target_type)
        return f"(({mapped})({value}))"
    
    def translate_struct_literal(
        self,
        struct_type: str,
        fields: Dict[str, str]
    ) -> str:
        """Translate a struct literal to C99 (designated initializer)."""
        field_strs = [f".{k} = {v}" for k, v in fields.items()]
        return f"({struct_type}){{{', '.join(field_strs)}}}"
    
    def translate_index(self, base: str, index: str) -> str:
        """Translate array indexing to C99."""
        return f"{base}[{index}]"
    
    def translate_member_access(self, base: str, member: str) -> str:
        """Translate member access to C99."""
        # Check if it's a pointer (simple heuristic)
        if '->' in base or base.endswith('*'):
            return f"{base}->{member}"
        return f"{base}.{member}"
    
    def map_type(self, ir_type: str) -> str:
        """Map IR type to C99 type."""
        return self.TYPE_MAP.get(ir_type, ir_type)


class C99StatementTranslator(StatementTranslator):
    """C99-specific statement translator."""
    
    TARGET = 'c99'
    STATEMENT_TERMINATOR = ';'
    
    # C99 type mapping (same as expression translator)
    TYPE_MAP = C99ExpressionTranslator.TYPE_MAP
    
    def translate_variable_declaration(
        self,
        var_name: str,
        var_type: str,
        init_value: Optional[str],
        mutable: bool = True,
        indent: int = 0
    ) -> str:
        """Translate a variable declaration to C99."""
        ind = self.get_indent(indent)
        mapped_type = self.map_type(var_type)
        
        const_kw = '' if mutable else 'const '
        
        if init_value is not None:
            return f"{ind}{const_kw}{mapped_type} {var_name} = {init_value};"
        else:
            # C99 allows uninitialized local variables
            return f"{ind}{const_kw}{mapped_type} {var_name};"
    
    def translate_assignment(
        self,
        target: str,
        value: str,
        indent: int = 0
    ) -> str:
        """Translate an assignment to C99."""
        ind = self.get_indent(indent)
        return f"{ind}{target} = {value};"
    
    def translate_compound_assignment(
        self,
        target: str,
        op: str,
        value: str,
        indent: int = 0
    ) -> str:
        """Translate a compound assignment to C99."""
        ind = self.get_indent(indent)
        return f"{ind}{target} {op} {value};"
    
    def translate_return(
        self,
        value: Optional[str],
        indent: int = 0
    ) -> str:
        """Translate a return statement to C99."""
        ind = self.get_indent(indent)
        if value is None:
            return f"{ind}return;"
        return f"{ind}return {value};"
    
    def translate_expression_statement(
        self,
        expr: str,
        indent: int = 0
    ) -> str:
        """Translate an expression statement to C99."""
        ind = self.get_indent(indent)
        return f"{ind}{expr};"
    
    def _empty_body(self, indent: int = 0) -> str:
        """Generate C99 empty body (empty statement or comment)."""
        ind = self.get_indent(indent)
        return f"{ind}/* empty */"
    
    def _translate_none(self) -> str:
        """Translate None value for C99."""
        return "NULL"
    
    def _translate_bool(self, value: bool) -> str:
        """Translate boolean value for C99."""
        return "true" if value else "false"
    
    def _translate_list(self, value: list) -> str:
        """Translate a list literal for C99 (array initializer)."""
        items = [self._translate_value(v) for v in value]
        return f"{{{', '.join(items)}}}"
    
    def map_type(self, ir_type: str) -> str:
        """Map IR type to C99 type."""
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
        """Translate an if statement to C99."""
        ind = self.get_indent(indent)
        lines = [f"{ind}if ({condition}) {{"]
        
        if then_block:
            lines.append(self.translate_statements(then_block, indent + 1))
        else:
            lines.append(f"{self.get_indent(indent + 1)}/* empty */")
        
        for elif_branch in elif_blocks:
            elif_cond = self._translate_value(elif_branch.get('condition'))
            elif_body = elif_branch.get('body', [])
            lines.append(f"{ind}}} else if ({elif_cond}) {{")
            if elif_body:
                lines.append(self.translate_statements(elif_body, indent + 1))
            else:
                lines.append(f"{self.get_indent(indent + 1)}/* empty */")
        
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
        """Translate a while loop to C99."""
        ind = self.get_indent(indent)
        lines = [f"{ind}while ({condition}) {{"]
        
        if body:
            lines.append(self.translate_statements(body, indent + 1))
        else:
            lines.append(f"{self.get_indent(indent + 1)}/* empty */")
        
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
        """Translate a C-style for loop to C99."""
        ind = self.get_indent(indent)
        lines = [f"{ind}for ({init}; {condition}; {update}) {{"]
        
        if body:
            lines.append(self.translate_statements(body, indent + 1))
        else:
            lines.append(f"{self.get_indent(indent + 1)}/* empty */")
        
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
        """Translate a for-each loop to C99 (manual iteration)."""
        # C99 doesn't have for-each, generate index-based loop
        ind = self.get_indent(indent)
        mapped_type = self.map_type(var_type) if var_type else 'int'
        
        lines = [
            f"{ind}for (size_t _i = 0; _i < sizeof({iterable})/sizeof({iterable}[0]); _i++) {{",
            f"{self.get_indent(indent + 1)}{mapped_type} {var_name} = {iterable}[_i];"
        ]
        
        if body:
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
        """Translate a range-based for loop to C99."""
        ind = self.get_indent(indent)
        
        if step:
            update = f"{var_name} += {step}"
        else:
            update = f"{var_name}++"
        
        lines = [f"{ind}for (int {var_name} = {start}; {var_name} < {end}; {update}) {{"]
        
        if body:
            lines.append(self.translate_statements(body, indent + 1))
        else:
            lines.append(f"{self.get_indent(indent + 1)}/* empty */")
        
        lines.append(f"{ind}}}")
        return '\n'.join(lines)
    
    def translate_do_while(
        self,
        condition: str,
        body: List[Dict[str, Any]],
        indent: int = 0
    ) -> str:
        """Translate a do-while loop to C99."""
        ind = self.get_indent(indent)
        lines = [f"{ind}do {{"]
        
        if body:
            lines.append(self.translate_statements(body, indent + 1))
        else:
            lines.append(f"{self.get_indent(indent + 1)}/* empty */")
        
        lines.append(f"{ind}}} while ({condition});")
        return '\n'.join(lines)
    
    def translate_switch_statement(
        self,
        value: str,
        cases: List[Dict[str, Any]],
        default: Optional[List[Dict[str, Any]]],
        indent: int = 0
    ) -> str:
        """Translate a switch statement to C99."""
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
                lines.append(f"{self.get_indent(indent + 2)}break;")
        
        if default:
            lines.append(f"{case_ind}default:")
            lines.append(self.translate_statements(default, indent + 2))
            lines.append(f"{self.get_indent(indent + 2)}break;")
        
        lines.append(f"{ind}}}")
        return '\n'.join(lines)
    
    def translate_block(
        self,
        statements: List[Dict[str, Any]],
        indent: int = 0
    ) -> str:
        """Translate a block of statements to C99."""
        ind = self.get_indent(indent)
        lines = [f"{ind}{{"]
        if statements:
            lines.append(self.translate_statements(statements, indent + 1))
        else:
            lines.append(f"{self.get_indent(indent + 1)}/* empty */")
        lines.append(f"{ind}}}")
        return '\n'.join(lines)
    
    def translate_infinite_loop(
        self,
        body: List[Dict[str, Any]],
        indent: int = 0
    ) -> str:
        """Translate an infinite loop to C99 while(1)."""
        return self.translate_while_loop("1", body, indent)


class C99CodeGenerator:
    """Complete C99 code generator combining statement and expression translators.
    
    This class provides a high-level interface for generating C99 code from
    STUNIR IR, including function definitions, struct definitions, and header
    file generation.
    
    Attributes:
        enhancement_context: Optional EnhancementContext for type info.
        expr_translator: C99 expression translator.
        stmt_translator: C99 statement translator.
        indent_size: Number of spaces for indentation.
    """
    
    TARGET = 'c99'
    FILE_EXTENSION = 'c'
    HEADER_EXTENSION = 'h'
    
    def __init__(
        self,
        enhancement_context: Optional['EnhancementContext'] = None,
        indent_size: int = 4
    ):
        """Initialize the C99 code generator.
        
        Args:
            enhancement_context: Optional EnhancementContext for type info.
            indent_size: Number of spaces for indentation.
        """
        self.enhancement_context = enhancement_context
        self.indent_size = indent_size
        
        # Create translators
        self.expr_translator = C99ExpressionTranslator(enhancement_context)
        self.stmt_translator = C99StatementTranslator(
            enhancement_context,
            indent_size=indent_size
        )
        
        # Link translators
        self.stmt_translator.set_expression_translator(self.expr_translator)
    
    def generate_function(self, func_ir: Dict[str, Any], declaration_only: bool = False) -> str:
        """Generate C99 function code from IR.
        
        Args:
            func_ir: Function IR dictionary.
            declaration_only: If True, generate only the declaration (for headers).
            
        Returns:
            Complete C99 function definition or declaration.
        """
        name = func_ir.get('name', 'unnamed')
        params = func_ir.get('params', [])
        return_type = func_ir.get('return_type', 'void')
        body = func_ir.get('body', [])
        static = func_ir.get('static', False)
        inline = func_ir.get('inline', False)
        docstring = func_ir.get('docstring', func_ir.get('description'))
        
        lines = []
        
        # Doc comment
        if docstring:
            lines.append(f"/* {docstring} */")
        
        # Function signature
        prefix = ''
        if static:
            prefix += 'static '
        if inline:
            prefix += 'inline '
        
        params_str = self._format_params(params)
        ret_type_str = self.stmt_translator.map_type(return_type)
        
        if declaration_only:
            lines.append(f"{prefix}{ret_type_str} {name}({params_str});")
            return '\n'.join(lines)
        
        lines.append(f"{prefix}{ret_type_str} {name}({params_str}) {{")
        
        # Body
        if body:
            body_code = self.stmt_translator.translate_statements(body, indent=1)
            lines.append(body_code)
        else:
            indent = ' ' * self.indent_size
            if ret_type_str == 'void':
                lines.append(f"{indent}/* empty */")
            else:
                # Need to return something
                default_ret = self._default_return_value(ret_type_str)
                lines.append(f"{indent}return {default_ret};")
        
        lines.append("}")
        
        return '\n'.join(lines)
    
    def generate_module(self, module_ir: Dict[str, Any]) -> str:
        """Generate C99 source file code from IR.
        
        Args:
            module_ir: Module IR dictionary.
            
        Returns:
            Complete C99 source file code.
        """
        lines = []
        
        # File header comment
        module_name = module_ir.get('name', module_ir.get('ir_module', 'module'))
        lines.append(f"/**")
        lines.append(f" * @file {module_name}.c")
        lines.append(f" * @brief Generated by STUNIR C99 Code Generator")
        lines.append(f" */")
        lines.append('')
        
        # Standard includes
        lines.append('#include <stdint.h>')
        lines.append('#include <stdbool.h>')
        lines.append('#include <stddef.h>')
        lines.append('#include <stdlib.h>')
        lines.append('')
        
        # Custom includes
        imports = module_ir.get('imports', [])
        if imports:
            for imp in imports:
                if isinstance(imp, dict):
                    mod = imp.get('module', '')
                    is_system = imp.get('system', False)
                    if is_system:
                        lines.append(f'#include <{mod}>')
                    else:
                        lines.append(f'#include "{mod}"')
                else:
                    lines.append(f'#include "{imp}"')
            lines.append('')
        
        # Include own header
        lines.append(f'#include "{module_name}.h"')
        lines.append('')
        
        # Type definitions (forward declarations already in header)
        types = module_ir.get('types', module_ir.get('ir_types', []))
        # Types are usually in header, but can put helper functions here
        
        # Static/private functions first
        functions = module_ir.get('functions', module_ir.get('ir_functions', []))
        static_funcs = [f for f in functions if f.get('static', False)]
        public_funcs = [f for f in functions if not f.get('static', False)]
        
        # Static function declarations
        if static_funcs:
            lines.append('/* Static function declarations */')
            for func in static_funcs:
                lines.append(self.generate_function(func, declaration_only=True))
            lines.append('')
        
        # All function implementations
        for func in static_funcs + public_funcs:
            lines.append(self.generate_function(func))
            lines.append('')
        
        return '\n'.join(lines)
    
    def generate_header(self, module_ir: Dict[str, Any]) -> str:
        """Generate C99 header file from IR.
        
        Args:
            module_ir: Module IR dictionary.
            
        Returns:
            Complete C99 header file code.
        """
        lines = []
        
        module_name = module_ir.get('name', module_ir.get('ir_module', 'module'))
        guard_name = f"{module_name.upper().replace('.', '_').replace('-', '_')}_H"
        
        # Header guard
        lines.append(f"#ifndef {guard_name}")
        lines.append(f"#define {guard_name}")
        lines.append('')
        
        # File header comment
        lines.append(f"/**")
        lines.append(f" * @file {module_name}.h")
        lines.append(f" * @brief Generated by STUNIR C99 Code Generator")
        lines.append(f" */")
        lines.append('')
        
        # Standard includes
        lines.append('#include <stdint.h>')
        lines.append('#include <stdbool.h>')
        lines.append('#include <stddef.h>')
        lines.append('')
        
        # Custom includes
        imports = module_ir.get('imports', [])
        if imports:
            for imp in imports:
                if isinstance(imp, dict):
                    mod = imp.get('module', '')
                    is_system = imp.get('system', False)
                    if is_system:
                        lines.append(f'#include <{mod}>')
                    else:
                        lines.append(f'#include "{mod}"')
                else:
                    lines.append(f'#include "{imp}"')
            lines.append('')
        
        # Type definitions
        types = module_ir.get('types', module_ir.get('ir_types', []))
        for type_def in types:
            lines.append(self._generate_struct(type_def))
            lines.append('')
        
        # Function declarations (only public/exported functions)
        functions = module_ir.get('functions', module_ir.get('ir_functions', []))
        exports = {e.get('name', e) if isinstance(e, dict) else e 
                   for e in module_ir.get('exports', module_ir.get('ir_exports', []))}
        
        public_funcs = [f for f in functions 
                        if not f.get('static', False) and 
                        (not exports or f.get('name') in exports)]
        
        if public_funcs:
            lines.append('/* Public function declarations */')
            for func in public_funcs:
                lines.append(self.generate_function(func, declaration_only=True))
            lines.append('')
        
        # Close header guard
        lines.append(f"#endif /* {guard_name} */")
        
        return '\n'.join(lines)
    
    def _format_params(self, params: List[Dict[str, Any]]) -> str:
        """Format function parameters with types.
        
        Args:
            params: List of parameter IR dictionaries.
            
        Returns:
            Formatted parameter string.
        """
        if not params:
            return 'void'
        
        formatted = []
        for param in params:
            if isinstance(param, dict):
                name = param.get('name', 'arg')
                p_type = param.get('type', 'int')
                mapped_type = self.stmt_translator.map_type(p_type)
                
                # Handle pointer types
                if p_type.endswith('*') or mapped_type.endswith('*'):
                    if not mapped_type.endswith('*'):
                        mapped_type = mapped_type + '*'
                
                const = 'const ' if param.get('const', False) else ''
                formatted.append(f"{const}{mapped_type} {name}")
            else:
                formatted.append(str(param))
        
        return ', '.join(formatted)
    
    def _generate_struct(self, type_def: Dict[str, Any]) -> str:
        """Generate C99 struct from type definition.
        
        Args:
            type_def: Type definition IR dictionary.
            
        Returns:
            C99 struct definition string.
        """
        name = type_def.get('name', 'MyStruct')
        fields = type_def.get('fields', [])
        
        lines = [f"typedef struct {name} {{"]
        
        indent = ' ' * self.indent_size
        for field in fields:
            if isinstance(field, dict):
                f_name = field.get('name', 'field')
                f_type = self.stmt_translator.map_type(field.get('type', 'int'))
                
                # Handle arrays
                array_size = field.get('array_size')
                if array_size:
                    lines.append(f"{indent}{f_type} {f_name}[{array_size}];")
                else:
                    lines.append(f"{indent}{f_type} {f_name};")
            else:
                lines.append(f"{indent}int {field};")
        
        lines.append(f"}} {name};")
        
        return '\n'.join(lines)
    
    def _default_return_value(self, c_type: str) -> str:
        """Get default return value for a C type.
        
        Args:
            c_type: C type string.
            
        Returns:
            Default value for the type.
        """
        if c_type in ('void',):
            return ''
        if c_type in ('bool',):
            return 'false'
        if c_type.endswith('*'):
            return 'NULL'
        if c_type in ('float', 'double'):
            return '0.0'
        return '0'
