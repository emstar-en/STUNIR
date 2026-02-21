#!/usr/bin/env python3
"""STUNIR Java Code Generator.

This module implements Java-specific code generation by extending the
statement and expression translators.

Part of Phase 3 (Advanced Code Generation) of STUNIR Enhancement Integration.

Features:
- Java 8+ features (lambdas, streams)
- Proper class structure generation
- Generic type support
- Full control flow support
- JavaDoc generation

Usage:
    from tools.codegen.java_generator import JavaCodeGenerator
    
    generator = JavaCodeGenerator(enhancement_context=ctx)
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


class JavaExpressionTranslator(ExpressionTranslator):
    """Java-specific expression translator."""
    
    TARGET = 'java'
    LOGICAL_AND = '&&'
    LOGICAL_OR = '||'
    LOGICAL_NOT = '!'
    
    # Java type mapping
    TYPE_MAP = {
        'i8': 'byte',
        'i16': 'short',
        'i32': 'int',
        'i64': 'long',
        'u8': 'byte',
        'u16': 'char',
        'u32': 'int',
        'u64': 'long',
        'f32': 'float',
        'f64': 'double',
        'bool': 'boolean',
        'string': 'String',
        'void': 'void',
        'unit': 'void',
        'char': 'char',
        'byte': 'byte',
        'usize': 'long',
        'isize': 'long',
        'any': 'Object',
        'object': 'Object',
    }
    
    # Boxed type mapping for generics
    BOXED_TYPES = {
        'byte': 'Byte',
        'short': 'Short',
        'int': 'Integer',
        'long': 'Long',
        'float': 'Float',
        'double': 'Double',
        'boolean': 'Boolean',
        'char': 'Character',
    }
    
    def _build_operator_map(self) -> Dict[str, str]:
        """Build Java-specific operator mappings."""
        base = super()._build_operator_map()
        base['//'] = '/'  # Integer division (Java divides ints as ints)
        base['**'] = 'Math.pow'  # Power
        base['and'] = '&&'
        base['&&'] = '&&'
        base['or'] = '||'
        base['||'] = '||'
        base['not'] = '!'
        base['!'] = '!'
        return base
    
    def translate_literal(self, value: Any, lit_type: str) -> str:
        """Translate a literal value to Java."""
        if value is None:
            return 'null'
        
        if lit_type == 'bool' or isinstance(value, bool):
            return 'true' if value else 'false'
        
        if lit_type == 'string' or isinstance(value, str):
            if isinstance(value, str):
                escaped = value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
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
        
        if lit_type == 'i64' or lit_type == 'u64':
            return f"{int(value)}L"
        
        if isinstance(value, int):
            return str(value)
        
        return str(value)
    
    def translate_variable(self, name: str, var_type: Optional[str] = None) -> str:
        """Translate a variable reference to Java."""
        return name
    
    def translate_binary_op(self, left: str, op: str, right: str) -> str:
        """Translate a binary operation to Java."""
        if op == 'Math.pow':
            return f"Math.pow({left}, {right})"
        return f"({left} {op} {right})"
    
    def translate_unary_op(self, op: str, operand: str) -> str:
        """Translate a unary operation to Java."""
        return f"({op}{operand})"
    
    def translate_function_call(
        self,
        func_name: str,
        args: List[str],
        receiver: Optional[str] = None
    ) -> str:
        """Translate a function call to Java."""
        args_str = ', '.join(args)
        if receiver:
            return f"{receiver}.{func_name}({args_str})"
        return f"{func_name}({args_str})"
    
    def translate_ternary(self, condition: str, then_val: str, else_val: str) -> str:
        """Translate a ternary expression to Java."""
        return f"({condition} ? {then_val} : {else_val})"
    
    def translate_cast(self, value: str, target_type: str) -> str:
        """Translate a type cast to Java."""
        mapped = self.map_type(target_type)
        return f"(({mapped}) {value})"
    
    def translate_struct_literal(
        self,
        struct_type: str,
        fields: Dict[str, str]
    ) -> str:
        """Translate a struct literal to Java (new object with builder pattern)."""
        # Java doesn't have struct literals, use Map.of or constructor
        field_strs = [f'"{k}", {v}' for k, v in fields.items()]
        return f"Map.of({', '.join(field_strs)})"
    
    def translate_lambda(
        self,
        params: List[Dict[str, Any]],
        body: Any,
        return_type: Optional[str] = None
    ) -> str:
        """Translate a lambda to Java lambda expression."""
        param_parts = []
        for p in params:
            if isinstance(p, dict):
                name = p.get('name', 'x')
                p_type = self.map_type(p.get('type', 'Object'))
                param_parts.append(f"{p_type} {name}")
            else:
                param_parts.append(str(p))
        params_str = ', '.join(param_parts)
        
        if isinstance(body, dict):
            body_code = self.translate_expression(body)
            return f"({params_str}) -> {body_code}"
        elif isinstance(body, list):
            return f"({params_str}) -> {{ /* ... */ }}"
        else:
            return f"({params_str}) -> {body}"
    
    def translate_new(self, class_name: str, args: List[str]) -> str:
        """Translate new expression to Java."""
        args_str = ', '.join(args)
        return f"new {class_name}({args_str})"
    
    def translate_typeof(self, target: str) -> str:
        """Translate typeof to Java instanceof check (limited)."""
        return f"{target}.getClass()"
    
    def translate_reference(self, target: str, mutable: bool = False) -> str:
        """Java doesn't have references in same sense - return as is."""
        return target
    
    def translate_dereference(self, target: str) -> str:
        """Java doesn't have pointers - return as is."""
        return target
    
    def map_type(self, ir_type: str) -> str:
        """Map IR type to Java type."""
        # Handle array types
        if ir_type.endswith('[]'):
            base = ir_type[:-2]
            mapped_base = self.TYPE_MAP.get(base, base)
            return f"{mapped_base}[]"
        # Handle generic types
        if '<' in ir_type:
            return ir_type
        return self.TYPE_MAP.get(ir_type, ir_type)
    
    def get_boxed_type(self, primitive: str) -> str:
        """Get boxed type for a primitive."""
        return self.BOXED_TYPES.get(primitive, primitive)


class JavaStatementTranslator(StatementTranslator):
    """Java-specific statement translator."""
    
    TARGET = 'java'
    STATEMENT_TERMINATOR = ';'
    
    TYPE_MAP = JavaExpressionTranslator.TYPE_MAP
    
    def translate_variable_declaration(
        self,
        var_name: str,
        var_type: str,
        init_value: Optional[str],
        mutable: bool = True,
        indent: int = 0
    ) -> str:
        """Translate a variable declaration to Java."""
        ind = self.get_indent(indent)
        mapped_type = self.map_type(var_type)
        
        final_kw = '' if mutable else 'final '
        
        if init_value is not None:
            return f"{ind}{final_kw}{mapped_type} {var_name} = {init_value};"
        else:
            return f"{ind}{final_kw}{mapped_type} {var_name};"
    
    def translate_assignment(
        self,
        target: str,
        value: str,
        indent: int = 0
    ) -> str:
        """Translate an assignment to Java."""
        ind = self.get_indent(indent)
        return f"{ind}{target} = {value};"
    
    def translate_compound_assignment(
        self,
        target: str,
        op: str,
        value: str,
        indent: int = 0
    ) -> str:
        """Translate a compound assignment to Java."""
        ind = self.get_indent(indent)
        return f"{ind}{target} {op} {value};"
    
    def translate_return(
        self,
        value: Optional[str],
        indent: int = 0
    ) -> str:
        """Translate a return statement to Java."""
        ind = self.get_indent(indent)
        if value is None:
            return f"{ind}return;"
        return f"{ind}return {value};"
    
    def translate_expression_statement(
        self,
        expr: str,
        indent: int = 0
    ) -> str:
        """Translate an expression statement to Java."""
        ind = self.get_indent(indent)
        return f"{ind}{expr};"
    
    def _empty_body(self, indent: int = 0) -> str:
        """Generate Java empty body."""
        ind = self.get_indent(indent)
        return f"{ind}// empty"
    
    def _translate_none(self) -> str:
        """Translate None value for Java."""
        return "null"
    
    def _translate_bool(self, value: bool) -> str:
        """Translate boolean value for Java."""
        return "true" if value else "false"
    
    def _translate_list(self, value: list) -> str:
        """Translate a list literal for Java (Arrays.asList)."""
        items = [self._translate_value(v) for v in value]
        return f"Arrays.asList({', '.join(items)})"
    
    def map_type(self, ir_type: str) -> str:
        """Map IR type to Java type."""
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
        """Translate an if statement to Java."""
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
        """Translate a while loop to Java."""
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
        """Translate a C-style for loop to Java."""
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
        """Translate a for-each loop to Java enhanced for."""
        ind = self.get_indent(indent)
        mapped_type = self.map_type(var_type) if var_type else 'var'
        
        lines = [f"{ind}for ({mapped_type} {var_name} : {iterable}) {{"]
        
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
        """Translate a range-based for loop to Java."""
        ind = self.get_indent(indent)
        
        if step:
            update = f"{var_name} += {step}"
        else:
            update = f"{var_name}++"
        
        lines = [f"{ind}for (int {var_name} = {start}; {var_name} < {end}; {update}) {{"]
        
        if body:
            lines.append(self.translate_statements(body, indent + 1))
        else:
            lines.append(f"{self.get_indent(indent + 1)}// empty")
        
        lines.append(f"{ind}}}")
        return '\n'.join(lines)
    
    def translate_do_while(
        self,
        condition: str,
        body: List[Dict[str, Any]],
        indent: int = 0
    ) -> str:
        """Translate a do-while loop to Java."""
        ind = self.get_indent(indent)
        lines = [f"{ind}do {{"]
        
        if body:
            lines.append(self.translate_statements(body, indent + 1))
        else:
            lines.append(f"{self.get_indent(indent + 1)}// empty")
        
        lines.append(f"{ind}}} while ({condition});")
        return '\n'.join(lines)
    
    def translate_switch_statement(
        self,
        value: str,
        cases: List[Dict[str, Any]],
        default: Optional[List[Dict[str, Any]]],
        indent: int = 0
    ) -> str:
        """Translate a switch statement to Java."""
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
        """Translate a block of statements to Java."""
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
        """Translate an infinite loop to Java while(true)."""
        return self.translate_while_loop("true", body, indent)


class JavaCodeGenerator:
    """Complete Java code generator."""
    
    TARGET = 'java'
    FILE_EXTENSION = 'java'
    
    def __init__(
        self,
        enhancement_context: Optional['EnhancementContext'] = None,
        indent_size: int = 4
    ):
        """Initialize the Java code generator."""
        self.enhancement_context = enhancement_context
        self.indent_size = indent_size
        
        self.expr_translator = JavaExpressionTranslator(enhancement_context)
        self.stmt_translator = JavaStatementTranslator(
            enhancement_context,
            indent_size=indent_size
        )
        
        self.stmt_translator.set_expression_translator(self.expr_translator)
    
    def generate_function(self, func_ir: Dict[str, Any]) -> str:
        """Generate Java method code from IR."""
        name = func_ir.get('name', 'unnamed')
        params = func_ir.get('params', [])
        return_type = func_ir.get('return_type', 'void')
        body = func_ir.get('body', [])
        visibility = func_ir.get('visibility', 'public')
        is_static = func_ir.get('static', False)
        docstring = func_ir.get('docstring', func_ir.get('description'))
        throws = func_ir.get('throws', [])
        
        lines = []
        indent = ' ' * self.indent_size
        
        # JavaDoc comment
        if docstring or params or return_type != 'void':
            lines.append('/**')
            if docstring:
                lines.append(f' * {docstring}')
            for param in params:
                if isinstance(param, dict):
                    p_name = param.get('name', 'arg')
                    lines.append(f' * @param {p_name}')
            if return_type != 'void':
                lines.append(f' * @return')
            for exc in throws:
                lines.append(f' * @throws {exc}')
            lines.append(' */')
        
        # Method signature
        params_str = self._format_params(params)
        ret_type_str = self.stmt_translator.map_type(return_type)
        static_kw = 'static ' if is_static else ''
        throws_clause = f" throws {', '.join(throws)}" if throws else ''
        
        lines.append(f"{visibility} {static_kw}{ret_type_str} {name}({params_str}){throws_clause} {{")
        
        # Body
        if body:
            body_code = self.stmt_translator.translate_statements(body, indent=1)
            lines.append(body_code)
        else:
            lines.append(f"{indent}// empty")
        
        lines.append("}")
        
        return '\n'.join(lines)
    
    def generate_class(self, class_ir: Dict[str, Any]) -> str:
        """Generate Java class from IR."""
        name = class_ir.get('name', 'MyClass')
        fields = class_ir.get('fields', [])
        methods = class_ir.get('methods', class_ir.get('functions', []))
        visibility = class_ir.get('visibility', 'public')
        extends = class_ir.get('extends')
        implements = class_ir.get('implements', [])
        is_abstract = class_ir.get('abstract', False)
        
        lines = []
        indent = ' ' * self.indent_size
        
        # Class declaration
        abstract_kw = 'abstract ' if is_abstract else ''
        extends_clause = f" extends {extends}" if extends else ''
        implements_clause = f" implements {', '.join(implements)}" if implements else ''
        
        lines.append(f"{visibility} {abstract_kw}class {name}{extends_clause}{implements_clause} {{")
        lines.append('')
        
        # Fields
        for field in fields:
            if isinstance(field, dict):
                f_name = field.get('name', 'field')
                f_type = self.stmt_translator.map_type(field.get('type', 'Object'))
                f_vis = field.get('visibility', 'private')
                f_static = 'static ' if field.get('static', False) else ''
                f_final = 'final ' if not field.get('mutable', True) else ''
                init = field.get('init')
                if init:
                    init_code = self.expr_translator.translate_expression(init)
                    lines.append(f"{indent}{f_vis} {f_static}{f_final}{f_type} {f_name} = {init_code};")
                else:
                    lines.append(f"{indent}{f_vis} {f_static}{f_final}{f_type} {f_name};")
        
        if fields:
            lines.append('')
        
        # Methods
        for method in methods:
            method_code = self.generate_function(method)
            # Add indent to method
            indented_method = '\n'.join(f"{indent}{line}" for line in method_code.split('\n'))
            lines.append(indented_method)
            lines.append('')
        
        lines.append("}")
        
        return '\n'.join(lines)
    
    def generate_module(self, module_ir: Dict[str, Any]) -> str:
        """Generate Java source file from IR."""
        lines = []
        
        # Package
        package = module_ir.get('package', module_ir.get('name', module_ir.get('ir_module', '')))
        if package:
            package = package.replace('-', '_').replace('/', '.')
            lines.append(f"package {package};")
            lines.append('')
        
        # Imports
        imports = module_ir.get('imports', [])
        if imports:
            for imp in imports:
                if isinstance(imp, dict):
                    mod = imp.get('module', '')
                    items = imp.get('items', [])
                    if items:
                        for item in items:
                            lines.append(f"import {mod}.{item};")
                    else:
                        lines.append(f"import {mod}.*;")
                else:
                    lines.append(f"import {imp};")
            lines.append('')
        
        # Default imports
        lines.append("import java.util.*;")
        lines.append('')
        
        # File comment
        module_name = module_ir.get('name', module_ir.get('ir_module', 'Module'))
        lines.append(f"/**")
        lines.append(f" * Module: {module_name}")
        lines.append(f" * Generated by STUNIR Java Code Generator")
        lines.append(f" */")
        
        # Classes
        types = module_ir.get('types', module_ir.get('ir_types', []))
        functions = module_ir.get('functions', module_ir.get('ir_functions', []))
        
        # If there are standalone functions, wrap in a utility class
        if functions and not types:
            class_name = ''.join(word.capitalize() for word in module_name.split('_'))
            if not class_name:
                class_name = 'Module'
            
            class_ir = {
                'name': class_name,
                'methods': [dict(f, static=True) for f in functions],
                'fields': [],
            }
            lines.append(self.generate_class(class_ir))
        else:
            for type_def in types:
                lines.append(self.generate_class(type_def))
                lines.append('')
        
        return '\n'.join(lines)
    
    def _format_params(self, params: List[Dict[str, Any]]) -> str:
        """Format method parameters with types."""
        formatted = []
        for param in params:
            if isinstance(param, dict):
                name = param.get('name', 'arg')
                p_type = self.stmt_translator.map_type(param.get('type', 'Object'))
                final_kw = 'final ' if param.get('final', False) else ''
                formatted.append(f"{final_kw}{p_type} {name}")
            else:
                formatted.append(str(param))
        return ', '.join(formatted)
