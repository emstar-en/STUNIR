#!/usr/bin/env python3
"""STUNIR C++ Code Generator.

This module implements C++-specific code generation by extending the
statement and expression translators.

Part of Phase 3 (Advanced Code Generation) of STUNIR Enhancement Integration.

Features:
- Modern C++17/20 features
- Smart pointer support
- Template support
- RAII patterns
- Full control flow support
- Range-based for loops

Usage:
    from tools.codegen.cpp_generator import CppCodeGenerator
    
    generator = CppCodeGenerator(enhancement_context=ctx)
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


class CppExpressionTranslator(ExpressionTranslator):
    """C++-specific expression translator."""
    
    TARGET = 'cpp'
    LOGICAL_AND = '&&'
    LOGICAL_OR = '||'
    LOGICAL_NOT = '!'
    
    # C++ type mapping
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
        'string': 'std::string',
        'void': 'void',
        'unit': 'void',
        'char': 'char',
        'byte': 'uint8_t',
        'usize': 'size_t',
        'isize': 'ptrdiff_t',
        'any': 'std::any',
        'auto': 'auto',
    }
    
    def _build_operator_map(self) -> Dict[str, str]:
        """Build C++-specific operator mappings."""
        base = super()._build_operator_map()
        base['//'] = '/'  # Integer division
        base['**'] = 'std::pow'
        base['and'] = '&&'
        base['&&'] = '&&'
        base['or'] = '||'
        base['||'] = '||'
        base['not'] = '!'
        base['!'] = '!'
        return base
    
    def translate_literal(self, value: Any, lit_type: str) -> str:
        """Translate a literal value to C++."""
        if value is None:
            return 'nullptr'
        
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
        
        if lit_type in ('i64', 'u64'):
            suffix = 'LL' if lit_type == 'i64' else 'ULL'
            return f"{int(value)}{suffix}"
        
        if lit_type in ('u8', 'u16', 'u32'):
            return f"{int(value)}U"
        
        if isinstance(value, int):
            return str(value)
        
        return str(value)
    
    def translate_variable(self, name: str, var_type: Optional[str] = None) -> str:
        """Translate a variable reference to C++."""
        return name
    
    def translate_binary_op(self, left: str, op: str, right: str) -> str:
        """Translate a binary operation to C++."""
        if op == 'std::pow':
            return f"std::pow({left}, {right})"
        return f"({left} {op} {right})"
    
    def translate_unary_op(self, op: str, operand: str) -> str:
        """Translate a unary operation to C++."""
        return f"({op}{operand})"
    
    def translate_function_call(
        self,
        func_name: str,
        args: List[str],
        receiver: Optional[str] = None
    ) -> str:
        """Translate a function call to C++."""
        args_str = ', '.join(args)
        if receiver:
            return f"{receiver}.{func_name}({args_str})"
        return f"{func_name}({args_str})"
    
    def translate_ternary(self, condition: str, then_val: str, else_val: str) -> str:
        """Translate a ternary expression to C++."""
        return f"(({condition}) ? ({then_val}) : ({else_val}))"
    
    def translate_cast(self, value: str, target_type: str) -> str:
        """Translate a type cast to C++ (static_cast)."""
        mapped = self.map_type(target_type)
        return f"static_cast<{mapped}>({value})"
    
    def translate_struct_literal(
        self,
        struct_type: str,
        fields: Dict[str, str]
    ) -> str:
        """Translate a struct literal to C++ aggregate initialization."""
        field_strs = [f".{k} = {v}" for k, v in fields.items()]
        return f"{struct_type}{{{', '.join(field_strs)}}}"
    
    def translate_lambda(
        self,
        params: List[Dict[str, Any]],
        body: Any,
        return_type: Optional[str] = None
    ) -> str:
        """Translate a lambda to C++ lambda expression."""
        param_parts = []
        for p in params:
            if isinstance(p, dict):
                name = p.get('name', 'x')
                p_type = self.map_type(p.get('type', 'auto'))
                param_parts.append(f"{p_type} {name}")
            else:
                param_parts.append(f"auto {p}")
        params_str = ', '.join(param_parts)
        
        ret_annotation = ''
        if return_type:
            ret_annotation = f" -> {self.map_type(return_type)}"
        
        if isinstance(body, dict):
            body_code = self.translate_expression(body)
            return f"[&]({params_str}){ret_annotation} {{ return {body_code}; }}"
        elif isinstance(body, list):
            return f"[&]({params_str}){ret_annotation} {{ /* ... */ }}"
        else:
            return f"[&]({params_str}){ret_annotation} {{ return {body}; }}"
    
    def translate_new(self, class_name: str, args: List[str]) -> str:
        """Translate new expression to C++ (prefer make_unique)."""
        args_str = ', '.join(args)
        return f"std::make_unique<{class_name}>({args_str})"
    
    def translate_reference(self, target: str, mutable: bool = False) -> str:
        """Translate reference to C++."""
        return f"&{target}"
    
    def translate_dereference(self, target: str) -> str:
        """Translate dereference to C++."""
        return f"*{target}"
    
    def translate_sizeof(self, target: str) -> str:
        """Translate sizeof to C++."""
        return f"sizeof({target})"
    
    def translate_typeof(self, target: str) -> str:
        """Translate typeof to C++ decltype."""
        return f"decltype({target})"
    
    def map_type(self, ir_type: str) -> str:
        """Map IR type to C++ type."""
        # Handle pointer types
        if ir_type.endswith('*'):
            base = ir_type[:-1].strip()
            mapped_base = self.TYPE_MAP.get(base, base)
            return f"{mapped_base}*"
        # Handle reference types
        if ir_type.endswith('&'):
            base = ir_type[:-1].strip()
            mapped_base = self.TYPE_MAP.get(base, base)
            return f"{mapped_base}&"
        # Handle array types
        if ir_type.endswith('[]'):
            base = ir_type[:-2]
            mapped_base = self.TYPE_MAP.get(base, base)
            return f"std::vector<{mapped_base}>"
        # Handle generic types
        if '<' in ir_type:
            return ir_type
        return self.TYPE_MAP.get(ir_type, ir_type)


class CppStatementTranslator(StatementTranslator):
    """C++-specific statement translator."""
    
    TARGET = 'cpp'
    STATEMENT_TERMINATOR = ';'
    
    TYPE_MAP = CppExpressionTranslator.TYPE_MAP
    
    def __init__(
        self,
        enhancement_context: Optional['EnhancementContext'] = None,
        indent_size: int = 4,
        use_auto: bool = True
    ):
        """Initialize C++ statement translator."""
        super().__init__(enhancement_context, indent_size)
        self.use_auto = use_auto
    
    def translate_variable_declaration(
        self,
        var_name: str,
        var_type: str,
        init_value: Optional[str],
        mutable: bool = True,
        indent: int = 0
    ) -> str:
        """Translate a variable declaration to C++."""
        ind = self.get_indent(indent)
        mapped_type = self.map_type(var_type)
        
        const_kw = '' if mutable else 'const '
        
        if init_value is not None:
            if self.use_auto and mapped_type != 'auto':
                # Use auto with initialization for type inference
                return f"{ind}{const_kw}auto {var_name} = {init_value};"
            return f"{ind}{const_kw}{mapped_type} {var_name} = {init_value};"
        else:
            return f"{ind}{const_kw}{mapped_type} {var_name};"
    
    def translate_assignment(
        self,
        target: str,
        value: str,
        indent: int = 0
    ) -> str:
        """Translate an assignment to C++."""
        ind = self.get_indent(indent)
        return f"{ind}{target} = {value};"
    
    def translate_compound_assignment(
        self,
        target: str,
        op: str,
        value: str,
        indent: int = 0
    ) -> str:
        """Translate a compound assignment to C++."""
        ind = self.get_indent(indent)
        return f"{ind}{target} {op} {value};"
    
    def translate_return(
        self,
        value: Optional[str],
        indent: int = 0
    ) -> str:
        """Translate a return statement to C++."""
        ind = self.get_indent(indent)
        if value is None:
            return f"{ind}return;"
        return f"{ind}return {value};"
    
    def translate_expression_statement(
        self,
        expr: str,
        indent: int = 0
    ) -> str:
        """Translate an expression statement to C++."""
        ind = self.get_indent(indent)
        return f"{ind}{expr};"
    
    def _empty_body(self, indent: int = 0) -> str:
        """Generate C++ empty body."""
        ind = self.get_indent(indent)
        return f"{ind}// empty"
    
    def _translate_none(self) -> str:
        """Translate None value for C++."""
        return "nullptr"
    
    def _translate_bool(self, value: bool) -> str:
        """Translate boolean value for C++."""
        return "true" if value else "false"
    
    def _translate_list(self, value: list) -> str:
        """Translate a list literal for C++ (initializer list)."""
        items = [self._translate_value(v) for v in value]
        return f"{{{', '.join(items)}}}"
    
    def map_type(self, ir_type: str) -> str:
        """Map IR type to C++ type."""
        if ir_type.endswith('*'):
            base = ir_type[:-1].strip()
            mapped_base = self.TYPE_MAP.get(base, base)
            return f"{mapped_base}*"
        if ir_type.endswith('&'):
            base = ir_type[:-1].strip()
            mapped_base = self.TYPE_MAP.get(base, base)
            return f"{mapped_base}&"
        if ir_type.endswith('[]'):
            base = ir_type[:-2]
            mapped_base = self.TYPE_MAP.get(base, base)
            return f"std::vector<{mapped_base}>"
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
        """Translate an if statement to C++."""
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
        """Translate a while loop to C++."""
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
        """Translate a C-style for loop to C++."""
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
        """Translate a for-each loop to C++ range-based for."""
        ind = self.get_indent(indent)
        
        if var_type:
            mapped_type = self.map_type(var_type)
            lines = [f"{ind}for (const {mapped_type}& {var_name} : {iterable}) {{"]
        else:
            lines = [f"{ind}for (const auto& {var_name} : {iterable}) {{"]
        
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
        """Translate a range-based for loop to C++."""
        ind = self.get_indent(indent)
        
        if step:
            update = f"{var_name} += {step}"
        else:
            update = f"++{var_name}"
        
        lines = [f"{ind}for (auto {var_name} = {start}; {var_name} < {end}; {update}) {{"]
        
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
        """Translate a do-while loop to C++."""
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
        """Translate a switch statement to C++."""
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
            else:
                lines.append(f"{self.get_indent(indent + 2)}[[fallthrough]];")
        
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
        """Translate a block of statements to C++."""
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
        """Translate an infinite loop to C++ while(true)."""
        return self.translate_while_loop("true", body, indent)


class CppCodeGenerator:
    """Complete C++ code generator."""
    
    TARGET = 'cpp'
    FILE_EXTENSION = 'cpp'
    HEADER_EXTENSION = 'hpp'
    
    def __init__(
        self,
        enhancement_context: Optional['EnhancementContext'] = None,
        indent_size: int = 4,
        use_auto: bool = True
    ):
        """Initialize the C++ code generator."""
        self.enhancement_context = enhancement_context
        self.indent_size = indent_size
        self.use_auto = use_auto
        
        self.expr_translator = CppExpressionTranslator(enhancement_context)
        self.stmt_translator = CppStatementTranslator(
            enhancement_context,
            indent_size=indent_size,
            use_auto=use_auto
        )
        
        self.stmt_translator.set_expression_translator(self.expr_translator)
    
    def generate_function(self, func_ir: Dict[str, Any], declaration_only: bool = False) -> str:
        """Generate C++ function code from IR."""
        name = func_ir.get('name', 'unnamed')
        params = func_ir.get('params', [])
        return_type = func_ir.get('return_type', 'void')
        body = func_ir.get('body', [])
        is_static = func_ir.get('static', False)
        is_inline = func_ir.get('inline', False)
        is_constexpr = func_ir.get('constexpr', False)
        is_noexcept = func_ir.get('noexcept', False)
        docstring = func_ir.get('docstring', func_ir.get('description'))
        template_params = func_ir.get('template', [])
        
        lines = []
        indent = ' ' * self.indent_size
        
        # Doxygen comment
        if docstring:
            lines.append('/**')
            lines.append(f' * @brief {docstring}')
            for param in params:
                if isinstance(param, dict):
                    p_name = param.get('name', 'arg')
                    lines.append(f' * @param {p_name}')
            if return_type != 'void':
                lines.append(f' * @return')
            lines.append(' */')
        
        # Template declaration
        if template_params:
            template_str = ', '.join([f"typename {t}" for t in template_params])
            lines.append(f"template<{template_str}>")
        
        # Function signature
        params_str = self._format_params(params)
        ret_type_str = self.stmt_translator.map_type(return_type)
        
        prefix = ''
        if is_static:
            prefix += 'static '
        if is_inline:
            prefix += 'inline '
        if is_constexpr:
            prefix += 'constexpr '
        
        suffix = ''
        if is_noexcept:
            suffix += ' noexcept'
        
        if declaration_only:
            lines.append(f"{prefix}{ret_type_str} {name}({params_str}){suffix};")
            return '\n'.join(lines)
        
        lines.append(f"{prefix}{ret_type_str} {name}({params_str}){suffix} {{")
        
        # Body
        if body:
            body_code = self.stmt_translator.translate_statements(body, indent=1)
            lines.append(body_code)
        else:
            lines.append(f"{indent}// empty")
        
        lines.append("}")
        
        return '\n'.join(lines)
    
    def generate_class(self, class_ir: Dict[str, Any]) -> str:
        """Generate C++ class from IR."""
        name = class_ir.get('name', 'MyClass')
        fields = class_ir.get('fields', [])
        methods = class_ir.get('methods', class_ir.get('functions', []))
        base_classes = class_ir.get('bases', class_ir.get('extends', []))
        template_params = class_ir.get('template', [])
        
        lines = []
        indent = ' ' * self.indent_size
        
        # Template declaration
        if template_params:
            template_str = ', '.join([f"typename {t}" for t in template_params])
            lines.append(f"template<{template_str}>")
        
        # Class declaration
        base_clause = ''
        if base_classes:
            if isinstance(base_classes, list):
                base_clause = ' : ' + ', '.join([f"public {b}" for b in base_classes])
            else:
                base_clause = f" : public {base_classes}"
        
        lines.append(f"class {name}{base_clause} {{")
        
        # Public section
        public_fields = [f for f in fields if isinstance(f, dict) and f.get('visibility', 'private') == 'public']
        public_methods = [m for m in methods if isinstance(m, dict) and m.get('visibility', 'public') == 'public']
        
        if public_fields or public_methods:
            lines.append("public:")
            for field in public_fields:
                f_name = field.get('name', 'field')
                f_type = self.stmt_translator.map_type(field.get('type', 'int'))
                f_static = 'static ' if field.get('static', False) else ''
                f_const = 'const ' if not field.get('mutable', True) else ''
                lines.append(f"{indent}{f_static}{f_const}{f_type} {f_name};")
            for method in public_methods:
                method_code = self.generate_function(method, declaration_only=True)
                lines.append(f"{indent}{method_code}")
        
        # Private section
        private_fields = [f for f in fields if isinstance(f, dict) and f.get('visibility', 'private') == 'private']
        private_methods = [m for m in methods if isinstance(m, dict) and m.get('visibility', 'public') == 'private']
        
        if private_fields or private_methods:
            lines.append("")
            lines.append("private:")
            for field in private_fields:
                f_name = field.get('name', 'field')
                f_type = self.stmt_translator.map_type(field.get('type', 'int'))
                f_static = 'static ' if field.get('static', False) else ''
                f_const = 'const ' if not field.get('mutable', True) else ''
                lines.append(f"{indent}{f_static}{f_const}{f_type} {f_name}_;")
            for method in private_methods:
                method_code = self.generate_function(method, declaration_only=True)
                lines.append(f"{indent}{method_code}")
        
        lines.append("};")
        
        return '\n'.join(lines)
    
    def generate_module(self, module_ir: Dict[str, Any]) -> str:
        """Generate C++ source file from IR."""
        lines = []
        
        module_name = module_ir.get('name', module_ir.get('ir_module', 'module'))
        lines.append(f"/**")
        lines.append(f" * @file {module_name}.cpp")
        lines.append(f" * @brief Generated by STUNIR C++ Code Generator")
        lines.append(f" */")
        lines.append('')
        
        # Standard includes
        lines.append('#include <cstdint>')
        lines.append('#include <string>')
        lines.append('#include <vector>')
        lines.append('#include <memory>')
        lines.append('#include <cmath>')
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
        
        # Namespace
        namespace = module_ir.get('namespace', module_name.replace('-', '_').replace('.', '_'))
        lines.append(f"namespace {namespace} {{")
        lines.append('')
        
        # Types/Classes
        types = module_ir.get('types', module_ir.get('ir_types', []))
        for type_def in types:
            lines.append(self.generate_class(type_def))
            lines.append('')
        
        # Functions
        functions = module_ir.get('functions', module_ir.get('ir_functions', []))
        for func in functions:
            lines.append(self.generate_function(func))
            lines.append('')
        
        lines.append(f"}} // namespace {namespace}")
        
        return '\n'.join(lines)
    
    def generate_header(self, module_ir: Dict[str, Any]) -> str:
        """Generate C++ header file from IR."""
        lines = []
        
        module_name = module_ir.get('name', module_ir.get('ir_module', 'module'))
        guard_name = f"{module_name.upper().replace('.', '_').replace('-', '_')}_HPP"
        
        lines.append(f"#ifndef {guard_name}")
        lines.append(f"#define {guard_name}")
        lines.append('')
        lines.append(f"/**")
        lines.append(f" * @file {module_name}.hpp")
        lines.append(f" * @brief Generated by STUNIR C++ Code Generator")
        lines.append(f" */")
        lines.append('')
        
        # Standard includes
        lines.append('#include <cstdint>')
        lines.append('#include <string>')
        lines.append('#include <vector>')
        lines.append('#include <memory>')
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
        
        # Namespace
        namespace = module_ir.get('namespace', module_name.replace('-', '_').replace('.', '_'))
        lines.append(f"namespace {namespace} {{")
        lines.append('')
        
        # Types/Classes
        types = module_ir.get('types', module_ir.get('ir_types', []))
        for type_def in types:
            lines.append(self.generate_class(type_def))
            lines.append('')
        
        # Function declarations
        functions = module_ir.get('functions', module_ir.get('ir_functions', []))
        for func in functions:
            lines.append(self.generate_function(func, declaration_only=True))
            lines.append('')
        
        lines.append(f"}} // namespace {namespace}")
        lines.append('')
        lines.append(f"#endif // {guard_name}")
        
        return '\n'.join(lines)
    
    def _format_params(self, params: List[Dict[str, Any]]) -> str:
        """Format function parameters with types."""
        if not params:
            return ''
        
        formatted = []
        for param in params:
            if isinstance(param, dict):
                name = param.get('name', 'arg')
                p_type = self.stmt_translator.map_type(param.get('type', 'int'))
                const = 'const ' if param.get('const', False) else ''
                ref = '&' if param.get('ref', False) else ''
                default = param.get('default')
                if default is not None:
                    default_str = self.expr_translator.translate_expression(default)
                    formatted.append(f"{const}{p_type}{ref} {name} = {default_str}")
                else:
                    formatted.append(f"{const}{p_type}{ref} {name}")
            else:
                formatted.append(str(param))
        
        return ', '.join(formatted)
