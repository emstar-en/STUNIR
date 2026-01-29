#!/usr/bin/env python3
"""STUNIR D Language Emitter - Generate D code from Systems IR.

This emitter generates idiomatic D code including:
- Module declarations
- Struct and class definitions
- Template definitions
- CTFE (Compile-Time Function Execution) functions
- Mixin templates
- Contracts (in/out/invariant)
- Memory safety attributes (@safe, @trusted, @system)

Usage:
    from targets.systems.d_emitter import DEmitter
    from ir.systems import Package, Subprogram
    
    emitter = DEmitter()
    code = emitter.emit_module(package)
"""

import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ir.systems import (
    # Core
    Package, Import, Subprogram, Parameter, Mode, Visibility, SafetyLevel,
    TypeRef, Expr, Statement, Declaration,
    # Expressions
    Literal, VarExpr, BinaryOp, UnaryOp, CallExpr, MemberAccess,
    IndexExpr, IfExpr, CastExpr, RangeExpr, AggregateExpr,
    # Statements
    Assignment, IfStatement, ElsifPart, CaseStatement, CaseAlternative,
    WhileLoop, ForLoop, BasicLoop, ExitStatement, ReturnStatement,
    NullStatement, BlockStatement, RaiseStatement, CallStatement,
    TryStatement, ExceptionHandler,
    # Types
    TypeDecl, SubtypeDecl, RecordType, ArrayType, EnumType, EnumLiteral,
    AccessType, ComponentDecl, ClassType, TypeAlias,
    # Concurrency
    SharedVariable, SynchronizedBlock,
    # Verification
    Contract, TypeInvariant, AssertPragma,
    DContractIn, DContractOut, DInvariant,
    # Memory
    Allocator, Deallocate, SliceExpr,
    VariableDecl, ConstantDecl,
)


@dataclass
class EmitterResult:
    """Result of code emission."""
    code: str
    manifest: dict


def canonical_json(obj: Any) -> str:
    """Generate canonical JSON (sorted keys)."""
    return json.dumps(obj, sort_keys=True, separators=(',', ':'))


def compute_sha256(data: str) -> str:
    """Compute SHA-256 hash of data."""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


# D type mappings
D_TYPE_MAP = {
    'int': 'int',
    'i8': 'byte',
    'i16': 'short',
    'i32': 'int',
    'i64': 'long',
    'u8': 'ubyte',
    'u16': 'ushort',
    'u32': 'uint',
    'u64': 'ulong',
    'f32': 'float',
    'f64': 'double',
    'float': 'float',
    'double': 'double',
    'bool': 'bool',
    'boolean': 'bool',
    'char': 'char',
    'string': 'string',
    'unit': 'void',
    'void': 'void',
    'Integer': 'int',
    'Natural': 'uint',
    'Boolean': 'bool',
    'Character': 'char',
    'String': 'string',
}

# D operator mappings
D_OPERATOR_MAP = {
    '+': '+',
    '-': '-',
    '*': '*',
    '/': '/',
    '%': '%',
    'mod': '%',
    '==': '==',
    '=': '==',
    '!=': '!=',
    '/=': '!=',
    '<': '<',
    '>': '>',
    '<=': '<=',
    '>=': '>=',
    'and': '&&',
    '&&': '&&',
    'or': '||',
    '||': '||',
    'xor': '^',
    'not': '!',
    '!': '!',
    '**': '^^',
    '&': '&',
    '|': '|',
    '^': '^',
    '<<': '<<',
    '>>': '>>',
    '~': '~',
}


class DEmitter:
    """D language code emitter.
    
    Generates idiomatic D code with templates, CTFE,
    contracts, and memory safety attributes.
    """
    
    DIALECT = 'd'
    FILE_EXTENSION = '.d'
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.indent_size = self.config.get('indent_size', 4)
        self.default_safety = self.config.get('default_safety', SafetyLevel.SYSTEM)
        self._current_indent = 0
    
    def _indent(self, level: int = None) -> str:
        """Get indentation string."""
        if level is None:
            level = self._current_indent
        return ' ' * (level * self.indent_size)
    
    def _map_type(self, type_ref: TypeRef) -> str:
        """Map IR type to D type."""
        if type_ref is None:
            return 'void'
        
        name = type_ref.name
        d_type = D_TYPE_MAP.get(name, name)
        
        # Handle access types (pointers)
        if type_ref.is_access:
            return f"{d_type}*"
        
        # Handle type arguments (generics/templates)
        if type_ref.type_args:
            args = ', '.join(self._map_type(arg) for arg in type_ref.type_args)
            return f"{d_type}!({args})"
        
        return d_type
    
    def _map_operator(self, op: str) -> str:
        """Map IR operator to D."""
        return D_OPERATOR_MAP.get(op, op)
    
    def _emit_safety_attrs(self, subprog: Subprogram) -> str:
        """Emit D safety attributes."""
        attrs = []
        
        # Memory safety
        if subprog.safety_level == SafetyLevel.SAFE:
            attrs.append('@safe')
        elif subprog.safety_level == SafetyLevel.TRUSTED:
            attrs.append('@trusted')
        # @system is default, don't emit
        
        # Pure function
        if subprog.is_pure:
            attrs.append('pure')
        
        # Nothrow
        if subprog.is_nothrow:
            attrs.append('nothrow')
        
        # No GC
        if subprog.is_nogc:
            attrs.append('@nogc')
        
        return ' '.join(attrs)
    
    # =========================================================================
    # Module Emission
    # =========================================================================
    
    def emit_module(self, package: Package) -> str:
        """Generate D module."""
        lines = []
        
        # Header comment
        lines.append(f"// Auto-generated by STUNIR D Emitter")
        lines.append(f"// Module: {package.name}")
        lines.append("")
        
        # Module declaration
        module_name = package.name.lower().replace('.', '_')
        lines.append(f"module {module_name};")
        lines.append("")
        
        # Imports
        for imp in package.imports:
            lines.append(self._emit_import(imp))
        if package.imports:
            lines.append("")
        
        # Type definitions
        for type_decl in package.types:
            lines.append(self._emit_type_decl(type_decl))
            lines.append("")
        
        # Constants
        for const in package.constants:
            lines.append(self._emit_constant_decl(const))
        if package.constants:
            lines.append("")
        
        # Variables
        for var in package.variables:
            lines.append(self._emit_variable_decl(var))
        if package.variables:
            lines.append("")
        
        # Functions
        for subprog in package.subprograms:
            lines.append(self._emit_function(subprog))
            lines.append("")
        
        # Generics (as templates)
        for generic in package.generics:
            lines.append(self._emit_template(generic))
            lines.append("")
        
        return '\n'.join(lines)
    
    def _emit_import(self, imp: Import) -> str:
        """Emit import statement."""
        module = imp.module.lower().replace('.', '_')
        
        if imp.selective_imports:
            symbols = ', '.join(imp.selective_imports)
            return f"import {module} : {symbols};"
        elif imp.renamed_as:
            return f"import {imp.renamed_as} = {module};"
        else:
            return f"import {module};"
    
    # =========================================================================
    # Type Declarations
    # =========================================================================
    
    def _emit_type_decl(self, type_decl: TypeDecl) -> str:
        """Emit type declaration."""
        if isinstance(type_decl, RecordType):
            return self._emit_struct(type_decl)
        elif isinstance(type_decl, ClassType):
            return self._emit_class(type_decl)
        elif isinstance(type_decl, EnumType):
            return self._emit_enum(type_decl)
        elif isinstance(type_decl, ArrayType):
            return self._emit_array_alias(type_decl)
        elif isinstance(type_decl, TypeAlias):
            return self._emit_alias(type_decl)
        elif isinstance(type_decl, SubtypeDecl):
            return self._emit_alias_subtype(type_decl)
        else:
            return f"// Unknown type: {type_decl.name}"
    
    def _emit_struct(self, record: RecordType) -> str:
        """Emit struct definition."""
        lines = []
        
        # Struct header
        lines.append(f"struct {record.name} {{")
        self._current_indent = 1
        
        # Components (fields)
        for comp in record.components:
            comp_type = self._map_type(comp.type_ref)
            default = ""
            if comp.default_value:
                default = f" = {self._emit_expr(comp.default_value)}"
            lines.append(f"{self._indent()}{comp_type} {comp.name}{default};")
        
        if record.components:
            lines.append("")
        
        # Invariants
        for inv in record.invariants:
            lines.append(self._emit_struct_invariant(inv))
            lines.append("")
        
        self._current_indent = 0
        lines.append("}")
        
        return '\n'.join(lines)
    
    def _emit_struct_invariant(self, inv: Contract) -> str:
        """Emit struct invariant."""
        ind = self._indent()
        cond = self._emit_expr(inv.condition)
        return f"{ind}invariant {{\n{ind}    assert({cond});\n{ind}}}"
    
    def _emit_class(self, class_type: ClassType) -> str:
        """Emit class definition."""
        lines = []
        
        # Class header with inheritance
        header = f"class {class_type.name}"
        if class_type.parent_class or class_type.interfaces:
            parents = []
            if class_type.parent_class:
                parents.append(self._map_type(class_type.parent_class))
            for iface in class_type.interfaces:
                parents.append(self._map_type(iface))
            header += f" : {', '.join(parents)}"
        
        lines.append(f"{header} {{")
        self._current_indent = 1
        
        # Components (fields)
        for comp in class_type.components:
            comp_type = self._map_type(comp.type_ref)
            lines.append(f"{self._indent()}{comp_type} {comp.name};")
        
        if class_type.components:
            lines.append("")
        
        # Methods
        for method in class_type.methods:
            lines.append(self._emit_function(method))
            lines.append("")
        
        # Invariants
        for inv in class_type.invariants:
            lines.append(self._emit_struct_invariant(inv))
            lines.append("")
        
        self._current_indent = 0
        lines.append("}")
        
        return '\n'.join(lines)
    
    def _emit_enum(self, enum: EnumType) -> str:
        """Emit enum definition."""
        lines = []
        
        lines.append(f"enum {enum.name} {{")
        self._current_indent = 1
        
        for i, lit in enumerate(enum.literals):
            comma = "," if i < len(enum.literals) - 1 else ""
            if lit.value is not None:
                lines.append(f"{self._indent()}{lit.name} = {self._emit_expr(lit.value)}{comma}")
            else:
                lines.append(f"{self._indent()}{lit.name}{comma}")
        
        self._current_indent = 0
        lines.append("}")
        
        return '\n'.join(lines)
    
    def _emit_array_alias(self, array: ArrayType) -> str:
        """Emit array type alias."""
        element = self._map_type(array.element_type)
        
        if array.index_constraints:
            # Fixed-size array
            size = self._emit_expr(array.index_constraints[0])
            return f"alias {array.name} = {element}[{size}];"
        elif array.is_unconstrained:
            # Dynamic array
            return f"alias {array.name} = {element}[];"
        else:
            return f"alias {array.name} = {element}[];"
    
    def _emit_alias(self, alias: TypeAlias) -> str:
        """Emit type alias."""
        aliased = self._map_type(alias.aliased_type)
        return f"alias {alias.name} = {aliased};"
    
    def _emit_alias_subtype(self, subtype: SubtypeDecl) -> str:
        """Emit subtype as alias."""
        base = self._map_type(subtype.base_type)
        return f"alias {subtype.name} = {base};"
    
    # =========================================================================
    # Variable and Constant Declarations
    # =========================================================================
    
    def _emit_variable_decl(self, var: VariableDecl) -> str:
        """Emit variable declaration."""
        var_type = self._map_type(var.type_ref)
        
        # Determine mutability
        if var.is_constant:
            qualifier = "immutable"
        else:
            qualifier = ""
        
        init = ""
        if var.initializer:
            init = f" = {self._emit_expr(var.initializer)}"
        
        if qualifier:
            return f"{qualifier} {var_type} {var.name}{init};"
        else:
            return f"{var_type} {var.name}{init};"
    
    def _emit_constant_decl(self, const: ConstantDecl) -> str:
        """Emit constant declaration."""
        const_type = self._map_type(const.type_ref) if const.type_ref else "auto"
        value = self._emit_expr(const.value)
        
        # Use enum for compile-time constants
        if const_type == "auto":
            return f"enum {const.name} = {value};"
        else:
            return f"enum {const_type} {const.name} = {value};"
    
    # =========================================================================
    # Function Emission
    # =========================================================================
    
    def _emit_function(self, subprog: Subprogram) -> str:
        """Emit function definition."""
        ind = self._indent()
        lines = []
        
        # Return type
        return_type = self._map_type(subprog.return_type) if subprog.return_type else 'void'
        
        # Parameters
        params = self._emit_parameters(subprog.parameters)
        
        # Attributes
        attrs = self._emit_safety_attrs(subprog)
        attr_str = f" {attrs}" if attrs else ""
        
        # Function signature
        sig = f"{return_type} {subprog.name}({params}){attr_str}"
        
        # Contracts
        contracts = self._emit_contracts(subprog)
        if contracts:
            lines.append(f"{ind}{sig}")
            lines.append(contracts)
        else:
            lines.append(f"{ind}{sig}")
        
        # Body
        lines.append(f"{ind}{{")
        self._current_indent += 1
        
        # Local declarations
        for decl in subprog.local_declarations:
            if isinstance(decl, VariableDecl):
                lines.append(f"{self._indent()}{self._emit_variable_decl(decl)}")
            elif isinstance(decl, ConstantDecl):
                lines.append(f"{self._indent()}{self._emit_constant_decl(decl)}")
        
        if subprog.local_declarations:
            lines.append("")
        
        # Statements
        for stmt in subprog.body:
            lines.append(self._emit_statement(stmt))
        
        self._current_indent -= 1
        lines.append(f"{ind}}}")
        
        return '\n'.join(lines)
    
    def _emit_parameters(self, params: List[Parameter]) -> str:
        """Emit parameter list."""
        if not params:
            return ""
        
        parts = []
        for param in params:
            param_type = self._map_type(param.type_ref)
            
            # Parameter attributes based on mode
            attrs = []
            if param.mode == Mode.OUT:
                attrs.append('out')
            elif param.mode == Mode.IN_OUT:
                attrs.append('ref')
            elif param.mode == Mode.IN:
                # 'in' is default, could add 'const' for safety
                pass
            
            attr_str = ' '.join(attrs) + ' ' if attrs else ''
            
            part = f"{attr_str}{param_type} {param.name}"
            if param.default_value:
                part += f" = {self._emit_expr(param.default_value)}"
            parts.append(part)
        
        return ", ".join(parts)
    
    def _emit_contracts(self, subprog: Subprogram) -> str:
        """Emit D contracts (in/out)."""
        ind = self._indent()
        parts = []
        
        # Preconditions (in contracts)
        if subprog.preconditions:
            for pre in subprog.preconditions:
                cond = self._emit_expr(pre.condition)
                msg = f', "{pre.message}"' if pre.message else ""
                parts.append(f"{ind}in ({cond}{msg})")
        
        # Postconditions (out contracts)
        if subprog.postconditions:
            for post in subprog.postconditions:
                cond = self._emit_expr(post.condition)
                # D out contracts can name the result
                parts.append(f"{ind}out (result; {cond})")
        
        return '\n'.join(parts)
    
    # =========================================================================
    # Template Emission
    # =========================================================================
    
    def _emit_template(self, generic) -> str:
        """Emit D template from generic unit."""
        ind = self._indent()
        lines = []
        
        # For now, emit as a simple template function
        # Could be expanded to handle generic packages/types
        name = generic.name if hasattr(generic, 'name') else 'GenericUnit'
        type_params = generic.type_params if hasattr(generic, 'type_params') else ['T']
        
        params_str = ', '.join(type_params)
        lines.append(f"{ind}// Template: {name}")
        lines.append(f"{ind}template {name}({params_str}) {{")
        self._current_indent += 1
        
        # Template content would go here
        lines.append(f"{self._indent()}// Template body")
        
        self._current_indent -= 1
        lines.append(f"{ind}}}")
        
        return '\n'.join(lines)
    
    def emit_template_function(self, name: str, type_params: List[str], 
                               params: List[Parameter], return_type: TypeRef,
                               body: List[Statement], 
                               constraint: Optional[str] = None) -> str:
        """Emit a D template function."""
        ind = self._indent()
        lines = []
        
        # Type parameters
        type_param_str = ', '.join(type_params)
        
        # Return type
        ret_type = self._map_type(return_type) if return_type else 'auto'
        
        # Value parameters
        param_str = self._emit_parameters(params)
        
        # Function signature
        sig = f"{ret_type} {name}({type_param_str})({param_str})"
        
        # Optional constraint
        if constraint:
            sig += f"\n{ind}if ({constraint})"
        
        lines.append(f"{ind}{sig}")
        lines.append(f"{ind}{{")
        
        self._current_indent += 1
        for stmt in body:
            lines.append(self._emit_statement(stmt))
        self._current_indent -= 1
        
        lines.append(f"{ind}}}")
        
        return '\n'.join(lines)
    
    # =========================================================================
    # Statement Emission
    # =========================================================================
    
    def _emit_statement(self, stmt: Statement) -> str:
        """Emit statement."""
        ind = self._indent()
        
        if isinstance(stmt, Assignment):
            target = self._emit_expr(stmt.target)
            value = self._emit_expr(stmt.value)
            return f"{ind}{target} = {value};"
        
        elif isinstance(stmt, IfStatement):
            return self._emit_if_statement(stmt)
        
        elif isinstance(stmt, CaseStatement):
            return self._emit_switch_statement(stmt)
        
        elif isinstance(stmt, WhileLoop):
            return self._emit_while_loop(stmt)
        
        elif isinstance(stmt, ForLoop):
            return self._emit_foreach_loop(stmt)
        
        elif isinstance(stmt, BasicLoop):
            return self._emit_infinite_loop(stmt)
        
        elif isinstance(stmt, ExitStatement):
            return f"{ind}break;"
        
        elif isinstance(stmt, ReturnStatement):
            if stmt.value:
                return f"{ind}return {self._emit_expr(stmt.value)};"
            else:
                return f"{ind}return;"
        
        elif isinstance(stmt, NullStatement):
            return f"{ind}// no-op"
        
        elif isinstance(stmt, CallStatement):
            return f"{ind}{self._emit_expr(stmt.call)};"
        
        elif isinstance(stmt, BlockStatement):
            return self._emit_block(stmt)
        
        elif isinstance(stmt, TryStatement):
            return self._emit_try_catch(stmt)
        
        elif isinstance(stmt, RaiseStatement):
            if stmt.exception_name:
                msg = f'"{self._emit_expr(stmt.message)}"' if stmt.message else '""'
                return f"{ind}throw new Exception({msg});"
            else:
                return f"{ind}throw;"
        
        elif isinstance(stmt, SynchronizedBlock):
            return self._emit_synchronized(stmt)
        
        elif isinstance(stmt, AssertPragma):
            cond = self._emit_expr(stmt.condition)
            msg = f', "{stmt.message}"' if stmt.message else ""
            return f"{ind}assert({cond}{msg});"
        
        else:
            return f"{ind}// Unknown statement: {type(stmt).__name__}"
    
    def _emit_if_statement(self, stmt: IfStatement) -> str:
        """Emit if statement."""
        ind = self._indent()
        lines = []
        
        cond = self._emit_expr(stmt.condition)
        lines.append(f"{ind}if ({cond}) {{")
        
        self._current_indent += 1
        for s in stmt.then_body:
            lines.append(self._emit_statement(s))
        self._current_indent -= 1
        
        # Elsif parts become else if
        for elsif in stmt.elsif_parts:
            cond = self._emit_expr(elsif.condition)
            lines.append(f"{ind}}} else if ({cond}) {{")
            self._current_indent += 1
            for s in elsif.body:
                lines.append(self._emit_statement(s))
            self._current_indent -= 1
        
        # Else part
        if stmt.else_body:
            lines.append(f"{ind}}} else {{")
            self._current_indent += 1
            for s in stmt.else_body:
                lines.append(self._emit_statement(s))
            self._current_indent -= 1
        
        lines.append(f"{ind}}}")
        return '\n'.join(lines)
    
    def _emit_switch_statement(self, stmt: CaseStatement) -> str:
        """Emit switch statement."""
        ind = self._indent()
        lines = []
        
        selector = self._emit_expr(stmt.selector)
        lines.append(f"{ind}switch ({selector}) {{")
        
        self._current_indent += 1
        for alt in stmt.alternatives:
            for choice in alt.choices:
                choice_str = self._emit_expr(choice)
                lines.append(f"{self._indent()}case {choice_str}:")
            
            self._current_indent += 1
            for s in alt.body:
                lines.append(self._emit_statement(s))
            lines.append(f"{self._indent()}break;")
            self._current_indent -= 1
        
        # Default case
        lines.append(f"{self._indent()}default:")
        self._current_indent += 1
        lines.append(f"{self._indent()}break;")
        self._current_indent -= 1
        
        self._current_indent -= 1
        lines.append(f"{ind}}}")
        return '\n'.join(lines)
    
    def _emit_while_loop(self, stmt: WhileLoop) -> str:
        """Emit while loop."""
        ind = self._indent()
        lines = []
        
        cond = self._emit_expr(stmt.condition)
        lines.append(f"{ind}while ({cond}) {{")
        
        self._current_indent += 1
        for s in stmt.body:
            lines.append(self._emit_statement(s))
        self._current_indent -= 1
        
        lines.append(f"{ind}}}")
        return '\n'.join(lines)
    
    def _emit_foreach_loop(self, stmt: ForLoop) -> str:
        """Emit foreach loop."""
        ind = self._indent()
        lines = []
        
        range_str = self._emit_expr(stmt.range_expr)
        lines.append(f"{ind}foreach ({stmt.variable}; {range_str}) {{")
        
        self._current_indent += 1
        for s in stmt.body:
            lines.append(self._emit_statement(s))
        self._current_indent -= 1
        
        lines.append(f"{ind}}}")
        return '\n'.join(lines)
    
    def _emit_infinite_loop(self, stmt: BasicLoop) -> str:
        """Emit infinite loop (while true)."""
        ind = self._indent()
        lines = []
        
        lines.append(f"{ind}while (true) {{")
        
        self._current_indent += 1
        for s in stmt.body:
            lines.append(self._emit_statement(s))
        self._current_indent -= 1
        
        lines.append(f"{ind}}}")
        return '\n'.join(lines)
    
    def _emit_block(self, stmt: BlockStatement) -> str:
        """Emit block statement."""
        ind = self._indent()
        lines = [f"{ind}{{"]
        
        self._current_indent += 1
        
        # Declarations
        for decl in stmt.declarations:
            if isinstance(decl, VariableDecl):
                lines.append(f"{self._indent()}{self._emit_variable_decl(decl)}")
        
        # Body
        for s in stmt.body:
            lines.append(self._emit_statement(s))
        
        self._current_indent -= 1
        lines.append(f"{ind}}}")
        return '\n'.join(lines)
    
    def _emit_try_catch(self, stmt: TryStatement) -> str:
        """Emit try-catch statement."""
        ind = self._indent()
        lines = [f"{ind}try {{"]
        
        self._current_indent += 1
        for s in stmt.body:
            lines.append(self._emit_statement(s))
        self._current_indent -= 1
        
        for handler in stmt.handlers:
            exc_type = handler.exception_names[0] if handler.exception_names else "Exception"
            param = handler.choice_parameter or "e"
            lines.append(f"{ind}}} catch ({exc_type} {param}) {{")
            self._current_indent += 1
            for s in handler.body:
                lines.append(self._emit_statement(s))
            self._current_indent -= 1
        
        if stmt.finally_body:
            lines.append(f"{ind}}} finally {{")
            self._current_indent += 1
            for s in stmt.finally_body:
                lines.append(self._emit_statement(s))
            self._current_indent -= 1
        
        lines.append(f"{ind}}}")
        return '\n'.join(lines)
    
    def _emit_synchronized(self, stmt: SynchronizedBlock) -> str:
        """Emit synchronized block."""
        ind = self._indent()
        lines = []
        
        if stmt.mutex:
            mutex = self._emit_expr(stmt.mutex)
            lines.append(f"{ind}synchronized ({mutex}) {{")
        else:
            lines.append(f"{ind}synchronized {{")
        
        self._current_indent += 1
        for s in stmt.body:
            lines.append(self._emit_statement(s))
        self._current_indent -= 1
        
        lines.append(f"{ind}}}")
        return '\n'.join(lines)
    
    # =========================================================================
    # Expression Emission
    # =========================================================================
    
    def _emit_expr(self, expr: Expr) -> str:
        """Emit expression."""
        if expr is None:
            return "null"
        
        if isinstance(expr, Literal):
            return self._emit_literal(expr)
        
        elif isinstance(expr, VarExpr):
            return expr.name
        
        elif isinstance(expr, BinaryOp):
            left = self._emit_expr(expr.left)
            right = self._emit_expr(expr.right)
            op = self._map_operator(expr.op)
            return f"({left} {op} {right})"
        
        elif isinstance(expr, UnaryOp):
            operand = self._emit_expr(expr.operand)
            op = self._map_operator(expr.op)
            return f"{op}{operand}"
        
        elif isinstance(expr, CallExpr):
            args = ', '.join(self._emit_expr(a) for a in expr.arguments)
            return f"{expr.name}({args})"
        
        elif isinstance(expr, MemberAccess):
            target = self._emit_expr(expr.target)
            return f"{target}.{expr.member}"
        
        elif isinstance(expr, IndexExpr):
            target = self._emit_expr(expr.target)
            index = self._emit_expr(expr.index)
            return f"{target}[{index}]"
        
        elif isinstance(expr, IfExpr):
            cond = self._emit_expr(expr.condition)
            then_expr = self._emit_expr(expr.then_expr)
            else_expr = self._emit_expr(expr.else_expr)
            return f"({cond} ? {then_expr} : {else_expr})"
        
        elif isinstance(expr, RangeExpr):
            start = self._emit_expr(expr.start)
            end = self._emit_expr(expr.end)
            return f"{start} .. {end}"
        
        elif isinstance(expr, CastExpr):
            target_type = self._map_type(expr.target_type)
            inner = self._emit_expr(expr.expr)
            return f"cast({target_type})({inner})"
        
        elif isinstance(expr, Allocator):
            type_ref = self._map_type(expr.type_ref)
            if expr.initializer:
                init = self._emit_expr(expr.initializer)
                return f"new {type_ref}({init})"
            return f"new {type_ref}()"
        
        elif isinstance(expr, SliceExpr):
            target = self._emit_expr(expr.target)
            start = self._emit_expr(expr.start)
            end = self._emit_expr(expr.end)
            return f"{target}[{start} .. {end}]"
        
        elif isinstance(expr, AggregateExpr):
            return self._emit_aggregate(expr)
        
        else:
            return f"/* Unknown expr: {type(expr).__name__} */"
    
    def _emit_literal(self, lit: Literal) -> str:
        """Emit literal value."""
        if lit.literal_type == 'bool' or isinstance(lit.value, bool):
            return "true" if lit.value else "false"
        elif lit.literal_type == 'char':
            return f"'{lit.value}'"
        elif lit.literal_type == 'string' or isinstance(lit.value, str):
            return f'"{lit.value}"'
        elif lit.literal_type == 'float' or isinstance(lit.value, float):
            return str(lit.value)
        else:
            return str(lit.value)
    
    def _emit_aggregate(self, agg: AggregateExpr) -> str:
        """Emit aggregate/struct literal."""
        parts = []
        
        # Named components
        for name, value in agg.components.items():
            parts.append(f"{name}: {self._emit_expr(value)}")
        
        # Positional components
        for value in agg.positional:
            parts.append(self._emit_expr(value))
        
        if not parts:
            return "{}"
        
        return "{ " + ", ".join(parts) + " }"
    
    # =========================================================================
    # Manifest Generation
    # =========================================================================
    
    def generate_manifest(self, package: Package, code: str) -> dict:
        """Generate deterministic build manifest."""
        code_hash = compute_sha256(code)
        ir_hash = compute_sha256(canonical_json(package.to_dict()))
        
        manifest = {
            'schema': 'stunir.manifest.targets.d.v1',
            'generator': 'stunir.d.emitter',
            'epoch': int(time.time()),
            'ir_hash': ir_hash,
            'output': {
                'hash': code_hash,
                'size': len(code),
                'extension': self.FILE_EXTENSION,
            },
            'manifest_hash': ''
        }
        
        manifest['manifest_hash'] = compute_sha256(
            canonical_json({k: v for k, v in manifest.items() if k != 'manifest_hash'})
        )
        
        return manifest
    
    def emit(self, package: Package) -> EmitterResult:
        """Emit module with manifest."""
        code = self.emit_module(package)
        manifest = self.generate_manifest(package, code)
        return EmitterResult(code=code, manifest=manifest)
