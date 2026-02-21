#!/usr/bin/env python3
"""STUNIR Ada Emitter - Generate Ada code with SPARK support.

This emitter generates Ada 2012+ code including:
- Package specifications (.ads) and bodies (.adb)
- Type declarations (records, arrays, enums, access types)
- Subprogram declarations with contracts
- Task types and bodies
- Protected types and bodies
- SPARK annotations (Pre, Post, Global, Depends, etc.)
- Loop invariants and variants
- Ghost code for verification

Usage:
    from targets.systems.ada_emitter import AdaEmitter
    from ir.systems import Package, Subprogram
    
    emitter = AdaEmitter()
    spec, body = emitter.emit_package(package)
"""

import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ir.systems import (
    # Core
    Package, Import, Subprogram, Parameter, Mode, Visibility,
    TypeRef, Expr, Statement, Declaration,
    # Expressions
    Literal, VarExpr, BinaryOp, UnaryOp, CallExpr, MemberAccess,
    IndexExpr, IfExpr, CastExpr, RangeExpr, AggregateExpr, AttributeExpr,
    QualifiedExpr,
    # Statements
    Assignment, IfStatement, ElsifPart, CaseStatement, CaseAlternative,
    WhileLoop, ForLoop, BasicLoop, ExitStatement, ReturnStatement,
    NullStatement, BlockStatement, RaiseStatement, CallStatement,
    TryStatement, ExceptionHandler,
    # Types
    TypeDecl, SubtypeDecl, DerivedTypeDecl, RecordType, ArrayType,
    EnumType, EnumLiteral, IntegerType, ModularType, AccessType,
    ComponentDecl, Discriminant, VariantPart, Variant, RangeConstraint,
    # Concurrency
    TaskType, Entry, ProtectedType, AcceptStatement, SelectStatement,
    SelectAlternative, AcceptAlternative, DelayAlternative, TerminateAlternative,
    EntryCallStatement, DelayStatement,
    # Verification
    Contract, ContractCase, GlobalSpec, DependsSpec, LoopInvariant,
    LoopVariant, VariantExpr, GhostCode, GhostVariable, AssertPragma,
    AssumePragma, QuantifiedExpr, OldExpr, ResultExpr,
    # Memory
    Allocator, Deallocate, AddressOf, Dereference,
    VariableDecl, ConstantDecl,
)


@dataclass
class EmitterResult:
    """Result of code emission."""
    spec_code: str
    body_code: str
    manifest: dict


def canonical_json(obj: Any) -> str:
    """Generate canonical JSON (sorted keys)."""
    return json.dumps(obj, sort_keys=True, separators=(',', ':'))


def compute_sha256(data: str) -> str:
    """Compute SHA-256 hash of data."""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


# Ada type mappings
ADA_TYPE_MAP = {
    'int': 'Integer',
    'i8': 'Interfaces.Integer_8',
    'i16': 'Interfaces.Integer_16',
    'i32': 'Interfaces.Integer_32',
    'i64': 'Interfaces.Integer_64',
    'u8': 'Interfaces.Unsigned_8',
    'u16': 'Interfaces.Unsigned_16',
    'u32': 'Interfaces.Unsigned_32',
    'u64': 'Interfaces.Unsigned_64',
    'f32': 'Float',
    'f64': 'Long_Float',
    'float': 'Float',
    'double': 'Long_Float',
    'bool': 'Boolean',
    'boolean': 'Boolean',
    'char': 'Character',
    'string': 'String',
    'unit': 'null',
    'void': 'null',
}

# Ada operator mappings
ADA_OPERATOR_MAP = {
    '+': '+',
    '-': '-',
    '*': '*',
    '/': '/',
    'mod': 'mod',
    '%': 'mod',
    'rem': 'rem',
    '==': '=',
    '=': '=',
    '!=': '/=',
    '/=': '/=',
    '<': '<',
    '>': '>',
    '<=': '<=',
    '>=': '>=',
    'and': 'and',
    '&&': 'and',
    'or': 'or',
    '||': 'or',
    'xor': 'xor',
    'not': 'not',
    '!': 'not',
    '**': '**',
    '&': '&',  # Concatenation
}


class AdaEmitter:
    """Ada code emitter with SPARK support.
    
    Generates Ada 2012+ code with optional SPARK annotations
    for formal verification.
    """
    
    DIALECT = 'ada'
    SPEC_EXTENSION = '.ads'
    BODY_EXTENSION = '.adb'
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.indent_size = self.config.get('indent_size', 3)
        self.spark_mode_default = self.config.get('spark_mode', False)
        self._current_indent = 0
    
    def _indent(self, level: int = None) -> str:
        """Get indentation string."""
        if level is None:
            level = self._current_indent
        return ' ' * (level * self.indent_size)
    
    def _map_type(self, type_ref: TypeRef) -> str:
        """Map IR type to Ada type."""
        if type_ref is None:
            return 'null'
        
        name = type_ref.name
        ada_type = ADA_TYPE_MAP.get(name.lower(), name)
        
        # Handle access types
        if type_ref.is_access:
            prefix = 'not null ' if type_ref.not_null else ''
            return f"{prefix}access {ada_type}"
        
        # Handle type arguments (generics)
        if type_ref.type_args:
            args = ', '.join(self._map_type(arg) for arg in type_ref.type_args)
            return f"{ada_type} ({args})"
        
        return ada_type
    
    def _map_mode(self, mode: Mode) -> str:
        """Map parameter mode to Ada."""
        mode_map = {
            Mode.IN: 'in',
            Mode.OUT: 'out',
            Mode.IN_OUT: 'in out',
            Mode.ACCESS: 'access',
        }
        return mode_map.get(mode, 'in')
    
    def _map_operator(self, op: str) -> str:
        """Map IR operator to Ada."""
        return ADA_OPERATOR_MAP.get(op, op)
    
    # =========================================================================
    # Package Emission
    # =========================================================================
    
    def emit_package(self, package: Package) -> Tuple[str, str]:
        """Emit Ada package specification and body.
        
        Returns:
            Tuple of (specification, body) strings
        """
        spec = self.emit_package_spec(package)
        body = self.emit_package_body(package)
        return spec, body
    
    def emit_package_spec(self, package: Package) -> str:
        """Generate Ada package specification (.ads)."""
        lines = []
        
        # Header comment
        lines.append(f"--  Auto-generated by STUNIR Ada Emitter")
        lines.append(f"--  Package: {package.name}")
        lines.append("")
        
        # SPARK_Mode pragma
        if package.spark_mode or self.spark_mode_default:
            lines.append("pragma SPARK_Mode (On);")
            lines.append("")
        
        # With clauses
        for imp in package.imports:
            lines.append(self._emit_import(imp))
        if package.imports:
            lines.append("")
        
        # Package specification
        lines.append(f"package {package.name} is")
        lines.append("")
        self._current_indent = 1
        
        # Type declarations
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
        
        # Subprogram specifications
        for subprog in package.subprograms:
            lines.append(self._emit_subprogram_spec(subprog))
            lines.append("")
        
        # Task types
        for task in package.tasks:
            lines.append(self._emit_task_type_spec(task))
            lines.append("")
        
        # Protected types
        for protected in package.protected_types:
            lines.append(self._emit_protected_type_spec(protected))
            lines.append("")
        
        # Exceptions
        for exc in package.exceptions:
            lines.append(f"{self._indent()}{exc.name} : exception;")
        if package.exceptions:
            lines.append("")
        
        # Private section
        if package.private_types or package.private_variables:
            lines.append("private")
            lines.append("")
            for type_decl in package.private_types:
                lines.append(self._emit_type_decl(type_decl))
                lines.append("")
            for var in package.private_variables:
                lines.append(self._emit_variable_decl(var))
            if package.private_variables:
                lines.append("")
        
        self._current_indent = 0
        lines.append(f"end {package.name};")
        
        return '\n'.join(lines)
    
    def emit_package_body(self, package: Package) -> str:
        """Generate Ada package body (.adb)."""
        lines = []
        
        # Header comment
        lines.append(f"--  Auto-generated by STUNIR Ada Emitter")
        lines.append(f"--  Package body: {package.name}")
        lines.append("")
        
        # SPARK_Mode pragma
        if package.spark_mode or self.spark_mode_default:
            lines.append("pragma SPARK_Mode (On);")
            lines.append("")
        
        # Package body
        lines.append(f"package body {package.name} is")
        lines.append("")
        self._current_indent = 1
        
        # Local declarations in body
        for decl in package.body_declarations:
            if isinstance(decl, VariableDecl):
                lines.append(self._emit_variable_decl(decl))
            elif isinstance(decl, ConstantDecl):
                lines.append(self._emit_constant_decl(decl))
        if package.body_declarations:
            lines.append("")
        
        # Subprogram bodies
        for subprog in package.subprograms:
            if subprog.body:  # Has implementation
                lines.append(self._emit_subprogram_body(subprog))
                lines.append("")
        
        # Task bodies
        for task in package.tasks:
            lines.append(self._emit_task_body(task))
            lines.append("")
        
        # Protected bodies
        for protected in package.protected_types:
            lines.append(self._emit_protected_body(protected))
            lines.append("")
        
        # Package initialization
        if package.initialization:
            lines.append("begin")
            self._current_indent = 1
            for stmt in package.initialization:
                lines.append(self._emit_statement(stmt))
            self._current_indent = 0
        
        self._current_indent = 0
        lines.append(f"end {package.name};")
        
        return '\n'.join(lines)
    
    def _emit_import(self, imp: Import) -> str:
        """Emit with/use clause."""
        result = f"with {imp.module};"
        if imp.use_clause:
            result += f"\nuse {imp.module};"
        return result
    
    # =========================================================================
    # Type Declarations
    # =========================================================================
    
    def _emit_type_decl(self, type_decl: TypeDecl) -> str:
        """Emit type declaration."""
        ind = self._indent()
        
        if isinstance(type_decl, SubtypeDecl):
            return self._emit_subtype_decl(type_decl)
        elif isinstance(type_decl, DerivedTypeDecl):
            return self._emit_derived_type_decl(type_decl)
        elif isinstance(type_decl, RecordType):
            return self._emit_record_type(type_decl)
        elif isinstance(type_decl, ArrayType):
            return self._emit_array_type(type_decl)
        elif isinstance(type_decl, EnumType):
            return self._emit_enum_type(type_decl)
        elif isinstance(type_decl, IntegerType):
            return self._emit_integer_type(type_decl)
        elif isinstance(type_decl, ModularType):
            return self._emit_modular_type(type_decl)
        else:
            return f"{ind}--  Unknown type: {type_decl.name}"
    
    def _emit_subtype_decl(self, subtype: SubtypeDecl) -> str:
        """Emit subtype declaration."""
        ind = self._indent()
        base = self._map_type(subtype.base_type)
        result = f"{ind}subtype {subtype.name} is {base}"
        
        if subtype.constraint:
            if isinstance(subtype.constraint, RangeConstraint):
                lower = self._emit_expr(subtype.constraint.lower)
                upper = self._emit_expr(subtype.constraint.upper)
                result += f" range {lower} .. {upper}"
        
        result += ";"
        return result
    
    def _emit_derived_type_decl(self, derived: DerivedTypeDecl) -> str:
        """Emit derived type declaration."""
        ind = self._indent()
        parent = self._map_type(derived.parent_type)
        
        abstract = "abstract " if derived.is_abstract else ""
        result = f"{ind}type {derived.name} is {abstract}new {parent}"
        
        if derived.extension:
            if derived.extension.is_null_extension:
                result += " with null record"
            elif derived.extension.components:
                result += " with record\n"
                self._current_indent += 1
                for comp in derived.extension.components:
                    result += f"{self._indent()}{comp.name} : {self._map_type(comp.type_ref)};\n"
                self._current_indent -= 1
                result += f"{ind}end record"
        
        result += ";"
        return result
    
    def _emit_record_type(self, record: RecordType) -> str:
        """Emit record type declaration."""
        ind = self._indent()
        lines = []
        
        # Type header
        header = f"{ind}type {record.name}"
        
        # Discriminants
        if record.discriminants:
            disc_parts = []
            for disc in record.discriminants:
                disc_type = self._map_type(disc.type_ref)
                disc_parts.append(f"{disc.name} : {disc_type}")
            header += f" ({'; '.join(disc_parts)})"
        
        # Tagged/limited
        modifiers = []
        if record.is_abstract:
            modifiers.append("abstract")
        if record.is_tagged:
            modifiers.append("tagged")
        if record.is_limited:
            modifiers.append("limited")
        
        mod_str = ' '.join(modifiers) + ' ' if modifiers else ''
        
        if not record.components and not record.variant_part:
            lines.append(f"{header} is {mod_str}null record;")
        else:
            lines.append(f"{header} is {mod_str}record")
            self._current_indent += 1
            
            for comp in record.components:
                comp_type = self._map_type(comp.type_ref)
                default = ""
                if comp.default_value:
                    default = f" := {self._emit_expr(comp.default_value)}"
                lines.append(f"{self._indent()}{comp.name} : {comp_type}{default};")
            
            # Variant part
            if record.variant_part:
                lines.append(self._emit_variant_part(record.variant_part))
            
            self._current_indent -= 1
            lines.append(f"{ind}end record;")
        
        return '\n'.join(lines)
    
    def _emit_variant_part(self, variant_part: VariantPart) -> str:
        """Emit variant part of a record."""
        ind = self._indent()
        lines = [f"{ind}case {variant_part.discriminant_name} is"]
        self._current_indent += 1
        
        for variant in variant_part.variants:
            choices = ' | '.join(self._emit_expr(c) for c in variant.choices)
            lines.append(f"{self._indent()}when {choices} =>")
            self._current_indent += 1
            for comp in variant.components:
                comp_type = self._map_type(comp.type_ref)
                lines.append(f"{self._indent()}{comp.name} : {comp_type};")
            self._current_indent -= 1
        
        self._current_indent -= 1
        lines.append(f"{ind}end case;")
        return '\n'.join(lines)
    
    def _emit_array_type(self, array: ArrayType) -> str:
        """Emit array type declaration."""
        ind = self._indent()
        
        # Index specification
        if array.is_unconstrained:
            indices = ', '.join(
                f"{self._map_type(idx)} range <>" 
                for idx in array.index_types
            )
        else:
            indices = ', '.join(
                self._emit_expr(idx) if isinstance(idx, Expr) else self._map_type(idx)
                for idx in (array.index_constraints or array.index_types)
            )
        
        element = self._map_type(array.element_type)
        aliased = "aliased " if array.is_aliased_components else ""
        
        return f"{ind}type {array.name} is array ({indices}) of {aliased}{element};"
    
    def _emit_enum_type(self, enum: EnumType) -> str:
        """Emit enumeration type declaration."""
        ind = self._indent()
        literals = ', '.join(lit.name for lit in enum.literals)
        return f"{ind}type {enum.name} is ({literals});"
    
    def _emit_integer_type(self, int_type: IntegerType) -> str:
        """Emit integer type declaration."""
        ind = self._indent()
        lower = self._emit_expr(int_type.range_constraint.lower)
        upper = self._emit_expr(int_type.range_constraint.upper)
        return f"{ind}type {int_type.name} is range {lower} .. {upper};"
    
    def _emit_modular_type(self, mod_type: ModularType) -> str:
        """Emit modular type declaration."""
        ind = self._indent()
        modulus = self._emit_expr(mod_type.modulus)
        return f"{ind}type {mod_type.name} is mod {modulus};"
    
    # =========================================================================
    # Variable and Constant Declarations
    # =========================================================================
    
    def _emit_variable_decl(self, var: VariableDecl) -> str:
        """Emit variable declaration."""
        ind = self._indent()
        var_type = self._map_type(var.type_ref)
        
        modifiers = []
        if var.is_constant:
            modifiers.append("constant")
        if var.is_aliased:
            modifiers.append("aliased")
        
        mod_str = ' '.join(modifiers) + ' ' if modifiers else ''
        
        result = f"{ind}{var.name} : {mod_str}{var_type}"
        
        if var.initializer:
            result += f" := {self._emit_expr(var.initializer)}"
        
        # Ghost aspect
        if var.is_ghost:
            result += " with Ghost"
        
        result += ";"
        return result
    
    def _emit_constant_decl(self, const: ConstantDecl) -> str:
        """Emit constant declaration."""
        ind = self._indent()
        const_type = self._map_type(const.type_ref) if const.type_ref else ""
        type_part = f" : {const_type}" if const_type else ""
        value = self._emit_expr(const.value)
        return f"{ind}{const.name}{type_part} : constant := {value};"
    
    # =========================================================================
    # Subprogram Emission
    # =========================================================================
    
    def _emit_subprogram_spec(self, subprog: Subprogram) -> str:
        """Emit subprogram specification."""
        ind = self._indent()
        
        # Determine if procedure or function
        is_function = subprog.return_type is not None
        keyword = "function" if is_function else "procedure"
        
        # Parameters
        params = self._emit_parameters(subprog.parameters)
        params_str = f" ({params})" if params else ""
        
        # Return type
        return_str = ""
        if is_function:
            return_str = f" return {self._map_type(subprog.return_type)}"
        
        result = f"{ind}{keyword} {subprog.name}{params_str}{return_str}"
        
        # Contracts and SPARK annotations
        aspects = self._emit_contracts(subprog)
        if aspects:
            result += "\n" + aspects
        
        result += ";"
        return result
    
    def _emit_subprogram_body(self, subprog: Subprogram) -> str:
        """Emit subprogram body."""
        ind = self._indent()
        
        # Determine if procedure or function
        is_function = subprog.return_type is not None
        keyword = "function" if is_function else "procedure"
        
        # Parameters
        params = self._emit_parameters(subprog.parameters)
        params_str = f" ({params})" if params else ""
        
        # Return type
        return_str = ""
        if is_function:
            return_str = f" return {self._map_type(subprog.return_type)}"
        
        lines = [f"{ind}{keyword} {subprog.name}{params_str}{return_str}"]
        
        # Contracts
        aspects = self._emit_contracts(subprog)
        if aspects:
            lines.append(aspects)
        
        lines.append(f"{ind}is")
        
        # Local declarations
        self._current_indent += 1
        for decl in subprog.local_declarations:
            if isinstance(decl, VariableDecl):
                lines.append(self._emit_variable_decl(decl))
            elif isinstance(decl, ConstantDecl):
                lines.append(self._emit_constant_decl(decl))
        self._current_indent -= 1
        
        lines.append(f"{ind}begin")
        
        # Body statements
        self._current_indent += 1
        for stmt in subprog.body:
            lines.append(self._emit_statement(stmt))
        self._current_indent -= 1
        
        lines.append(f"{ind}end {subprog.name};")
        
        return '\n'.join(lines)
    
    def _emit_parameters(self, params: List[Parameter]) -> str:
        """Emit parameter list."""
        if not params:
            return ""
        
        parts = []
        for param in params:
            mode = self._map_mode(param.mode)
            param_type = self._map_type(param.type_ref)
            
            mode_str = f"{mode} " if mode != 'in' else ""
            
            part = f"{param.name} : {mode_str}{param_type}"
            if param.default_value:
                part += f" := {self._emit_expr(param.default_value)}"
            parts.append(part)
        
        return "; ".join(parts)
    
    def _emit_contracts(self, subprog: Subprogram) -> str:
        """Emit contracts and SPARK annotations."""
        ind = self._indent()
        aspects = []
        
        # SPARK_Mode
        if subprog.spark_mode:
            aspects.append("SPARK_Mode => On")
        
        # Preconditions
        for pre in subprog.preconditions:
            cond = self._emit_expr(pre.condition)
            aspects.append(f"Pre => {cond}")
        
        # Postconditions
        for post in subprog.postconditions:
            cond = self._emit_expr(post.condition)
            aspects.append(f"Post => {cond}")
        
        # Contract_Cases
        if subprog.contract_cases:
            cases = []
            for case in subprog.contract_cases:
                guard = self._emit_expr(case.guard)
                conseq = self._emit_expr(case.consequence)
                cases.append(f"{guard} => {conseq}")
            aspects.append(f"Contract_Cases => ({', '.join(cases)})")
        
        # Global
        if subprog.global_spec:
            aspects.append(self._emit_global_spec(subprog.global_spec))
        
        # Depends
        if subprog.depends_spec:
            aspects.append(self._emit_depends_spec(subprog.depends_spec))
        
        if not aspects:
            return ""
        
        # Format with 'with' clause
        if len(aspects) == 1:
            return f"{ind}   with {aspects[0]}"
        else:
            result = f"{ind}   with {aspects[0]}"
            for aspect in aspects[1:]:
                result += f",\n{ind}        {aspect}"
            return result
    
    def _emit_global_spec(self, global_spec: GlobalSpec) -> str:
        """Emit SPARK Global specification."""
        if global_spec.is_null:
            return "Global => null"
        
        parts = []
        if global_spec.inputs:
            items = ', '.join(global_spec.inputs)
            parts.append(f"Input => ({items})" if len(global_spec.inputs) > 1 else f"Input => {items}")
        if global_spec.outputs:
            items = ', '.join(global_spec.outputs)
            parts.append(f"Output => ({items})" if len(global_spec.outputs) > 1 else f"Output => {items}")
        if global_spec.in_outs:
            items = ', '.join(global_spec.in_outs)
            parts.append(f"In_Out => ({items})" if len(global_spec.in_outs) > 1 else f"In_Out => {items}")
        if global_spec.proof_ins:
            items = ', '.join(global_spec.proof_ins)
            parts.append(f"Proof_In => ({items})" if len(global_spec.proof_ins) > 1 else f"Proof_In => {items}")
        
        if not parts:
            return "Global => null"
        
        return f"Global => ({', '.join(parts)})"
    
    def _emit_depends_spec(self, depends_spec: DependsSpec) -> str:
        """Emit SPARK Depends specification."""
        deps = []
        for output, inputs in depends_spec.dependencies.items():
            if len(inputs) == 0:
                deps.append(f"{output} => null")
            elif len(inputs) == 1:
                deps.append(f"{output} => {inputs[0]}")
            else:
                deps.append(f"{output} => ({', '.join(inputs)})")
        
        return f"Depends => ({', '.join(deps)})"
    
    # =========================================================================
    # Task Type Emission
    # =========================================================================
    
    def _emit_task_type_spec(self, task: TaskType) -> str:
        """Emit task type specification."""
        ind = self._indent()
        lines = []
        
        # Task type header
        disc_str = ""
        if task.discriminants:
            discs = '; '.join(
                f"{d.name} : {self._map_type(d.type_ref)}"
                for d in task.discriminants
            )
            disc_str = f" ({discs})"
        
        lines.append(f"{ind}task type {task.name}{disc_str} is")
        self._current_indent += 1
        
        # Entries
        for entry in task.entries:
            lines.append(self._emit_entry_spec(entry))
        
        self._current_indent -= 1
        lines.append(f"{ind}end {task.name};")
        
        return '\n'.join(lines)
    
    def _emit_entry_spec(self, entry: Entry) -> str:
        """Emit entry specification."""
        ind = self._indent()
        params = self._emit_parameters(entry.parameters)
        params_str = f" ({params})" if params else ""
        
        family_str = ""
        if entry.family_index:
            family_str = f" (for {self._map_type(entry.family_index)})"
        
        return f"{ind}entry {entry.name}{family_str}{params_str};"
    
    def _emit_task_body(self, task: TaskType) -> str:
        """Emit task body."""
        ind = self._indent()
        lines = [f"{ind}task body {task.name} is"]
        
        # Local declarations
        self._current_indent += 1
        for decl in task.local_declarations:
            if isinstance(decl, VariableDecl):
                lines.append(self._emit_variable_decl(decl))
        self._current_indent -= 1
        
        lines.append(f"{ind}begin")
        
        # Body
        self._current_indent += 1
        for stmt in task.body:
            lines.append(self._emit_statement(stmt))
        self._current_indent -= 1
        
        lines.append(f"{ind}end {task.name};")
        
        return '\n'.join(lines)
    
    # =========================================================================
    # Protected Type Emission
    # =========================================================================
    
    def _emit_protected_type_spec(self, protected: ProtectedType) -> str:
        """Emit protected type specification."""
        ind = self._indent()
        lines = [f"{ind}protected type {protected.name} is"]
        self._current_indent += 1
        
        # Entries
        for entry in protected.entries:
            lines.append(self._emit_entry_spec(entry))
        
        # Procedures
        for proc in protected.procedures:
            lines.append(self._emit_subprogram_spec(proc))
        
        # Functions
        for func in protected.functions:
            lines.append(self._emit_subprogram_spec(func))
        
        # Private components
        if protected.private_components:
            self._current_indent -= 1
            lines.append(f"{ind}private")
            self._current_indent += 1
            
            for comp in protected.private_components:
                comp_type = self._map_type(comp.type_ref)
                default = ""
                if comp.default_value:
                    default = f" := {self._emit_expr(comp.default_value)}"
                lines.append(f"{self._indent()}{comp.name} : {comp_type}{default};")
        
        self._current_indent -= 1
        lines.append(f"{ind}end {protected.name};")
        
        return '\n'.join(lines)
    
    def _emit_protected_body(self, protected: ProtectedType) -> str:
        """Emit protected body."""
        ind = self._indent()
        lines = [f"{ind}protected body {protected.name} is"]
        lines.append("")
        self._current_indent += 1
        
        # Procedure bodies
        for proc in protected.procedures:
            if proc.body:
                lines.append(self._emit_subprogram_body(proc))
                lines.append("")
        
        # Function bodies
        for func in protected.functions:
            if func.body:
                lines.append(self._emit_subprogram_body(func))
                lines.append("")
        
        self._current_indent -= 1
        lines.append(f"{ind}end {protected.name};")
        
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
            return f"{ind}{target} := {value};"
        
        elif isinstance(stmt, IfStatement):
            return self._emit_if_statement(stmt)
        
        elif isinstance(stmt, CaseStatement):
            return self._emit_case_statement(stmt)
        
        elif isinstance(stmt, WhileLoop):
            return self._emit_while_loop(stmt)
        
        elif isinstance(stmt, ForLoop):
            return self._emit_for_loop(stmt)
        
        elif isinstance(stmt, BasicLoop):
            return self._emit_basic_loop(stmt)
        
        elif isinstance(stmt, ExitStatement):
            return self._emit_exit_statement(stmt)
        
        elif isinstance(stmt, ReturnStatement):
            if stmt.value:
                return f"{ind}return {self._emit_expr(stmt.value)};"
            else:
                return f"{ind}return;"
        
        elif isinstance(stmt, NullStatement):
            return f"{ind}null;"
        
        elif isinstance(stmt, RaiseStatement):
            if stmt.exception_name:
                msg = f' with "{self._emit_expr(stmt.message)}"' if stmt.message else ""
                return f"{ind}raise {stmt.exception_name}{msg};"
            else:
                return f"{ind}raise;"
        
        elif isinstance(stmt, CallStatement):
            return f"{ind}{self._emit_expr(stmt.call)};"
        
        elif isinstance(stmt, BlockStatement):
            return self._emit_block_statement(stmt)
        
        elif isinstance(stmt, AcceptStatement):
            return self._emit_accept_statement(stmt)
        
        elif isinstance(stmt, SelectStatement):
            return self._emit_select_statement(stmt)
        
        elif isinstance(stmt, DelayStatement):
            return self._emit_delay_statement(stmt)
        
        elif isinstance(stmt, TryStatement):
            return self._emit_try_statement(stmt)
        
        elif isinstance(stmt, LoopInvariant):
            cond = self._emit_expr(stmt.condition)
            return f"{ind}pragma Loop_Invariant ({cond});"
        
        elif isinstance(stmt, LoopVariant):
            return self._emit_loop_variant(stmt)
        
        elif isinstance(stmt, AssertPragma):
            cond = self._emit_expr(stmt.condition)
            msg = f', "{stmt.message}"' if stmt.message else ""
            return f"{ind}pragma Assert ({cond}{msg});"
        
        elif isinstance(stmt, AssumePragma):
            cond = self._emit_expr(stmt.condition)
            return f"{ind}pragma Assume ({cond});"
        
        else:
            return f"{ind}--  Unknown statement: {type(stmt).__name__}"
    
    def _emit_if_statement(self, stmt: IfStatement) -> str:
        """Emit if statement."""
        ind = self._indent()
        lines = []
        
        cond = self._emit_expr(stmt.condition)
        lines.append(f"{ind}if {cond} then")
        
        self._current_indent += 1
        for s in stmt.then_body:
            lines.append(self._emit_statement(s))
        self._current_indent -= 1
        
        # Elsif parts
        for elsif in stmt.elsif_parts:
            cond = self._emit_expr(elsif.condition)
            lines.append(f"{ind}elsif {cond} then")
            self._current_indent += 1
            for s in elsif.body:
                lines.append(self._emit_statement(s))
            self._current_indent -= 1
        
        # Else part
        if stmt.else_body:
            lines.append(f"{ind}else")
            self._current_indent += 1
            for s in stmt.else_body:
                lines.append(self._emit_statement(s))
            self._current_indent -= 1
        
        lines.append(f"{ind}end if;")
        return '\n'.join(lines)
    
    def _emit_case_statement(self, stmt: CaseStatement) -> str:
        """Emit case statement."""
        ind = self._indent()
        lines = []
        
        selector = self._emit_expr(stmt.selector)
        lines.append(f"{ind}case {selector} is")
        
        self._current_indent += 1
        for alt in stmt.alternatives:
            choices = ' | '.join(self._emit_expr(c) for c in alt.choices)
            lines.append(f"{self._indent()}when {choices} =>")
            self._current_indent += 1
            for s in alt.body:
                lines.append(self._emit_statement(s))
            self._current_indent -= 1
        self._current_indent -= 1
        
        lines.append(f"{ind}end case;")
        return '\n'.join(lines)
    
    def _emit_while_loop(self, stmt: WhileLoop) -> str:
        """Emit while loop."""
        ind = self._indent()
        lines = []
        
        label = f"{stmt.loop_name} : " if stmt.loop_name else ""
        cond = self._emit_expr(stmt.condition)
        lines.append(f"{ind}{label}while {cond} loop")
        
        self._current_indent += 1
        for s in stmt.body:
            lines.append(self._emit_statement(s))
        self._current_indent -= 1
        
        end_label = f" {stmt.loop_name}" if stmt.loop_name else ""
        lines.append(f"{ind}end loop{end_label};")
        return '\n'.join(lines)
    
    def _emit_for_loop(self, stmt: ForLoop) -> str:
        """Emit for loop."""
        ind = self._indent()
        lines = []
        
        label = f"{stmt.loop_name} : " if stmt.loop_name else ""
        range_str = self._emit_expr(stmt.range_expr)
        reverse_str = " reverse" if stmt.reverse else ""
        lines.append(f"{ind}{label}for {stmt.variable} in{reverse_str} {range_str} loop")
        
        self._current_indent += 1
        for s in stmt.body:
            lines.append(self._emit_statement(s))
        self._current_indent -= 1
        
        end_label = f" {stmt.loop_name}" if stmt.loop_name else ""
        lines.append(f"{ind}end loop{end_label};")
        return '\n'.join(lines)
    
    def _emit_basic_loop(self, stmt: BasicLoop) -> str:
        """Emit basic loop."""
        ind = self._indent()
        lines = []
        
        label = f"{stmt.loop_name} : " if stmt.loop_name else ""
        lines.append(f"{ind}{label}loop")
        
        self._current_indent += 1
        for s in stmt.body:
            lines.append(self._emit_statement(s))
        self._current_indent -= 1
        
        end_label = f" {stmt.loop_name}" if stmt.loop_name else ""
        lines.append(f"{ind}end loop{end_label};")
        return '\n'.join(lines)
    
    def _emit_exit_statement(self, stmt: ExitStatement) -> str:
        """Emit exit statement."""
        ind = self._indent()
        loop_name = f" {stmt.loop_name}" if stmt.loop_name else ""
        when_clause = f" when {self._emit_expr(stmt.condition)}" if stmt.condition else ""
        return f"{ind}exit{loop_name}{when_clause};"
    
    def _emit_block_statement(self, stmt: BlockStatement) -> str:
        """Emit block statement."""
        ind = self._indent()
        lines = []
        
        label = f"{stmt.name} : " if stmt.name else ""
        
        if stmt.declarations:
            lines.append(f"{ind}{label}declare")
            self._current_indent += 1
            for decl in stmt.declarations:
                if isinstance(decl, VariableDecl):
                    lines.append(self._emit_variable_decl(decl))
            self._current_indent -= 1
            lines.append(f"{ind}begin")
        else:
            lines.append(f"{ind}{label}begin")
        
        self._current_indent += 1
        for s in stmt.body:
            lines.append(self._emit_statement(s))
        self._current_indent -= 1
        
        end_label = f" {stmt.name}" if stmt.name else ""
        lines.append(f"{ind}end{end_label};")
        return '\n'.join(lines)
    
    def _emit_accept_statement(self, stmt: AcceptStatement) -> str:
        """Emit accept statement."""
        ind = self._indent()
        lines = []
        
        params = self._emit_parameters(stmt.parameters)
        params_str = f" ({params})" if params else ""
        
        if stmt.body:
            lines.append(f"{ind}accept {stmt.entry_name}{params_str} do")
            self._current_indent += 1
            for s in stmt.body:
                lines.append(self._emit_statement(s))
            self._current_indent -= 1
            lines.append(f"{ind}end {stmt.entry_name};")
        else:
            lines.append(f"{ind}accept {stmt.entry_name}{params_str};")
        
        return '\n'.join(lines)
    
    def _emit_select_statement(self, stmt: SelectStatement) -> str:
        """Emit select statement."""
        ind = self._indent()
        lines = [f"{ind}select"]
        
        first = True
        for alt in stmt.alternatives:
            if not first:
                lines.append(f"{ind}or")
            first = False
            
            guard = ""
            if alt.guard:
                guard = f"when {self._emit_expr(alt.guard)} =>\n{self._indent(self._current_indent + 1)}"
            
            self._current_indent += 1
            if isinstance(alt, AcceptAlternative):
                lines.append(f"{self._indent()}{guard}{self._emit_accept_statement(alt.accept_statement).strip()}")
                for s in alt.statements:
                    lines.append(self._emit_statement(s))
            elif isinstance(alt, DelayAlternative):
                delay_kw = "delay until" if alt.is_until else "delay"
                lines.append(f"{self._indent()}{guard}{delay_kw} {self._emit_expr(alt.delay_expr)};")
                for s in alt.statements:
                    lines.append(self._emit_statement(s))
            elif isinstance(alt, TerminateAlternative):
                lines.append(f"{self._indent()}{guard}terminate;")
            self._current_indent -= 1
        
        if stmt.else_part:
            lines.append(f"{ind}else")
            self._current_indent += 1
            for s in stmt.else_part:
                lines.append(self._emit_statement(s))
            self._current_indent -= 1
        
        lines.append(f"{ind}end select;")
        return '\n'.join(lines)
    
    def _emit_delay_statement(self, stmt: DelayStatement) -> str:
        """Emit delay statement."""
        ind = self._indent()
        delay_kw = "delay until" if stmt.is_until else "delay"
        return f"{ind}{delay_kw} {self._emit_expr(stmt.duration)};"
    
    def _emit_try_statement(self, stmt: TryStatement) -> str:
        """Emit try (begin/exception) statement."""
        ind = self._indent()
        lines = [f"{ind}begin"]
        
        self._current_indent += 1
        for s in stmt.body:
            lines.append(self._emit_statement(s))
        self._current_indent -= 1
        
        if stmt.handlers:
            lines.append(f"{ind}exception")
            self._current_indent += 1
            for handler in stmt.handlers:
                exceptions = ' | '.join(handler.exception_names) if handler.exception_names else "others"
                param = f" : {handler.choice_parameter}" if handler.choice_parameter else ""
                lines.append(f"{self._indent()}when {exceptions}{param} =>")
                self._current_indent += 1
                for s in handler.body:
                    lines.append(self._emit_statement(s))
                self._current_indent -= 1
            self._current_indent -= 1
        
        lines.append(f"{ind}end;")
        return '\n'.join(lines)
    
    def _emit_loop_variant(self, stmt: LoopVariant) -> str:
        """Emit loop variant pragma."""
        ind = self._indent()
        parts = []
        for v in stmt.expressions:
            direction = "Decreases" if v.direction == 'decreases' else "Increases"
            parts.append(f"{direction} => {self._emit_expr(v.expr)}")
        return f"{ind}pragma Loop_Variant ({', '.join(parts)});"
    
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
            return f"{op} {operand}"
        
        elif isinstance(expr, CallExpr):
            args = ', '.join(self._emit_expr(a) for a in expr.arguments)
            return f"{expr.name} ({args})" if args else expr.name
        
        elif isinstance(expr, MemberAccess):
            target = self._emit_expr(expr.target)
            return f"{target}.{expr.member}"
        
        elif isinstance(expr, IndexExpr):
            target = self._emit_expr(expr.target)
            index = self._emit_expr(expr.index)
            return f"{target} ({index})"
        
        elif isinstance(expr, IfExpr):
            cond = self._emit_expr(expr.condition)
            then_expr = self._emit_expr(expr.then_expr)
            else_expr = self._emit_expr(expr.else_expr)
            return f"(if {cond} then {then_expr} else {else_expr})"
        
        elif isinstance(expr, RangeExpr):
            start = self._emit_expr(expr.start)
            end = self._emit_expr(expr.end)
            return f"{start} .. {end}"
        
        elif isinstance(expr, AggregateExpr):
            return self._emit_aggregate(expr)
        
        elif isinstance(expr, AttributeExpr):
            target = self._emit_expr(expr.target)
            return f"{target}'{expr.attribute}"
        
        elif isinstance(expr, QualifiedExpr):
            type_mark = self._map_type(expr.type_mark)
            inner = self._emit_expr(expr.expr)
            return f"{type_mark}'({inner})"
        
        elif isinstance(expr, CastExpr):
            target_type = self._map_type(expr.target_type)
            inner = self._emit_expr(expr.expr)
            return f"{target_type} ({inner})"
        
        elif isinstance(expr, Allocator):
            type_ref = self._map_type(expr.type_ref)
            if expr.initializer:
                init = self._emit_expr(expr.initializer)
                return f"new {type_ref}'({init})"
            return f"new {type_ref}"
        
        elif isinstance(expr, Dereference):
            target = self._emit_expr(expr.target)
            return f"{target}.all"
        
        elif isinstance(expr, AddressOf):
            target = self._emit_expr(expr.target)
            return f"{target}'Address"
        
        elif isinstance(expr, QuantifiedExpr):
            quant = "for all" if expr.quantifier == 'all' else "for some"
            range_str = self._emit_expr(expr.range_expr)
            cond = self._emit_expr(expr.condition)
            return f"({quant} {expr.variable} in {range_str} => {cond})"
        
        elif isinstance(expr, OldExpr):
            inner = self._emit_expr(expr.expr)
            return f"{inner}'Old"
        
        elif isinstance(expr, ResultExpr):
            func_name = expr.function_name or ""
            return f"{func_name}'Result"
        
        else:
            return f"--  Unknown expr: {type(expr).__name__}"
    
    def _emit_literal(self, lit: Literal) -> str:
        """Emit literal value."""
        if lit.literal_type == 'bool' or isinstance(lit.value, bool):
            return "True" if lit.value else "False"
        elif lit.literal_type == 'char':
            return f"'{lit.value}'"
        elif lit.literal_type == 'string' or isinstance(lit.value, str):
            return f'"{lit.value}"'
        elif lit.literal_type == 'float' or isinstance(lit.value, float):
            return str(lit.value)
        else:
            return str(lit.value)
    
    def _emit_aggregate(self, agg: AggregateExpr) -> str:
        """Emit aggregate expression."""
        parts = []
        
        # Named components
        for name, value in agg.components.items():
            parts.append(f"{name} => {self._emit_expr(value)}")
        
        # Positional components
        for value in agg.positional:
            parts.append(self._emit_expr(value))
        
        if not parts:
            return "(others => <>)"
        
        return f"({', '.join(parts)})"
    
    # =========================================================================
    # Manifest Generation
    # =========================================================================
    
    def generate_manifest(self, package: Package, spec_code: str, body_code: str) -> dict:
        """Generate deterministic build manifest."""
        spec_hash = compute_sha256(spec_code)
        body_hash = compute_sha256(body_code)
        ir_hash = compute_sha256(canonical_json(package.to_dict()))
        
        manifest = {
            'schema': 'stunir.manifest.targets.ada.v1',
            'generator': 'stunir.ada.emitter',
            'epoch': int(time.time()),
            'ir_hash': ir_hash,
            'outputs': {
                'spec': {
                    'hash': spec_hash,
                    'size': len(spec_code),
                    'extension': self.SPEC_EXTENSION,
                },
                'body': {
                    'hash': body_hash,
                    'size': len(body_code),
                    'extension': self.BODY_EXTENSION,
                },
            },
            'manifest_hash': ''
        }
        
        manifest['manifest_hash'] = compute_sha256(
            canonical_json({k: v for k, v in manifest.items() if k != 'manifest_hash'})
        )
        
        return manifest
    
    def emit(self, package: Package) -> EmitterResult:
        """Emit package with manifest."""
        spec_code, body_code = self.emit_package(package)
        manifest = self.generate_manifest(package, spec_code, body_code)
        return EmitterResult(spec_code=spec_code, body_code=body_code, manifest=manifest)
