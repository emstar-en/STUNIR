#!/usr/bin/env python3
"""STUNIR F# Emitter - Generate F# code from Functional IR.

This emitter generates idiomatic F# code including:
- Discriminated unions (sum types)
- Record types
- Function definitions with pattern matching
- Computation expressions (async, seq, query)
- Units of measure
- Active patterns
- .NET interoperability (classes, interfaces)
- Module definitions

Usage:
    from targets.functional.fsharp_emitter import FSharpEmitter
    from ir.functional import Module
    
    emitter = FSharpEmitter()
    code = emitter.emit_module(module)
"""

import sys
import json
from pathlib import Path
from typing import Optional, Dict, List, Any, Union

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from targets.functional.base import FunctionalEmitterBase, EmitterResult, compute_sha256
from ir.functional import (
    # Types
    TypeExpr, TypeVar, TypeCon, FunctionType, TupleType, ListType,
    # Expressions
    Expr, LiteralExpr, VarExpr, AppExpr, LambdaExpr, LetExpr, IfExpr,
    CaseBranch, CaseExpr, ListGenerator, ListExpr, TupleExpr,
    BinaryOpExpr, UnaryOpExpr,
    # Patterns
    Pattern, WildcardPattern, VarPattern, LiteralPattern,
    ConstructorPattern, TuplePattern, ListPattern, AsPattern, OrPattern,
    # ADTs
    TypeParameter, DataConstructor, DataType, TypeAlias,
    RecordField, RecordType,
    Import, FunctionClause, FunctionDef, Module,
    # F# specific
    ComputationExpr, ComputationStatement, ComputationLet, ComputationDo,
    ComputationReturn, ComputationYield, ComputationFor, ComputationWhile,
    ComputationIf, ComputationTry,
    MeasureType, MeasureExpr, MeasureUnit, MeasureProd, MeasureDiv,
    MeasurePow, MeasureOne, MeasureDeclaration,
    ActivePattern, ActivePatternMatch, ParameterizedActivePattern,
    Attribute, ClassMember, ClassField, ClassMethod, ClassProperty,
    ClassEvent, Constructor, ClassDef, InterfaceMember, InterfaceDef,
    StructDef, ObjectExpr, UseExpr, PipelineExpr, CompositionExpr,
    QuotationExpr,
)


# F#-specific type mappings
FSHARP_TYPE_MAP = {
    # Numeric types
    'int': 'int',
    'i8': 'int8',
    'i16': 'int16',
    'i32': 'int',
    'i64': 'int64',
    'u8': 'uint8',
    'u16': 'uint16',
    'u32': 'uint32',
    'u64': 'uint64',
    'float': 'float',
    'f32': 'float32',
    'f64': 'float',
    'double': 'float',
    'decimal': 'decimal',
    
    # Other primitives
    'bool': 'bool',
    'string': 'string',
    'char': 'char',
    'unit': 'unit',
    'void': 'unit',
    'byte': 'byte',
    'sbyte': 'sbyte',
    
    # Collections
    'list': 'list',
    'array': 'array',
    'seq': 'seq',
    'option': 'option',
    'result': 'Result',
    'map': 'Map',
    'set': 'Set',
    
    # .NET types
    'object': 'obj',
    'Object': 'obj',
    'String': 'string',
    'Int32': 'int',
    'Int64': 'int64',
    'DateTime': 'System.DateTime',
    'Guid': 'System.Guid',
    'TimeSpan': 'System.TimeSpan',
}

# F# operator mappings
FSHARP_OPERATOR_MAP = {
    # Arithmetic
    '+': '+',
    '-': '-',
    '*': '*',
    '/': '/',
    '%': '%',
    '**': '**',
    'mod': '%',
    
    # Comparison
    '==': '=',
    '=': '=',
    '!=': '<>',
    '/=': '<>',
    '<>': '<>',
    '<': '<',
    '>': '>',
    '<=': '<=',
    '>=': '>=',
    
    # Logical
    '&&': '&&',
    '||': '||',
    'and': '&&',
    'or': '||',
    'not': 'not',
    
    # List/Sequence
    '::': '::',
    '@': '@',
    '++': '@',
    
    # Pipeline
    '|>': '|>',
    '<|': '<|',
    '>>': '>>',
    '<<': '<<',
    
    # Reference
    ':=': ':=',
    '!': '!',
}


class FSharpEmitter(FunctionalEmitterBase):
    """Emitter for F# code generation.
    
    Generates idiomatic F# code from Functional IR.
    """
    
    DIALECT = 'fsharp'
    FILE_EXTENSION = '.fs'
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.indent_size = self.config.get('indent_size', 4)
        self.generate_fsproj = self.config.get('generate_fsproj', False)
        self.target_framework = self.config.get('target_framework', 'net8.0')
    
    def emit_module(self, module: Module) -> str:
        """Emit complete F# module.
        
        Args:
            module: Module IR node
            
        Returns:
            Generated F# code
        """
        lines = []
        
        # Module declaration (namespace or module)
        if '.' in module.name:
            namespace = '.'.join(module.name.split('.')[:-1])
            mod_name = module.name.split('.')[-1]
            lines.append(f"namespace {namespace}")
            lines.append("")
            lines.append(f"module {mod_name} =")
            indent = "    "
        else:
            lines.append(f"module {module.name}")
            lines.append("")
            indent = ""
        
        # Open statements (imports)
        for imp in module.imports:
            lines.append(f"{indent}{self._emit_import(imp)}")
        if module.imports:
            lines.append("")
        
        # Type definitions
        for type_def in module.type_definitions:
            type_code = self._emit_type_definition(type_def)
            for line in type_code.split('\n'):
                lines.append(f"{indent}{line}" if line else "")
            lines.append("")
        
        # Functions
        for func in module.functions:
            func_code = self._emit_function(func)
            for line in func_code.split('\n'):
                lines.append(f"{indent}{line}" if line else "")
            lines.append("")
        
        return '\n'.join(lines)
    
    def _emit_import(self, imp: Import) -> str:
        """Emit open statement."""
        if imp.alias:
            return f"module {imp.alias} = {imp.module}"
        return f"open {imp.module}"
    
    def _emit_type_definition(self, type_def) -> str:
        """Emit type definition."""
        if isinstance(type_def, DataType):
            return self._emit_discriminated_union(type_def)
        elif isinstance(type_def, RecordType):
            return self._emit_record_type(type_def)
        elif isinstance(type_def, TypeAlias):
            return self._emit_type_alias(type_def)
        elif isinstance(type_def, MeasureDeclaration):
            return self._emit_measure_declaration(type_def)
        elif isinstance(type_def, ClassDef):
            return self._emit_class(type_def)
        elif isinstance(type_def, InterfaceDef):
            return self._emit_interface(type_def)
        elif isinstance(type_def, StructDef):
            return self._emit_struct(type_def)
        return ""
    
    def _emit_discriminated_union(self, data: DataType) -> str:
        """Emit F# discriminated union (sum type)."""
        params = self._emit_type_params(data.type_params)
        if params:
            header = f"type {data.name}<{params}> ="
        else:
            header = f"type {data.name} ="
        
        constructors = []
        for con in data.constructors:
            con_str = self._emit_du_case(con)
            constructors.append(f"    | {con_str}")
        
        return header + '\n' + '\n'.join(constructors)
    
    def _emit_du_case(self, con: DataConstructor) -> str:
        """Emit discriminated union case."""
        if not con.fields:
            return con.name
        if con.field_names:
            # Named fields (record-like)
            fields = '; '.join(
                f"{name}: {self._emit_type(ftype)}"
                for name, ftype in zip(con.field_names, con.fields)
            )
            return f"{con.name} of {{ {fields} }}"
        else:
            # Positional fields
            if len(con.fields) == 1:
                return f"{con.name} of {self._emit_type(con.fields[0])}"
            field_types = ' * '.join(self._emit_type(f) for f in con.fields)
            return f"{con.name} of {field_types}"
    
    def _emit_record_type(self, record: RecordType) -> str:
        """Emit F# record type."""
        params = self._emit_type_params(record.type_params)
        if params:
            header = f"type {record.name}<{params}> ="
        else:
            header = f"type {record.name} ="
        
        lines = [header, "    {"]
        for field in record.fields:
            mutable = "mutable " if field.mutable else ""
            lines.append(f"        {mutable}{field.name}: {self._emit_type(field.field_type)}")
        lines.append("    }")
        
        return '\n'.join(lines)
    
    def _emit_type_alias(self, alias: TypeAlias) -> str:
        """Emit type alias."""
        params = self._emit_type_params(alias.type_params)
        if params:
            return f"type {alias.name}<{params}> = {self._emit_type(alias.target)}"
        return f"type {alias.name} = {self._emit_type(alias.target)}"
    
    def _emit_type_params(self, type_params: List[TypeParameter]) -> str:
        """Emit type parameter list."""
        if not type_params:
            return ''
        return ', '.join(f"'{p.name}" for p in type_params)
    
    def _emit_type(self, type_expr: TypeExpr) -> str:
        """Emit F# type expression."""
        if type_expr is None:
            return "'a"
        
        if isinstance(type_expr, TypeVar):
            return f"'{type_expr.name}"
        elif isinstance(type_expr, TypeCon):
            mapped = FSHARP_TYPE_MAP.get(type_expr.name.lower(), type_expr.name)
            mapped = FSHARP_TYPE_MAP.get(type_expr.name, mapped)  # Try exact match too
            if type_expr.args:
                if len(type_expr.args) == 1:
                    arg = self._emit_type(type_expr.args[0])
                    # F# uses postfix for single-arg types: 'a list, int option
                    if mapped in ('list', 'option', 'seq', 'array'):
                        return f"{arg} {mapped}"
                    return f"{mapped}<{arg}>"
                args = ', '.join(self._emit_type(a) for a in type_expr.args)
                return f"{mapped}<{args}>"
            return mapped
        elif isinstance(type_expr, FunctionType):
            param = self._emit_type(type_expr.param_type)
            ret = self._emit_type(type_expr.return_type)
            # Wrap complex types in parens
            if isinstance(type_expr.param_type, FunctionType):
                param = f"({param})"
            return f"{param} -> {ret}"
        elif isinstance(type_expr, TupleType):
            elems = ' * '.join(self._emit_type(e) for e in type_expr.elements)
            return f"({elems})"
        elif isinstance(type_expr, ListType):
            return f"{self._emit_type(type_expr.element_type)} list"
        elif isinstance(type_expr, MeasureType):
            return self._emit_measure_type(type_expr)
        return "'a"
    
    # =========================================================================
    # Function Emission
    # =========================================================================
    
    def _emit_function(self, func: FunctionDef) -> str:
        """Emit F# function definition."""
        is_rec = self._is_recursive(func)
        rec_kw = "rec " if is_rec else ""
        
        # Simple value binding (no patterns)
        if len(func.clauses) == 1 and not func.clauses[0].patterns:
            clause = func.clauses[0]
            body = self._emit_expr(clause.body)
            return f"let {rec_kw}{func.name} = {body}"
        
        # Single clause function
        if len(func.clauses) == 1:
            clause = func.clauses[0]
            params = ' '.join(self._emit_pattern(p) for p in clause.patterns)
            body = self._emit_expr(clause.body)
            
            # Type signature if available
            sig = ""
            if func.type_signature:
                sig = f" : {self._emit_type(func.type_signature)}"
            
            return f"let {rec_kw}{func.name} {params}{sig} = {body}"
        
        # Multi-clause function using match
        param_count = len(func.clauses[0].patterns) if func.clauses else 0
        params = ' '.join(f"__arg{i}" for i in range(param_count))
        
        lines = [f"let {rec_kw}{func.name} {params} ="]
        
        if param_count == 1:
            match_expr = "__arg0"
        else:
            match_expr = '(' + ', '.join(f"__arg{i}" for i in range(param_count)) + ')'
        
        lines.append(f"    match {match_expr} with")
        
        for clause in func.clauses:
            if param_count == 1:
                pattern = self._emit_pattern(clause.patterns[0])
            else:
                pattern = '(' + ', '.join(self._emit_pattern(p) for p in clause.patterns) + ')'
            
            body = self._emit_expr(clause.body)
            
            if clause.guard:
                guard = self._emit_expr(clause.guard)
                lines.append(f"    | {pattern} when {guard} -> {body}")
            else:
                lines.append(f"    | {pattern} -> {body}")
        
        return '\n'.join(lines)
    
    def _is_recursive(self, func: FunctionDef) -> bool:
        """Check if function is recursive."""
        for clause in func.clauses:
            if self._uses_name(clause.body, func.name):
                return True
        return False
    
    def _uses_name(self, expr: Expr, name: str) -> bool:
        """Check if expression uses a name (for recursion detection)."""
        if expr is None:
            return False
        
        if isinstance(expr, VarExpr):
            return expr.name == name
        elif isinstance(expr, AppExpr):
            return self._uses_name(expr.func, name) or self._uses_name(expr.arg, name)
        elif isinstance(expr, LambdaExpr):
            return self._uses_name(expr.body, name)
        elif isinstance(expr, LetExpr):
            return self._uses_name(expr.value, name) or self._uses_name(expr.body, name)
        elif isinstance(expr, IfExpr):
            return (self._uses_name(expr.condition, name) or
                    self._uses_name(expr.then_branch, name) or
                    self._uses_name(expr.else_branch, name))
        elif isinstance(expr, CaseExpr):
            if self._uses_name(expr.scrutinee, name):
                return True
            return any(self._uses_name(b.body, name) for b in expr.branches)
        elif isinstance(expr, BinaryOpExpr):
            return self._uses_name(expr.left, name) or self._uses_name(expr.right, name)
        elif isinstance(expr, ListExpr):
            return any(self._uses_name(e, name) for e in expr.elements)
        elif isinstance(expr, TupleExpr):
            return any(self._uses_name(e, name) for e in expr.elements)
        elif isinstance(expr, ComputationExpr):
            return any(self._computation_uses_name(s, name) for s in expr.body)
        return False
    
    def _computation_uses_name(self, stmt: ComputationStatement, name: str) -> bool:
        """Check if computation statement uses a name."""
        if isinstance(stmt, ComputationLet):
            return self._uses_name(stmt.value, name)
        elif isinstance(stmt, ComputationDo):
            return self._uses_name(stmt.expr, name)
        elif isinstance(stmt, ComputationReturn):
            return self._uses_name(stmt.value, name)
        elif isinstance(stmt, ComputationYield):
            return self._uses_name(stmt.value, name)
        elif isinstance(stmt, ComputationFor):
            return (self._uses_name(stmt.source, name) or
                    any(self._computation_uses_name(s, name) for s in stmt.body))
        return False
    
    # =========================================================================
    # Pattern Emission
    # =========================================================================
    
    def _emit_pattern(self, pattern: Pattern) -> str:
        """Emit F# pattern."""
        if pattern is None:
            return "_"
        
        if isinstance(pattern, WildcardPattern):
            return "_"
        elif isinstance(pattern, VarPattern):
            return pattern.name
        elif isinstance(pattern, LiteralPattern):
            return self._emit_literal_value(pattern.value, pattern.literal_type)
        elif isinstance(pattern, ConstructorPattern):
            if pattern.args:
                args = ', '.join(self._emit_pattern(a) for a in pattern.args)
                return f"{pattern.constructor}({args})"
            return pattern.constructor
        elif isinstance(pattern, TuplePattern):
            elems = ', '.join(self._emit_pattern(e) for e in pattern.elements)
            return f"({elems})"
        elif isinstance(pattern, ListPattern):
            if pattern.rest:
                if pattern.elements:
                    elems = '::'.join(self._emit_pattern(e) for e in pattern.elements)
                    return f"{elems}::{self._emit_pattern(pattern.rest)}"
                return self._emit_pattern(pattern.rest)
            elems = '; '.join(self._emit_pattern(e) for e in pattern.elements)
            return f"[{elems}]"
        elif isinstance(pattern, AsPattern):
            return f"{self._emit_pattern(pattern.pattern)} as {pattern.name}"
        elif isinstance(pattern, OrPattern):
            return f"({self._emit_pattern(pattern.left)} | {self._emit_pattern(pattern.right)})"
        elif isinstance(pattern, ActivePatternMatch):
            if pattern.args:
                args = ', '.join(self._emit_pattern(a) for a in pattern.args)
                return f"{pattern.pattern_name}({args})"
            return pattern.pattern_name
        return "_"
    
    # =========================================================================
    # Expression Emission
    # =========================================================================
    
    def _emit_expr(self, expr: Expr) -> str:
        """Emit F# expression."""
        if expr is None:
            return "()"
        
        if isinstance(expr, LiteralExpr):
            return self._emit_literal_value(expr.value, expr.literal_type)
        elif isinstance(expr, VarExpr):
            return expr.name
        elif isinstance(expr, AppExpr):
            return self._emit_app(expr)
        elif isinstance(expr, LambdaExpr):
            return f"(fun {expr.param} -> {self._emit_expr(expr.body)})"
        elif isinstance(expr, LetExpr):
            return self._emit_let(expr)
        elif isinstance(expr, IfExpr):
            return self._emit_if(expr)
        elif isinstance(expr, CaseExpr):
            return self._emit_match(expr)
        elif isinstance(expr, ListExpr):
            return self._emit_list(expr)
        elif isinstance(expr, TupleExpr):
            return self._emit_tuple(expr)
        elif isinstance(expr, BinaryOpExpr):
            return self._emit_binary_op(expr)
        elif isinstance(expr, UnaryOpExpr):
            return self._emit_unary_op(expr)
        elif isinstance(expr, ComputationExpr):
            return self._emit_computation_expr(expr)
        elif isinstance(expr, UseExpr):
            return self._emit_use_expr(expr)
        elif isinstance(expr, PipelineExpr):
            return self._emit_pipeline_expr(expr)
        elif isinstance(expr, CompositionExpr):
            return self._emit_composition_expr(expr)
        elif isinstance(expr, ObjectExpr):
            return self._emit_object_expr(expr)
        elif isinstance(expr, QuotationExpr):
            return self._emit_quotation_expr(expr)
        return "()"
    
    def _emit_app(self, expr: AppExpr) -> str:
        """Emit function application."""
        func = self._emit_expr(expr.func)
        if expr.arg:
            arg = self._emit_expr(expr.arg)
            # Wrap tuple arguments
            if isinstance(expr.arg, TupleExpr) and len(expr.arg.elements) > 1:
                return f"({func} {arg})"
            return f"({func} {arg})"
        return func
    
    def _emit_let(self, expr: LetExpr) -> str:
        """Emit let binding."""
        rec = "rec " if expr.is_recursive else ""
        value = self._emit_expr(expr.value)
        body = self._emit_expr(expr.body)
        return f"let {rec}{expr.name} = {value} in {body}"
    
    def _emit_if(self, expr: IfExpr) -> str:
        """Emit if expression."""
        cond = self._emit_expr(expr.condition)
        then_b = self._emit_expr(expr.then_branch)
        else_b = self._emit_expr(expr.else_branch)
        return f"if {cond} then {then_b} else {else_b}"
    
    def _emit_match(self, case: CaseExpr) -> str:
        """Emit match expression."""
        scrutinee = self._emit_expr(case.scrutinee)
        lines = [f"match {scrutinee} with"]
        for branch in case.branches:
            pattern = self._emit_pattern(branch.pattern)
            body = self._emit_expr(branch.body)
            if branch.guard:
                guard = self._emit_expr(branch.guard)
                lines.append(f"    | {pattern} when {guard} -> {body}")
            else:
                lines.append(f"    | {pattern} -> {body}")
        return '\n'.join(lines)
    
    def _emit_list(self, expr: ListExpr) -> str:
        """Emit list expression."""
        if expr.is_comprehension:
            return self._emit_list_comprehension(expr)
        elems = '; '.join(self._emit_expr(e) for e in expr.elements)
        return f"[{elems}]"
    
    def _emit_list_comprehension(self, expr: ListExpr) -> str:
        """Emit list comprehension as sequence expression."""
        # F# uses sequence expressions instead of list comprehensions
        if not expr.generators:
            return "[]"
        
        gen = expr.generators[0]
        pattern = self._emit_pattern(gen.pattern)
        source = self._emit_expr(gen.source)
        
        elem = self._emit_expr(expr.elements[0]) if expr.elements else "_"
        
        # Add conditions
        conditions = ""
        if gen.conditions:
            conds = ' && '.join(self._emit_expr(c) for c in gen.conditions)
            conditions = f" if {conds}"
        
        return f"[ for {pattern} in {source}{conditions} -> {elem} ]"
    
    def _emit_tuple(self, expr: TupleExpr) -> str:
        """Emit tuple expression."""
        elems = ', '.join(self._emit_expr(e) for e in expr.elements)
        return f"({elems})"
    
    def _emit_binary_op(self, expr: BinaryOpExpr) -> str:
        """Emit binary operation."""
        left = self._emit_expr(expr.left)
        right = self._emit_expr(expr.right)
        op = FSHARP_OPERATOR_MAP.get(expr.op, expr.op)
        return f"({left} {op} {right})"
    
    def _emit_unary_op(self, expr: UnaryOpExpr) -> str:
        """Emit unary operation."""
        operand = self._emit_expr(expr.operand)
        if expr.op == 'not':
            return f"(not {operand})"
        elif expr.op == '-':
            return f"(-{operand})"
        elif expr.op == '!':
            return f"(!{operand})"
        return f"({expr.op}{operand})"
    
    def _emit_literal_value(self, value: Any, literal_type: str) -> str:
        """Emit literal value."""
        if literal_type == 'bool':
            return 'true' if value else 'false'
        elif literal_type == 'char':
            return f"'{value}'"
        elif literal_type == 'string':
            escaped = str(value).replace('\\', '\\\\').replace('"', '\\"')
            return f'"{escaped}"'
        elif literal_type == 'float':
            s = str(float(value))
            if '.' not in s and 'e' not in s.lower():
                s += '.0'
            return s
        elif literal_type == 'unit':
            return "()"
        return str(value)
    
    # =========================================================================
    # Computation Expression Emission
    # =========================================================================
    
    def _emit_computation_expr(self, expr: ComputationExpr) -> str:
        """Emit F# computation expression."""
        lines = [f"{expr.builder} {{"]
        
        for stmt in expr.body:
            stmt_code = self._emit_computation_statement(stmt)
            for line in stmt_code.split('\n'):
                lines.append(f"    {line}")
        
        lines.append("}")
        return '\n'.join(lines)
    
    def _emit_computation_statement(self, stmt: ComputationStatement) -> str:
        """Emit computation expression statement."""
        if isinstance(stmt, ComputationLet):
            bang = "!" if stmt.is_bang else ""
            pattern = self._emit_pattern(stmt.pattern)
            value = self._emit_expr(stmt.value)
            return f"let{bang} {pattern} = {value}"
        
        elif isinstance(stmt, ComputationDo):
            bang = "!" if stmt.is_bang else ""
            return f"do{bang} {self._emit_expr(stmt.expr)}"
        
        elif isinstance(stmt, ComputationReturn):
            bang = "!" if stmt.is_bang else ""
            return f"return{bang} {self._emit_expr(stmt.value)}"
        
        elif isinstance(stmt, ComputationYield):
            bang = "!" if stmt.is_bang else ""
            return f"yield{bang} {self._emit_expr(stmt.value)}"
        
        elif isinstance(stmt, ComputationFor):
            pattern = self._emit_pattern(stmt.pattern)
            source = self._emit_expr(stmt.source)
            body_lines = [self._emit_computation_statement(s) for s in stmt.body]
            body = '\n        '.join(body_lines)
            return f"for {pattern} in {source} do\n        {body}"
        
        elif isinstance(stmt, ComputationWhile):
            cond = self._emit_expr(stmt.condition)
            body_lines = [self._emit_computation_statement(s) for s in stmt.body]
            body = '\n        '.join(body_lines)
            return f"while {cond} do\n        {body}"
        
        elif isinstance(stmt, ComputationIf):
            cond = self._emit_expr(stmt.condition)
            then_lines = [self._emit_computation_statement(s) for s in stmt.then_body]
            then_body = '\n        '.join(then_lines)
            if stmt.else_body:
                else_lines = [self._emit_computation_statement(s) for s in stmt.else_body]
                else_body = '\n        '.join(else_lines)
                return f"if {cond} then\n        {then_body}\n    else\n        {else_body}"
            return f"if {cond} then\n        {then_body}"
        
        elif isinstance(stmt, ComputationTry):
            body_lines = [self._emit_computation_statement(s) for s in stmt.body]
            body = '\n        '.join(body_lines)
            lines = [f"try\n        {body}"]
            
            for pattern, handlers in stmt.handlers:
                handler_lines = [self._emit_computation_statement(s) for s in handlers]
                handler_body = '\n        '.join(handler_lines)
                lines.append(f"    with {self._emit_pattern(pattern)} ->\n        {handler_body}")
            
            if stmt.finally_body:
                finally_lines = [self._emit_computation_statement(s) for s in stmt.finally_body]
                finally_body = '\n        '.join(finally_lines)
                lines.append(f"    finally\n        {finally_body}")
            
            return '\n'.join(lines)
        
        return ""
    
    # =========================================================================
    # Units of Measure Emission
    # =========================================================================
    
    def _emit_measure_declaration(self, measure: MeasureDeclaration) -> str:
        """Emit unit of measure declaration."""
        if measure.base_measure:
            base = self._emit_measure_expr(measure.base_measure)
            if measure.conversion_factor:
                return f"[<Measure>] type {measure.name} = {measure.conversion_factor} * {base}"
            return f"[<Measure>] type {measure.name} = {base}"
        return f"[<Measure>] type {measure.name}"
    
    def _emit_measure_type(self, measure_type: MeasureType) -> str:
        """Emit measure type annotation."""
        base = self._emit_type(measure_type.base_type)
        measure = self._emit_measure_expr(measure_type.measure)
        return f"{base}<{measure}>"
    
    def _emit_measure_expr(self, measure: MeasureExpr) -> str:
        """Emit measure expression."""
        if isinstance(measure, MeasureUnit):
            return measure.name
        elif isinstance(measure, MeasureProd):
            left = self._emit_measure_expr(measure.left)
            right = self._emit_measure_expr(measure.right)
            return f"{left} * {right}"
        elif isinstance(measure, MeasureDiv):
            num = self._emit_measure_expr(measure.numerator)
            den = self._emit_measure_expr(measure.denominator)
            return f"{num} / {den}"
        elif isinstance(measure, MeasurePow):
            base = self._emit_measure_expr(measure.base)
            return f"{base}^{measure.power}"
        elif isinstance(measure, MeasureOne):
            return "1"
        return ""
    
    # =========================================================================
    # Active Pattern Emission
    # =========================================================================
    
    def _emit_active_pattern(self, ap: ActivePattern) -> str:
        """Emit F# active pattern definition."""
        cases = '|'.join(ap.cases)
        if ap.is_partial:
            cases += '|_'
        pattern_sig = f"(|{cases}|)"
        
        params = ' '.join(ap.params)
        body = self._emit_expr(ap.body)
        
        return f"let {pattern_sig} {params} = {body}"
    
    def _emit_parameterized_active_pattern(self, ap: ParameterizedActivePattern) -> str:
        """Emit parameterized active pattern."""
        cases = '|'.join(ap.cases)
        if ap.is_partial:
            cases += '|_'
        pattern_sig = f"(|{cases}|)"
        
        pattern_params = ' '.join(ap.pattern_params)
        additional = ' '.join(ap.additional_params)
        all_params = f"{pattern_params} {additional}".strip()
        body = self._emit_expr(ap.body)
        
        return f"let {pattern_sig} {all_params} = {body}"
    
    # =========================================================================
    # .NET Interop Emission
    # =========================================================================
    
    def _emit_class(self, cls: ClassDef) -> str:
        """Emit F# class definition."""
        lines = []
        
        # Attributes
        for attr in cls.attributes:
            attr_str = self._emit_attribute(attr)
            lines.append(attr_str)
        
        # Class header
        params = self._emit_type_params(cls.type_params)
        header = f"type {cls.name}"
        if params:
            header += f"<{params}>"
        
        # Primary constructor
        if cls.primary_constructor:
            ctor_params = ', '.join(
                f"{name}: {self._emit_type(ptype)}"
                for name, ptype in cls.primary_constructor
            )
            header += f"({ctor_params})"
        else:
            header += "()"
        
        # Base class
        if cls.base_class:
            header += " ="
            lines.append(header)
            lines.append(f"    inherit {cls.base_class}()")
        else:
            header += " ="
            lines.append(header)
        
        # Interface implementations
        for interface in cls.interfaces:
            lines.append(f"    interface {interface} with")
        
        # Members
        for member in cls.members:
            member_code = self._emit_class_member(member)
            for line in member_code.split('\n'):
                lines.append(f"    {line}")
        
        return '\n'.join(lines)
    
    def _emit_class_member(self, member: ClassMember) -> str:
        """Emit class member."""
        # Attributes first
        attr_lines = []
        for attr in member.attributes:
            attr_lines.append(self._emit_attribute(attr))
        attr_prefix = '\n'.join(attr_lines) + '\n' if attr_lines else ''
        
        if isinstance(member, ClassField):
            mutable = "mutable " if member.is_mutable else ""
            static = "static " if member.is_static else ""
            if member.default_value:
                default = self._emit_expr(member.default_value)
                return f"{attr_prefix}let {static}{mutable}{member.name} = {default}"
            type_ann = f": {self._emit_type(member.field_type)}" if member.field_type else ""
            return f"{attr_prefix}let {static}{mutable}{member.name}{type_ann}"
        
        elif isinstance(member, ClassMethod):
            static = "static " if member.is_static else ""
            override = "override " if member.is_override else ""
            abstract = "abstract " if member.is_abstract else ""
            
            if member.params:
                params = ', '.join(f"{n}: {self._emit_type(t)}" for n, t in member.params)
            else:
                params = "()"
            
            self_ref = "this" if not member.is_static else ""
            
            if member.is_abstract:
                ret_type = self._emit_type(member.return_type) if member.return_type else "unit"
                return f"{attr_prefix}abstract member {member.name}: {ret_type}"
            
            body = self._emit_expr(member.body)
            return f"{attr_prefix}{static}{override}member {self_ref}.{member.name}({params}) = {body}"
        
        elif isinstance(member, ClassProperty):
            static = "static " if member.is_static else ""
            override = "override " if member.is_override else ""
            self_ref = "this" if not member.is_static else "_"
            
            lines = [f"{attr_prefix}{static}{override}member {self_ref}.{member.name}"]
            if member.getter:
                lines.append(f"    with get() = {self._emit_expr(member.getter)}")
            if member.setter:
                lines.append(f"    and set(value) = {self._emit_expr(member.setter)}")
            return '\n'.join(lines)
        
        elif isinstance(member, ClassEvent):
            type_str = self._emit_type(member.event_type) if member.event_type else "IEvent<_>"
            return f"{attr_prefix}[<CLIEvent>] member this.{member.name}: {type_str}"
        
        return ""
    
    def _emit_attribute(self, attr: Attribute) -> str:
        """Emit F# attribute."""
        if attr.args or attr.named_args:
            args = []
            args.extend(str(a) for a in attr.args)
            args.extend(f"{k}={v}" for k, v in attr.named_args.items())
            return f"[<{attr.name}({', '.join(args)})>]"
        return f"[<{attr.name}>]"
    
    def _emit_interface(self, iface: InterfaceDef) -> str:
        """Emit F# interface definition."""
        lines = []
        
        # Attributes
        for attr in iface.attributes:
            lines.append(self._emit_attribute(attr))
        
        # Interface header
        params = self._emit_type_params(iface.type_params)
        header = f"type {iface.name}"
        if params:
            header += f"<{params}>"
        header += " ="
        lines.append(header)
        
        # Base interfaces
        for base in iface.base_interfaces:
            lines.append(f"    inherit {base}")
        
        # Members
        for member in iface.members:
            member_type = self._emit_type(member.member_type)
            lines.append(f"    abstract member {member.name}: {member_type}")
        
        return '\n'.join(lines)
    
    def _emit_struct(self, struct: StructDef) -> str:
        """Emit F# struct definition."""
        lines = ["[<Struct>]"]
        
        params = self._emit_type_params(struct.type_params)
        if params:
            header = f"type {struct.name}<{params}> ="
        else:
            header = f"type {struct.name} ="
        lines.append(header)
        
        for field in struct.fields:
            mutable = "mutable " if field.is_mutable else ""
            type_str = self._emit_type(field.field_type) if field.field_type else "'a"
            lines.append(f"    val {mutable}{field.name}: {type_str}")
        
        return '\n'.join(lines)
    
    # =========================================================================
    # Additional F# Constructs
    # =========================================================================
    
    def _emit_use_expr(self, expr: UseExpr) -> str:
        """Emit use binding."""
        value = self._emit_expr(expr.value)
        body = self._emit_expr(expr.body)
        return f"use {expr.name} = {value} in {body}"
    
    def _emit_pipeline_expr(self, expr: PipelineExpr) -> str:
        """Emit pipeline expression."""
        left = self._emit_expr(expr.left)
        right = self._emit_expr(expr.right)
        op = '|>' if expr.direction == 'forward' else '<|'
        return f"({left} {op} {right})"
    
    def _emit_composition_expr(self, expr: CompositionExpr) -> str:
        """Emit function composition."""
        left = self._emit_expr(expr.left)
        right = self._emit_expr(expr.right)
        op = '>>' if expr.direction == 'forward' else '<<'
        return f"({left} {op} {right})"
    
    def _emit_object_expr(self, expr: ObjectExpr) -> str:
        """Emit object expression."""
        lines = ["{"]
        
        if expr.base_call:
            base_type, args = expr.base_call
            args_str = ', '.join(self._emit_expr(a) for a in args)
            lines.append(f"    new {base_type}({args_str}) with")
        elif expr.interface_type:
            iface = self._emit_type(expr.interface_type)
            lines.append(f"    new {iface} with")
        
        for member in expr.members:
            member_code = self._emit_class_member(member)
            for line in member_code.split('\n'):
                lines.append(f"        {line}")
        
        lines.append("}")
        return '\n'.join(lines)
    
    def _emit_quotation_expr(self, expr: QuotationExpr) -> str:
        """Emit code quotation."""
        body = self._emit_expr(expr.body)
        if expr.is_typed:
            return f"<@ {body} @>"
        return f"<@@ {body} @@>"
    
    # =========================================================================
    # Project File Generation
    # =========================================================================
    
    def emit_fsproj(self, module_name: str, files: List[str]) -> str:
        """Generate F# project file content.
        
        Args:
            module_name: Project name
            files: List of source files
            
        Returns:
            fsproj XML content
        """
        lines = [
            '<Project Sdk="Microsoft.NET.Sdk">',
            '  <PropertyGroup>',
            f'    <TargetFramework>{self.target_framework}</TargetFramework>',
            '    <OutputType>Library</OutputType>',
            '  </PropertyGroup>',
            '  <ItemGroup>',
        ]
        
        for f in files:
            lines.append(f'    <Compile Include="{f}" />')
        
        lines.extend([
            '  </ItemGroup>',
            '</Project>',
        ])
        
        return '\n'.join(lines)


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='STUNIR F# Emitter')
    parser.add_argument('ir_file', nargs='?', help='Input IR JSON file')
    parser.add_argument('--output', '-o', help='Output file')
    parser.add_argument('--fsproj', action='store_true', help='Generate .fsproj file')
    args = parser.parse_args()
    
    if args.ir_file:
        # Load IR
        with open(args.ir_file, 'r') as f:
            ir_data = json.load(f)
        
        print(f"Loaded IR from {args.ir_file}")
        print(f"Would generate F# code to {args.output or 'stdout'}")
    else:
        # Demo mode
        emitter = FSharpEmitter()
        
        # Create a sample module
        module = Module(
            name='Sample',
            imports=[Import(module='System')],
            type_definitions=[
                DataType(
                    name='Option',
                    type_params=[TypeParameter(name='a')],
                    constructors=[
                        DataConstructor(name='None'),
                        DataConstructor(name='Some', fields=[TypeVar(name='a')])
                    ]
                )
            ],
            functions=[
                FunctionDef(
                    name='add',
                    clauses=[FunctionClause(
                        patterns=[VarPattern(name='x'), VarPattern(name='y')],
                        body=BinaryOpExpr(
                            op='+',
                            left=VarExpr(name='x'),
                            right=VarExpr(name='y')
                        )
                    )]
                )
            ]
        )
        
        code = emitter.emit_module(module)
        print("Generated F# code:")
        print(code)


if __name__ == '__main__':
    main()
