#!/usr/bin/env python3
"""STUNIR OCaml Emitter - Generate OCaml code from Functional IR.

This emitter generates idiomatic OCaml code including:
- Variant type declarations
- Record type declarations
- Function definitions with pattern matching
- Module definitions and signatures
- Functors
- Imperative features (ref, mutable)

Usage:
    from targets.functional.ocaml_emitter import OCamlEmitter
    from ir.functional import Module
    
    emitter = OCamlEmitter()
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
)


# OCaml-specific type mappings
OCAML_TYPE_MAP = {
    'int': 'int',
    'i32': 'int32',
    'i64': 'int64',
    'float': 'float',
    'f32': 'float',
    'f64': 'float',
    'bool': 'bool',
    'string': 'string',
    'char': 'char',
    'unit': 'unit',
    'void': 'unit',
}

# OCaml operator mappings
OCAML_OPERATOR_MAP = {
    '+': '+',
    '-': '-',
    '*': '*',
    '/': '/',
    '%': 'mod',
    '==': '=',
    '=': '=',
    '!=': '<>',
    '/=': '<>',
    '<>': '<>',
    '<': '<',
    '>': '>',
    '<=': '<=',
    '>=': '>=',
    '&&': '&&',
    '||': '||',
    '++': '@',
    '@': '@',
    ':': '::',
    '::': '::',
    '+.': '+.',
    '-.': '-.',
    '*.': '*.',
    '/.': '/.',
    '^': '^',  # String concatenation
}


class OCamlEmitter(FunctionalEmitterBase):
    """Emitter for OCaml code generation.
    
    Generates idiomatic OCaml code from Functional IR.
    """
    
    DIALECT = 'ocaml'
    FILE_EXTENSION = '.ml'
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.generate_mli = self.config.get('generate_mli', False)
    
    def emit_module(self, module: Module) -> str:
        """Emit complete OCaml module.
        
        Args:
            module: Module IR node
            
        Returns:
            Generated OCaml code
        """
        lines = []
        
        # Imports (open statements)
        for imp in module.imports:
            lines.append(self._emit_import(imp))
        if module.imports:
            lines.append("")
        
        # Type definitions
        for type_def in module.type_definitions:
            lines.append(self._emit_type_definition(type_def))
            lines.append("")
        
        # Functions
        for func in module.functions:
            lines.append(self._emit_function(func))
            lines.append("")
        
        return '\n'.join(lines)
    
    def emit_module_signature(self, module: Module) -> str:
        """Emit OCaml module signature (.mli file).
        
        Args:
            module: Module IR node
            
        Returns:
            Generated OCaml signature
        """
        lines = []
        
        # Type signatures
        for type_def in module.type_definitions:
            lines.append(self._emit_type_signature(type_def))
        
        # Function signatures
        for func in module.functions:
            if func.type_signature:
                lines.append(f"val {func.name} : {self._emit_type(func.type_signature)}")
        
        return '\n'.join(lines)
    
    def _emit_import(self, imp: Import) -> str:
        """Emit import statement."""
        if imp.qualified:
            alias = imp.alias or imp.module.split('.')[-1]
            return f"module {alias} = {imp.module}"
        return f"open {imp.module}"
    
    def _emit_type_definition(self, type_def: Union[DataType, RecordType, TypeAlias]) -> str:
        """Emit type definition."""
        if isinstance(type_def, DataType):
            return self._emit_variant_type(type_def)
        elif isinstance(type_def, RecordType):
            return self._emit_record_type(type_def)
        elif isinstance(type_def, TypeAlias):
            return self._emit_type_alias(type_def)
        return ""
    
    def _emit_type_signature(self, type_def: Union[DataType, RecordType, TypeAlias]) -> str:
        """Emit type signature for .mli file."""
        if isinstance(type_def, DataType):
            params = ', '.join(f"'{p.name}" for p in type_def.type_params)
            if params:
                return f"type ({params}) {type_def.name}"
            return f"type {type_def.name}"
        elif isinstance(type_def, RecordType):
            params = ', '.join(f"'{p.name}" for p in type_def.type_params)
            if params:
                return f"type ({params}) {type_def.name}"
            return f"type {type_def.name}"
        elif isinstance(type_def, TypeAlias):
            return self._emit_type_alias(type_def)
        return ""
    
    def _emit_variant_type(self, data: DataType) -> str:
        """Emit variant (sum) type definition."""
        params = ', '.join(f"'{p.name}" for p in data.type_params)
        if params:
            header = f"type ({params}) {data.name} ="
        else:
            header = f"type {data.name} ="
        
        constructors = []
        for con in data.constructors:
            con_str = self._emit_variant_constructor(con)
            constructors.append(f"  | {con_str}")
        
        return header + '\n' + '\n'.join(constructors)
    
    def _emit_variant_constructor(self, con: DataConstructor) -> str:
        """Emit variant constructor."""
        if con.fields:
            if con.field_names:
                # Record-style constructor
                fields = []
                for name, ftype in zip(con.field_names, con.fields):
                    fields.append(f"{name}: {self._emit_type(ftype)}")
                return f"{con.name} of {{ {'; '.join(fields)} }}"
            else:
                # Tuple-style constructor
                field_types = ' * '.join(self._emit_type(f) for f in con.fields)
                return f"{con.name} of {field_types}"
        return con.name
    
    def _emit_record_type(self, record: RecordType) -> str:
        """Emit record type definition."""
        params = ', '.join(f"'{p.name}" for p in record.type_params)
        if params:
            header = f"type ({params}) {record.name} = {{"
        else:
            header = f"type {record.name} = {{"
        
        fields = []
        for fld in record.fields:
            mutable = "mutable " if fld.mutable else ""
            fields.append(f"  {mutable}{fld.name}: {self._emit_type(fld.field_type)};")
        
        return header + '\n' + '\n'.join(fields) + '\n}'
    
    def _emit_type_alias(self, alias: TypeAlias) -> str:
        """Emit type alias."""
        params = ', '.join(f"'{p.name}" for p in alias.type_params)
        if params:
            return f"type ({params}) {alias.name} = {self._emit_type(alias.target)}"
        return f"type {alias.name} = {self._emit_type(alias.target)}"
    
    def _emit_type(self, type_expr: TypeExpr) -> str:
        """Emit type expression."""
        if type_expr is None:
            return "'a"  # Default type variable
            
        if isinstance(type_expr, TypeVar):
            return f"'{type_expr.name}"
        elif isinstance(type_expr, TypeCon):
            mapped = OCAML_TYPE_MAP.get(type_expr.name.lower(), type_expr.name)
            if type_expr.args:
                if len(type_expr.args) == 1:
                    return f"{self._emit_type(type_expr.args[0])} {mapped}"
                args = ', '.join(self._emit_type(a) for a in type_expr.args)
                return f"({args}) {mapped}"
            return mapped
        elif isinstance(type_expr, FunctionType):
            param = self._emit_type(type_expr.param_type)
            ret = self._emit_type(type_expr.return_type)
            return f"({param} -> {ret})"
        elif isinstance(type_expr, TupleType):
            elems = ' * '.join(self._emit_type(e) for e in type_expr.elements)
            return f"({elems})"
        elif isinstance(type_expr, ListType):
            return f"{self._emit_type(type_expr.element_type)} list"
        return "'a"
    
    def _emit_function(self, func: FunctionDef) -> str:
        """Emit function definition."""
        # Determine if recursive
        is_rec = len(func.clauses) > 1 or any(
            self._uses_recursion(clause.body, func.name)
            for clause in func.clauses
        )
        
        rec_kw = "rec " if is_rec else ""
        
        if len(func.clauses) == 1:
            # Single clause - direct definition
            clause = func.clauses[0]
            patterns = ' '.join(self._emit_pattern(p) for p in clause.patterns)
            body = self._emit_expr(clause.body)
            return f"let {rec_kw}{func.name} {patterns} = {body}"
        else:
            # Multiple clauses - use match
            param_count = len(func.clauses[0].patterns) if func.clauses else 0
            params = ' '.join(f"__arg{i}" for i in range(param_count))
            
            lines = [f"let {rec_kw}{func.name} {params} ="]
            
            if param_count == 1:
                match_expr = "__arg0"
            else:
                match_expr = '(' + ', '.join(f"__arg{i}" for i in range(param_count)) + ')'
            
            lines.append(f"  match {match_expr} with")
            
            for clause in func.clauses:
                if param_count == 1:
                    patterns = self._emit_pattern(clause.patterns[0])
                else:
                    patterns = '(' + ', '.join(self._emit_pattern(p) for p in clause.patterns) + ')'
                
                body = self._emit_expr(clause.body)
                
                if clause.guard:
                    guard = self._emit_expr(clause.guard)
                    lines.append(f"  | {patterns} when {guard} -> {body}")
                else:
                    lines.append(f"  | {patterns} -> {body}")
            
            return '\n'.join(lines)
    
    def _uses_recursion(self, expr: Expr, func_name: str) -> bool:
        """Check if expression uses recursion."""
        if expr is None:
            return False
            
        if isinstance(expr, VarExpr):
            return expr.name == func_name
        elif isinstance(expr, AppExpr):
            return self._uses_recursion(expr.func, func_name) or \
                   self._uses_recursion(expr.arg, func_name)
        elif isinstance(expr, LambdaExpr):
            return self._uses_recursion(expr.body, func_name)
        elif isinstance(expr, LetExpr):
            return self._uses_recursion(expr.value, func_name) or \
                   self._uses_recursion(expr.body, func_name)
        elif isinstance(expr, IfExpr):
            return self._uses_recursion(expr.condition, func_name) or \
                   self._uses_recursion(expr.then_branch, func_name) or \
                   self._uses_recursion(expr.else_branch, func_name)
        elif isinstance(expr, CaseExpr):
            if self._uses_recursion(expr.scrutinee, func_name):
                return True
            return any(self._uses_recursion(b.body, func_name) for b in expr.branches)
        elif isinstance(expr, BinaryOpExpr):
            return self._uses_recursion(expr.left, func_name) or \
                   self._uses_recursion(expr.right, func_name)
        elif isinstance(expr, ListExpr):
            return any(self._uses_recursion(e, func_name) for e in expr.elements)
        elif isinstance(expr, TupleExpr):
            return any(self._uses_recursion(e, func_name) for e in expr.elements)
        return False
    
    def _emit_pattern(self, pattern: Pattern) -> str:
        """Emit pattern."""
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
                if len(pattern.args) == 1:
                    return f"{pattern.constructor} {self._emit_pattern(pattern.args[0])}"
                args = ', '.join(self._emit_pattern(a) for a in pattern.args)
                return f"{pattern.constructor} ({args})"
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
            return f"({self._emit_pattern(pattern.pattern)} as {pattern.name})"
        elif isinstance(pattern, OrPattern):
            return f"({self._emit_pattern(pattern.left)} | {self._emit_pattern(pattern.right)})"
        return "_"
    
    def _emit_expr(self, expr: Expr) -> str:
        """Emit expression."""
        if expr is None:
            return "()"
            
        if isinstance(expr, LiteralExpr):
            return self._emit_literal_value(expr.value, expr.literal_type)
        elif isinstance(expr, VarExpr):
            return expr.name
        elif isinstance(expr, AppExpr):
            func = self._emit_expr(expr.func)
            if expr.arg:
                arg = self._emit_expr(expr.arg)
                return f"({func} {arg})"
            return func
        elif isinstance(expr, LambdaExpr):
            return f"(fun {expr.param} -> {self._emit_expr(expr.body)})"
        elif isinstance(expr, LetExpr):
            rec = "rec " if expr.is_recursive else ""
            return f"let {rec}{expr.name} = {self._emit_expr(expr.value)} in {self._emit_expr(expr.body)}"
        elif isinstance(expr, IfExpr):
            cond = self._emit_expr(expr.condition)
            then_b = self._emit_expr(expr.then_branch)
            else_b = self._emit_expr(expr.else_branch)
            return f"if {cond} then {then_b} else {else_b}"
        elif isinstance(expr, CaseExpr):
            return self._emit_match(expr)
        elif isinstance(expr, ListExpr):
            elems = '; '.join(self._emit_expr(e) for e in expr.elements)
            return f"[{elems}]"
        elif isinstance(expr, TupleExpr):
            elems = ', '.join(self._emit_expr(e) for e in expr.elements)
            return f"({elems})"
        elif isinstance(expr, BinaryOpExpr):
            left = self._emit_expr(expr.left)
            right = self._emit_expr(expr.right)
            op = OCAML_OPERATOR_MAP.get(expr.op, expr.op)
            return f"({left} {op} {right})"
        elif isinstance(expr, UnaryOpExpr):
            operand = self._emit_expr(expr.operand)
            if expr.op == 'not':
                return f"(not {operand})"
            return f"({expr.op}{operand})"
        return "()"
    
    def _emit_match(self, case: CaseExpr) -> str:
        """Emit match expression."""
        scrutinee = self._emit_expr(case.scrutinee)
        lines = [f"match {scrutinee} with"]
        for branch in case.branches:
            pattern = self._emit_pattern(branch.pattern)
            body = self._emit_expr(branch.body)
            if branch.guard:
                guard = self._emit_expr(branch.guard)
                lines.append(f"  | {pattern} when {guard} -> {body}")
            else:
                lines.append(f"  | {pattern} -> {body}")
        return '\n'.join(lines)
    
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
                s += '.'
            return s
        return str(value)
    
    def emit_functor(self, name: str, param_name: str, param_sig: str,
                     body: Module) -> str:
        """Emit OCaml functor.
        
        Args:
            name: Functor name
            param_name: Parameter module name
            param_sig: Parameter signature type
            body: Module body
            
        Returns:
            Functor definition string
        """
        lines = [f"module {name} ({param_name} : {param_sig}) = struct"]
        
        for type_def in body.type_definitions:
            type_str = self._emit_type_definition(type_def)
            for line in type_str.split('\n'):
                lines.append(f"  {line}")
        
        for func in body.functions:
            func_str = self._emit_function(func)
            for line in func_str.split('\n'):
                lines.append(f"  {line}")
        
        lines.append("end")
        return '\n'.join(lines)
    
    def emit_ref_operations(self, var_name: str, value: Expr) -> str:
        """Emit reference creation.
        
        Args:
            var_name: Variable name
            value: Initial value expression
            
        Returns:
            Reference creation code
        """
        return f"let {var_name} = ref {self._emit_expr(value)}"
    
    def emit_assignment(self, var_name: str, value: Expr) -> str:
        """Emit reference assignment.
        
        Args:
            var_name: Reference variable name
            value: Value to assign
            
        Returns:
            Assignment code
        """
        return f"{var_name} := {self._emit_expr(value)}"
    
    def emit_deref(self, var_name: str) -> str:
        """Emit dereference.
        
        Args:
            var_name: Reference variable name
            
        Returns:
            Dereference expression
        """
        return f"!{var_name}"


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='STUNIR OCaml Emitter')
    parser.add_argument('ir_file', help='Input IR JSON file')
    parser.add_argument('--output', '-o', help='Output file')
    parser.add_argument('--mli', action='store_true', help='Generate .mli signature')
    args = parser.parse_args()
    
    # Load IR
    with open(args.ir_file, 'r') as f:
        ir_data = json.load(f)
    
    # Create emitter and generate code
    emitter = OCamlEmitter()
    
    # For now, just demonstrate the emitter
    print(f"Loaded IR from {args.ir_file}")
    print(f"Would generate OCaml code to {args.output or 'stdout'}")
    if args.mli:
        print("Would also generate .mli signature file")


if __name__ == '__main__':
    main()
