#!/usr/bin/env python3
"""STUNIR Haskell Emitter - Generate Haskell code from Functional IR.

This emitter generates idiomatic Haskell code including:
- Data type declarations
- Function definitions with pattern matching
- Type signatures and type classes
- Monadic do notation
- List comprehensions

Usage:
    from targets.functional.haskell_emitter import HaskellEmitter
    from ir.functional import Module
    
    emitter = HaskellEmitter()
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
    DoExpr, DoBindStatement, DoLetStatement, DoExprStatement,
    # Patterns
    Pattern, WildcardPattern, VarPattern, LiteralPattern,
    ConstructorPattern, TuplePattern, ListPattern, AsPattern,
    # ADTs
    TypeParameter, DataConstructor, DataType, TypeAlias, NewType,
    MethodSignature, TypeClass, TypeClassInstance,
    Import, FunctionClause, FunctionDef, Module,
)


# Haskell-specific type mappings
HASKELL_TYPE_MAP = {
    'int': 'Int',
    'i8': 'Int8',
    'i16': 'Int16',
    'i32': 'Int32',
    'i64': 'Int64',
    'float': 'Float',
    'f32': 'Float',
    'f64': 'Double',
    'bool': 'Bool',
    'string': 'String',
    'char': 'Char',
    'unit': '()',
    'void': '()',
}

# Haskell operator mappings
HASKELL_OPERATOR_MAP = {
    '+': '+',
    '-': '-',
    '*': '*',
    '/': 'div',
    '//': 'div',
    '%': 'mod',
    '==': '==',
    '!=': '/=',
    '/=': '/=',
    '<': '<',
    '>': '>',
    '<=': '<=',
    '>=': '>=',
    '&&': '&&',
    '||': '||',
    '++': '++',
    ':': ':',
    '.': '.',
    '$': '$',
    '<$>': '<$>',
    '<*>': '<*>',
    '>>=': '>>=',
    '>>': '>>',
}


class HaskellEmitter(FunctionalEmitterBase):
    """Emitter for Haskell code generation.
    
    Generates idiomatic Haskell code from Functional IR.
    """
    
    DIALECT = 'haskell'
    FILE_EXTENSION = '.hs'
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.use_qualified_imports = self.config.get('qualified_imports', False)
    
    def emit_module(self, module: Module) -> str:
        """Emit complete Haskell module.
        
        Args:
            module: Module IR node
            
        Returns:
            Generated Haskell code
        """
        lines = []
        
        # Module header
        if module.exports:
            exports = ', '.join(module.exports)
            lines.append(f"module {module.name} ({exports}) where")
        else:
            lines.append(f"module {module.name} where")
        lines.append("")
        
        # Imports
        for imp in module.imports:
            lines.append(self._emit_import(imp))
        if module.imports:
            lines.append("")
        
        # Type definitions
        for type_def in module.type_definitions:
            lines.append(self._emit_type_definition(type_def))
            lines.append("")
        
        # Type classes
        for type_class in module.type_classes:
            lines.append(self._emit_type_class(type_class))
            lines.append("")
        
        # Instances
        for instance in module.instances:
            lines.append(self._emit_instance(instance))
            lines.append("")
        
        # Functions
        for func in module.functions:
            lines.append(self._emit_function(func))
            lines.append("")
        
        return '\n'.join(lines)
    
    def _emit_import(self, imp: Import) -> str:
        """Emit import statement."""
        parts = ["import"]
        if imp.qualified:
            parts.append("qualified")
        parts.append(imp.module)
        if imp.alias:
            parts.append(f"as {imp.alias}")
        if imp.items:
            items = ', '.join(imp.items)
            parts.append(f"({items})")
        if imp.hiding:
            hidden = ', '.join(imp.hiding)
            parts.append(f"hiding ({hidden})")
        return ' '.join(parts)
    
    def _emit_type_definition(self, type_def: Union[DataType, TypeAlias, NewType]) -> str:
        """Emit type definition."""
        if isinstance(type_def, DataType):
            return self._emit_data_type(type_def)
        elif isinstance(type_def, NewType):
            return self._emit_newtype(type_def)
        elif isinstance(type_def, TypeAlias):
            return self._emit_type_alias(type_def)
        return ""
    
    def _emit_data_type(self, data: DataType) -> str:
        """Emit data type definition."""
        params = ' '.join(p.name for p in data.type_params)
        header = f"data {data.name}"
        if params:
            header += f" {params}"
        
        constructors = []
        for i, con in enumerate(data.constructors):
            prefix = "=" if i == 0 else "|"
            con_str = self._emit_constructor(con)
            constructors.append(f"  {prefix} {con_str}")
        
        result = header + "\n" + '\n'.join(constructors)
        
        if data.deriving:
            deriving = ', '.join(data.deriving)
            result += f"\n  deriving ({deriving})"
        
        return result
    
    def _emit_constructor(self, con: DataConstructor) -> str:
        """Emit data constructor."""
        if con.field_names:
            # Record syntax
            fields = []
            for name, ftype in zip(con.field_names, con.fields):
                fields.append(f"{name} :: {self._emit_type(ftype)}")
            return f"{con.name} {{ {', '.join(fields)} }}"
        else:
            # Regular syntax
            if con.fields:
                field_types = ' '.join(self._emit_type(f) for f in con.fields)
                return f"{con.name} {field_types}"
            return con.name
    
    def _emit_newtype(self, newtype: NewType) -> str:
        """Emit newtype definition."""
        params = ' '.join(p.name for p in newtype.type_params)
        con = self._emit_constructor(newtype.constructor)
        header = f"newtype {newtype.name}"
        if params:
            header += f" {params}"
        result = f"{header} = {con}"
        if newtype.deriving:
            deriving = ', '.join(newtype.deriving)
            result += f" deriving ({deriving})"
        return result
    
    def _emit_type_alias(self, alias: TypeAlias) -> str:
        """Emit type alias."""
        params = ' '.join(p.name for p in alias.type_params)
        header = f"type {alias.name}"
        if params:
            header += f" {params}"
        return f"{header} = {self._emit_type(alias.target)}"
    
    def _emit_type(self, type_expr: TypeExpr) -> str:
        """Emit type expression."""
        if type_expr is None:
            return "a"  # Default type variable
            
        if isinstance(type_expr, TypeVar):
            return type_expr.name
        elif isinstance(type_expr, TypeCon):
            mapped = HASKELL_TYPE_MAP.get(type_expr.name.lower(), type_expr.name)
            if type_expr.args:
                args = ' '.join(self._emit_type(a) for a in type_expr.args)
                return f"({mapped} {args})"
            return mapped
        elif isinstance(type_expr, FunctionType):
            param = self._emit_type(type_expr.param_type)
            ret = self._emit_type(type_expr.return_type)
            return f"({param} -> {ret})"
        elif isinstance(type_expr, TupleType):
            elems = ', '.join(self._emit_type(e) for e in type_expr.elements)
            return f"({elems})"
        elif isinstance(type_expr, ListType):
            return f"[{self._emit_type(type_expr.element_type)}]"
        return "a"
    
    def _emit_type_class(self, tc: TypeClass) -> str:
        """Emit type class definition."""
        params = ' '.join(p.name for p in tc.type_params)
        header = "class"
        if tc.superclasses:
            constraints = ', '.join(f"{sc} {params}" for sc in tc.superclasses)
            header += f" ({constraints}) =>"
        header += f" {tc.name} {params} where"
        
        methods = []
        for method in tc.methods:
            methods.append(f"  {method.name} :: {self._emit_type(method.type_signature)}")
        
        return header + '\n' + '\n'.join(methods)
    
    def _emit_instance(self, inst: TypeClassInstance) -> str:
        """Emit type class instance."""
        args = ' '.join(self._emit_type(a) for a in inst.type_args)
        header = f"instance {inst.class_name} {args} where"
        
        impls = []
        for name, expr in inst.implementations.items():
            impls.append(f"  {name} = {self._emit_expr(expr)}")
        
        return header + '\n' + '\n'.join(impls)
    
    def _emit_function(self, func: FunctionDef) -> str:
        """Emit function definition."""
        lines = []
        
        # Type signature
        if func.type_signature:
            lines.append(f"{func.name} :: {self._emit_type(func.type_signature)}")
        
        # Function clauses
        for clause in func.clauses:
            patterns = ' '.join(self._emit_pattern(p) for p in clause.patterns)
            line = f"{func.name}"
            if patterns:
                line += f" {patterns}"
            
            if clause.guard:
                line += f"\n  | {self._emit_expr(clause.guard)}"
            
            line += f" = {self._emit_expr(clause.body)}"
            lines.append(line)
        
        # Where clause
        if func.where_bindings:
            lines.append("  where")
            for binding in func.where_bindings:
                lines.append(f"    {binding.name} = {self._emit_expr(binding.value)}")
        
        return '\n'.join(lines)
    
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
                args = ' '.join(self._emit_pattern(a) for a in pattern.args)
                return f"({pattern.constructor} {args})"
            return pattern.constructor
        elif isinstance(pattern, TuplePattern):
            elems = ', '.join(self._emit_pattern(e) for e in pattern.elements)
            return f"({elems})"
        elif isinstance(pattern, ListPattern):
            if pattern.rest:
                if pattern.elements:
                    elems = ':'.join(self._emit_pattern(e) for e in pattern.elements)
                    return f"({elems}:{self._emit_pattern(pattern.rest)})"
                return self._emit_pattern(pattern.rest)
            elems = ', '.join(self._emit_pattern(e) for e in pattern.elements)
            return f"[{elems}]"
        elif isinstance(pattern, AsPattern):
            return f"{pattern.name}@{self._emit_pattern(pattern.pattern)}"
        return "_"
    
    def _emit_expr(self, expr: Expr) -> str:
        """Emit expression."""
        if expr is None:
            return "_"
            
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
            return f"(\\{expr.param} -> {self._emit_expr(expr.body)})"
        elif isinstance(expr, LetExpr):
            rec = "rec " if expr.is_recursive else ""
            return f"let {rec}{expr.name} = {self._emit_expr(expr.value)} in {self._emit_expr(expr.body)}"
        elif isinstance(expr, IfExpr):
            cond = self._emit_expr(expr.condition)
            then_b = self._emit_expr(expr.then_branch)
            else_b = self._emit_expr(expr.else_branch)
            return f"if {cond} then {then_b} else {else_b}"
        elif isinstance(expr, CaseExpr):
            return self._emit_case(expr)
        elif isinstance(expr, ListExpr):
            if expr.is_comprehension:
                return self._emit_list_comprehension(expr)
            elems = ', '.join(self._emit_expr(e) for e in expr.elements)
            return f"[{elems}]"
        elif isinstance(expr, TupleExpr):
            elems = ', '.join(self._emit_expr(e) for e in expr.elements)
            return f"({elems})"
        elif isinstance(expr, BinaryOpExpr):
            left = self._emit_expr(expr.left)
            right = self._emit_expr(expr.right)
            op = HASKELL_OPERATOR_MAP.get(expr.op, expr.op)
            return f"({left} {op} {right})"
        elif isinstance(expr, UnaryOpExpr):
            operand = self._emit_expr(expr.operand)
            if expr.op == 'not':
                return f"(not {operand})"
            return f"({expr.op}{operand})"
        elif isinstance(expr, DoExpr):
            return self._emit_do(expr)
        return "_"
    
    def _emit_case(self, case: CaseExpr) -> str:
        """Emit case expression."""
        scrutinee = self._emit_expr(case.scrutinee)
        lines = [f"case {scrutinee} of"]
        for branch in case.branches:
            pattern = self._emit_pattern(branch.pattern)
            body = self._emit_expr(branch.body)
            if branch.guard:
                guard = self._emit_expr(branch.guard)
                lines.append(f"  {pattern} | {guard} -> {body}")
            else:
                lines.append(f"  {pattern} -> {body}")
        return '\n'.join(lines)
    
    def _emit_do(self, do_expr: DoExpr) -> str:
        """Emit do notation."""
        lines = ["do"]
        for stmt in do_expr.statements:
            if isinstance(stmt, DoBindStatement):
                pattern = self._emit_pattern(stmt.pattern)
                action = self._emit_expr(stmt.action)
                lines.append(f"  {pattern} <- {action}")
            elif isinstance(stmt, DoLetStatement):
                value = self._emit_expr(stmt.value)
                lines.append(f"  let {stmt.name} = {value}")
            elif isinstance(stmt, DoExprStatement):
                lines.append(f"  {self._emit_expr(stmt.expr)}")
        return '\n'.join(lines)
    
    def _emit_list_comprehension(self, list_expr: ListExpr) -> str:
        """Emit list comprehension."""
        elem = self._emit_expr(list_expr.elements[0]) if list_expr.elements else "_"
        generators = []
        for gen in list_expr.generators:
            pattern = self._emit_pattern(gen.pattern)
            source = self._emit_expr(gen.source)
            generators.append(f"{pattern} <- {source}")
            for cond in gen.conditions:
                generators.append(self._emit_expr(cond))
        return f"[{elem} | {', '.join(generators)}]"
    
    def _emit_literal_value(self, value: Any, literal_type: str) -> str:
        """Emit literal value."""
        if literal_type == 'bool':
            return 'True' if value else 'False'
        elif literal_type == 'char':
            return f"'{value}'"
        elif literal_type == 'string':
            escaped = str(value).replace('\\', '\\\\').replace('"', '\\"')
            return f'"{escaped}"'
        elif literal_type == 'float':
            return str(float(value))
        return str(value)


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='STUNIR Haskell Emitter')
    parser.add_argument('ir_file', help='Input IR JSON file')
    parser.add_argument('--output', '-o', help='Output file')
    args = parser.parse_args()
    
    # Load IR
    with open(args.ir_file, 'r') as f:
        ir_data = json.load(f)
    
    # Create emitter and generate code
    emitter = HaskellEmitter()
    
    # For now, just demonstrate the emitter
    print(f"Loaded IR from {args.ir_file}")
    print(f"Would generate Haskell code to {args.output or 'stdout'}")


if __name__ == '__main__':
    main()
