#!/usr/bin/env python3
"""STUNIR Scheme (R7RS) Emitter.

Generates R7RS-compliant Scheme code from STUNIR IR.
Supports define-library, type comments, and tail-call optimization hints.

Part of Phase 5A: Core Lisp Implementation.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..base import LispEmitterBase, LispEmitterConfig, EmitterResult, compute_sha256, canonical_json
from .types import SchemeTypeMapper, SCHEME_TYPES


@dataclass
class SchemeEmitterConfig(LispEmitterConfig):
    """Scheme specific configuration."""
    module_prefix: str = "stunir"
    r7rs_library: bool = True  # Use R7RS library syntax
    implementation: str = "chibi"  # chibi, guile, chicken, etc.
    tail_call_optimize: bool = True
    emit_type_comments: bool = True


class SchemeEmitter(LispEmitterBase):
    """Emitter for R7RS Scheme code.
    
    Generates valid Scheme code with:
    - R7RS define-library syntax
    - Type signature comments
    - Named let for loops
    - Proper #t/#f booleans
    """
    
    DIALECT = "scheme"
    FILE_EXTENSION = ".scm"
    
    # Scheme-specific operators
    BINARY_OPS = {
        '+': '+',
        '-': '-',
        '*': '*',
        '/': '/',
        '%': 'modulo',
        '==': '=',
        '!=': 'not-equal?',  # Will need helper
        '<': '<',
        '>': '>',
        '<=': '<=',
        '>=': '>=',
        'and': 'and',
        'or': 'or',
        'eq': 'eq?',
        'equal': 'equal?',
    }
    
    def __init__(self, config: SchemeEmitterConfig):
        """Initialize Scheme emitter.
        
        Args:
            config: Scheme emitter configuration.
        """
        super().__init__(config)
        self.config: SchemeEmitterConfig = config
        self.type_mapper = SchemeTypeMapper(config.emit_type_comments)
        self._exports: Set[str] = set()
    
    def emit(self, ir: Dict[str, Any]) -> EmitterResult:
        """Emit Scheme code from IR.
        
        Args:
            ir: STUNIR IR dictionary.
            
        Returns:
            EmitterResult with generated code.
        """
        module_name = ir.get('module', 'unnamed')
        
        # Collect exports
        self._exports = set()
        for func in ir.get('functions', []):
            if func.get('exported', True):
                self._exports.add(self._lisp_name(func.get('name', '')))
        for export in ir.get('exports', []):
            self._exports.add(self._lisp_name(export))
        
        # Build code
        if self.config.r7rs_library:
            code = self._emit_r7rs_library(ir)
        else:
            code = self._emit_simple_module(ir)
        
        # Write file
        main_file = f"{self._lisp_name(module_name)}.scm"
        self._write_file(main_file, code)
        
        return EmitterResult(
            code=code,
            files={main_file: code},
            manifest=self._generate_manifest(ir)
        )
    
    def _emit_r7rs_library(self, ir: Dict[str, Any]) -> str:
        """Emit R7RS library definition."""
        module_name = ir.get('module', 'unnamed')
        library_name = f"({self.config.module_prefix} {self._lisp_name(module_name)})"
        
        lines = [
            self._emit_header(module_name),
            f"(define-library {library_name}",
            "  (import (scheme base)",
            "          (scheme write))",
        ]
        
        # Exports
        if self._exports:
            export_list = ' '.join(sorted(self._exports))
            lines.append(f"  (export {export_list})")
        
        lines.append("")
        lines.append("  (begin")
        
        # Functions
        for func in ir.get('functions', []):
            func_code = self._emit_function(func)
            # Indent for library body
            indented = '\n'.join('    ' + line if line else '' for line in func_code.split('\n'))
            lines.append(indented)
            lines.append("")
        
        lines.append("))")  # Close begin and define-library
        
        return '\n'.join(lines)
    
    def _emit_simple_module(self, ir: Dict[str, Any]) -> str:
        """Emit simple Scheme module without library wrapper."""
        module_name = ir.get('module', 'unnamed')
        
        lines = [self._emit_header(module_name)]
        
        # Functions
        for func in ir.get('functions', []):
            lines.append(self._emit_function(func))
            lines.append("")
        
        return '\n'.join(lines)
    
    def _emit_function(self, func: Dict[str, Any]) -> str:
        """Emit a Scheme function definition."""
        name = self._lisp_name(func.get('name', 'unnamed'))
        params = func.get('params', [])
        body = func.get('body', [])
        docstring = func.get('docstring', '')
        return_type = func.get('return_type', 'any')
        
        # Build parameter list
        param_names = ' '.join(self._lisp_name(p.get('name', '_')) for p in params)
        
        lines = []
        
        # Type comment
        if self.config.emit_type_comments and params:
            param_types = [p.get('type', 'any') for p in params]
            type_comment = self.type_mapper.emit_type_comment(name, param_types, return_type)
            if type_comment:
                lines.append(type_comment)
        
        # Function definition
        if param_names:
            lines.append(f"(define ({name} {param_names})")
        else:
            lines.append(f"(define ({name})")
        
        # Body - last expression is implicit return
        if body:
            for stmt in body:
                stmt_str = self._emit_statement(stmt)
                lines.append(f"  {stmt_str}")
        else:
            lines.append("  (void)")
        
        lines.append(")")
        
        return '\n'.join(lines)
    
    def _emit_literal(self, value: Any) -> str:
        """Emit a Scheme literal."""
        if value is None:
            return "'()"
        if isinstance(value, bool):
            return "#t" if value else "#f"
        if isinstance(value, str):
            escaped = value.replace('\\', '\\\\').replace('"', '\\"')
            return f'"{escaped}"'
        return str(value)
    
    def _emit_lambda(self, data: Dict[str, Any]) -> str:
        """Emit a Scheme lambda."""
        params = data.get('params', [])
        body = data.get('body', [])
        
        param_names = ' '.join(self._lisp_name(p.get('name', '_')) for p in params)
        
        if body:
            body_str = ' '.join(self._emit_statement(stmt) for stmt in body)
        else:
            body_str = "(void)"
        
        return f"(lambda ({param_names}) {body_str})"
    
    def _emit_var_decl(self, stmt: Dict[str, Any]) -> str:
        """Emit variable declaration."""
        name = self._lisp_name(stmt.get('name', '_'))
        value = stmt.get('value', stmt.get('init'))
        
        if value is not None:
            val_str = self._emit_expression(value)
            return f"(define {name} {val_str})"
        return f"(define {name} #f)"
    
    def _emit_assignment(self, stmt: Dict[str, Any]) -> str:
        """Emit assignment statement."""
        name = self._lisp_name(stmt.get('name', stmt.get('target', '_')))
        value = self._emit_expression(stmt.get('value', {}))
        return f"(set! {name} {value})"
    
    def _emit_if_stmt(self, stmt: Dict[str, Any]) -> str:
        """Emit if statement."""
        cond = self._emit_expression(stmt.get('condition', stmt.get('cond', {})))
        then_body = stmt.get('then', stmt.get('consequent', []))
        else_body = stmt.get('else', stmt.get('alternate'))
        
        if isinstance(then_body, list):
            if len(then_body) == 1:
                then_str = self._emit_statement(then_body[0])
            elif then_body:
                then_str = f"(begin {' '.join(self._emit_statement(s) for s in then_body)})"
            else:
                then_str = "(void)"
        else:
            then_str = self._emit_statement(then_body)
        
        if else_body:
            if isinstance(else_body, list):
                if len(else_body) == 1:
                    else_str = self._emit_statement(else_body[0])
                else:
                    else_str = f"(begin {' '.join(self._emit_statement(s) for s in else_body)})"
            else:
                else_str = self._emit_statement(else_body)
            return f"(if {cond}\n    {then_str}\n    {else_str})"
        
        return f"(when {cond}\n    {then_str})"
    
    def _emit_while(self, stmt: Dict[str, Any]) -> str:
        """Emit while loop as named let."""
        cond = self._emit_expression(stmt.get('condition', stmt.get('cond', {})))
        body = stmt.get('body', [])
        
        body_strs = [self._emit_statement(s) for s in body] if body else ['(void)']
        body_str = '\n        '.join(body_strs)
        
        return f"""(let loop ()
    (when {cond}
        {body_str}
        (loop)))"""
    
    def _emit_for(self, stmt: Dict[str, Any]) -> str:
        """Emit for loop as do or named let."""
        var = self._lisp_name(stmt.get('var', 'i'))
        start = self._emit_expression(stmt.get('start', {'kind': 'literal', 'value': 0}))
        end = self._emit_expression(stmt.get('end', {'kind': 'literal', 'value': 10}))
        body = stmt.get('body', [])
        
        body_strs = [self._emit_statement(s) for s in body] if body else ['(void)']
        body_str = ' '.join(body_strs)
        
        return f"""(do (({var} {start} (+ {var} 1)))
    ((>= {var} {end}))
    {body_str})"""
    
    def _emit_block(self, stmt: Dict[str, Any]) -> str:
        """Emit a block of statements."""
        body = stmt.get('body', stmt.get('statements', []))
        if len(body) == 0:
            return "(void)"
        if len(body) == 1:
            return self._emit_statement(body[0])
        stmts = ' '.join(self._emit_statement(s) for s in body)
        return f"(begin {stmts})"
    
    def _emit_binary_op(self, data: Dict[str, Any]) -> str:
        """Emit a binary operation."""
        op = data.get('op', '+')
        left = self._emit_expression(data.get('left', {}))
        right = self._emit_expression(data.get('right', {}))
        
        # Handle != specially
        if op == '!=':
            return f"(not (= {left} {right}))"
        
        scheme_op = self.BINARY_OPS.get(op, op)
        return f"({scheme_op} {left} {right})"
    
    def _map_type(self, ir_type: str) -> str:
        """Map IR type to Scheme type."""
        return self.type_mapper.map_type(ir_type)
