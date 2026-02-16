#!/usr/bin/env python3
"""STUNIR Clojure Emitter.

Generates idiomatic Clojure code from STUNIR IR.
Supports type hints, clojure.spec, and namespace management.

Part of Phase 5A: Core Lisp Implementation.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..base import LispEmitterBase, LispEmitterConfig, EmitterResult, compute_sha256, canonical_json
from .types import ClojureTypeMapper, CLOJURE_TYPES, CLOJURE_TYPE_HINTS


@dataclass
class ClojureEmitterConfig(LispEmitterConfig):
    """Clojure specific configuration."""
    module_prefix: str = "stunir"
    use_spec: bool = True  # Generate clojure.spec definitions
    use_type_hints: bool = True  # Add ^Type hints
    target_version: str = "1.11"  # Clojure version
    emit_docstrings: bool = True


class ClojureEmitter(LispEmitterBase):
    """Emitter for Clojure code.
    
    Generates idiomatic Clojure with:
    - Namespace definitions (ns)
    - Type hints for JVM performance
    - clojure.spec definitions
    - Proper true/false/nil
    - Vector binding forms [x 1]
    """
    
    DIALECT = "clojure"
    FILE_EXTENSION = ".clj"
    
    # Clojure-specific operators
    BINARY_OPS = {
        '+': '+',
        '-': '-',
        '*': '*',
        '/': '/',
        '%': 'mod',
        '==': '=',
        '!=': 'not=',
        '<': '<',
        '>': '>',
        '<=': '<=',
        '>=': '>=',
        'and': 'and',
        'or': 'or',
        '&': 'bit-and',
        '|': 'bit-or',
        '^': 'bit-xor',
    }
    
    def __init__(self, config: ClojureEmitterConfig):
        """Initialize Clojure emitter.
        
        Args:
            config: Clojure emitter configuration.
        """
        super().__init__(config)
        self.config: ClojureEmitterConfig = config
        self.type_mapper = ClojureTypeMapper(config.use_type_hints, config.use_spec)
        self._exports: Set[str] = set()
    
    def emit(self, ir: Dict[str, Any]) -> EmitterResult:
        """Emit Clojure code from IR.
        
        Args:
            ir: STUNIR IR dictionary.
            
        Returns:
            EmitterResult with generated code.
        """
        module_name = ir.get('module', 'unnamed')
        ns_name = f"{self.config.module_prefix}.{self._lisp_name(module_name).replace('-', '.')}"
        
        # Build code sections
        sections = []
        
        # Header
        sections.append(self._emit_header(module_name))
        
        # Namespace
        sections.append(self._emit_namespace(ns_name))
        
        # Specs (if enabled)
        if self.config.use_spec:
            for func in ir.get('functions', []):
                spec = self._emit_spec(func)
                if spec:
                    sections.append(spec)
        
        # Functions
        for func in ir.get('functions', []):
            sections.append(self._emit_function(func))
        
        code = "\n\n".join(filter(None, sections))
        
        # Write file
        main_file = f"{self._lisp_name(module_name).replace('-', '_')}.clj"
        self._write_file(main_file, code)
        
        return EmitterResult(
            code=code,
            files={main_file: code},
            manifest=self._generate_manifest(ir)
        )
    
    def _emit_namespace(self, ns_name: str) -> str:
        """Emit namespace definition."""
        lines = [f"(ns {ns_name}"]
        
        if self.config.use_spec:
            lines.append("  (:require [clojure.spec.alpha :as s]))")
        else:
            lines.append(")")
        
        return "\n".join(lines)
    
    def _emit_spec(self, func: Dict[str, Any]) -> str:
        """Emit clojure.spec fdef for function."""
        if not self.config.use_spec:
            return ""
        
        name = self._lisp_name(func.get('name', 'unnamed'))
        params = func.get('params', [])
        return_type = func.get('return_type', 'any')
        
        param_types = [p.get('type', 'any') for p in params]
        
        return self.type_mapper.emit_spec_fdef(name, param_types, return_type)
    
    def _emit_function(self, func: Dict[str, Any]) -> str:
        """Emit a Clojure function definition."""
        name = self._lisp_name(func.get('name', 'unnamed'))
        params = func.get('params', [])
        body = func.get('body', [])
        docstring = func.get('docstring', func.get('doc', ''))
        return_type = func.get('return_type', 'any')
        
        lines = []
        
        # Function with potential return type hint
        if self.config.use_type_hints and return_type:
            ret_hint = self.type_mapper.get_type_hint(return_type)
            if ret_hint:
                lines.append(f"(defn {ret_hint} {name}")
            else:
                lines.append(f"(defn {name}")
        else:
            lines.append(f"(defn {name}")
        
        # Docstring
        if docstring and self.config.emit_docstrings:
            lines.append(f'  "{docstring}"')
        
        # Parameters with type hints (vector syntax)
        param_strs = []
        for p in params:
            pname = self._lisp_name(p.get('name', '_'))
            ptype = p.get('type', 'any')
            typed_param = self.type_mapper.emit_typed_param(pname, ptype)
            param_strs.append(typed_param)
        
        lines.append(f"  [{' '.join(param_strs)}]")
        
        # Body
        if body:
            for stmt in body:
                stmt_str = self._emit_statement(stmt)
                lines.append(f"  {stmt_str}")
        else:
            lines.append("  nil")
        
        lines.append(")")
        
        return "\n".join(lines)
    
    def _emit_literal(self, value: Any) -> str:
        """Emit a Clojure literal."""
        if value is None:
            return "nil"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            escaped = value.replace('\\', '\\\\').replace('"', '\\"')
            return f'"{escaped}"'
        return str(value)
    
    def _emit_lambda(self, data: Dict[str, Any]) -> str:
        """Emit a Clojure anonymous function."""
        params = data.get('params', [])
        body = data.get('body', [])
        
        # Use vector syntax for params
        param_names = ' '.join(self._lisp_name(p.get('name', '_')) for p in params)
        
        if body:
            body_str = ' '.join(self._emit_statement(stmt) for stmt in body)
        else:
            body_str = "nil"
        
        return f"(fn [{param_names}] {body_str})"
    
    def _emit_var_decl(self, stmt: Dict[str, Any]) -> str:
        """Emit variable declaration."""
        name = self._lisp_name(stmt.get('name', '_'))
        value = stmt.get('value', stmt.get('init'))
        
        if value is not None:
            val_str = self._emit_expression(value)
            # Top-level def or local let binding
            return f"(def {name} {val_str})"
        return f"(def {name} nil)"
    
    def _emit_let_binding(self, bindings: List[Dict], body: List) -> str:
        """Emit a let binding with vector syntax."""
        binding_strs = []
        for b in bindings:
            name = self._lisp_name(b.get('name', '_'))
            value = self._emit_expression(b.get('value', {'kind': 'literal', 'value': None}))
            binding_strs.extend([name, value])
        
        binding_vec = ' '.join(binding_strs)
        body_str = ' '.join(self._emit_statement(s) for s in body) if body else 'nil'
        
        return f"(let [{binding_vec}] {body_str})"
    
    def _emit_assignment(self, stmt: Dict[str, Any]) -> str:
        """Emit assignment statement.
        
        Note: Clojure is immutable, so we use atom + reset! for mutable state.
        """
        name = self._lisp_name(stmt.get('name', stmt.get('target', '_')))
        value = self._emit_expression(stmt.get('value', {}))
        # For simplicity, emit as def (rebinding) or comment about atoms
        return f";; Note: Clojure prefers immutability\n(def {name} {value})"
    
    def _emit_if_stmt(self, stmt: Dict[str, Any]) -> str:
        """Emit if statement."""
        cond = self._emit_expression(stmt.get('condition', stmt.get('cond', {})))
        then_body = stmt.get('then', stmt.get('consequent', []))
        else_body = stmt.get('else', stmt.get('alternate'))
        
        if isinstance(then_body, list):
            if len(then_body) == 1:
                then_str = self._emit_statement(then_body[0])
            elif then_body:
                then_str = f"(do {' '.join(self._emit_statement(s) for s in then_body)})"
            else:
                then_str = "nil"
        else:
            then_str = self._emit_statement(then_body)
        
        if else_body:
            if isinstance(else_body, list):
                if len(else_body) == 1:
                    else_str = self._emit_statement(else_body[0])
                else:
                    else_str = f"(do {' '.join(self._emit_statement(s) for s in else_body)})"
            else:
                else_str = self._emit_statement(else_body)
            return f"(if {cond}\n    {then_str}\n    {else_str})"
        
        return f"(when {cond}\n    {then_str})"
    
    def _emit_while(self, stmt: Dict[str, Any]) -> str:
        """Emit while loop as loop/recur."""
        cond = self._emit_expression(stmt.get('condition', stmt.get('cond', {})))
        body = stmt.get('body', [])
        
        body_strs = [self._emit_statement(s) for s in body] if body else ['nil']
        body_str = '\n        '.join(body_strs)
        
        return f"""(loop []
    (when {cond}
        {body_str}
        (recur)))"""
    
    def _emit_for(self, stmt: Dict[str, Any]) -> str:
        """Emit for loop as doseq or loop/recur."""
        var = self._lisp_name(stmt.get('var', 'i'))
        start = self._emit_expression(stmt.get('start', {'kind': 'literal', 'value': 0}))
        end = self._emit_expression(stmt.get('end', {'kind': 'literal', 'value': 10}))
        body = stmt.get('body', [])
        
        body_strs = [self._emit_statement(s) for s in body] if body else ['nil']
        body_str = ' '.join(body_strs)
        
        return f"(doseq [{var} (range {start} {end})]\n    {body_str})"
    
    def _emit_block(self, stmt: Dict[str, Any]) -> str:
        """Emit a block of statements."""
        body = stmt.get('body', stmt.get('statements', []))
        if len(body) == 0:
            return "nil"
        if len(body) == 1:
            return self._emit_statement(body[0])
        stmts = ' '.join(self._emit_statement(s) for s in body)
        return f"(do {stmts})"
    
    def _map_type(self, ir_type: str) -> str:
        """Map IR type to Clojure type."""
        return self.type_mapper.map_type(ir_type)
