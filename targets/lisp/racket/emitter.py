#!/usr/bin/env python3
"""STUNIR Racket Emitter.

Generates idiomatic Racket code from STUNIR IR.
Supports both untyped Racket with contracts and Typed Racket.

Part of Phase 5A: Core Lisp Implementation.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..base import LispEmitterBase, LispEmitterConfig, EmitterResult, compute_sha256, canonical_json
from .types import RacketTypeMapper, RACKET_TYPES, TYPED_RACKET_TYPES, RACKET_CONTRACTS


@dataclass
class RacketEmitterConfig(LispEmitterConfig):
    """Racket specific configuration."""
    module_prefix: str = "stunir"
    lang: str = "racket"  # racket, typed/racket, racket/base
    use_contracts: bool = True  # Generate contracts (for untyped)
    use_typed: bool = False  # Use Typed Racket
    provide_all: bool = False  # Use (provide (all-defined-out))
    emit_structs: bool = True  # Generate struct definitions


class RacketEmitter(LispEmitterBase):
    """Emitter for Racket code.
    
    Generates Racket code with:
    - #lang directive
    - Contract-based type checking (untyped)
    - Type annotations (Typed Racket)
    - Struct definitions
    - Module provides
    """
    
    DIALECT = "racket"
    FILE_EXTENSION = ".rkt"
    
    # Racket-specific operators
    BINARY_OPS = {
        '+': '+',
        '-': '-',
        '*': '*',
        '/': '/',
        '%': 'modulo',
        '==': '=',
        '!=': 'not-equal?',  # We'll handle this specially
        '<': '<',
        '>': '>',
        '<=': '<=',
        '>=': '>=',
        'and': 'and',
        'or': 'or',
        '&': 'bitwise-and',
        '|': 'bitwise-ior',
        '^': 'bitwise-xor',
    }
    
    def __init__(self, config: RacketEmitterConfig):
        """Initialize Racket emitter.
        
        Args:
            config: Racket emitter configuration.
        """
        super().__init__(config)
        self.config: RacketEmitterConfig = config
        self.type_mapper = RacketTypeMapper(config.use_typed, config.use_contracts)
        self._exports: Set[str] = set()
    
    def emit(self, ir: Dict[str, Any]) -> EmitterResult:
        """Emit Racket code from IR.
        
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
        
        # Build code sections
        sections = []
        
        # #lang directive
        sections.append(self._emit_lang_line())
        
        # Header comment
        sections.append(self._emit_header(module_name))
        
        # Provide
        sections.append(self._emit_provide(ir))
        
        # Structs (type definitions)
        if self.config.emit_structs:
            for type_def in ir.get('types', []):
                sections.append(self._emit_struct(type_def))
        
        # Functions
        for func in ir.get('functions', []):
            # Type annotation for Typed Racket
            if self.config.use_typed:
                type_ann = self._emit_type_annotation(func)
                if type_ann:
                    sections.append(type_ann)
            
            sections.append(self._emit_function(func))
        
        code = "\n\n".join(filter(None, sections))
        
        # Write file
        main_file = f"{self._lisp_name(module_name)}.rkt"
        self._write_file(main_file, code)
        
        return EmitterResult(
            code=code,
            files={main_file: code},
            manifest=self._generate_manifest(ir)
        )
    
    def _emit_lang_line(self) -> str:
        """Emit #lang directive."""
        lang = self.config.lang
        if self.config.use_typed and lang == "racket":
            lang = "typed/racket"
        return f"#lang {lang}"
    
    def _emit_provide(self, ir: Dict[str, Any]) -> str:
        """Emit provide declaration."""
        if self.config.provide_all:
            return "(provide (all-defined-out))"
        
        if not self._exports:
            return ""
        
        if self.config.use_contracts and not self.config.use_typed:
            # Use contract-out for contracts
            contracts = []
            for func in ir.get('functions', []):
                name = self._lisp_name(func.get('name', ''))
                if name in self._exports:
                    params = func.get('params', [])
                    return_type = func.get('return_type', 'any')
                    param_types = [p.get('type', 'any') for p in params]
                    contract = self.type_mapper.emit_contract(param_types, return_type)
                    contracts.append(f"[{name} {contract}]")
            
            if contracts:
                contract_list = "\n  ".join(contracts)
                return f"(provide\n (contract-out\n  {contract_list}))"
        
        # Simple provide
        exports = ' '.join(sorted(self._exports))
        return f"(provide {exports})"
    
    def _emit_type_annotation(self, func: Dict[str, Any]) -> str:
        """Emit Typed Racket type annotation."""
        if not self.config.use_typed:
            return ""
        
        name = self._lisp_name(func.get('name', 'unnamed'))
        params = func.get('params', [])
        return_type = func.get('return_type', 'any')
        param_types = [p.get('type', 'any') for p in params]
        
        return self.type_mapper.emit_type_annotation(name, param_types, return_type)
    
    def _emit_function(self, func: Dict[str, Any]) -> str:
        """Emit a Racket function definition."""
        name = self._lisp_name(func.get('name', 'unnamed'))
        params = func.get('params', [])
        body = func.get('body', [])
        docstring = func.get('docstring', func.get('doc', ''))
        
        lines = []
        
        # Comment with docstring
        if docstring and self.config.emit_comments:
            lines.append(f";; {docstring}")
        
        # Parameter list
        if self.config.use_typed:
            # Typed Racket: [name : Type]
            param_strs = []
            for p in params:
                pname = self._lisp_name(p.get('name', '_'))
                ptype = TYPED_RACKET_TYPES.get(p.get('type', 'any'), 'Any')
                param_strs.append(f"[{pname} : {ptype}]")
            param_list = ' '.join(param_strs)
        else:
            param_strs = [self._lisp_name(p.get('name', '_')) for p in params]
            param_list = ' '.join(param_strs)
        
        lines.append(f"(define ({name} {param_list})")
        
        # Body
        if body:
            for stmt in body:
                stmt_str = self._emit_statement(stmt)
                lines.append(f"  {stmt_str}")
        else:
            lines.append("  (void)")
        
        lines.append(")")
        
        return "\n".join(lines)
    
    def _emit_struct(self, type_def: Dict[str, Any]) -> str:
        """Emit a struct definition."""
        name = self._lisp_name(type_def.get('name', 'unnamed'))
        fields = type_def.get('fields', [])
        
        if self.config.use_typed:
            # Typed Racket struct
            field_strs = []
            for f in fields:
                fname = self._lisp_name(f.get('name', 'field'))
                ftype = TYPED_RACKET_TYPES.get(f.get('type', 'any'), 'Any')
                field_strs.append(f"[{fname} : {ftype}]")
            fields_str = ' '.join(field_strs)
            return f"(struct {name} ({fields_str}) #:transparent)"
        else:
            # Untyped struct
            field_names = [self._lisp_name(f.get('name', 'field')) for f in fields]
            fields_str = ' '.join(field_names)
            return f"(struct {name} ({fields_str}) #:transparent)"
    
    def _emit_literal(self, value: Any) -> str:
        """Emit a Racket literal."""
        if value is None:
            return "#f"  # Racket uses #f for false/null in boolean context
        if isinstance(value, bool):
            return "#t" if value else "#f"
        if isinstance(value, str):
            escaped = value.replace('\\', '\\\\').replace('"', '\\"')
            return f'"{escaped}"'
        return str(value)
    
    def _emit_lambda(self, data: Dict[str, Any]) -> str:
        """Emit a Racket lambda."""
        params = data.get('params', [])
        body = data.get('body', [])
        
        param_names = ' '.join(self._lisp_name(p.get('name', '_')) for p in params)
        
        if body:
            body_str = ' '.join(self._emit_statement(stmt) for stmt in body)
        else:
            body_str = "(void)"
        
        # Can use λ or lambda
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
        """Emit for loop using Racket's for form."""
        var = self._lisp_name(stmt.get('var', 'i'))
        start = self._emit_expression(stmt.get('start', {'kind': 'literal', 'value': 0}))
        end = self._emit_expression(stmt.get('end', {'kind': 'literal', 'value': 10}))
        body = stmt.get('body', [])
        
        body_strs = [self._emit_statement(s) for s in body] if body else ['(void)']
        body_str = ' '.join(body_strs)
        
        return f"(for ([{var} (in-range {start} {end})])\n    {body_str})"
    
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
        
        racket_op = self.BINARY_OPS.get(op, op)
        return f"({racket_op} {left} {right})"
    
    def _map_type(self, ir_type: str) -> str:
        """Map IR type to Racket type."""
        return self.type_mapper.map_type(ir_type)
