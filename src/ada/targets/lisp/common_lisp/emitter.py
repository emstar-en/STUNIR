#!/usr/bin/env python3
"""STUNIR Common Lisp Emitter.

Generates ANSI Common Lisp code from STUNIR IR.
Supports type declarations, CLOS classes, and ASDF system definitions.

Part of Phase 5A: Core Lisp Implementation.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..base import LispEmitterBase, LispEmitterConfig, EmitterResult, compute_sha256, canonical_json
from .types import CommonLispTypeMapper, COMMON_LISP_TYPES


@dataclass
class CommonLispConfig(LispEmitterConfig):
    """Common Lisp specific configuration."""
    package_prefix: str = "stunir"
    use_declarations: bool = True
    use_the: bool = True  # Use THE for type assertions
    emit_asdf: bool = True  # Generate ASDF system definition
    emit_clos: bool = True  # Generate CLOS classes for types


class CommonLispEmitter(LispEmitterBase):
    """Emitter for ANSI Common Lisp code.
    
    Generates valid Common Lisp code with:
    - Package definitions
    - Type declarations (declaim, declare)
    - CLOS class generation
    - ASDF system definitions
    """
    
    DIALECT = "common-lisp"
    FILE_EXTENSION = ".lisp"
    
    # Common Lisp specific operators
    BINARY_OPS = {
        '+': '+',
        '-': '-',
        '*': '*',
        '/': '/',
        '%': 'mod',
        '==': '=',
        '!=': '/=',
        '<': '<',
        '>': '>',
        '<=': '<=',
        '>=': '>=',
        'and': 'and',
        'or': 'or',
        '&': 'logand',
        '|': 'logior',
        '^': 'logxor',
        '<<': 'ash',
    }
    
    def __init__(self, config: CommonLispConfig):
        """Initialize Common Lisp emitter.
        
        Args:
            config: Common Lisp emitter configuration.
        """
        super().__init__(config)
        self.config: CommonLispConfig = config
        self.type_mapper = CommonLispTypeMapper(config.use_declarations)
        self._exports: Set[str] = set()
    
    def emit(self, ir: Dict[str, Any]) -> EmitterResult:
        """Emit Common Lisp code from IR.
        
        Args:
            ir: STUNIR IR dictionary.
            
        Returns:
            EmitterResult with generated code.
        """
        module_name = ir.get('module', 'unnamed')
        package_name = f"{self.config.package_prefix}.{self._lisp_name(module_name)}"
        
        # Collect exports
        self._exports = set()
        for func in ir.get('functions', []):
            if func.get('exported', True):
                self._exports.add(self._lisp_name(func.get('name', '')))
        for export in ir.get('exports', []):
            self._exports.add(self._lisp_name(export))
        
        # Build code sections
        sections = []
        
        # Header
        sections.append(self._emit_header(module_name))
        
        # Package definition
        sections.append(self._emit_package(package_name))
        
        # Type declarations (ftype)
        for func in ir.get('functions', []):
            ftype = self._emit_ftype(func)
            if ftype:
                sections.append(ftype)
        
        # Type definitions (CLOS classes)
        if self.config.emit_clos:
            for type_def in ir.get('types', []):
                sections.append(self._emit_class(type_def))
        
        # Functions
        for func in ir.get('functions', []):
            sections.append(self._emit_function(func))
        
        code = "\n\n".join(filter(None, sections))
        
        # Write main file
        main_file = f"{self._lisp_name(module_name)}.lisp"
        self._write_file(main_file, code)
        
        # Generate ASDF system if enabled
        asdf_code = ""
        if self.config.emit_asdf:
            asdf_code = self._emit_asdf_system(ir)
            self._write_file(f"{self._lisp_name(module_name)}.asd", asdf_code)
        
        return EmitterResult(
            code=code,
            files={main_file: code, f"{self._lisp_name(module_name)}.asd": asdf_code} if asdf_code else {main_file: code},
            manifest=self._generate_manifest(ir)
        )
    
    def _emit_package(self, package_name: str) -> str:
        """Emit package definition."""
        exports = " ".join(f"#:{name}" for name in sorted(self._exports))
        
        lines = [
            f"(defpackage #:{package_name}",
            "  (:use #:cl)",
        ]
        
        if exports:
            lines.append(f"  (:export {exports}))")
        else:
            lines.append(")")
        
        lines.append("")
        lines.append(f"(in-package #:{package_name})")
        
        return "\n".join(lines)
    
    def _emit_ftype(self, func: Dict[str, Any]) -> str:
        """Emit function type declaration."""
        if not self.config.use_declarations:
            return ""
        
        name = self._lisp_name(func.get('name', 'unnamed'))
        params = func.get('params', [])
        return_type = func.get('return_type', 'any')
        
        param_types = [p.get('type', 'any') for p in params]
        
        return self.type_mapper.emit_ftype(name, param_types, return_type)
    
    def _emit_function(self, func: Dict[str, Any]) -> str:
        """Emit a function definition."""
        name = self._lisp_name(func.get('name', 'unnamed'))
        params = func.get('params', [])
        body = func.get('body', [])
        docstring = func.get('docstring', func.get('doc', ''))
        
        # Parameter list
        param_strs = []
        for p in params:
            pname = self._lisp_name(p.get('name', '_'))
            param_strs.append(pname)
        
        param_list = ' '.join(param_strs)
        
        # Build function
        lines = [f"(defun {name} ({param_list})"]
        
        # Docstring
        if docstring:
            lines.append(f'  "{docstring}"')
        
        # Type declarations for parameters
        if self.config.use_declarations and params:
            decls = []
            for p in params:
                pname = self._lisp_name(p.get('name', '_'))
                ptype = self.type_mapper.map_type(p.get('type', 'any'))
                decls.append(f"(type {ptype} {pname})")
            if decls:
                lines.append(f"  (declare {' '.join(decls)})")
        
        # Body
        if body:
            for i, stmt in enumerate(body):
                stmt_str = self._emit_statement(stmt)
                # Apply THE to return value if enabled
                if i == len(body) - 1 and self.config.use_the:
                    return_type = func.get('return_type')
                    if return_type and return_type != 'void':
                        stmt_str = self.type_mapper.emit_the(return_type, stmt_str)
                lines.append(f"  {stmt_str}")
        else:
            lines.append("  nil")
        
        lines.append(")")
        
        return "\n".join(lines)
    
    def _emit_class(self, type_def: Dict[str, Any]) -> str:
        """Emit a CLOS class definition."""
        name = self._lisp_name(type_def.get('name', 'unnamed'))
        fields = type_def.get('fields', [])
        docstring = type_def.get('docstring', '')
        
        lines = [f"(defclass {name} ()"]
        
        # Slots
        slots = []
        for field in fields:
            fname = self._lisp_name(field.get('name', 'field'))
            ftype = self.type_mapper.map_type(field.get('type', 'any'))
            
            slot = f"  (({fname}"
            slot += f" :initarg :{fname}"
            slot += f" :accessor {name}-{fname}"
            if 'default' in field:
                default = self._emit_literal(field['default'])
                slot += f" :initform {default}"
            slot += f" :type {ftype}))"
            slots.append(slot)
        
        if slots:
            lines.extend(slots)
        
        # Docstring
        if docstring:
            lines.append(f'  (:documentation "{docstring}"))')
        else:
            lines.append(")")
        
        return "\n".join(lines)
    
    def _emit_asdf_system(self, ir: Dict[str, Any]) -> str:
        """Emit ASDF system definition."""
        module_name = self._lisp_name(ir.get('module', 'unnamed'))
        
        lines = [
            f";;; ASDF System Definition for {module_name}",
            "",
            "(asdf:defsystem #:" + module_name,
            '  :description "Generated by STUNIR"',
            f'  :version "1.0.0"',
            "  :serial t",
            "  :components",
            f'  ((:file "{module_name}")))',
        ]
        
        return "\n".join(lines)
    
    def _emit_var_decl(self, stmt: Dict[str, Any]) -> str:
        """Emit variable declaration."""
        name = self._lisp_name(stmt.get('name', '_'))
        value = stmt.get('value', stmt.get('init'))
        
        if value is not None:
            val_str = self._emit_expression(value)
            return f"(let (({name} {val_str})))"
        return f"(let (({name} nil)))"
    
    def _emit_assignment(self, stmt: Dict[str, Any]) -> str:
        """Emit assignment statement."""
        name = self._lisp_name(stmt.get('name', stmt.get('target', '_')))
        value = self._emit_expression(stmt.get('value', {}))
        return f"(setf {name} {value})"
    
    def _emit_if_stmt(self, stmt: Dict[str, Any]) -> str:
        """Emit if statement."""
        cond = self._emit_expression(stmt.get('condition', stmt.get('cond', {})))
        then_body = stmt.get('then', stmt.get('consequent', []))
        else_body = stmt.get('else', stmt.get('alternate', []))
        
        # Handle body as list or single statement
        if isinstance(then_body, list):
            then_str = ' '.join(self._emit_statement(s) for s in then_body) if then_body else 'nil'
        else:
            then_str = self._emit_statement(then_body)
        
        if else_body:
            if isinstance(else_body, list):
                else_str = ' '.join(self._emit_statement(s) for s in else_body)
            else:
                else_str = self._emit_statement(else_body)
            return f"(if {cond}\n    {then_str}\n    {else_str})"
        
        return f"(when {cond}\n    {then_str})"
    
    def _emit_while(self, stmt: Dict[str, Any]) -> str:
        """Emit while loop."""
        cond = self._emit_expression(stmt.get('condition', stmt.get('cond', {})))
        body = stmt.get('body', [])
        
        body_strs = [self._emit_statement(s) for s in body] if body else ['nil']
        body_str = '\n      '.join(body_strs)
        
        return f"(loop while {cond}\n      do {body_str})"
    
    def _emit_for(self, stmt: Dict[str, Any]) -> str:
        """Emit for loop."""
        var = self._lisp_name(stmt.get('var', 'i'))
        start = self._emit_expression(stmt.get('start', {'kind': 'literal', 'value': 0}))
        end = self._emit_expression(stmt.get('end', {'kind': 'literal', 'value': 10}))
        body = stmt.get('body', [])
        
        body_strs = [self._emit_statement(s) for s in body] if body else ['nil']
        body_str = '\n      '.join(body_strs)
        
        return f"(loop for {var} from {start} below {end}\n      do {body_str})"
    
    def _emit_literal(self, value: Any) -> str:
        """Emit a Common Lisp literal."""
        if value is None:
            return "nil"
        if isinstance(value, bool):
            return "t" if value else "nil"
        if isinstance(value, str):
            escaped = value.replace('\\', '\\\\').replace('"', '\\"')
            return f'"{escaped}"'
        return str(value)
    
    def _map_type(self, ir_type: str) -> str:
        """Map IR type to Common Lisp type."""
        return self.type_mapper.map_type(ir_type)
