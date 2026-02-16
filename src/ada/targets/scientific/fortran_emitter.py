#!/usr/bin/env python3
"""STUNIR Fortran Emitter - Generate Fortran code from Scientific IR.

This emitter generates modern Fortran (2003/2008/2018) code including:
- Programs, modules, and submodules
- Subroutines and functions with intent/attributes
- Derived types with type-bound procedures
- Array operations with slicing and whole-array operations
- DO CONCURRENT loops for parallelism
- Coarray support for parallel programming
- Interface blocks for generic procedures

Usage:
    from targets.scientific.fortran_emitter import FortranEmitter
    from ir.scientific import Module, Subprogram
    
    emitter = FortranEmitter()
    result = emitter.emit(ir_dict)
    print(result.code)
"""

import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple, Union
from dataclasses import dataclass

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ir.scientific import (
    # Enumerations
    Visibility, Intent, ParameterMode, ArrayOrder,
    # Core
    Module, Program, Import, Subprogram, Parameter, Interface,
    TypeRef, VariableDecl, ConstantDecl,
    # Types
    RecordType, FieldDecl, EnumType, ArrayType, ArrayDimension,
    # Statements
    Assignment, IfStatement, ElseIfPart, ForLoop, WhileLoop,
    CaseStatement, CaseItem, CallStatement, ReturnStatement,
    BlockStatement, ExitStatement, ContinueStatement,
    OpenStatement, CloseStatement, ReadStatement, WriteStatement,
    AllocateStatement, DeallocateStatement,
    # Expressions
    Literal, VarRef, BinaryOp, UnaryOp, FunctionCall,
    ArrayAccess, FieldAccess, RangeExpr, TypeCast,
    # Arrays
    ArraySlice, SliceSpec, ArrayIntrinsic, ArrayConstructor, ImpliedDo,
    WhereStatement, ForallStatement, ForallIndex,
    # Numerical
    MathIntrinsic, ComplexLiteral, INTRINSIC_MAP,
    DoConcurrent, LoopIndex, LocalitySpec, ReduceSpec,
    Coarray, CoarrayAccess, SyncAll, SyncImages, CriticalBlock,
    NamelistGroup, NamelistRead, NamelistWrite,
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


# =============================================================================
# Type Mappings
# =============================================================================

FORTRAN_TYPE_MAP = {
    # Integers
    'int': 'INTEGER',
    'integer': 'INTEGER',
    'i8': 'INTEGER(KIND=1)',
    'i16': 'INTEGER(KIND=2)',
    'i32': 'INTEGER(KIND=4)',
    'i64': 'INTEGER(KIND=8)',
    # Reals
    'f32': 'REAL(KIND=4)',
    'f64': 'REAL(KIND=8)',
    'f128': 'REAL(KIND=16)',
    'float': 'REAL',
    'real': 'REAL',
    'double': 'DOUBLE PRECISION',
    # Complex
    'complex': 'COMPLEX',
    'complex32': 'COMPLEX(KIND=4)',
    'complex64': 'COMPLEX(KIND=8)',
    # Logical
    'bool': 'LOGICAL',
    'boolean': 'LOGICAL',
    'logical': 'LOGICAL',
    # Character
    'char': 'CHARACTER',
    'character': 'CHARACTER',
    'string': 'CHARACTER(LEN=*)',
}

FORTRAN_OP_MAP = {
    # Arithmetic
    '+': '+',
    '-': '-',
    '*': '*',
    '/': '/',
    '**': '**',
    'power': '**',
    'mod': 'MOD',
    # Comparison
    '==': '==',
    '/=': '/=',
    '!=': '/=',
    '<': '<',
    '>': '>',
    '<=': '<=',
    '>=': '>=',
    'eq': '==',
    'ne': '/=',
    'lt': '<',
    'gt': '>',
    'le': '<=',
    'ge': '>=',
    # Logical
    'and': '.AND.',
    'or': '.OR.',
    'not': '.NOT.',
    'eqv': '.EQV.',
    'neqv': '.NEQV.',
    '.and.': '.AND.',
    '.or.': '.OR.',
    '.not.': '.NOT.',
    # String
    '//': '//',
    'concat': '//',
}


class FortranEmitter:
    """Emit Fortran code from Scientific IR."""
    
    DIALECT = 'fortran'
    FILE_EXT = '.f90'
    
    def __init__(self, config: dict = None):
        """Initialize the Fortran emitter.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._indent = 0
        self._output: List[str] = []
        self._indent_str = '  '
        self._free_form = self.config.get('free_form', True)
    
    def emit(self, ir: dict) -> EmitterResult:
        """Emit Fortran code from IR dictionary.
        
        Args:
            ir: IR dictionary with 'kind' key indicating node type
            
        Returns:
            EmitterResult with code and manifest
        """
        self._output = []
        self._indent = 0
        
        kind = ir.get('kind', '')
        if kind == 'program':
            code = self.emit_program(ir)
        elif kind == 'module':
            code = self.emit_module(ir)
        else:
            code = self._emit_top_level(ir)
        
        manifest = self._generate_manifest(ir, code)
        return EmitterResult(code=code, manifest=manifest)
    
    def _line(self, text: str = ''):
        """Add a line with current indentation."""
        if text:
            self._output.append(self._indent_str * self._indent + text)
        else:
            self._output.append('')
    
    def _indent_inc(self):
        """Increase indentation."""
        self._indent += 1
    
    def _indent_dec(self):
        """Decrease indentation."""
        self._indent = max(0, self._indent - 1)
    
    def _get_output(self) -> str:
        """Get accumulated output."""
        return '\n'.join(self._output)
    
    # =========================================================================
    # Program/Module Emission
    # =========================================================================
    
    def emit_program(self, prog: dict) -> str:
        """Emit a PROGRAM unit."""
        self._output = []
        name = prog.get('name', 'main')
        
        self._line(f'PROGRAM {name}')
        self._indent_inc()
        
        # USE statements
        for use in prog.get('uses', []):
            self._emit_use(use)
        
        self._line('IMPLICIT NONE')
        self._line()
        
        # Variable declarations
        for var in prog.get('variables', []):
            self._emit_variable_decl(var)
        
        # Type declarations
        for typ in prog.get('types', []):
            self._emit_type_decl(typ)
        
        if prog.get('variables') or prog.get('types'):
            self._line()
        
        # Body statements
        for stmt in prog.get('body', []):
            self._emit_statement(stmt)
        
        # Internal subprograms
        if prog.get('subprograms'):
            self._line()
            self._line('CONTAINS')
            self._line()
            for sub in prog.get('subprograms', []):
                self._emit_subprogram(sub)
        
        self._indent_dec()
        self._line(f'END PROGRAM {name}')
        
        return self._get_output()
    
    def emit_module(self, mod: dict) -> str:
        """Emit a MODULE unit."""
        self._output = []
        name = mod.get('name', 'unnamed_module')
        is_submodule = mod.get('is_submodule', False)
        
        if is_submodule:
            parent = mod.get('parent_module', 'parent')
            self._line(f'SUBMODULE ({parent}) {name}')
        else:
            self._line(f'MODULE {name}')
        
        self._indent_inc()
        
        # USE statements
        for use in mod.get('imports', []):
            self._emit_use(use)
        
        self._line('IMPLICIT NONE')
        
        # Visibility
        if mod.get('visibility') == 'private' or mod.get('visibility') == Visibility.PRIVATE:
            self._line('PRIVATE')
        
        # Public exports
        exports = mod.get('exports', [])
        if exports:
            self._line(f"PUBLIC :: {', '.join(exports)}")
        
        self._line()
        
        # Type declarations
        for typ in mod.get('types', []):
            self._emit_type_decl(typ)
        
        # Variable declarations
        for var in mod.get('variables', []):
            self._emit_variable_decl(var)
        
        # Interface blocks
        for iface in mod.get('interfaces', []):
            self._emit_interface(iface)
        
        # Subprograms
        if mod.get('subprograms'):
            self._line()
            self._line('CONTAINS')
            self._line()
            for sub in mod.get('subprograms', []):
                self._emit_subprogram(sub)
        
        self._indent_dec()
        
        if is_submodule:
            self._line(f'END SUBMODULE {name}')
        else:
            self._line(f'END MODULE {name}')
        
        return self._get_output()
    
    def _emit_use(self, use: dict):
        """Emit USE statement."""
        mod_name = use.get('module_name', '')
        only = use.get('only', [])
        rename = use.get('rename', {})
        
        if only:
            items = ', '.join(only)
            self._line(f'USE {mod_name}, ONLY: {items}')
        elif rename:
            items = ', '.join(f'{local} => {orig}' for local, orig in rename.items())
            self._line(f'USE {mod_name}, {items}')
        else:
            self._line(f'USE {mod_name}')
    
    # =========================================================================
    # Subprogram Emission
    # =========================================================================
    
    def _emit_subprogram(self, sub: dict):
        """Emit SUBROUTINE or FUNCTION."""
        name = sub.get('name', 'unnamed')
        is_function = sub.get('is_function', False)
        params = sub.get('parameters', [])
        return_type = sub.get('return_type')
        result_var = sub.get('result_var')
        
        # Attributes
        attrs = []
        if sub.get('is_pure'):
            attrs.append('PURE')
        if sub.get('is_elemental'):
            attrs.append('ELEMENTAL')
        if sub.get('is_recursive'):
            attrs.append('RECURSIVE')
        
        attr_str = ' '.join(attrs) + ' ' if attrs else ''
        param_names = ', '.join(p.get('name', '') for p in params)
        
        if is_function:
            type_str = self._map_type(return_type) if return_type else 'INTEGER'
            result_clause = f' RESULT({result_var})' if result_var else ''
            self._line(f'{attr_str}{type_str} FUNCTION {name}({param_names}){result_clause}')
        else:
            self._line(f'{attr_str}SUBROUTINE {name}({param_names})')
        
        self._indent_inc()
        
        # Parameter declarations
        for param in params:
            self._emit_parameter_decl(param)
        
        # Local variable declarations
        for var in sub.get('local_vars', []):
            self._emit_variable_decl(var)
        
        # Local type declarations
        for typ in sub.get('local_types', []):
            self._emit_type_decl(typ)
        
        if params or sub.get('local_vars') or sub.get('local_types'):
            self._line()
        
        # Body
        for stmt in sub.get('body', []):
            self._emit_statement(stmt)
        
        self._indent_dec()
        
        if is_function:
            self._line(f'END FUNCTION {name}')
        else:
            self._line(f'END SUBROUTINE {name}')
        
        self._line()
    
    def _emit_parameter_decl(self, param: dict):
        """Emit parameter declaration."""
        name = param.get('name', '')
        type_ref = param.get('type_ref', {})
        intent = param.get('intent', 'in')
        is_optional = param.get('is_optional', False)
        
        type_str = self._map_type(type_ref)
        
        # Build attributes
        attrs = []
        
        # Intent
        if isinstance(intent, Intent):
            intent = intent.value
        intent_map = {'in': 'IN', 'out': 'OUT', 'inout': 'INOUT', 'in_out': 'INOUT'}
        attrs.append(f"INTENT({intent_map.get(intent, 'IN')})")
        
        if is_optional:
            attrs.append('OPTIONAL')
        
        attr_str = ', '.join(attrs)
        self._line(f'{type_str}, {attr_str} :: {name}')
    
    def _emit_interface(self, iface: dict):
        """Emit INTERFACE block."""
        name = iface.get('name')
        procedures = iface.get('procedures', [])
        is_abstract = iface.get('is_abstract', False)
        
        if is_abstract:
            self._line('ABSTRACT INTERFACE')
        elif name:
            self._line(f'INTERFACE {name}')
        else:
            self._line('INTERFACE')
        
        self._indent_inc()
        for proc in procedures:
            self._line(f'MODULE PROCEDURE {proc}')
        self._indent_dec()
        
        if name:
            self._line(f'END INTERFACE {name}')
        else:
            self._line('END INTERFACE')
        self._line()
    
    # =========================================================================
    # Type Emission
    # =========================================================================
    
    def _emit_type_decl(self, typ: dict):
        """Emit type declaration."""
        kind = typ.get('kind', '')
        
        if kind == 'record_type':
            self._emit_derived_type(typ)
        elif kind == 'enum_type':
            self._emit_enum_type(typ)
        else:
            # Generic type alias
            name = typ.get('name', '')
            if name:
                self._line(f'! Type: {name}')
    
    def _emit_derived_type(self, typ: dict):
        """Emit derived TYPE definition."""
        name = typ.get('name', 'unnamed_type')
        fields = typ.get('fields', [])
        extends = typ.get('extends')
        is_sequence = typ.get('is_sequence', False)
        is_bind_c = typ.get('is_bind_c', False)
        
        # Type header
        attrs = []
        if extends:
            attrs.append(f'EXTENDS({extends})')
        if is_bind_c:
            attrs.append('BIND(C)')
        
        attr_str = ', '.join(attrs)
        if attr_str:
            self._line(f'TYPE, {attr_str} :: {name}')
        else:
            self._line(f'TYPE :: {name}')
        
        self._indent_inc()
        
        if is_sequence:
            self._line('SEQUENCE')
        
        # Fields
        for fld in fields:
            self._emit_field_decl(fld)
        
        self._indent_dec()
        self._line(f'END TYPE {name}')
        self._line()
    
    def _emit_field_decl(self, fld: dict):
        """Emit field declaration in derived type."""
        name = fld.get('name', '')
        type_ref = fld.get('type_ref', {})
        is_pointer = fld.get('is_pointer', False)
        is_allocatable = fld.get('is_allocatable', False)
        
        type_str = self._map_type(type_ref)
        
        attrs = []
        if is_pointer:
            attrs.append('POINTER')
        if is_allocatable:
            attrs.append('ALLOCATABLE')
        
        if attrs:
            attr_str = ', ' + ', '.join(attrs)
        else:
            attr_str = ''
        
        self._line(f'{type_str}{attr_str} :: {name}')
    
    def _emit_enum_type(self, typ: dict):
        """Emit ENUM definition."""
        name = typ.get('name', '')
        values = typ.get('values', [])
        
        self._line(f'ENUM, BIND(C) :: {name}')
        self._indent_inc()
        for i, val in enumerate(values):
            self._line(f'ENUMERATOR :: {val} = {i}')
        self._indent_dec()
        self._line(f'END ENUM')
        self._line()
    
    # =========================================================================
    # Variable Declaration
    # =========================================================================
    
    def _emit_variable_decl(self, var: dict):
        """Emit variable declaration."""
        name = var.get('name', '')
        type_ref = var.get('type_ref', {})
        initial = var.get('initial_value')
        is_constant = var.get('is_constant', False)
        is_save = var.get('is_save', False)
        is_volatile = var.get('is_volatile', False)
        is_target = var.get('is_target', False)
        
        type_str = self._map_type(type_ref)
        
        attrs = []
        if is_constant:
            attrs.append('PARAMETER')
        if is_save:
            attrs.append('SAVE')
        if is_volatile:
            attrs.append('VOLATILE')
        if is_target:
            attrs.append('TARGET')
        if type_ref and type_ref.get('is_allocatable'):
            attrs.append('ALLOCATABLE')
        if type_ref and type_ref.get('is_pointer'):
            attrs.append('POINTER')
        
        if attrs:
            attr_str = ', ' + ', '.join(attrs)
        else:
            attr_str = ''
        
        if initial:
            init_str = ' = ' + self._emit_expr(initial)
        else:
            init_str = ''
        
        self._line(f'{type_str}{attr_str} :: {name}{init_str}')
    
    # =========================================================================
    # Statement Emission
    # =========================================================================
    
    def _emit_statement(self, stmt: dict):
        """Emit a statement."""
        kind = stmt.get('kind', '')
        
        method = getattr(self, f'_emit_{kind}', None)
        if method:
            method(stmt)
        else:
            self._line(f'! Unknown statement: {kind}')
    
    def _emit_assignment(self, stmt: dict):
        """Emit assignment statement."""
        target = self._emit_expr(stmt.get('target', {}))
        value = self._emit_expr(stmt.get('value', {}))
        self._line(f'{target} = {value}')
    
    def _emit_if_statement(self, stmt: dict):
        """Emit IF statement."""
        cond = self._emit_expr(stmt.get('condition', {}))
        self._line(f'IF ({cond}) THEN')
        self._indent_inc()
        for s in stmt.get('then_body', []):
            self._emit_statement(s)
        self._indent_dec()
        
        for elseif in stmt.get('elseif_parts', []):
            cond = self._emit_expr(elseif.get('condition', {}))
            self._line(f'ELSE IF ({cond}) THEN')
            self._indent_inc()
            for s in elseif.get('body', []):
                self._emit_statement(s)
            self._indent_dec()
        
        if stmt.get('else_body'):
            self._line('ELSE')
            self._indent_inc()
            for s in stmt.get('else_body', []):
                self._emit_statement(s)
            self._indent_dec()
        
        self._line('END IF')
    
    def _emit_for_loop(self, stmt: dict):
        """Emit DO loop."""
        var = stmt.get('variable', 'i')
        start = self._emit_expr(stmt.get('start', {}))
        end = self._emit_expr(stmt.get('end', {}))
        step = stmt.get('step')
        
        if step:
            step_str = ', ' + self._emit_expr(step)
        else:
            step_str = ''
        
        self._line(f'DO {var} = {start}, {end}{step_str}')
        self._indent_inc()
        for s in stmt.get('body', []):
            self._emit_statement(s)
        self._indent_dec()
        self._line('END DO')
    
    def _emit_while_loop(self, stmt: dict):
        """Emit DO WHILE loop."""
        cond = self._emit_expr(stmt.get('condition', {}))
        self._line(f'DO WHILE ({cond})')
        self._indent_inc()
        for s in stmt.get('body', []):
            self._emit_statement(s)
        self._indent_dec()
        self._line('END DO')
    
    def _emit_case_statement(self, stmt: dict):
        """Emit SELECT CASE statement."""
        selector = self._emit_expr(stmt.get('selector', {}))
        self._line(f'SELECT CASE ({selector})')
        self._indent_inc()
        
        for case in stmt.get('cases', []):
            values = case.get('values', [])
            val_strs = ', '.join(self._emit_expr(v) for v in values)
            self._line(f'CASE ({val_strs})')
            self._indent_inc()
            for s in case.get('body', []):
                self._emit_statement(s)
            self._indent_dec()
        
        if stmt.get('default_body'):
            self._line('CASE DEFAULT')
            self._indent_inc()
            for s in stmt.get('default_body', []):
                self._emit_statement(s)
            self._indent_dec()
        
        self._indent_dec()
        self._line('END SELECT')
    
    def _emit_call_statement(self, stmt: dict):
        """Emit CALL statement."""
        name = stmt.get('name', '')
        args = ', '.join(self._emit_expr(a) for a in stmt.get('arguments', []))
        self._line(f'CALL {name}({args})')
    
    def _emit_return_statement(self, stmt: dict):
        """Emit RETURN statement."""
        value = stmt.get('value')
        if value:
            self._line(f'RETURN {self._emit_expr(value)}')
        else:
            self._line('RETURN')
    
    def _emit_exit_statement(self, stmt: dict):
        """Emit EXIT statement."""
        loop_name = stmt.get('loop_name')
        if loop_name:
            self._line(f'EXIT {loop_name}')
        else:
            self._line('EXIT')
    
    def _emit_continue_statement(self, stmt: dict):
        """Emit CYCLE statement."""
        loop_name = stmt.get('loop_name')
        if loop_name:
            self._line(f'CYCLE {loop_name}')
        else:
            self._line('CYCLE')
    
    def _emit_block_statement(self, stmt: dict):
        """Emit BLOCK construct."""
        self._line('BLOCK')
        self._indent_inc()
        for var in stmt.get('declarations', []):
            self._emit_variable_decl(var)
        for s in stmt.get('body', []):
            self._emit_statement(s)
        self._indent_dec()
        self._line('END BLOCK')
    
    def _emit_null_statement(self, stmt: dict):
        """Emit CONTINUE (no-op) statement."""
        self._line('CONTINUE')
    
    # =========================================================================
    # I/O Statements
    # =========================================================================
    
    def _emit_open_statement(self, stmt: dict):
        """Emit OPEN statement."""
        parts = []
        if stmt.get('unit'):
            parts.append(f"UNIT={self._emit_expr(stmt['unit'])}")
        if stmt.get('file_name'):
            parts.append(f"FILE={self._emit_expr(stmt['file_name'])}")
        if stmt.get('status'):
            parts.append(f"STATUS='{stmt['status']}'")
        if stmt.get('access'):
            parts.append(f"ACCESS='{stmt['access']}'")
        if stmt.get('form'):
            parts.append(f"FORM='{stmt['form']}'")
        
        self._line(f"OPEN({', '.join(parts)})")
    
    def _emit_close_statement(self, stmt: dict):
        """Emit CLOSE statement."""
        unit = self._emit_expr(stmt.get('unit', {}))
        self._line(f'CLOSE({unit})')
    
    def _emit_read_statement(self, stmt: dict):
        """Emit READ statement."""
        unit = stmt.get('unit')
        fmt = stmt.get('format_spec')
        items = ', '.join(self._emit_expr(i) for i in stmt.get('items', []))
        
        if unit:
            unit_str = self._emit_expr(unit)
        else:
            unit_str = '*'
        
        if fmt:
            fmt_str = self._emit_expr(fmt)
        else:
            fmt_str = '*'
        
        self._line(f'READ({unit_str}, {fmt_str}) {items}')
    
    def _emit_write_statement(self, stmt: dict):
        """Emit WRITE/PRINT statement."""
        unit = stmt.get('unit')
        fmt = stmt.get('format_spec')
        items = ', '.join(self._emit_expr(i) for i in stmt.get('items', []))
        
        if unit:
            unit_str = self._emit_expr(unit)
        else:
            unit_str = '*'
        
        if fmt:
            fmt_str = self._emit_expr(fmt)
        else:
            fmt_str = '*'
        
        self._line(f'WRITE({unit_str}, {fmt_str}) {items}')
    
    # =========================================================================
    # Allocation Statements
    # =========================================================================
    
    def _emit_allocate_statement(self, stmt: dict):
        """Emit ALLOCATE statement."""
        allocs = stmt.get('allocations', [])
        alloc_strs = []
        for alloc in allocs:
            target = alloc.get('target', '')
            shape = alloc.get('shape', [])
            if shape:
                shape_str = ', '.join(self._emit_expr(s) for s in shape)
                alloc_strs.append(f'{target}({shape_str})')
            else:
                alloc_strs.append(target)
        
        parts = [', '.join(alloc_strs)]
        if stmt.get('stat_var'):
            parts.append(f"STAT={stmt['stat_var']}")
        if stmt.get('source_expr'):
            parts.append(f"SOURCE={self._emit_expr(stmt['source_expr'])}")
        
        self._line(f"ALLOCATE({', '.join(parts)})")
    
    def _emit_deallocate_statement(self, stmt: dict):
        """Emit DEALLOCATE statement."""
        vars_str = ', '.join(stmt.get('variables', []))
        parts = [vars_str]
        if stmt.get('stat_var'):
            parts.append(f"STAT={stmt['stat_var']}")
        
        self._line(f"DEALLOCATE({', '.join(parts)})")
    
    # =========================================================================
    # Array Statements
    # =========================================================================
    
    def _emit_where_statement(self, stmt: dict):
        """Emit WHERE construct."""
        mask = self._emit_expr(stmt.get('mask', {}))
        self._line(f'WHERE ({mask})')
        self._indent_inc()
        for s in stmt.get('body', []):
            self._emit_statement(s)
        self._indent_dec()
        
        if stmt.get('elsewhere_body'):
            self._line('ELSEWHERE')
            self._indent_inc()
            for s in stmt.get('elsewhere_body', []):
                self._emit_statement(s)
            self._indent_dec()
        
        self._line('END WHERE')
    
    def _emit_forall_statement(self, stmt: dict):
        """Emit FORALL construct."""
        indices = stmt.get('indices', [])
        idx_strs = []
        for idx in indices:
            var = idx.get('variable', 'i')
            lower = self._emit_expr(idx.get('lower', {}))
            upper = self._emit_expr(idx.get('upper', {}))
            stride = idx.get('stride')
            if stride:
                idx_strs.append(f'{var} = {lower}:{upper}:{self._emit_expr(stride)}')
            else:
                idx_strs.append(f'{var} = {lower}:{upper}')
        
        header = ', '.join(idx_strs)
        mask = stmt.get('mask')
        if mask:
            header += f', {self._emit_expr(mask)}'
        
        self._line(f'FORALL ({header})')
        self._indent_inc()
        for s in stmt.get('body', []):
            self._emit_statement(s)
        self._indent_dec()
        self._line('END FORALL')
    
    # =========================================================================
    # Parallel Constructs
    # =========================================================================
    
    def _emit_do_concurrent(self, stmt: dict):
        """Emit DO CONCURRENT loop."""
        indices = stmt.get('indices', [])
        idx_strs = []
        for idx in indices:
            var = idx.get('variable', 'i')
            start = self._emit_expr(idx.get('start', {}))
            end = self._emit_expr(idx.get('end', {}))
            stride = idx.get('stride')
            if stride:
                idx_strs.append(f'{var} = {start}:{end}:{self._emit_expr(stride)}')
            else:
                idx_strs.append(f'{var} = {start}:{end}')
        
        header = ', '.join(idx_strs)
        
        # Mask
        mask = stmt.get('mask')
        if mask:
            header += f', {self._emit_expr(mask)}'
        
        self._line(f'DO CONCURRENT ({header})')
        
        # Locality specification
        locality = stmt.get('locality')
        if locality:
            self._indent_inc()
            if locality.get('local_vars'):
                self._line(f"LOCAL({', '.join(locality['local_vars'])})")
            if locality.get('local_init'):
                self._line(f"LOCAL_INIT({', '.join(locality['local_init'])})")
            if locality.get('shared'):
                self._line(f"SHARED({', '.join(locality['shared'])})")
            for reduce in locality.get('reduce_ops', []):
                self._line(f"REDUCE({reduce['op']}:{reduce['variable']})")
            self._indent_dec()
        
        self._indent_inc()
        for s in stmt.get('body', []):
            self._emit_statement(s)
        self._indent_dec()
        self._line('END DO')
    
    def _emit_sync_all(self, stmt: dict):
        """Emit SYNC ALL statement."""
        parts = []
        if stmt.get('stat_var'):
            parts.append(f"STAT={stmt['stat_var']}")
        if stmt.get('errmsg_var'):
            parts.append(f"ERRMSG={stmt['errmsg_var']}")
        
        if parts:
            self._line(f"SYNC ALL({', '.join(parts)})")
        else:
            self._line('SYNC ALL')
    
    def _emit_sync_images(self, stmt: dict):
        """Emit SYNC IMAGES statement."""
        images = stmt.get('images', [])
        if images:
            img_str = ', '.join(self._emit_expr(i) for i in images)
        else:
            img_str = '*'
        
        parts = [img_str]
        if stmt.get('stat_var'):
            parts.append(f"STAT={stmt['stat_var']}")
        
        self._line(f"SYNC IMAGES({', '.join(parts)})")
    
    def _emit_critical_block(self, stmt: dict):
        """Emit CRITICAL block."""
        name = stmt.get('name')
        if name:
            self._line(f'{name}: CRITICAL')
        else:
            self._line('CRITICAL')
        
        self._indent_inc()
        for s in stmt.get('body', []):
            self._emit_statement(s)
        self._indent_dec()
        
        if name:
            self._line(f'END CRITICAL {name}')
        else:
            self._line('END CRITICAL')
    
    # =========================================================================
    # Expression Emission
    # =========================================================================
    
    def _emit_expr(self, expr: dict) -> str:
        """Emit an expression."""
        if not expr:
            return ''
        
        if isinstance(expr, str):
            return expr
        if isinstance(expr, (int, float)):
            return str(expr)
        
        kind = expr.get('kind', '')
        method = getattr(self, f'_emit_expr_{kind}', None)
        if method:
            return method(expr)
        
        # Fallback for simple values
        if 'value' in expr:
            return str(expr['value'])
        if 'name' in expr:
            return expr['name']
        
        return f'/* unknown expr: {kind} */'
    
    def _emit_expr_literal(self, expr: dict) -> str:
        """Emit literal value."""
        value = expr.get('value')
        type_hint = expr.get('type_hint', '')
        kind_param = expr.get('kind_param')
        
        if isinstance(value, bool):
            return '.TRUE.' if value else '.FALSE.'
        elif isinstance(value, str):
            return f"'{value}'"
        elif isinstance(value, float):
            result = repr(value)
            if 'e' not in result.lower() and '.' not in result:
                result += '.0'
            if kind_param:
                result += f'_{kind_param}'
            return result
        elif isinstance(value, int):
            if kind_param:
                return f'{value}_{kind_param}'
            return str(value)
        else:
            return str(value)
    
    def _emit_expr_var_ref(self, expr: dict) -> str:
        """Emit variable reference."""
        return expr.get('name', '')
    
    def _emit_expr_binary_op(self, expr: dict) -> str:
        """Emit binary operation."""
        op = expr.get('op', '+')
        left = self._emit_expr(expr.get('left', {}))
        right = self._emit_expr(expr.get('right', {}))
        
        fortran_op = FORTRAN_OP_MAP.get(op, op)
        
        # Handle special cases
        if op == 'mod':
            return f'MOD({left}, {right})'
        elif op == '**' or op == 'power':
            return f'({left})**({right})'
        else:
            return f'({left} {fortran_op} {right})'
    
    def _emit_expr_unary_op(self, expr: dict) -> str:
        """Emit unary operation."""
        op = expr.get('op', '-')
        operand = self._emit_expr(expr.get('operand', {}))
        
        fortran_op = FORTRAN_OP_MAP.get(op, op)
        if op in ('not', '.not.'):
            return f'.NOT. {operand}'
        else:
            return f'{fortran_op}({operand})'
    
    def _emit_expr_function_call(self, expr: dict) -> str:
        """Emit function call."""
        name = expr.get('name', '')
        args = ', '.join(self._emit_expr(a) for a in expr.get('arguments', []))
        return f'{name}({args})'
    
    def _emit_expr_array_access(self, expr: dict) -> str:
        """Emit array element access."""
        array = self._emit_expr(expr.get('array', {}))
        indices = ', '.join(self._emit_expr(i) for i in expr.get('indices', []))
        return f'{array}({indices})'
    
    def _emit_expr_array_slice(self, expr: dict) -> str:
        """Emit array slice."""
        array = self._emit_expr(expr.get('array', {}))
        slices = expr.get('slices', [])
        
        slice_strs = []
        for s in slices:
            start = s.get('start')
            stop = s.get('stop')
            stride = s.get('stride')
            
            if start is None and stop is None and stride is None:
                slice_strs.append(':')
            else:
                parts = []
                parts.append(self._emit_expr(start) if start else '')
                parts.append(self._emit_expr(stop) if stop else '')
                if stride:
                    parts.append(self._emit_expr(stride))
                slice_strs.append(':'.join(parts))
        
        return f"{array}({', '.join(slice_strs)})"
    
    def _emit_expr_field_access(self, expr: dict) -> str:
        """Emit record field access."""
        record = self._emit_expr(expr.get('record', {}))
        field = expr.get('field_name', '')
        return f'{record}%{field}'
    
    def _emit_expr_range_expr(self, expr: dict) -> str:
        """Emit range expression."""
        start = self._emit_expr(expr.get('start', {}))
        end = self._emit_expr(expr.get('end', {}))
        return f'{start}:{end}'
    
    def _emit_expr_type_cast(self, expr: dict) -> str:
        """Emit type cast."""
        target = expr.get('target_type', {})
        value = self._emit_expr(expr.get('expr', {}))
        
        type_name = target.get('name', 'integer')
        
        # Use appropriate conversion function
        convert_map = {
            'integer': 'INT',
            'int': 'INT',
            'real': 'REAL',
            'double': 'DBLE',
            'complex': 'CMPLX',
            'logical': 'LOGICAL',
        }
        
        func = convert_map.get(type_name.lower(), 'INT')
        return f'{func}({value})'
    
    def _emit_expr_complex_literal(self, expr: dict) -> str:
        """Emit complex literal."""
        real = self._emit_expr(expr.get('real_part', {}))
        imag = self._emit_expr(expr.get('imag_part', {}))
        return f'({real}, {imag})'
    
    def _emit_expr_array_constructor(self, expr: dict) -> str:
        """Emit array constructor."""
        elements = expr.get('elements', [])
        elem_strs = ', '.join(self._emit_expr(e) for e in elements)
        return f'[{elem_strs}]'
    
    def _emit_expr_implied_do(self, expr: dict) -> str:
        """Emit implied DO loop."""
        inner = self._emit_expr(expr.get('expr', {}))
        var = expr.get('variable', 'i')
        start = self._emit_expr(expr.get('start', {}))
        end = self._emit_expr(expr.get('end', {}))
        step = expr.get('step')
        
        if step:
            return f'({inner}, {var} = {start}, {end}, {self._emit_expr(step)})'
        else:
            return f'({inner}, {var} = {start}, {end})'
    
    def _emit_expr_array_intrinsic(self, expr: dict) -> str:
        """Emit array intrinsic function."""
        name = expr.get('name', '').upper()
        array = self._emit_expr(expr.get('array', {}))
        
        args = [array]
        if expr.get('dim'):
            args.append(f"DIM={self._emit_expr(expr['dim'])}")
        if expr.get('mask'):
            args.append(f"MASK={self._emit_expr(expr['mask'])}")
        
        return f"{name}({', '.join(args)})"
    
    def _emit_expr_math_intrinsic(self, expr: dict) -> str:
        """Emit mathematical intrinsic function."""
        name = expr.get('name', '')
        args = ', '.join(self._emit_expr(a) for a in expr.get('arguments', []))
        
        # Map to Fortran intrinsic
        fortran_name = INTRINSIC_MAP.get(name, {}).get('fortran', name.upper())
        return f'{fortran_name}({args})'
    
    def _emit_expr_coarray_access(self, expr: dict) -> str:
        """Emit coarray access."""
        coarray = self._emit_expr(expr.get('coarray', {}))
        indices = ', '.join(self._emit_expr(i) for i in expr.get('indices', []))
        coindices = ', '.join(self._emit_expr(c) for c in expr.get('coindices', []))
        
        if indices:
            return f'{coarray}({indices})[{coindices}]'
        else:
            return f'{coarray}[{coindices}]'
    
    # =========================================================================
    # Type Mapping
    # =========================================================================
    
    def _map_type(self, type_ref: dict) -> str:
        """Map IR type to Fortran type."""
        if not type_ref:
            return 'INTEGER'
        
        if isinstance(type_ref, str):
            return FORTRAN_TYPE_MAP.get(type_ref, type_ref.upper())
        
        kind = type_ref.get('kind', '')
        
        if kind == 'array_type':
            return self._map_array_type(type_ref)
        
        name = type_ref.get('name', '')
        kind_param = type_ref.get('kind_param')
        len_param = type_ref.get('len_param')
        
        base = FORTRAN_TYPE_MAP.get(name, name.upper())
        
        if kind_param:
            if '(' in base:
                # Already has KIND
                pass
            else:
                base = f'{base}(KIND={kind_param})'
        
        if len_param:
            if name in ('char', 'character', 'string'):
                base = f'CHARACTER(LEN={len_param})'
        
        return base
    
    def _map_array_type(self, type_ref: dict) -> str:
        """Map array type to Fortran declaration."""
        elem = type_ref.get('element_type', {})
        dims = type_ref.get('dimensions', [])
        allocatable = type_ref.get('allocatable', False)
        pointer = type_ref.get('pointer', False)
        
        base_type = self._map_type(elem)
        
        # Build dimension string
        dim_strs = []
        for dim in dims:
            if dim.get('is_deferred') or allocatable or pointer:
                dim_strs.append(':')
            elif dim.get('is_assumed_shape'):
                dim_strs.append(':')
            elif dim.get('is_assumed'):
                dim_strs.append('*')
            else:
                lower = dim.get('lower')
                upper = dim.get('upper')
                if lower and upper:
                    dim_strs.append(f'{self._emit_expr(lower)}:{self._emit_expr(upper)}')
                elif upper:
                    dim_strs.append(self._emit_expr(upper))
                else:
                    dim_strs.append(':')
        
        attrs = []
        if dim_strs:
            attrs.append(f"DIMENSION({', '.join(dim_strs)})")
        if allocatable:
            attrs.append('ALLOCATABLE')
        if pointer:
            attrs.append('POINTER')
        
        if attrs:
            return f"{base_type}, {', '.join(attrs)}"
        return base_type
    
    # =========================================================================
    # Manifest Generation
    # =========================================================================
    
    def _generate_manifest(self, ir: dict, code: str) -> dict:
        """Generate build manifest."""
        return {
            'schema': 'stunir.manifest.targets.v1',
            'generator': 'stunir.fortran.emitter',
            'epoch': int(time.time()),
            'ir_hash': compute_sha256(canonical_json(ir)),
            'output': {
                'hash': compute_sha256(code),
                'size': len(code),
                'format': 'fortran',
            }
        }
    
    def _emit_top_level(self, ir: dict) -> str:
        """Emit top-level construct from IR."""
        kind = ir.get('kind', '')
        
        if kind == 'subprogram':
            self._output = []
            self._emit_subprogram(ir)
            return self._get_output()
        elif kind == 'record_type':
            self._output = []
            self._emit_derived_type(ir)
            return self._get_output()
        else:
            return f'! Unknown top-level kind: {kind}'


# CLI entry point
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: fortran_emitter.py <ir.json> [output.f90]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    with open(input_path) as f:
        ir = json.load(f)
    
    emitter = FortranEmitter()
    result = emitter.emit(ir)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(result.code)
        print(f"Generated: {output_path}")
    else:
        print(result.code)
