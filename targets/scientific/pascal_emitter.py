#!/usr/bin/env python3
"""STUNIR Pascal Emitter - Generate Pascal code from Scientific IR.

This emitter generates Pascal code including:
- Programs and units
- Procedures and functions
- Records and variant records
- Object Pascal classes (Delphi/FPC)
- Set types and operations
- File operations
- Dynamic arrays

Usage:
    from targets.scientific.pascal_emitter import PascalEmitter
    from ir.scientific import Module, Subprogram
    
    emitter = PascalEmitter()
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
    Module, Program, Import, Subprogram, Parameter,
    TypeRef, VariableDecl, ConstantDecl,
    # Types
    RecordType, VariantRecord, RecordVariant, FieldDecl,
    EnumType, SetType, RangeType, PointerType, FileType,
    ClassType, MethodDecl, PropertyDecl,
    # Statements
    Assignment, IfStatement, ElseIfPart, ForLoop, WhileLoop, RepeatLoop,
    CaseStatement, CaseItem, CallStatement, ReturnStatement,
    BlockStatement, ExitStatement, ContinueStatement, GotoStatement,
    WithStatement, TryStatement, ExceptionHandler, RaiseStatement,
    # Expressions
    Literal, VarRef, BinaryOp, UnaryOp, FunctionCall,
    ArrayAccess, FieldAccess, SetExpr, SetOp, RangeExpr, TypeCast,
    PointerDeref, AddressOf,
    # Arrays
    PascalArrayType, DynamicArray, IndexRange, SetLength, Length, High, Low,
    # Numerical
    MathIntrinsic, INTRINSIC_MAP,
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

PASCAL_TYPE_MAP = {
    # Integers
    'int': 'Integer',
    'integer': 'Integer',
    'i8': 'ShortInt',
    'i16': 'SmallInt',
    'i32': 'LongInt',
    'i64': 'Int64',
    'u8': 'Byte',
    'u16': 'Word',
    'u32': 'LongWord',
    'u64': 'QWord',
    # Reals
    'f32': 'Single',
    'f64': 'Double',
    'f80': 'Extended',
    'float': 'Real',
    'real': 'Real',
    'double': 'Double',
    # Others
    'bool': 'Boolean',
    'boolean': 'Boolean',
    'logical': 'Boolean',
    'char': 'Char',
    'character': 'Char',
    'string': 'String',
    # Special
    'pointer': 'Pointer',
    'variant': 'Variant',
}

PASCAL_OP_MAP = {
    # Arithmetic
    '+': '+',
    '-': '-',
    '*': '*',
    '/': '/',
    'div': 'div',
    'mod': 'mod',
    # Comparison
    '==': '=',
    '=': '=',
    '<>': '<>',
    '!=': '<>',
    '/=': '<>',
    '<': '<',
    '>': '>',
    '<=': '<=',
    '>=': '>=',
    # Logical
    'and': 'and',
    'or': 'or',
    'not': 'not',
    'xor': 'xor',
    '.and.': 'and',
    '.or.': 'or',
    '.not.': 'not',
    # Bit operations
    'shl': 'shl',
    'shr': 'shr',
    # String
    'concat': '+',
    '//': '+',
    # Set operations
    'union': '+',
    'intersection': '*',
    'difference': '-',
    'in': 'in',
}


class PascalEmitter:
    """Emit Pascal code from Scientific IR."""
    
    DIALECT = 'pascal'
    FILE_EXT = '.pas'
    
    def __init__(self, config: dict = None):
        """Initialize the Pascal emitter.
        
        Args:
            config: Optional configuration dictionary
                - object_pascal: Enable Object Pascal features (default: True)
                - fpc_mode: Free Pascal Compiler mode (default: True)
        """
        self.config = config or {}
        self._indent = 0
        self._output: List[str] = []
        self._indent_str = '  '
        self._object_pascal = self.config.get('object_pascal', True)
        self._fpc_mode = self.config.get('fpc_mode', True)
    
    def emit(self, ir: dict) -> EmitterResult:
        """Emit Pascal code from IR dictionary.
        
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
            code = self.emit_unit(ir)
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
    # Program/Unit Emission
    # =========================================================================
    
    def emit_program(self, prog: dict) -> str:
        """Emit a Pascal PROGRAM."""
        self._output = []
        name = prog.get('name', 'Main')
        
        self._line(f'program {name};')
        self._line()
        
        # Uses clause
        uses = prog.get('uses', [])
        if uses:
            use_names = [u.get('module_name', '') for u in uses]
            self._line(f"uses {', '.join(use_names)};")
            self._line()
        
        # Type declarations
        types = prog.get('types', [])
        if types:
            self._line('type')
            self._indent_inc()
            for typ in types:
                self._emit_type_decl(typ)
            self._indent_dec()
            self._line()
        
        # Variable declarations
        variables = prog.get('variables', [])
        if variables:
            self._line('var')
            self._indent_inc()
            for var in variables:
                self._emit_variable_decl(var)
            self._indent_dec()
            self._line()
        
        # Subprograms
        for sub in prog.get('subprograms', []):
            self._emit_subprogram(sub)
            self._line()
        
        # Main body
        self._line('begin')
        self._indent_inc()
        for stmt in prog.get('body', []):
            self._emit_statement(stmt)
        self._indent_dec()
        self._line('end.')
        
        return self._get_output()
    
    def emit_unit(self, unit: dict) -> str:
        """Emit a Pascal UNIT."""
        self._output = []
        name = unit.get('name', 'UnitName')
        
        self._line(f'unit {name};')
        self._line()
        
        # Interface section
        self._line('interface')
        self._line()
        
        # Uses in interface
        uses = unit.get('imports', [])
        if uses:
            use_names = [u.get('module_name', '') for u in uses]
            self._line(f"uses {', '.join(use_names)};")
            self._line()
        
        # Public type declarations
        types = [t for t in unit.get('types', []) 
                 if t.get('visibility', 'public') != 'private']
        if types:
            self._line('type')
            self._indent_inc()
            for typ in types:
                self._emit_type_decl(typ)
            self._indent_dec()
            self._line()
        
        # Public variable declarations
        variables = [v for v in unit.get('variables', [])
                     if v.get('visibility', 'public') != 'private']
        if variables:
            self._line('var')
            self._indent_inc()
            for var in variables:
                self._emit_variable_decl(var)
            self._indent_dec()
            self._line()
        
        # Public subprogram declarations
        public_subs = [s for s in unit.get('subprograms', [])
                       if s.get('visibility', 'public') != 'private']
        for sub in public_subs:
            self._emit_subprogram_declaration(sub)
        
        self._line()
        
        # Implementation section
        self._line('implementation')
        self._line()
        
        # Private types
        private_types = [t for t in unit.get('types', [])
                         if t.get('visibility', 'public') == 'private']
        if private_types:
            self._line('type')
            self._indent_inc()
            for typ in private_types:
                self._emit_type_decl(typ)
            self._indent_dec()
            self._line()
        
        # Private variables
        private_vars = [v for v in unit.get('variables', [])
                        if v.get('visibility', 'public') == 'private']
        if private_vars:
            self._line('var')
            self._indent_inc()
            for var in private_vars:
                self._emit_variable_decl(var)
            self._indent_dec()
            self._line()
        
        # All subprogram implementations
        for sub in unit.get('subprograms', []):
            self._emit_subprogram(sub)
            self._line()
        
        self._line('end.')
        
        return self._get_output()
    
    # =========================================================================
    # Subprogram Emission
    # =========================================================================
    
    def _emit_subprogram_declaration(self, sub: dict):
        """Emit subprogram forward declaration."""
        name = sub.get('name', 'unnamed')
        is_function = sub.get('is_function', False)
        params = sub.get('parameters', [])
        return_type = sub.get('return_type')
        
        param_str = self._format_parameters(params)
        
        if is_function:
            ret_type = self._map_type(return_type) if return_type else 'Integer'
            self._line(f'function {name}({param_str}): {ret_type};')
        else:
            self._line(f'procedure {name}({param_str});')
    
    def _emit_subprogram(self, sub: dict):
        """Emit procedure or function."""
        name = sub.get('name', 'unnamed')
        is_function = sub.get('is_function', False)
        params = sub.get('parameters', [])
        return_type = sub.get('return_type')
        
        param_str = self._format_parameters(params)
        
        if is_function:
            ret_type = self._map_type(return_type) if return_type else 'Integer'
            self._line(f'function {name}({param_str}): {ret_type};')
        else:
            self._line(f'procedure {name}({param_str});')
        
        # Local variables
        local_vars = sub.get('local_vars', [])
        local_types = sub.get('local_types', [])
        
        if local_types:
            self._line('type')
            self._indent_inc()
            for typ in local_types:
                self._emit_type_decl(typ)
            self._indent_dec()
        
        if local_vars:
            self._line('var')
            self._indent_inc()
            for var in local_vars:
                self._emit_variable_decl(var)
            self._indent_dec()
        
        # Body
        self._line('begin')
        self._indent_inc()
        for stmt in sub.get('body', []):
            self._emit_statement(stmt)
        self._indent_dec()
        self._line('end;')
    
    def _format_parameters(self, params: List[dict]) -> str:
        """Format parameter list."""
        if not params:
            return ''
        
        parts = []
        for param in params:
            name = param.get('name', '')
            type_ref = param.get('type_ref', {})
            mode = param.get('mode', 'value')
            
            if isinstance(mode, ParameterMode):
                mode = mode.value
            
            type_str = self._map_type(type_ref)
            
            if mode == 'var':
                parts.append(f'var {name}: {type_str}')
            elif mode == 'const':
                parts.append(f'const {name}: {type_str}')
            elif mode == 'out':
                parts.append(f'out {name}: {type_str}')
            else:
                parts.append(f'{name}: {type_str}')
        
        return '; '.join(parts)
    
    # =========================================================================
    # Type Emission
    # =========================================================================
    
    def _emit_type_decl(self, typ: dict):
        """Emit type declaration."""
        kind = typ.get('kind', '')
        
        if kind == 'record_type':
            self._emit_record_type(typ)
        elif kind == 'variant_record':
            self._emit_variant_record(typ)
        elif kind == 'enum_type':
            self._emit_enum_type(typ)
        elif kind == 'set_type':
            self._emit_set_type(typ)
        elif kind == 'range_type':
            self._emit_range_type(typ)
        elif kind == 'pointer_type':
            self._emit_pointer_type(typ)
        elif kind == 'file_type':
            self._emit_file_type(typ)
        elif kind == 'class_type':
            self._emit_class_type(typ)
        elif kind == 'pascal_array_type':
            self._emit_pascal_array_type(typ)
        elif kind == 'dynamic_array':
            self._emit_dynamic_array_type(typ)
        else:
            name = typ.get('name', '')
            if name:
                self._line(f'{name} = { self._map_type(typ) };')
    
    def _emit_record_type(self, typ: dict):
        """Emit record type."""
        name = typ.get('name', 'TRecord')
        fields = typ.get('fields', [])
        
        self._line(f'{name} = record')
        self._indent_inc()
        for fld in fields:
            fld_name = fld.get('name', '')
            fld_type = self._map_type(fld.get('type_ref', {}))
            self._line(f'{fld_name}: {fld_type};')
        self._indent_dec()
        self._line('end;')
    
    def _emit_variant_record(self, typ: dict):
        """Emit variant record type."""
        name = typ.get('name', 'TVariantRecord')
        fixed_fields = typ.get('fixed_fields', [])
        tag_field = typ.get('tag_field', {})
        variants = typ.get('variants', [])
        
        self._line(f'{name} = record')
        self._indent_inc()
        
        # Fixed fields
        for fld in fixed_fields:
            fld_name = fld.get('name', '')
            fld_type = self._map_type(fld.get('type_ref', {}))
            self._line(f'{fld_name}: {fld_type};')
        
        # Variant part
        if tag_field:
            tag_name = tag_field.get('name', 'Kind')
            tag_type = self._map_type(tag_field.get('type_ref', {}))
            self._line(f'case {tag_name}: {tag_type} of')
            self._indent_inc()
            
            for variant in variants:
                tag_values = variant.get('tag_values', [])
                val_strs = ', '.join(self._emit_expr(v) for v in tag_values)
                self._line(f'{val_strs}: (')
                self._indent_inc()
                for fld in variant.get('fields', []):
                    fld_name = fld.get('name', '')
                    fld_type = self._map_type(fld.get('type_ref', {}))
                    self._line(f'{fld_name}: {fld_type};')
                self._indent_dec()
                self._line(');')
            
            self._indent_dec()
        
        self._indent_dec()
        self._line('end;')
    
    def _emit_enum_type(self, typ: dict):
        """Emit enumeration type."""
        name = typ.get('name', 'TEnum')
        values = typ.get('values', [])
        vals_str = ', '.join(values)
        self._line(f'{name} = ({vals_str});')
    
    def _emit_set_type(self, typ: dict):
        """Emit set type."""
        name = typ.get('name', 'TSet')
        base_type = self._map_type(typ.get('base_type', {}))
        self._line(f'{name} = set of {base_type};')
    
    def _emit_range_type(self, typ: dict):
        """Emit subrange type."""
        name = typ.get('name', 'TRange')
        lower = self._emit_expr(typ.get('lower', {}))
        upper = self._emit_expr(typ.get('upper', {}))
        self._line(f'{name} = {lower}..{upper};')
    
    def _emit_pointer_type(self, typ: dict):
        """Emit pointer type."""
        name = typ.get('name', 'PType')
        target = self._map_type(typ.get('target_type', {}))
        self._line(f'{name} = ^{target};')
    
    def _emit_file_type(self, typ: dict):
        """Emit file type."""
        name = typ.get('name', 'TFile')
        elem_type = typ.get('element_type')
        
        if elem_type:
            elem_str = self._map_type(elem_type)
            self._line(f'{name} = file of {elem_str};')
        else:
            self._line(f'{name} = Text;')
    
    def _emit_pascal_array_type(self, typ: dict):
        """Emit Pascal array type with index ranges."""
        name = typ.get('name', 'TArray')
        elem_type = self._map_type(typ.get('element_type', {}))
        index_ranges = typ.get('index_ranges', [])
        is_packed = typ.get('is_packed', False)
        
        ranges = []
        for r in index_ranges:
            lower = self._emit_expr(r.get('lower', {}))
            upper = self._emit_expr(r.get('upper', {}))
            ranges.append(f'{lower}..{upper}')
        
        range_str = ', '.join(ranges)
        packed = 'packed ' if is_packed else ''
        self._line(f'{name} = {packed}array[{range_str}] of {elem_type};')
    
    def _emit_dynamic_array_type(self, typ: dict):
        """Emit dynamic array type."""
        name = typ.get('name', 'TDynArray')
        elem_type = self._map_type(typ.get('element_type', {}))
        dims = typ.get('dimensions', 1)
        
        # Build nested array of
        arr_str = elem_type
        for _ in range(dims):
            arr_str = f'array of {arr_str}'
        
        self._line(f'{name} = {arr_str};')
    
    def _emit_class_type(self, typ: dict):
        """Emit Object Pascal class type."""
        name = typ.get('name', 'TClass')
        parent = typ.get('parent', 'TObject')
        fields = typ.get('fields', [])
        methods = typ.get('methods', [])
        properties = typ.get('properties', [])
        is_abstract = typ.get('is_abstract', False)
        
        abstract = 'abstract ' if is_abstract else ''
        self._line(f'{name} = {abstract}class({parent})')
        
        # Group by visibility
        for vis in ['private', 'protected', 'public', 'published']:
            vis_fields = [f for f in fields if f.get('visibility', 'public') == vis]
            vis_methods = [m for m in methods if m.get('visibility', 'public') == vis]
            vis_props = [p for p in properties if p.get('visibility', 'public') == vis]
            
            if vis_fields or vis_methods or vis_props:
                self._line(f'{vis}')
                self._indent_inc()
                
                for fld in vis_fields:
                    fld_name = fld.get('name', '')
                    fld_type = self._map_type(fld.get('type_ref', {}))
                    self._line(f'{fld_name}: {fld_type};')
                
                for meth in vis_methods:
                    self._emit_method_decl(meth)
                
                for prop in vis_props:
                    self._emit_property_decl(prop)
                
                self._indent_dec()
        
        self._line('end;')
    
    def _emit_method_decl(self, meth: dict):
        """Emit method declaration in class."""
        name = meth.get('name', '')
        is_function = meth.get('is_function', False)
        params = meth.get('parameters', [])
        return_type = meth.get('return_type')
        
        modifiers = []
        if meth.get('is_virtual'):
            modifiers.append('virtual')
        if meth.get('is_abstract'):
            modifiers.append('abstract')
        if meth.get('is_override'):
            modifiers.append('override')
        if meth.get('is_static'):
            modifiers.append('static')
        if meth.get('is_class_method'):
            modifiers.insert(0, 'class')
        
        param_str = self._format_parameters(params)
        mod_str = '; '.join(modifiers)
        if mod_str:
            mod_str = ' ' + mod_str + ';'
        
        if is_function:
            ret_type = self._map_type(return_type) if return_type else 'Integer'
            self._line(f'function {name}({param_str}): {ret_type};{mod_str}')
        else:
            self._line(f'procedure {name}({param_str});{mod_str}')
    
    def _emit_property_decl(self, prop: dict):
        """Emit property declaration."""
        name = prop.get('name', '')
        type_ref = prop.get('type_ref', {})
        getter = prop.get('getter')
        setter = prop.get('setter')
        is_default = prop.get('is_default', False)
        
        type_str = self._map_type(type_ref)
        
        parts = [f'property {name}: {type_str}']
        if getter:
            parts.append(f'read {getter}')
        if setter:
            parts.append(f'write {setter}')
        if is_default:
            parts.append('default')
        
        self._line(' '.join(parts) + ';')
    
    # =========================================================================
    # Variable Declaration
    # =========================================================================
    
    def _emit_variable_decl(self, var: dict):
        """Emit variable declaration."""
        name = var.get('name', '')
        type_ref = var.get('type_ref', {})
        initial = var.get('initial_value')
        is_constant = var.get('is_constant', False)
        
        type_str = self._map_type(type_ref)
        
        if is_constant:
            if initial:
                init_str = self._emit_expr(initial)
                self._line(f'{name} = {init_str};')
            else:
                self._line(f'{name}: {type_str};')
        else:
            if initial:
                init_str = self._emit_expr(initial)
                self._line(f'{name}: {type_str} = {init_str};')
            else:
                self._line(f'{name}: {type_str};')
    
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
            self._line(f'{{ Unknown statement: {kind} }}')
    
    def _emit_assignment(self, stmt: dict):
        """Emit assignment statement."""
        target = self._emit_expr(stmt.get('target', {}))
        value = self._emit_expr(stmt.get('value', {}))
        self._line(f'{target} := {value};')
    
    def _emit_if_statement(self, stmt: dict):
        """Emit IF statement."""
        cond = self._emit_expr(stmt.get('condition', {}))
        then_body = stmt.get('then_body', [])
        elseif_parts = stmt.get('elseif_parts', [])
        else_body = stmt.get('else_body', [])
        
        self._line(f'if {cond} then')
        self._line('begin')
        self._indent_inc()
        for s in then_body:
            self._emit_statement(s)
        self._indent_dec()
        self._line('end')
        
        for elseif in elseif_parts:
            cond = self._emit_expr(elseif.get('condition', {}))
            self._line(f'else if {cond} then')
            self._line('begin')
            self._indent_inc()
            for s in elseif.get('body', []):
                self._emit_statement(s)
            self._indent_dec()
            self._line('end')
        
        if else_body:
            self._line('else')
            self._line('begin')
            self._indent_inc()
            for s in else_body:
                self._emit_statement(s)
            self._indent_dec()
            self._line('end;')
        else:
            # Remove 'end' and add semicolon
            if self._output and self._output[-1].strip() == 'end':
                self._output[-1] = self._output[-1] + ';'
    
    def _emit_for_loop(self, stmt: dict):
        """Emit FOR loop."""
        var = stmt.get('variable', 'i')
        start = self._emit_expr(stmt.get('start', {}))
        end = self._emit_expr(stmt.get('end', {}))
        is_downto = stmt.get('is_downto', False)
        
        direction = 'downto' if is_downto else 'to'
        
        self._line(f'for {var} := {start} {direction} {end} do')
        self._line('begin')
        self._indent_inc()
        for s in stmt.get('body', []):
            self._emit_statement(s)
        self._indent_dec()
        self._line('end;')
    
    def _emit_while_loop(self, stmt: dict):
        """Emit WHILE loop."""
        cond = self._emit_expr(stmt.get('condition', {}))
        
        self._line(f'while {cond} do')
        self._line('begin')
        self._indent_inc()
        for s in stmt.get('body', []):
            self._emit_statement(s)
        self._indent_dec()
        self._line('end;')
    
    def _emit_repeat_loop(self, stmt: dict):
        """Emit REPEAT-UNTIL loop."""
        cond = self._emit_expr(stmt.get('condition', {}))
        
        self._line('repeat')
        self._indent_inc()
        for s in stmt.get('body', []):
            self._emit_statement(s)
        self._indent_dec()
        self._line(f'until {cond};')
    
    def _emit_case_statement(self, stmt: dict):
        """Emit CASE statement."""
        selector = self._emit_expr(stmt.get('selector', {}))
        
        self._line(f'case {selector} of')
        self._indent_inc()
        
        for case in stmt.get('cases', []):
            values = case.get('values', [])
            val_strs = ', '.join(self._emit_expr(v) for v in values)
            self._line(f'{val_strs}:')
            self._line('begin')
            self._indent_inc()
            for s in case.get('body', []):
                self._emit_statement(s)
            self._indent_dec()
            self._line('end;')
        
        if stmt.get('default_body'):
            self._line('else')
            self._line('begin')
            self._indent_inc()
            for s in stmt.get('default_body', []):
                self._emit_statement(s)
            self._indent_dec()
            self._line('end;')
        
        self._indent_dec()
        self._line('end;')
    
    def _emit_call_statement(self, stmt: dict):
        """Emit procedure call."""
        name = stmt.get('name', '')
        args = ', '.join(self._emit_expr(a) for a in stmt.get('arguments', []))
        
        if args:
            self._line(f'{name}({args});')
        else:
            self._line(f'{name};')
    
    def _emit_return_statement(self, stmt: dict):
        """Emit exit/result assignment."""
        value = stmt.get('value')
        if value:
            # In Pascal, assign to Result or function name
            self._line(f'Result := {self._emit_expr(value)};')
            self._line('Exit;')
        else:
            self._line('Exit;')
    
    def _emit_exit_statement(self, stmt: dict):
        """Emit Break statement."""
        self._line('Break;')
    
    def _emit_continue_statement(self, stmt: dict):
        """Emit Continue statement."""
        self._line('Continue;')
    
    def _emit_goto_statement(self, stmt: dict):
        """Emit GOTO statement."""
        label = stmt.get('label', '')
        self._line(f'goto {label};')
    
    def _emit_block_statement(self, stmt: dict):
        """Emit compound statement."""
        self._line('begin')
        self._indent_inc()
        for s in stmt.get('body', []):
            self._emit_statement(s)
        self._indent_dec()
        self._line('end;')
    
    def _emit_null_statement(self, stmt: dict):
        """Emit empty statement."""
        self._line('{ no-op };')
    
    def _emit_with_statement(self, stmt: dict):
        """Emit WITH statement."""
        records = stmt.get('record_vars', [])
        rec_strs = ', '.join(self._emit_expr(r) for r in records)
        
        self._line(f'with {rec_strs} do')
        self._line('begin')
        self._indent_inc()
        for s in stmt.get('body', []):
            self._emit_statement(s)
        self._indent_dec()
        self._line('end;')
    
    def _emit_try_statement(self, stmt: dict):
        """Emit TRY-EXCEPT-FINALLY statement."""
        self._line('try')
        self._indent_inc()
        for s in stmt.get('try_body', []):
            self._emit_statement(s)
        self._indent_dec()
        
        handlers = stmt.get('handlers', [])
        if handlers:
            self._line('except')
            self._indent_inc()
            for handler in handlers:
                exc_type = handler.get('exception_type')
                exc_var = handler.get('variable')
                
                if exc_type:
                    if exc_var:
                        self._line(f'on {exc_var}: {exc_type} do')
                    else:
                        self._line(f'on {exc_type} do')
                    self._line('begin')
                    self._indent_inc()
                    for s in handler.get('body', []):
                        self._emit_statement(s)
                    self._indent_dec()
                    self._line('end;')
                else:
                    # Catch-all
                    for s in handler.get('body', []):
                        self._emit_statement(s)
            self._indent_dec()
        
        finally_body = stmt.get('finally_body', [])
        if finally_body:
            self._line('finally')
            self._indent_inc()
            for s in finally_body:
                self._emit_statement(s)
            self._indent_dec()
        
        self._line('end;')
    
    def _emit_raise_statement(self, stmt: dict):
        """Emit RAISE statement."""
        exc = stmt.get('exception_expr')
        if exc:
            self._line(f'raise {self._emit_expr(exc)};')
        else:
            self._line('raise;')
    
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
        
        return f'{{ unknown expr: {kind} }}'
    
    def _emit_expr_literal(self, expr: dict) -> str:
        """Emit literal value."""
        value = expr.get('value')
        type_hint = expr.get('type_hint', '')
        
        if isinstance(value, bool):
            return 'True' if value else 'False'
        elif isinstance(value, str):
            # Escape single quotes
            escaped = value.replace("'", "''")
            return f"'{escaped}'"
        elif isinstance(value, float):
            return repr(value)
        elif isinstance(value, int):
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
        
        pascal_op = PASCAL_OP_MAP.get(op, op)
        
        return f'({left} {pascal_op} {right})'
    
    def _emit_expr_unary_op(self, expr: dict) -> str:
        """Emit unary operation."""
        op = expr.get('op', '-')
        operand = self._emit_expr(expr.get('operand', {}))
        
        pascal_op = PASCAL_OP_MAP.get(op, op)
        
        if op in ('not', '.not.'):
            return f'not {operand}'
        else:
            return f'{pascal_op}{operand}'
    
    def _emit_expr_function_call(self, expr: dict) -> str:
        """Emit function call."""
        name = expr.get('name', '')
        args = ', '.join(self._emit_expr(a) for a in expr.get('arguments', []))
        
        if args:
            return f'{name}({args})'
        else:
            return f'{name}'
    
    def _emit_expr_array_access(self, expr: dict) -> str:
        """Emit array element access."""
        array = self._emit_expr(expr.get('array', {}))
        indices = ', '.join(self._emit_expr(i) for i in expr.get('indices', []))
        return f'{array}[{indices}]'
    
    def _emit_expr_field_access(self, expr: dict) -> str:
        """Emit record field access."""
        record = self._emit_expr(expr.get('record', {}))
        field = expr.get('field_name', '')
        return f'{record}.{field}'
    
    def _emit_expr_set_expr(self, expr: dict) -> str:
        """Emit set literal."""
        elements = expr.get('elements', [])
        elem_strs = ', '.join(self._emit_expr(e) for e in elements)
        return f'[{elem_strs}]'
    
    def _emit_expr_set_op(self, expr: dict) -> str:
        """Emit set operation."""
        op = expr.get('op', 'union')
        left = self._emit_expr(expr.get('left', {}))
        right = self._emit_expr(expr.get('right', {}))
        
        pascal_op = PASCAL_OP_MAP.get(op, op)
        return f'({left} {pascal_op} {right})'
    
    def _emit_expr_range_expr(self, expr: dict) -> str:
        """Emit range expression."""
        start = self._emit_expr(expr.get('start', {}))
        end = self._emit_expr(expr.get('end', {}))
        return f'{start}..{end}'
    
    def _emit_expr_type_cast(self, expr: dict) -> str:
        """Emit type cast."""
        target = expr.get('target_type', {})
        value = self._emit_expr(expr.get('expr', {}))
        
        type_name = self._map_type(target)
        return f'{type_name}({value})'
    
    def _emit_expr_pointer_deref(self, expr: dict) -> str:
        """Emit pointer dereference."""
        pointer = self._emit_expr(expr.get('pointer', {}))
        return f'{pointer}^'
    
    def _emit_expr_address_of(self, expr: dict) -> str:
        """Emit address-of operator."""
        operand = self._emit_expr(expr.get('operand', {}))
        return f'@{operand}'
    
    def _emit_expr_math_intrinsic(self, expr: dict) -> str:
        """Emit mathematical intrinsic function."""
        name = expr.get('name', '')
        args = ', '.join(self._emit_expr(a) for a in expr.get('arguments', []))
        
        # Map to Pascal intrinsic
        pascal_name = INTRINSIC_MAP.get(name, {}).get('pascal', name.capitalize())
        return f'{pascal_name}({args})'
    
    def _emit_expr_set_length(self, expr: dict) -> str:
        """Emit SetLength call."""
        array = self._emit_expr(expr.get('array', {}))
        sizes = ', '.join(self._emit_expr(s) for s in expr.get('sizes', []))
        return f'SetLength({array}, {sizes})'
    
    def _emit_expr_length(self, expr: dict) -> str:
        """Emit Length function."""
        target = self._emit_expr(expr.get('target', {}))
        return f'Length({target})'
    
    def _emit_expr_high(self, expr: dict) -> str:
        """Emit High function."""
        target = self._emit_expr(expr.get('target', {}))
        return f'High({target})'
    
    def _emit_expr_low(self, expr: dict) -> str:
        """Emit Low function."""
        target = self._emit_expr(expr.get('target', {}))
        return f'Low({target})'
    
    # =========================================================================
    # Type Mapping
    # =========================================================================
    
    def _map_type(self, type_ref: dict) -> str:
        """Map IR type to Pascal type."""
        if not type_ref:
            return 'Integer'
        
        if isinstance(type_ref, str):
            return PASCAL_TYPE_MAP.get(type_ref, type_ref)
        
        kind = type_ref.get('kind', '')
        
        if kind == 'array_type':
            return self._map_array_type(type_ref)
        if kind == 'pascal_array_type':
            return self._map_pascal_array_type(type_ref)
        if kind == 'dynamic_array':
            return self._map_dynamic_array_type(type_ref)
        
        name = type_ref.get('name', '')
        return PASCAL_TYPE_MAP.get(name, name)
    
    def _map_array_type(self, type_ref: dict) -> str:
        """Map generic array type."""
        elem = type_ref.get('element_type', {})
        dims = type_ref.get('dimensions', [])
        
        elem_type = self._map_type(elem)
        
        if not dims:
            return f'array of {elem_type}'
        
        # Build dimension ranges
        ranges = []
        for dim in dims:
            lower = dim.get('lower')
            upper = dim.get('upper')
            
            if lower is not None and upper is not None:
                ranges.append(f'{self._emit_expr(lower)}..{self._emit_expr(upper)}')
            else:
                # Dynamic
                return f'array of {elem_type}'
        
        return f"array[{', '.join(ranges)}] of {elem_type}"
    
    def _map_pascal_array_type(self, type_ref: dict) -> str:
        """Map Pascal array type with index ranges."""
        elem = type_ref.get('element_type', {})
        index_ranges = type_ref.get('index_ranges', [])
        
        elem_type = self._map_type(elem)
        
        ranges = []
        for r in index_ranges:
            lower = self._emit_expr(r.get('lower', {}))
            upper = self._emit_expr(r.get('upper', {}))
            ranges.append(f'{lower}..{upper}')
        
        return f"array[{', '.join(ranges)}] of {elem_type}"
    
    def _map_dynamic_array_type(self, type_ref: dict) -> str:
        """Map dynamic array type."""
        elem = type_ref.get('element_type', {})
        dims = type_ref.get('dimensions', 1)
        
        elem_type = self._map_type(elem)
        
        result = elem_type
        for _ in range(dims):
            result = f'array of {result}'
        
        return result
    
    # =========================================================================
    # Manifest Generation
    # =========================================================================
    
    def _generate_manifest(self, ir: dict, code: str) -> dict:
        """Generate build manifest."""
        return {
            'schema': 'stunir.manifest.targets.v1',
            'generator': 'stunir.pascal.emitter',
            'epoch': int(time.time()),
            'ir_hash': compute_sha256(canonical_json(ir)),
            'output': {
                'hash': compute_sha256(code),
                'size': len(code),
                'format': 'pascal',
            }
        }
    
    def _emit_top_level(self, ir: dict) -> str:
        """Emit top-level construct from IR."""
        kind = ir.get('kind', '')
        
        if kind == 'subprogram':
            self._output = []
            self._emit_subprogram(ir)
            return self._get_output()
        elif kind in ('record_type', 'variant_record', 'class_type'):
            self._output = []
            self._line('type')
            self._indent_inc()
            self._emit_type_decl(ir)
            self._indent_dec()
            return self._get_output()
        else:
            return f'{{ Unknown top-level kind: {kind} }}'


# CLI entry point
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: pascal_emitter.py <ir.json> [output.pas]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    with open(input_path) as f:
        ir = json.load(f)
    
    emitter = PascalEmitter()
    result = emitter.emit(ir)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(result.code)
        print(f"Generated: {output_path}")
    else:
        print(result.code)
