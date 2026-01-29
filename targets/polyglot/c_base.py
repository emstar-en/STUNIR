#!/usr/bin/env python3
"""STUNIR C Emitter Base - Shared functionality for C89/C99 targets.

This module provides common functionality for C code generation,
with dialect-specific customization for C89 and C99 standards.

Usage:
    from c_base import CEmitterBase
    
    class C89Emitter(CEmitterBase):
        DIALECT = 'c89'
"""

import json
import hashlib
import time
from pathlib import Path
from abc import ABC, abstractmethod


def canonical_json(data):
    """Generate RFC 8785 / JCS subset canonical JSON."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'))


def compute_sha256(content):
    """Compute SHA256 hash of content."""
    if isinstance(content, str):
        content = content.encode('utf-8')
    return hashlib.sha256(content).hexdigest()


class CEmitterBase(ABC):
    """Base class for C code emitters (C89, C99, etc.)."""
    
    DIALECT = 'c'  # Override in subclass: 'c89' or 'c99'
    
    # Dialect-specific settings
    DIALECT_FEATURES = {
        'c89': {
            'inline_keyword': '',  # No inline in C89
            'comment_style': '/* */',  # Block comments only
            'declarations_at_start': True,
            'stdbool': False,
            'stdint': False,
        },
        'c99': {
            'inline_keyword': 'inline',
            'comment_style': '//',  # Line comments allowed
            'declarations_at_start': False,
            'stdbool': True,
            'stdint': True,
        }
    }
    
    def __init__(self, ir_data, out_dir, options=None):
        """Initialize C emitter.
        
        Args:
            ir_data: Dictionary containing IR data
            out_dir: Output directory path
            options: Optional dictionary of emitter options
        """
        self.ir_data = ir_data
        self.out_dir = Path(out_dir)
        self.options = options or {}
        self.features = self.DIALECT_FEATURES.get(self.DIALECT, self.DIALECT_FEATURES['c99'])
        self.generated_files = []
        self.epoch = int(time.time())
    
    def _write_file(self, path, content):
        """Write content to file, creating directories as needed."""
        full_path = self.out_dir / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding='utf-8', newline='\n')
        self.generated_files.append({
            'path': str(path),
            'sha256': compute_sha256(content),
            'size': len(content.encode('utf-8'))
        })
        return full_path
    
    def _emit_comment(self, text, multiline=False):
        """Emit a comment in dialect-appropriate style."""
        if self.features['comment_style'] == '//' and not multiline:
            return f'// {text}'
        else:
            return f'/* {text} */'
    
    def _emit_function(self, func):
        """Emit a C function."""
        name = func.get('name', 'unnamed')
        params = func.get('params', [])
        returns = func.get('returns', 'void')
        body = func.get('body', [])
        
        # Map IR types to C types
        c_return = self._map_type(returns)
        c_params = ', '.join([self._map_param(p) for p in params]) or 'void'
        
        lines = []
        lines.append(f'{c_return} {name}({c_params}) {{')
        
        # Declarations at start for C89
        if self.features['declarations_at_start']:
            # Collect all variable declarations first
            for stmt in body:
                if isinstance(stmt, dict) and stmt.get('type') == 'var_decl':
                    var_type = self._map_type(stmt.get('var_type', 'int'))
                    var_name = stmt.get('var_name', 'var')
                    lines.append(f'    {var_type} {var_name};')
            lines.append('')
        
        # Function body
        for stmt in body:
            if isinstance(stmt, dict):
                stmt_code = self._emit_statement(stmt)
                if stmt_code:
                    lines.append(f'    {stmt_code}')
            elif isinstance(stmt, str):
                lines.append(f'    {stmt}')
        
        # Return statement if needed
        if c_return != 'void' and not any('return' in str(s) for s in body):
            lines.append(f'    return ({c_return})0;')
        
        lines.append('}')
        return '\n'.join(lines)
    
    def _emit_statement(self, stmt):
        """Emit a statement."""
        stmt_type = stmt.get('type', 'expr')
        if stmt_type == 'var_decl':
            if not self.features['declarations_at_start']:
                var_type = self._map_type(stmt.get('var_type', 'int'))
                var_name = stmt.get('var_name', 'var')
                init = stmt.get('init', '')
                if init:
                    return f'{var_type} {var_name} = {init};'
                return f'{var_type} {var_name};'
            return None  # Already emitted at start
        elif stmt_type == 'return':
            value = stmt.get('value', '0')
            return f'return {value};'
        elif stmt_type == 'call':
            func = stmt.get('func', 'noop')
            args = ', '.join(stmt.get('args', []))
            return f'{func}({args});'
        elif stmt_type == 'assign':
            target = stmt.get('target', 'x')
            value = stmt.get('value', '0')
            return f'{target} = {value};'
        return str(stmt)
    
    def _map_type(self, ir_type):
        """Map IR type to C type."""
        type_map = {
            'void': 'void',
            'int': 'int',
            'i32': 'int',
            'i64': 'long long',
            'u32': 'unsigned int',
            'u64': 'unsigned long long',
            'f32': 'float',
            'f64': 'double',
            'bool': 'int' if not self.features['stdbool'] else 'bool',
            'string': 'const char*',
            'ptr': 'void*',
        }
        return type_map.get(ir_type, 'int')
    
    def _map_param(self, param):
        """Map IR parameter to C parameter."""
        if isinstance(param, dict):
            ptype = self._map_type(param.get('type', 'int'))
            pname = param.get('name', 'arg')
            return f'{ptype} {pname}'
        return f'int {param}'
    
    def emit(self):
        """Emit all C code files."""
        module_name = self.ir_data.get('ir_module', self.ir_data.get('module', 'module'))
        functions = self.ir_data.get('ir_functions', self.ir_data.get('functions', []))
        types = self.ir_data.get('ir_types', self.ir_data.get('types', []))
        
        # Header file
        header_lines = [
            self._emit_comment(f'STUNIR Generated {self.DIALECT.upper()} Header'),
            self._emit_comment(f'Module: {module_name}'),
            '',
            f'#ifndef {module_name.upper()}_H',
            f'#define {module_name.upper()}_H',
            '',
        ]
        
        # Include standard headers for C99
        if self.features['stdint']:
            header_lines.append('#include <stdint.h>')
        if self.features['stdbool']:
            header_lines.append('#include <stdbool.h>')
        if self.features['stdint'] or self.features['stdbool']:
            header_lines.append('')
        
        # Type definitions
        for typedef in types:
            if isinstance(typedef, dict):
                tname = typedef.get('name', 'T')
                tbase = self._map_type(typedef.get('base', 'int'))
                header_lines.append(f'typedef {tbase} {tname};')
        if types:
            header_lines.append('')
        
        # Function declarations
        for func in functions:
            name = func.get('name', 'unnamed')
            params = func.get('params', [])
            returns = func.get('returns', 'void')
            c_return = self._map_type(returns)
            c_params = ', '.join([self._map_param(p) for p in params]) or 'void'
            header_lines.append(f'{c_return} {name}({c_params});')
        
        header_lines.extend(['', f'#endif {self._emit_comment(f"{module_name.upper()}_H")}', ''])
        header_content = '\n'.join(header_lines)
        
        # Source file
        source_lines = [
            self._emit_comment(f'STUNIR Generated {self.DIALECT.upper()} Source'),
            self._emit_comment(f'Module: {module_name}'),
            '',
            f'#include "{module_name}.h"',
            '',
        ]
        
        # Function implementations
        for func in functions:
            source_lines.append(self._emit_function(func))
            source_lines.append('')
        
        source_content = '\n'.join(source_lines)
        
        # Write files
        self._write_file(f'{module_name}.h', header_content)
        self._write_file(f'{module_name}.c', source_content)
        
        # Generate Makefile
        makefile = self._emit_makefile(module_name)
        self._write_file('Makefile', makefile)
        
        # Generate README
        readme = self._emit_readme(module_name)
        self._write_file('README.md', readme)
        
        return source_content
    
    def _emit_makefile(self, module_name):
        """Generate a Makefile for the C project."""
        cc_flags = '-ansi -pedantic' if self.DIALECT == 'c89' else '-std=c99'
        return f"""# STUNIR Generated Makefile ({self.DIALECT.upper()})
CC = gcc
CFLAGS = {cc_flags} -Wall -Wextra
TARGET = {module_name}
SRCS = {module_name}.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
\t$(CC) $(CFLAGS) -o $@ $^

%.o: %.c
\t$(CC) $(CFLAGS) -c -o $@ $<

clean:
\trm -f $(TARGET) $(OBJS)

.PHONY: all clean
"""
    
    def _emit_readme(self, module_name):
        """Generate README for the C project."""
        return f"""# {module_name} ({self.DIALECT.upper()})

Generated by STUNIR Target Emitter.

## Standard

{self.DIALECT.upper()} ({"ANSI C" if self.DIALECT == 'c89' else "ISO C99"})

## Build

```bash
make
```

## Files

- `{module_name}.h` - Header file
- `{module_name}.c` - Implementation
- `Makefile` - Build configuration

## Schema

stunir.target.{self.DIALECT}.v1
"""
    
    def emit_manifest(self):
        """Generate target manifest."""
        return {
            'schema': f'stunir.target.{self.DIALECT}.manifest.v1',
            'dialect': self.DIALECT,
            'epoch': self.epoch,
            'files': sorted(self.generated_files, key=lambda f: f['path']),
            'file_count': len(self.generated_files)
        }
    
    def emit_receipt(self):
        """Generate target receipt."""
        manifest = self.emit_manifest()
        manifest_json = canonical_json(manifest)
        return {
            'schema': f'stunir.target.{self.DIALECT}.receipt.v1',
            'dialect': self.DIALECT,
            'epoch': self.epoch,
            'manifest_sha256': compute_sha256(manifest_json),
            'file_count': len(self.generated_files)
        }
