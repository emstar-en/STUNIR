#!/usr/bin/env python3
"""STUNIR ASM/IR Emitter - Emit IR in assembly-like format.

This tool is part of the targets → asm → ir pipeline stage.
It converts STUNIR IR to a deterministic assembly-like IR format.

Usage:
    emitter.py <ir.json> --output=<dir>
    emitter.py --help
"""

import json
import hashlib
import time
import sys
import os
from pathlib import Path


def canonical_json(data):
    """Generate RFC 8785 / JCS subset canonical JSON."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'))


def compute_sha256(content):
    """Compute SHA256 hash of content."""
    if isinstance(content, str):
        content = content.encode('utf-8')
    return hashlib.sha256(content).hexdigest()


class AsmIrEmitter:
    """Emitter for assembly-like IR format."""
    
    def __init__(self, ir_data, out_dir, options=None):
        """Initialize ASM/IR emitter.
        
        Args:
            ir_data: Dictionary containing IR data
            out_dir: Output directory path
            options: Optional dictionary of emitter options
        """
        self.ir_data = ir_data
        self.out_dir = Path(out_dir)
        self.options = options or {}
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
    
    def _emit_ir_instruction(self, stmt):
        """Convert IR statement to ASM-IR format."""
        if isinstance(stmt, dict):
            stmt_type = stmt.get('type', 'nop')
            if stmt_type == 'var_decl':
                var_type = stmt.get('var_type', 'i32')
                var_name = stmt.get('var_name', 'v0')
                init = stmt.get('init', '0')
                return f'    local.{var_type} {var_name} = {init}'
            elif stmt_type == 'return':
                value = stmt.get('value', '0')
                return f'    return {value}'
            elif stmt_type == 'call':
                func = stmt.get('func', 'noop')
                args = ', '.join(stmt.get('args', []))
                return f'    call {func}({args})'
            elif stmt_type == 'assign':
                target = stmt.get('target', 'v0')
                value = stmt.get('value', '0')
                return f'    store {target}, {value}'
            elif stmt_type == 'load':
                target = stmt.get('target', 'v0')
                source = stmt.get('source', 'v1')
                return f'    load {target}, {source}'
            elif stmt_type == 'add':
                dest = stmt.get('dest', 'v0')
                left = stmt.get('left', '0')
                right = stmt.get('right', '0')
                return f'    add {dest}, {left}, {right}'
            elif stmt_type == 'sub':
                dest = stmt.get('dest', 'v0')
                left = stmt.get('left', '0')
                right = stmt.get('right', '0')
                return f'    sub {dest}, {left}, {right}'
            elif stmt_type == 'mul':
                dest = stmt.get('dest', 'v0')
                left = stmt.get('left', '0')
                right = stmt.get('right', '0')
                return f'    mul {dest}, {left}, {right}'
            elif stmt_type == 'cmp':
                op = stmt.get('op', 'eq')
                left = stmt.get('left', '0')
                right = stmt.get('right', '0')
                return f'    cmp.{op} {left}, {right}'
            elif stmt_type == 'br':
                target = stmt.get('target', 'label0')
                return f'    br {target}'
            elif stmt_type == 'br_if':
                cond = stmt.get('cond', 'true')
                target = stmt.get('target', 'label0')
                return f'    br_if {cond}, {target}'
            elif stmt_type == 'label':
                name = stmt.get('name', 'label0')
                return f'{name}:'
            else:
                return f'    ; unknown: {stmt_type}'
        elif isinstance(stmt, str):
            return f'    ; {stmt}'
        return '    nop'
    
    def _emit_function(self, func):
        """Emit a function in ASM-IR format."""
        name = func.get('name', 'unnamed')
        params = func.get('params', [])
        returns = func.get('returns', 'void')
        body = func.get('body', [])
        
        lines = []
        lines.append(f'; Function: {name}')
        
        # Function signature
        param_str = ', '.join([
            f"{p.get('type', 'i32')} {p.get('name', f'arg{i}')}"
            if isinstance(p, dict) else f'i32 {p}'
            for i, p in enumerate(params)
        ])
        lines.append(f'.func {name}({param_str}) -> {returns}')
        lines.append('{')
        
        # Function body
        for stmt in body:
            lines.append(self._emit_ir_instruction(stmt))
        
        # Implicit return if needed
        if not any(isinstance(s, dict) and s.get('type') == 'return' for s in body):
            if returns != 'void':
                lines.append('    return 0')
            else:
                lines.append('    return')
        
        lines.append('}')
        lines.append('')
        return '\n'.join(lines)
    
    def emit(self):
        """Emit ASM/IR files."""
        module_name = self.ir_data.get('ir_module', self.ir_data.get('module', 'module'))
        functions = self.ir_data.get('ir_functions', self.ir_data.get('functions', []))
        types = self.ir_data.get('ir_types', self.ir_data.get('types', []))
        imports = self.ir_data.get('ir_imports', self.ir_data.get('imports', []))
        exports = self.ir_data.get('ir_exports', self.ir_data.get('exports', []))
        
        # ASM-IR header
        lines = [
            '; STUNIR ASM-IR Format',
            f'; Module: {module_name}',
            f'; Schema: stunir.asm.ir.v1',
            f'; Epoch: {self.epoch}',
            '',
        ]
        
        # Module directive
        lines.append(f'.module {module_name}')
        lines.append('')
        
        # Imports
        if imports:
            lines.append('; Imports')
            for imp in imports:
                if isinstance(imp, dict):
                    lines.append(f".import {imp.get('module', 'ext')}::{imp.get('name', 'func')}")
                else:
                    lines.append(f'.import {imp}')
            lines.append('')
        
        # Exports
        if exports:
            lines.append('; Exports')
            for exp in exports:
                if isinstance(exp, dict):
                    lines.append(f".export {exp.get('name', 'func')}")
                else:
                    lines.append(f'.export {exp}')
            lines.append('')
        
        # Type definitions
        if types:
            lines.append('; Types')
            for typedef in types:
                if isinstance(typedef, dict):
                    lines.append(f".type {typedef.get('name', 'T')} = {typedef.get('base', 'i32')}")
                else:
                    lines.append(f'.type {typedef}')
            lines.append('')
        
        # Functions
        lines.append('; Functions')
        for func in functions:
            lines.append(self._emit_function(func))
        
        ir_content = '\n'.join(lines)
        
        # Write main IR file
        self._write_file(f'{module_name}.ir', ir_content)
        
        # Write canonical JSON representation
        ir_json = {
            'schema': 'stunir.asm.ir.v1',
            'module': module_name,
            'epoch': self.epoch,
            'functions': functions,
            'types': types,
            'imports': imports,
            'exports': exports
        }
        self._write_file(f'{module_name}.ir.json', canonical_json(ir_json))
        
        # Generate README
        readme = self._emit_readme(module_name, len(functions))
        self._write_file('README.md', readme)
        
        return ir_content
    
    def _emit_readme(self, module_name, func_count):
        """Generate README for the ASM/IR output."""
        return f"""# {module_name} (ASM/IR)

Generated by STUNIR ASM/IR Emitter.

## Format

STUNIR ASM-IR is a low-level, human-readable intermediate representation
suitable for debugging and verification.

## Files

- `{module_name}.ir` - Human-readable ASM-IR format
- `{module_name}.ir.json` - Canonical JSON representation

## Statistics

- Functions: {func_count}
- Epoch: {self.epoch}

## Schema

stunir.asm.ir.v1
"""
    
    def emit_manifest(self):
        """Generate target manifest."""
        return {
            'schema': 'stunir.target.asm.ir.manifest.v1',
            'epoch': self.epoch,
            'files': sorted(self.generated_files, key=lambda f: f['path']),
            'file_count': len(self.generated_files)
        }
    
    def emit_receipt(self):
        """Generate target receipt."""
        manifest = self.emit_manifest()
        manifest_json = canonical_json(manifest)
        return {
            'schema': 'stunir.target.asm.ir.receipt.v1',
            'epoch': self.epoch,
            'manifest_sha256': compute_sha256(manifest_json),
            'file_count': len(self.generated_files)
        }


def parse_args(argv):
    """Parse command line arguments."""
    args = {'output': None, 'input': None}
    
    for arg in argv[1:]:
        if arg.startswith('--output='):
            args['output'] = arg.split('=', 1)[1]
        elif arg == '--help':
            print(__doc__)
            sys.exit(0)
        elif not arg.startswith('--'):
            args['input'] = arg
    
    return args


def main():
    args = parse_args(sys.argv)
    
    if not args['input']:
        print(f"Usage: {sys.argv[0]} <ir.json> --output=<dir>", file=sys.stderr)
        sys.exit(1)
    
    out_dir = args['output'] or 'asm_ir_output'
    
    try:
        with open(args['input'], 'r') as f:
            ir_data = json.load(f)
        
        emitter = AsmIrEmitter(ir_data, out_dir)
        content = emitter.emit()
        
        # Write manifest
        manifest = emitter.emit_manifest()
        manifest_path = Path(out_dir) / 'manifest.json'
        manifest_path.write_text(canonical_json(manifest), encoding='utf-8')
        
        print(f"ASM/IR emitted to {out_dir}/", file=sys.stderr)
        print(f"Files: {len(emitter.generated_files)}", file=sys.stderr)
        
    except json.JSONDecodeError as e:
        print(f"JSON Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
