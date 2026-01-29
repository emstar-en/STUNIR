#!/usr/bin/env python3
"""STUNIR WASM Emitter - Emit WebAssembly Text Format.

This tool is part of the targets → wasm pipeline stage.
It converts STUNIR IR to WebAssembly Text Format (WAT).

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


class WasmEmitter:
    """Emitter for WebAssembly Text Format (WAT)."""
    
    # Type mappings from STUNIR IR to WASM
    TYPE_MAP = {
        'i32': 'i32', 'i64': 'i64', 'f32': 'f32', 'f64': 'f64',
        'int': 'i32', 'long': 'i64', 'float': 'f32', 'double': 'f64',
        'void': None, 'bool': 'i32', 'byte': 'i32', 'char': 'i32'
    }
    
    def __init__(self, ir_data, out_dir, options=None):
        """Initialize WASM emitter."""
        self.ir_data = ir_data
        self.out_dir = Path(out_dir)
        self.options = options or {}
        self.generated_files = []
        self.epoch = int(time.time())
        self.memory_pages = options.get('memory_pages', 1) if options else 1
    
    def _write_file(self, path, content):
        """Write content to file."""
        full_path = self.out_dir / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding='utf-8', newline='\n')
        self.generated_files.append({
            'path': str(path),
            'sha256': compute_sha256(content),
            'size': len(content.encode('utf-8'))
        })
        return full_path
    
    def _map_type(self, ir_type):
        """Map IR type to WASM type."""
        return self.TYPE_MAP.get(ir_type, 'i32')
    
    def _emit_instruction(self, stmt):
        """Convert IR statement to WASM instruction."""
        if isinstance(stmt, dict):
            stmt_type = stmt.get('type', 'nop')
            if stmt_type == 'var_decl':
                var_type = self._map_type(stmt.get('var_type', 'i32'))
                return f'    (local ${stmt.get("var_name", "v0")} {var_type})'
            elif stmt_type == 'return':
                value = stmt.get('value', '0')
                if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                    return f'    (i32.const {value})'
                return f'    (local.get ${value})'
            elif stmt_type == 'assign':
                target = stmt.get('target', 'v0')
                value = stmt.get('value', '0')
                if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                    return f'    (local.set ${target} (i32.const {value}))'
                return f'    (local.set ${target} (local.get ${value}))'
            elif stmt_type == 'add':
                return f'    (i32.add (local.get ${stmt.get("left", "0")}) (local.get ${stmt.get("right", "0")}))'
            elif stmt_type == 'sub':
                return f'    (i32.sub (local.get ${stmt.get("left", "0")}) (local.get ${stmt.get("right", "0")}))'
            elif stmt_type == 'mul':
                return f'    (i32.mul (local.get ${stmt.get("left", "0")}) (local.get ${stmt.get("right", "0")}))'
            elif stmt_type == 'call':
                func = stmt.get('func', 'noop')
                args = stmt.get('args', [])
                arg_str = ' '.join([f'(local.get ${a})' for a in args])
                return f'    (call ${func} {arg_str})'
            else:
                return f'    ;; {stmt_type}: not implemented'
        return '    nop'
    
    def _emit_function(self, func):
        """Emit a function in WAT format."""
        name = func.get('name', 'unnamed')
        params = func.get('params', [])
        returns = func.get('returns', 'void')
        body = func.get('body', [])
        
        lines = []
        
        # Function signature
        param_str = ' '.join([
            f'(param ${p.get("name", f"arg{i}")} {self._map_type(p.get("type", "i32"))})'
            if isinstance(p, dict) else f'(param $arg{i} i32)'
            for i, p in enumerate(params)
        ])
        
        result_str = ''
        wasm_ret = self._map_type(returns)
        if wasm_ret:
            result_str = f' (result {wasm_ret})'
        
        lines.append(f'  (func ${name} {param_str}{result_str}')
        
        # Function body
        for stmt in body:
            lines.append(self._emit_instruction(stmt))
        
        lines.append('  )')
        return '\n'.join(lines)
    
    def emit(self):
        """Emit WAT files."""
        module_name = self.ir_data.get('ir_module', self.ir_data.get('module', 'module'))
        functions = self.ir_data.get('ir_functions', self.ir_data.get('functions', []))
        exports = self.ir_data.get('ir_exports', self.ir_data.get('exports', []))
        
        lines = [
            ';; STUNIR WebAssembly Module',
            f';; Module: {module_name}',
            f';; Schema: stunir.wasm.v1',
            f';; Epoch: {self.epoch}',
            '',
            '(module',
            f'  ;; Memory: {self.memory_pages} pages (64KB each)',
            f'  (memory (export "memory") {self.memory_pages})',
            ''
        ]
        
        # Functions
        for func in functions:
            lines.append(self._emit_function(func))
        
        # Exports
        for func in functions:
            name = func.get('name', 'unnamed')
            if not exports or name in exports or any(e.get('name') == name for e in exports if isinstance(e, dict)):
                lines.append(f'  (export "{name}" (func ${name}))')
        
        lines.append(')')
        
        wat_content = '\n'.join(lines)
        
        # Write WAT file
        self._write_file(f'{module_name}.wat', wat_content)
        
        # Write build script
        build_script = self._emit_build_script(module_name)
        self._write_file('build.sh', build_script)
        
        # Write README
        readme = self._emit_readme(module_name, len(functions))
        self._write_file('README.md', readme)
        
        return wat_content
    
    def _emit_build_script(self, module_name):
        """Generate build script for WASM compilation."""
        return f"""#!/bin/bash
# STUNIR WASM Build Script
# Compiles WAT to WASM binary

set -e

WAT_FILE="{module_name}.wat"
WASM_FILE="{module_name}.wasm"

# Check for wat2wasm (from WABT toolkit)
if command -v wat2wasm &> /dev/null; then
    echo "Compiling $WAT_FILE -> $WASM_FILE"
    wat2wasm "$WAT_FILE" -o "$WASM_FILE"
    echo "Generated: $WASM_FILE"
else
    echo "Error: wat2wasm not found. Install WABT toolkit."
    echo "  macOS: brew install wabt"
    echo "  Linux: apt install wabt"
    exit 1
fi
"""
    
    def _emit_readme(self, module_name, func_count):
        """Generate README."""
        return f"""# {module_name} (WebAssembly)

Generated by STUNIR WASM Emitter.

## Files

- `{module_name}.wat` - WebAssembly Text Format
- `build.sh` - Compilation script

## Build

```bash
chmod +x build.sh
./build.sh
```

Requires WABT toolkit (`wat2wasm`).

## Usage

```javascript
const wasmModule = await WebAssembly.instantiateStreaming(
  fetch('{module_name}.wasm')
);
```

## Statistics

- Functions: {func_count}
- Memory Pages: {self.memory_pages}
- Epoch: {self.epoch}

## Schema

stunir.wasm.v1
"""
    
    def emit_manifest(self):
        """Generate target manifest."""
        return {
            'schema': 'stunir.target.wasm.manifest.v1',
            'epoch': self.epoch,
            'files': sorted(self.generated_files, key=lambda f: f['path']),
            'file_count': len(self.generated_files)
        }
    
    def emit_receipt(self):
        """Generate target receipt."""
        manifest = self.emit_manifest()
        manifest_json = canonical_json(manifest)
        return {
            'schema': 'stunir.target.wasm.receipt.v1',
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
    
    out_dir = args['output'] or 'wasm_output'
    
    try:
        with open(args['input'], 'r') as f:
            ir_data = json.load(f)
        
        emitter = WasmEmitter(ir_data, out_dir)
        content = emitter.emit()
        
        # Write manifest
        manifest = emitter.emit_manifest()
        manifest_path = Path(out_dir) / 'manifest.json'
        manifest_path.write_text(canonical_json(manifest), encoding='utf-8')
        
        print(f"WASM emitted to {out_dir}/", file=sys.stderr)
        print(f"Files: {len(emitter.generated_files)}", file=sys.stderr)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
