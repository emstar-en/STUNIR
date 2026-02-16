#!/usr/bin/env python3
"""STUNIR Rust Target Emitter - Emit Rust code from IR.

This tool is part of the targets → polyglot → rust pipeline stage.
It converts STUNIR IR to idiomatic Rust code.

Usage:
    emitter.py <ir.json> --output=<dir>
    emitter.py --help
"""

import json
import hashlib
import time
import sys
from pathlib import Path


def canonical_json(data):
    """Generate RFC 8785 / JCS subset canonical JSON."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'))


def compute_sha256(content):
    """Compute SHA256 hash of content."""
    if isinstance(content, str):
        content = content.encode('utf-8')
    return hashlib.sha256(content).hexdigest()


class RustEmitter:
    """Emitter for Rust target code."""
    
    def __init__(self, ir_data, out_dir, options=None):
        """Initialize Rust emitter.
        
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
    
    def _map_type(self, ir_type):
        """Map IR type to Rust type."""
        type_map = {
            'void': '()',
            'int': 'i32',
            'i32': 'i32',
            'i64': 'i64',
            'u32': 'u32',
            'u64': 'u64',
            'f32': 'f32',
            'f64': 'f64',
            'bool': 'bool',
            'string': 'String',
            'str': '&str',
            'ptr': '*mut u8',
        }
        return type_map.get(ir_type, 'i32')
    
    def _map_param(self, param):
        """Map IR parameter to Rust parameter."""
        if isinstance(param, dict):
            ptype = self._map_type(param.get('type', 'i32'))
            pname = param.get('name', 'arg')
            return f'{pname}: {ptype}'
        return f'{param}: i32'
    
    def _emit_statement(self, stmt, indent=1):
        """Emit a statement in Rust."""
        ind = '    ' * indent
        if isinstance(stmt, dict):
            stmt_type = stmt.get('type', 'expr')
            if stmt_type == 'var_decl':
                var_type = self._map_type(stmt.get('var_type', 'i32'))
                var_name = stmt.get('var_name', 'var')
                init = stmt.get('init', '0')
                mutable = 'mut ' if stmt.get('mutable', True) else ''
                return f'{ind}let {mutable}{var_name}: {var_type} = {init};'
            elif stmt_type == 'return':
                value = stmt.get('value', '()')
                return f'{ind}return {value};'
            elif stmt_type == 'call':
                func = stmt.get('func', 'noop')
                args = ', '.join(stmt.get('args', []))
                return f'{ind}{func}({args});'
            elif stmt_type == 'assign':
                target = stmt.get('target', 'x')
                value = stmt.get('value', '0')
                return f'{ind}{target} = {value};'
            elif stmt_type == 'if':
                cond = stmt.get('cond', 'true')
                then_body = stmt.get('then', [])
                else_body = stmt.get('else', [])
                lines = [f'{ind}if {cond} {{']
                for s in then_body:
                    lines.append(self._emit_statement(s, indent + 1))
                if else_body:
                    lines.append(f'{ind}}} else {{')
                    for s in else_body:
                        lines.append(self._emit_statement(s, indent + 1))
                lines.append(f'{ind}}}')
                return '\n'.join(lines)
            elif stmt_type == 'loop':
                body = stmt.get('body', [])
                lines = [f'{ind}loop {{']
                for s in body:
                    lines.append(self._emit_statement(s, indent + 1))
                lines.append(f'{ind}}}')
                return '\n'.join(lines)
            elif stmt_type == 'break':
                return f'{ind}break;'
            elif stmt_type == 'continue':
                return f'{ind}continue;'
        return f'{ind}// {stmt}'
    
    def _emit_function(self, func):
        """Emit a Rust function."""
        name = func.get('name', 'unnamed')
        params = func.get('params', [])
        returns = func.get('returns', 'void')
        body = func.get('body', [])
        is_public = func.get('public', True)
        
        rust_return = self._map_type(returns)
        rust_params = ', '.join([self._map_param(p) for p in params])
        pub = 'pub ' if is_public else ''
        
        lines = []
        lines.append(f'{pub}fn {name}({rust_params}) -> {rust_return} {{')
        
        # Function body
        for stmt in body:
            lines.append(self._emit_statement(stmt))
        
        # Default return if needed
        if rust_return != '()' and not any(isinstance(s, dict) and s.get('type') == 'return' for s in body):
            lines.append(f'    0')
        
        lines.append('}')
        return '\n'.join(lines)
    
    def emit(self):
        """Emit all Rust files."""
        module_name = self.ir_data.get('ir_module', self.ir_data.get('module', 'stunir_module'))
        functions = self.ir_data.get('ir_functions', self.ir_data.get('functions', []))
        types = self.ir_data.get('ir_types', self.ir_data.get('types', []))
        
        # Cargo.toml with deterministic ordering
        cargo_toml = f'''[package]
edition = "2021"
name = "{module_name.replace('-', '_')}"
version = "0.1.0"

[dependencies]
'''
        self._write_file('Cargo.toml', cargo_toml)
        
        # Main library file
        lib_lines = [
            '//! STUNIR Generated Rust Module',
            f'//! Module: {module_name}',
            '//! Schema: stunir.target.rust.v1',
            '',
            '#![allow(unused_variables)]',
            '#![allow(dead_code)]',
            '',
        ]
        
        # Type definitions
        for typedef in types:
            if isinstance(typedef, dict):
                tname = typedef.get('name', 'T')
                tbase = self._map_type(typedef.get('base', 'i32'))
                lib_lines.append(f'pub type {tname} = {tbase};')
        if types:
            lib_lines.append('')
        
        # Functions
        for func in functions:
            lib_lines.append(self._emit_function(func))
            lib_lines.append('')
        
        lib_content = '\n'.join(lib_lines)
        
        # Write to src/lib.rs
        self._write_file('src/lib.rs', lib_content)
        
        # Main entry point
        main_content = f'''//! STUNIR Generated Rust Binary
//! Module: {module_name}

mod lib;

fn main() {{
    println!("STUNIR Rust Target: {module_name}");
}}
'''
        self._write_file('src/main.rs', main_content)
        
        # README
        readme = self._emit_readme(module_name, len(functions))
        self._write_file('README.md', readme)
        
        return lib_content
    
    def _emit_readme(self, module_name, func_count):
        """Generate README for Rust project."""
        return f"""# {module_name} (Rust)

Generated by STUNIR Rust Target Emitter.

## Build

```bash
cargo build --release
```

## Test

```bash
cargo test
```

## Files

- `Cargo.toml` - Package manifest
- `src/lib.rs` - Library module
- `src/main.rs` - Binary entry point

## Statistics

- Functions: {func_count}
- Epoch: {self.epoch}

## Schema

stunir.target.rust.v1
"""
    
    def emit_manifest(self):
        """Generate target manifest."""
        return {
            'schema': 'stunir.target.rust.manifest.v1',
            'epoch': self.epoch,
            'files': sorted(self.generated_files, key=lambda f: f['path']),
            'file_count': len(self.generated_files)
        }
    
    def emit_receipt(self):
        """Generate target receipt."""
        manifest = self.emit_manifest()
        manifest_json = canonical_json(manifest)
        return {
            'schema': 'stunir.target.rust.receipt.v1',
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
    
    out_dir = args['output'] or 'rust_output'
    
    try:
        with open(args['input'], 'r') as f:
            ir_data = json.load(f)
        
        emitter = RustEmitter(ir_data, out_dir)
        emitter.emit()
        
        manifest = emitter.emit_manifest()
        manifest_path = Path(out_dir) / 'manifest.json'
        manifest_path.write_text(canonical_json(manifest), encoding='utf-8')
        
        print(f"Rust code emitted to {out_dir}/", file=sys.stderr)
        print(f"Files: {len(emitter.generated_files)}", file=sys.stderr)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
