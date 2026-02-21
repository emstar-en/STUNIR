#!/usr/bin/env python3
"""STUNIR ARM Assembly Target Emitter - Emit ARM assembly from IR.

This tool is part of the targets → assembly → arm pipeline stage.
It converts STUNIR IR to ARM assembly code.

Usage:
    emitter.py <ir.json> --output=<dir> [--64bit]
    emitter.py --help
"""

import sys
import json
from pathlib import Path

# Add parent directory to path for base import
sys.path.insert(0, str(Path(__file__).parent.parent))
from base import AssemblyEmitterBase, canonical_json


class ARMEmitter(AssemblyEmitterBase):
    """ARM assembly code emitter."""
    
    ARCH = 'arm'
    
    def __init__(self, ir_data, out_dir, options=None):
        options = options or {}
        if options.get('64bit', False):
            self.ARCH = 'arm64'
        super().__init__(ir_data, out_dir, options)
    
    def _emit_comment(self, text):
        """Emit ARM-style comment."""
        return f'@ {text}'
    
    def _emit_function_prologue(self, name):
        """Emit ARM function prologue."""
        if self.ARCH == 'arm64':
            return [
                '    stp x29, x30, [sp, #-16]!',
                '    mov x29, sp',
            ]
        else:
            return [
                '    push {fp, lr}',
                '    mov fp, sp',
            ]
    
    def _emit_function_epilogue(self):
        """Emit ARM function epilogue."""
        if self.ARCH == 'arm64':
            return [
                '    mov sp, x29',
                '    ldp x29, x30, [sp], #16',
                '    ret',
            ]
        else:
            return [
                '    mov sp, fp',
                '    pop {fp, lr}',
                '    bx lr',
            ]
    
    def _emit_return(self, value=None):
        """Emit ARM return with value in r0/x0."""
        lines = []
        if value is not None:
            reg = 'x0' if self.ARCH == 'arm64' else 'r0'
            try:
                int(value)
                lines.append(f'    mov {reg}, #{value}')
            except (ValueError, TypeError):
                lines.append(self._emit_comment(f'return {value}'))
                lines.append(f'    mov {reg}, {value}')
        return lines
    
    def _emit_statement(self, stmt):
        """Emit ARM-specific statements."""
        lines = []
        stmt_type = stmt.get('type', 'nop')
        
        if stmt_type == 'return':
            value = stmt.get('value', '0')
            lines.extend(self._emit_return(value))
        elif stmt_type == 'call':
            func = stmt.get('func', 'noop')
            lines.append(self._emit_comment(f'call {func}'))
            lines.append(f'    bl {func}')
        elif stmt_type == 'var_decl':
            var_name = stmt.get('var_name', 'var')
            init = stmt.get('init', '0')
            lines.append(self._emit_comment(f'local {var_name} = {init}'))
            if self.ARCH == 'arm64':
                lines.append('    sub sp, sp, #16')
                try:
                    int(init)
                    lines.append(f'    mov x0, #{init}')
                except (ValueError, TypeError):
                    lines.append(f'    mov x0, {init}')
                lines.append('    str x0, [sp]')
            else:
                lines.append('    sub sp, sp, #4')
                try:
                    int(init)
                    lines.append(f'    mov r0, #{init}')
                except (ValueError, TypeError):
                    lines.append(f'    mov r0, {init}')
                lines.append('    str r0, [sp]')
        elif stmt_type == 'assign':
            target = stmt.get('target', 'r0')
            value = stmt.get('value', '0')
            try:
                int(value)
                lines.append(f'    mov {target}, #{value}')
            except (ValueError, TypeError):
                lines.append(f'    mov {target}, {value}')
        elif stmt_type == 'add':
            dest = stmt.get('dest', 'r0')
            left = stmt.get('left', 'r0')
            right = stmt.get('right', '0')
            try:
                int(right)
                lines.append(f'    add {dest}, {left}, #{right}')
            except (ValueError, TypeError):
                lines.append(f'    add {dest}, {left}, {right}')
        elif stmt_type == 'sub':
            dest = stmt.get('dest', 'r0')
            left = stmt.get('left', 'r0')
            right = stmt.get('right', '0')
            try:
                int(right)
                lines.append(f'    sub {dest}, {left}, #{right}')
            except (ValueError, TypeError):
                lines.append(f'    sub {dest}, {left}, {right}')
        elif stmt_type == 'nop':
            lines.append('    nop')
        else:
            lines.append(self._emit_comment(f'TODO: {stmt_type}'))
            lines.append('    nop')
        
        return lines
    
    def _emit_build_script(self, module_name):
        """Generate ARM build script."""
        ext = self.config['extension']
        if self.ARCH == 'arm64':
            return f"""#!/bin/bash
# STUNIR Generated ARM64 Build Script
set -e

# Assemble (requires cross-compiler or ARM64 system)
aarch64-linux-gnu-as -o {module_name}.o {module_name}.{ext}

# Link
aarch64-linux-gnu-ld -o {module_name} {module_name}.o

echo "Built: {module_name} (ARM64)"
"""
        else:
            return f"""#!/bin/bash
# STUNIR Generated ARM32 Build Script
set -e

# Assemble (requires cross-compiler or ARM system)
arm-linux-gnueabi-as -o {module_name}.o {module_name}.{ext}

# Link
arm-linux-gnueabi-ld -o {module_name} {module_name}.o

echo "Built: {module_name} (ARM32)"
"""


def parse_args(argv):
    """Parse command line arguments."""
    args = {'output': None, 'input': None, '64bit': False}
    for arg in argv[1:]:
        if arg.startswith('--output='):
            args['output'] = arg.split('=', 1)[1]
        elif arg == '--64bit':
            args['64bit'] = True
        elif arg == '--help':
            print(__doc__)
            sys.exit(0)
        elif not arg.startswith('--'):
            args['input'] = arg
    return args


def main():
    args = parse_args(sys.argv)
    if not args['input']:
        print(f"Usage: {sys.argv[0]} <ir.json> --output=<dir> [--64bit]", file=sys.stderr)
        sys.exit(1)
    
    out_dir = args['output'] or 'arm_output'
    
    try:
        with open(args['input'], 'r') as f:
            ir_data = json.load(f)
        
        emitter = ARMEmitter(ir_data, out_dir, {'64bit': args['64bit']})
        emitter.emit()
        
        manifest = emitter.emit_manifest()
        manifest_path = Path(out_dir) / 'manifest.json'
        manifest_path.write_text(canonical_json(manifest), encoding='utf-8')
        
        arch = 'ARM64' if args['64bit'] else 'ARM32'
        print(f"{arch} assembly emitted to {out_dir}/", file=sys.stderr)
        print(f"Files: {len(emitter.generated_files)}", file=sys.stderr)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
