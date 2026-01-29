#!/usr/bin/env python3
"""STUNIR x86 Assembly Target Emitter - Emit x86 assembly from IR.

This tool is part of the targets → assembly → x86 pipeline stage.
It converts STUNIR IR to x86 assembly code.

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


class X86Emitter(AssemblyEmitterBase):
    """x86 assembly code emitter."""
    
    ARCH = 'x86'
    
    def __init__(self, ir_data, out_dir, options=None):
        options = options or {}
        if options.get('64bit', False):
            self.ARCH = 'x86_64'
        super().__init__(ir_data, out_dir, options)
    
    def _emit_function_prologue(self, name):
        """Emit x86 function prologue."""
        if self.ARCH == 'x86_64':
            return [
                '    push rbp',
                '    mov rbp, rsp',
            ]
        else:
            return [
                '    push ebp',
                '    mov ebp, esp',
            ]
    
    def _emit_function_epilogue(self):
        """Emit x86 function epilogue."""
        if self.ARCH == 'x86_64':
            return [
                '    mov rsp, rbp',
                '    pop rbp',
                '    ret',
            ]
        else:
            return [
                '    mov esp, ebp',
                '    pop ebp',
                '    ret',
            ]
    
    def _emit_return(self, value=None):
        """Emit x86 return with value in eax/rax."""
        lines = []
        if value is not None:
            reg = 'rax' if self.ARCH == 'x86_64' else 'eax'
            try:
                # Numeric value
                int(value)
                lines.append(f'    mov {reg}, {value}')
            except (ValueError, TypeError):
                # Variable or expression
                lines.append(f'    ; return {value}')
                lines.append(f'    mov {reg}, {value}')
        return lines
    
    def _emit_statement(self, stmt):
        """Emit x86-specific statements."""
        lines = []
        stmt_type = stmt.get('type', 'nop')
        
        if stmt_type == 'return':
            value = stmt.get('value', '0')
            lines.extend(self._emit_return(value))
        elif stmt_type == 'call':
            func = stmt.get('func', 'noop')
            lines.append(self._emit_comment(f'call {func}'))
            lines.append(f'    call {func}')
        elif stmt_type == 'var_decl':
            # Stack allocation
            var_name = stmt.get('var_name', 'var')
            init = stmt.get('init', '0')
            lines.append(self._emit_comment(f'local {var_name} = {init}'))
            if self.ARCH == 'x86_64':
                lines.append('    sub rsp, 8')
                lines.append(f'    mov qword [rsp], {init}')
            else:
                lines.append('    sub esp, 4')
                lines.append(f'    mov dword [esp], {init}')
        elif stmt_type == 'assign':
            target = stmt.get('target', 'eax')
            value = stmt.get('value', '0')
            lines.append(f'    mov {target}, {value}')
        elif stmt_type == 'add':
            dest = stmt.get('dest', 'eax')
            left = stmt.get('left', '0')
            right = stmt.get('right', '0')
            lines.append(f'    mov {dest}, {left}')
            lines.append(f'    add {dest}, {right}')
        elif stmt_type == 'sub':
            dest = stmt.get('dest', 'eax')
            left = stmt.get('left', '0')
            right = stmt.get('right', '0')
            lines.append(f'    mov {dest}, {left}')
            lines.append(f'    sub {dest}, {right}')
        elif stmt_type == 'nop':
            lines.append('    nop')
        else:
            lines.append(self._emit_comment(f'TODO: {stmt_type}'))
            lines.append('    nop')
        
        return lines
    
    def _emit_build_script(self, module_name):
        """Generate x86 build script using NASM."""
        ext = self.config['extension']
        if self.ARCH == 'x86_64':
            return f"""#!/bin/bash
# STUNIR Generated x86-64 Build Script
set -e

# Assemble
nasm -f elf64 -o {module_name}.o {module_name}.{ext}

# Link
ld -o {module_name} {module_name}.o

echo "Built: {module_name} (x86-64)"
"""
        else:
            return f"""#!/bin/bash
# STUNIR Generated x86 Build Script
set -e

# Assemble
nasm -f elf32 -o {module_name}.o {module_name}.{ext}

# Link
ld -m elf_i386 -o {module_name} {module_name}.o

echo "Built: {module_name} (x86-32)"
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
    
    out_dir = args['output'] or 'x86_output'
    
    try:
        with open(args['input'], 'r') as f:
            ir_data = json.load(f)
        
        emitter = X86Emitter(ir_data, out_dir, {'64bit': args['64bit']})
        emitter.emit()
        
        manifest = emitter.emit_manifest()
        manifest_path = Path(out_dir) / 'manifest.json'
        manifest_path.write_text(canonical_json(manifest), encoding='utf-8')
        
        arch = 'x86-64' if args['64bit'] else 'x86-32'
        print(f"{arch} assembly emitted to {out_dir}/", file=sys.stderr)
        print(f"Files: {len(emitter.generated_files)}", file=sys.stderr)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
