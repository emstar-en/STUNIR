#!/usr/bin/env python3
"""STUNIR Bytecode Emitter - Emit stack-based bytecode.

This tool is part of the targets → bytecode pipeline stage.
It converts STUNIR IR to a stack-based bytecode format.

Usage:
    emitter.py <ir.json> --output=<dir>
    emitter.py --help
"""

import json
import hashlib
import time
import sys
import struct
from pathlib import Path


def canonical_json(data):
    """Generate RFC 8785 / JCS subset canonical JSON."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'))


def compute_sha256(content):
    """Compute SHA256 hash of content."""
    if isinstance(content, str):
        content = content.encode('utf-8')
    return hashlib.sha256(content).hexdigest()


# Bytecode opcodes
class Opcode:
    NOP = 0x00
    CONST_I32 = 0x01
    CONST_I64 = 0x02
    CONST_F32 = 0x03
    CONST_F64 = 0x04
    LOAD = 0x10
    STORE = 0x11
    ADD = 0x20
    SUB = 0x21
    MUL = 0x22
    DIV = 0x23
    CALL = 0x30
    RET = 0x31
    JMP = 0x40
    JMP_IF = 0x41
    CMP_EQ = 0x50
    CMP_LT = 0x51
    CMP_GT = 0x52
    HALT = 0xFF


class BytecodeEmitter:
    """Emitter for stack-based bytecode."""
    
    def __init__(self, ir_data, out_dir, options=None):
        """Initialize bytecode emitter."""
        self.ir_data = ir_data
        self.out_dir = Path(out_dir)
        self.options = options or {}
        self.generated_files = []
        self.epoch = int(time.time())
        self.constants = []  # Constant pool
        self.functions = {}  # Function table
    
    def _write_file(self, path, content, binary=False):
        """Write content to file."""
        full_path = self.out_dir / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        if binary:
            full_path.write_bytes(content)
            size = len(content)
        else:
            full_path.write_text(content, encoding='utf-8', newline='\n')
            size = len(content.encode('utf-8'))
        self.generated_files.append({
            'path': str(path),
            'sha256': compute_sha256(content),
            'size': size
        })
        return full_path
    
    def _add_constant(self, value):
        """Add value to constant pool, return index."""
        if value not in self.constants:
            self.constants.append(value)
        return self.constants.index(value)
    
    def _emit_instruction(self, stmt, var_map):
        """Convert IR statement to bytecode instructions."""
        bytecode = []
        if isinstance(stmt, dict):
            stmt_type = stmt.get('type', 'nop')
            if stmt_type == 'var_decl':
                var_name = stmt.get('var_name', 'v0')
                init = stmt.get('init', '0')
                var_map[var_name] = len(var_map)
                bytecode.append(Opcode.CONST_I32)
                bytecode.extend(self._encode_i32(int(init) if init.lstrip('-').isdigit() else 0))
                bytecode.append(Opcode.STORE)
                bytecode.extend(self._encode_i32(var_map[var_name]))
            elif stmt_type == 'return':
                value = stmt.get('value', '0')
                if value.lstrip('-').isdigit():
                    bytecode.append(Opcode.CONST_I32)
                    bytecode.extend(self._encode_i32(int(value)))
                else:
                    bytecode.append(Opcode.LOAD)
                    bytecode.extend(self._encode_i32(var_map.get(value, 0)))
                bytecode.append(Opcode.RET)
            elif stmt_type == 'assign':
                target = stmt.get('target', 'v0')
                value = stmt.get('value', '0')
                if value.lstrip('-').isdigit():
                    bytecode.append(Opcode.CONST_I32)
                    bytecode.extend(self._encode_i32(int(value)))
                else:
                    bytecode.append(Opcode.LOAD)
                    bytecode.extend(self._encode_i32(var_map.get(value, 0)))
                bytecode.append(Opcode.STORE)
                bytecode.extend(self._encode_i32(var_map.get(target, 0)))
            elif stmt_type in ('add', 'sub', 'mul'):
                left = stmt.get('left', '0')
                right = stmt.get('right', '0')
                dest = stmt.get('dest', 'v0')
                # Push left
                if left.lstrip('-').isdigit():
                    bytecode.append(Opcode.CONST_I32)
                    bytecode.extend(self._encode_i32(int(left)))
                else:
                    bytecode.append(Opcode.LOAD)
                    bytecode.extend(self._encode_i32(var_map.get(left, 0)))
                # Push right
                if right.lstrip('-').isdigit():
                    bytecode.append(Opcode.CONST_I32)
                    bytecode.extend(self._encode_i32(int(right)))
                else:
                    bytecode.append(Opcode.LOAD)
                    bytecode.extend(self._encode_i32(var_map.get(right, 0)))
                # Operation
                op_map = {'add': Opcode.ADD, 'sub': Opcode.SUB, 'mul': Opcode.MUL}
                bytecode.append(op_map.get(stmt_type, Opcode.ADD))
                # Store result
                bytecode.append(Opcode.STORE)
                bytecode.extend(self._encode_i32(var_map.get(dest, 0)))
            elif stmt_type == 'call':
                func = stmt.get('func', 'noop')
                func_idx = self._add_constant(func)
                bytecode.append(Opcode.CALL)
                bytecode.extend(self._encode_i32(func_idx))
            else:
                bytecode.append(Opcode.NOP)
        else:
            bytecode.append(Opcode.NOP)
        return bytecode
    
    def _encode_i32(self, value):
        """Encode 32-bit integer as bytes (little-endian)."""
        return list(struct.pack('<i', value))
    
    def _emit_function(self, func):
        """Emit bytecode for a function."""
        body = func.get('body', [])
        params = func.get('params', [])
        
        var_map = {}
        for i, p in enumerate(params):
            name = p.get('name', f'arg{i}') if isinstance(p, dict) else f'arg{i}'
            var_map[name] = i
        
        bytecode = []
        for stmt in body:
            bytecode.extend(self._emit_instruction(stmt, var_map))
        
        if not bytecode or bytecode[-1] != Opcode.RET:
            bytecode.append(Opcode.CONST_I32)
            bytecode.extend(self._encode_i32(0))
            bytecode.append(Opcode.RET)
        
        return bytes(bytecode)
    
    def emit(self):
        """Emit bytecode files."""
        module_name = self.ir_data.get('ir_module', self.ir_data.get('module', 'module'))
        functions = self.ir_data.get('ir_functions', self.ir_data.get('functions', []))
        
        # Build function table
        func_bytecodes = {}
        for func in functions:
            name = func.get('name', 'unnamed')
            func_bytecodes[name] = self._emit_function(func)
            self.functions[name] = len(self.functions)
        
        # Human-readable assembly
        asm_lines = [
            '; STUNIR Bytecode Assembly',
            f'; Module: {module_name}',
            f'; Schema: stunir.bytecode.v1',
            f'; Epoch: {self.epoch}',
            '',
            '; Constants Pool',
        ]
        for i, c in enumerate(self.constants):
            asm_lines.append(f'  const.{i}: {repr(c)}')
        asm_lines.append('')
        asm_lines.append('; Functions')
        for name, bc in func_bytecodes.items():
            asm_lines.append(f'.func {name}:')
            asm_lines.append(f'  ; {len(bc)} bytes')
            asm_lines.append(f'  ; hex: {bc.hex()}')
            asm_lines.append('')
        
        asm_content = '\n'.join(asm_lines)
        self._write_file(f'{module_name}.bc.asm', asm_content)
        
        # Binary bytecode
        all_bytecode = b''
        for name, bc in func_bytecodes.items():
            all_bytecode += bc
        self._write_file(f'{module_name}.bc', all_bytecode, binary=True)
        
        # JSON representation
        bc_json = {
            'schema': 'stunir.bytecode.v1',
            'module': module_name,
            'epoch': self.epoch,
            'constants': self.constants,
            'functions': {k: v.hex() for k, v in func_bytecodes.items()}
        }
        self._write_file(f'{module_name}.bc.json', canonical_json(bc_json))
        
        # README
        self._write_file('README.md', self._emit_readme(module_name, len(functions)))
        
        return asm_content
    
    def _emit_readme(self, module_name, func_count):
        """Generate README."""
        return f"""# {module_name} (Bytecode)

Generated by STUNIR Bytecode Emitter.

## Files

- `{module_name}.bc` - Binary bytecode
- `{module_name}.bc.asm` - Human-readable assembly
- `{module_name}.bc.json` - JSON representation

## Bytecode Format

Stack-based VM bytecode with the following opcodes:
- `0x01` CONST_I32 - Push 32-bit integer
- `0x10` LOAD - Load variable
- `0x11` STORE - Store variable
- `0x20` ADD - Add top two values
- `0x21` SUB - Subtract
- `0x22` MUL - Multiply
- `0x30` CALL - Call function
- `0x31` RET - Return

## Statistics

- Functions: {func_count}
- Constants: {len(self.constants)}
- Epoch: {self.epoch}

## Schema

stunir.bytecode.v1
"""
    
    def emit_manifest(self):
        """Generate target manifest."""
        return {
            'schema': 'stunir.target.bytecode.manifest.v1',
            'epoch': self.epoch,
            'files': sorted(self.generated_files, key=lambda f: f['path']),
            'file_count': len(self.generated_files)
        }
    
    def emit_receipt(self):
        """Generate target receipt."""
        manifest = self.emit_manifest()
        manifest_json = canonical_json(manifest)
        return {
            'schema': 'stunir.target.bytecode.receipt.v1',
            'epoch': self.epoch,
            'manifest_sha256': compute_sha256(manifest_json),
            'file_count': len(self.generated_files)
        }


def main():
    args = {'output': None, 'input': None}
    for arg in sys.argv[1:]:
        if arg.startswith('--output='):
            args['output'] = arg.split('=', 1)[1]
        elif arg == '--help':
            print(__doc__)
            sys.exit(0)
        elif not arg.startswith('--'):
            args['input'] = arg
    
    if not args['input']:
        print(f"Usage: {sys.argv[0]} <ir.json> --output=<dir>", file=sys.stderr)
        sys.exit(1)
    
    out_dir = args['output'] or 'bytecode_output'
    
    try:
        with open(args['input'], 'r') as f:
            ir_data = json.load(f)
        
        emitter = BytecodeEmitter(ir_data, out_dir)
        emitter.emit()
        
        manifest = emitter.emit_manifest()
        manifest_path = Path(out_dir) / 'manifest.json'
        manifest_path.write_text(canonical_json(manifest), encoding='utf-8')
        
        print(f"Bytecode emitted to {out_dir}/", file=sys.stderr)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
