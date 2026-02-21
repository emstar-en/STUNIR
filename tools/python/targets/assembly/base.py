#!/usr/bin/env python3
"""STUNIR Assembly Emitter Base - Shared functionality for x86/ARM targets.

This module provides common functionality for assembly code generation,
with architecture-specific customization for x86 and ARM.

Usage:
    from base import AssemblyEmitterBase

    class X86Emitter(AssemblyEmitterBase):
        ARCH = 'x86'
"""

import json
import hashlib
import time
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


def canonical_json(data: Any) -> str:
    """Generate RFC 8785 / JCS subset canonical JSON."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'))


def compute_sha256(content: Union[str, bytes]) -> str:
    """Compute SHA256 hash of content."""
    if isinstance(content, str):
        content = content.encode('utf-8')
    return hashlib.sha256(content).hexdigest()


class AssemblyEmitterBase(ABC):
    """Base class for assembly code emitters (x86, ARM, etc.)."""

    ARCH = 'generic'  # Override in subclass: 'x86', 'x86_64', 'arm', 'arm64'

    # Architecture-specific settings
    ARCH_CONFIG: Dict[str, Dict[str, Any]] = {
        'x86': {
            'syntax': 'intel',  # or 'att'
            'registers': ['eax', 'ebx', 'ecx', 'edx', 'esi', 'edi', 'esp', 'ebp'],
            'word_size': 32,
            'call_convention': 'cdecl',
            'data_section': '.data',
            'text_section': '.text',
            'extension': 'asm',
        },
        'x86_64': {
            'syntax': 'intel',
            'registers': ['rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'rsp', 'rbp', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15'],
            'word_size': 64,
            'call_convention': 'sysv_amd64',
            'data_section': '.data',
            'text_section': '.text',
            'extension': 'asm',
        },
        'arm': {
            'syntax': 'arm',
            'registers': ['r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'r11', 'r12', 'sp', 'lr', 'pc'],
            'word_size': 32,
            'call_convention': 'aapcs',
            'data_section': '.data',
            'text_section': '.text',
            'extension': 's',
        },
        'arm64': {
            'syntax': 'arm',
            'registers': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30', 'sp'],
            'word_size': 64,
            'call_convention': 'aapcs64',
            'data_section': '.data',
            'text_section': '.text',
            'extension': 's',
        },
    }

    def __init__(self, ir_data: Dict[str, Any], out_dir: Union[str, Path], options: Optional[Dict[str, Any]] = None) -> None:
        """Initialize assembly emitter.

        Args:
            ir_data: Dictionary containing IR data
            out_dir: Output directory path
            options: Optional dictionary of emitter options
        """
        self.ir_data = ir_data
        self.out_dir = Path(out_dir)
        self.options = options or {}
        self.config = self.ARCH_CONFIG.get(self.ARCH, self.ARCH_CONFIG['x86'])
        self.generated_files: List[Dict[str, Any]] = []
        self.epoch = int(time.time())

    def _write_file(self, path: Union[str, Path], content: str) -> Path:
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

    def _emit_comment(self, text: str) -> str:
        """Emit a comment in assembly syntax."""
        return f'; {text}'
    
    @abstractmethod
    def _emit_function_prologue(self, name: str) -> List[str]:
        """Emit function prologue (architecture-specific)."""
        pass

    @abstractmethod
    def _emit_function_epilogue(self) -> List[str]:
        """Emit function epilogue (architecture-specific)."""
        pass

    @abstractmethod
    def _emit_return(self, value: Optional[str] = None) -> List[str]:
        """Emit return instruction (architecture-specific)."""
        pass

    def _emit_function(self, func: Dict[str, Any]) -> List[str]:
        """Emit an assembly function."""
        name = func.get('name', 'unnamed')
        params = func.get('params', [])
        body = func.get('body', [])

        lines: List[str] = []
        lines.append(self._emit_comment(f'Function: {name}'))
        lines.append(f'global {name}')
        lines.append(f'{name}:')
        lines.extend(self._emit_function_prologue(name))

        # Function body
        for stmt in body:
            if isinstance(stmt, dict):
                stmt_lines = self._emit_statement(stmt)
                lines.extend(stmt_lines)
            elif isinstance(stmt, str):
                lines.append(f'    {stmt}')

        lines.extend(self._emit_function_epilogue())
        lines.append('')
        return lines

    def _emit_statement(self, stmt: Dict[str, Any]) -> List[str]:
        """Emit a statement as assembly. Override for architecture-specific."""
        lines: List[str] = []
        stmt_type = stmt.get('type', 'nop')

        if stmt_type == 'return':
            value = stmt.get('value', '0')
            lines.extend(self._emit_return(value))
        elif stmt_type == 'call':
            func = stmt.get('func', 'noop')
            lines.append(self._emit_comment(f'call {func}'))
            lines.append(f'    call {func}')
        elif stmt_type == 'nop':
            lines.append('    nop')
        else:
            lines.append(self._emit_comment(f'Unknown statement type: {stmt_type}'))

        return lines

    def emit(self) -> str:
        """Emit all assembly files."""
        module_name = self.ir_data.get('ir_module', self.ir_data.get('module', 'module'))
        functions = self.ir_data.get('ir_functions', self.ir_data.get('functions', []))

        ext = self.config['extension']

        # Assembly header
        lines = [
            self._emit_comment(f'STUNIR Generated {self.ARCH.upper()} Assembly'),
            self._emit_comment(f'Module: {module_name}'),
            self._emit_comment(f'Syntax: {self.config["syntax"]}'),
            '',
            f'section {self.config["data_section"]}',
            '',
            f'section {self.config["text_section"]}',
            '',
        ]

        # Function implementations
        for func in functions:
            lines.extend(self._emit_function(func))

        asm_content = '\n'.join(lines)

        # Write assembly file
        self._write_file(f'{module_name}.{ext}', asm_content)

        # Generate build script
        build_script = self._emit_build_script(module_name)
        self._write_file('build.sh', build_script)

        # Generate README
        readme = self._emit_readme(module_name)
        self._write_file('README.md', readme)

        return asm_content
    
    @abstractmethod
    def _emit_build_script(self, module_name: str) -> str:
        """Generate build script (architecture-specific)."""
        pass

    def _emit_readme(self, module_name: str) -> str:
        """Generate README for the assembly project."""
        return f"""# {module_name} ({self.ARCH.upper()} Assembly)

Generated by STUNIR Target Emitter.

## Architecture

- **Target:** {self.ARCH.upper()}
- **Syntax:** {self.config['syntax']}
- **Word Size:** {self.config['word_size']} bits
- **Calling Convention:** {self.config['call_convention']}

## Build

```bash
chmod +x build.sh
./build.sh
```

## Files

- `{module_name}.{self.config['extension']}` - Assembly source
- `build.sh` - Build script

## Schema

stunir.target.{self.ARCH}.v1
"""

    def emit_manifest(self) -> Dict[str, Any]:
        """Generate target manifest."""
        return {
            'schema': f'stunir.target.{self.ARCH}.manifest.v1',
            'arch': self.ARCH,
            'epoch': self.epoch,
            'files': sorted(self.generated_files, key=lambda f: f['path']),
            'file_count': len(self.generated_files)
        }

    def emit_receipt(self) -> Dict[str, Any]:
        """Generate target receipt."""
        manifest = self.emit_manifest()
        manifest_json = canonical_json(manifest)
        return {
            'schema': f'stunir.target.{self.ARCH}.receipt.v1',
            'arch': self.ARCH,
            'epoch': self.epoch,
            'manifest_sha256': compute_sha256(manifest_json),
            'file_count': len(self.generated_files)
        }
