#!/usr/bin/env python3
"""STUNIR Functional Emitter Base - Shared functionality for Haskell/OCaml.

This module provides common utilities and base classes for functional
language code generation.

Usage:
    from targets.functional.base import FunctionalEmitterBase
    
    class HaskellEmitter(FunctionalEmitterBase):
        DIALECT = 'haskell'
"""

import json
import hashlib
import time
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


def canonical_json(data: Any) -> str:
    """Generate RFC 8785 / JCS subset canonical JSON.
    
    Args:
        data: Data to serialize
        
    Returns:
        Canonical JSON string
    """
    return json.dumps(data, sort_keys=True, separators=(',', ':'))


def compute_sha256(content: Any) -> str:
    """Compute SHA256 hash of content.
    
    Args:
        content: String or bytes to hash
        
    Returns:
        Hex-encoded SHA256 hash
    """
    if isinstance(content, str):
        content = content.encode('utf-8')
    return hashlib.sha256(content).hexdigest()


class EmitterResult:
    """Result of code emission.
    
    Attributes:
        code: Generated code string
        manifest: Build manifest dictionary
        warnings: List of warning messages
    """
    
    def __init__(self, code: str = '', manifest: dict = None, warnings: list = None):
        self.code = code
        self.manifest = manifest or {}
        self.warnings = warnings or []


class FunctionalEmitterBase(ABC):
    """Base class for functional language emitters.
    
    Provides common functionality for Haskell and OCaml emitters.
    """
    
    DIALECT = 'functional'
    FILE_EXTENSION = '.txt'
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize emitter.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.indent_size = self.config.get('indent_size', 2)
        self.generated_files = []
        self._warnings = []
        self.epoch = int(time.time())
    
    def _indent(self, level: int) -> str:
        """Generate indentation string.
        
        Args:
            level: Indentation level
            
        Returns:
            Indentation string
        """
        return ' ' * (level * self.indent_size)
    
    def _write_file(self, path: Path, content: str) -> Path:
        """Write content to file.
        
        Args:
            path: Output path
            content: Content to write
            
        Returns:
            Full path of written file
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8', newline='\n')
        self.generated_files.append({
            'path': str(path),
            'sha256': compute_sha256(content),
            'size': len(content.encode('utf-8'))
        })
        return path
    
    def _generate_manifest(self, ir: dict, code: str) -> dict:
        """Generate build manifest.
        
        Args:
            ir: Input IR dictionary
            code: Generated code
            
        Returns:
            Manifest dictionary
        """
        code_hash = compute_sha256(code)
        ir_hash = compute_sha256(canonical_json(ir))
        
        manifest = {
            'schema': f'stunir.manifest.{self.DIALECT}.v1',
            'generator': f'stunir.{self.DIALECT}.emitter',
            'epoch': self.epoch,
            'ir_hash': ir_hash,
            'output': {
                'hash': code_hash,
                'size': len(code.encode('utf-8')),
                'format': self.DIALECT,
            },
            'manifest_hash': ''
        }
        
        # Compute manifest hash
        manifest_content = {k: v for k, v in manifest.items() if k != 'manifest_hash'}
        manifest['manifest_hash'] = compute_sha256(canonical_json(manifest_content))
        
        return manifest
    
    def warn(self, message: str):
        """Record a warning message.
        
        Args:
            message: Warning message
        """
        self._warnings.append(message)
    
    @abstractmethod
    def emit_module(self, module) -> str:
        """Emit complete module.
        
        Args:
            module: Module IR node
            
        Returns:
            Generated code string
        """
        pass


# Common type mappings
BASE_TYPE_MAP = {
    'int': 'Int',
    'i8': 'Int',
    'i16': 'Int',
    'i32': 'Int',
    'i64': 'Int',
    'float': 'Float',
    'f32': 'Float',
    'f64': 'Float',
    'bool': 'Bool',
    'string': 'String',
    'char': 'Char',
    'unit': '()',
    'void': '()',
}

# Common operators
BASE_OPERATOR_MAP = {
    '+': '+',
    '-': '-',
    '*': '*',
    '/': '/',
    '%': 'mod',
    '==': '==',
    '!=': '/=',
    '/=': '/=',
    '<': '<',
    '>': '>',
    '<=': '<=',
    '>=': '>=',
    '&&': '&&',
    '||': '||',
}
