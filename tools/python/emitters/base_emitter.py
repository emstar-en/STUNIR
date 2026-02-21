#!/usr/bin/env python3
"""STUNIR Base Emitter with Enhancement Context Support.

This module provides the base class for all STUNIR emitters with integrated
support for enhancement context. Emitters can access control flow graphs,
type information, semantic data, memory patterns, and optimization hints
through the enhancement context.

Part of Phase 1 (Foundation) of the STUNIR Enhancement-to-Emitter Integration.

Usage:
    from tools.emitters.base_emitter import BaseEmitter
    
    class PythonEmitter(BaseEmitter):
        def emit(self):
            # Access enhancement data
            cfg = self.get_function_cfg('main')
            loops = self.get_loops('main')
            var_type = self.lookup_variable_type('x')
            
            # Generate code...
            return self._generate_code()

Architecture:
    BaseEmitter receives an EnhancementContext that provides:
    - Control flow graphs for structured code generation
    - Type information for proper type mapping
    - Symbol tables for variable/function lookup
    - Memory patterns for appropriate memory management
    - Optimization hints for better code generation
"""

from __future__ import annotations

import json
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from tools.integration import EnhancementContext

logger = logging.getLogger(__name__)


def canonical_json(data: Any) -> str:
    """Generate RFC 8785 / JCS subset canonical JSON."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'))


def compute_sha256(content: Any) -> str:
    """Compute SHA256 hash of content."""
    if isinstance(content, str):
        content = content.encode('utf-8')
    return hashlib.sha256(content).hexdigest()


@dataclass
class EmitterConfig:
    """Configuration for emitters.
    
    Attributes:
        use_enhancements: Whether to use enhancement context data.
        emit_comments: Whether to emit comments in generated code.
        emit_debug_info: Whether to emit debug information.
        indent_size: Number of spaces for indentation.
        line_ending: Line ending style ('\\n' or '\\r\\n').
        max_line_length: Maximum line length (0 = no limit).
    """
    use_enhancements: bool = True
    emit_comments: bool = True
    emit_debug_info: bool = False
    indent_size: int = 4
    line_ending: str = '\n'
    max_line_length: int = 0


@dataclass
class EmittedFile:
    """Information about an emitted file.
    
    Attributes:
        path: Relative path of the file.
        sha256: SHA256 hash of the content.
        size: Size in bytes.
        line_count: Number of lines.
    """
    path: str
    sha256: str
    size: int
    line_count: int = 0


class BaseEmitter(ABC):
    """Base class for all STUNIR code emitters with enhancement support.
    
    This base class provides:
    - Enhancement context integration (optional, backward compatible)
    - Safe accessors for enhancement data
    - File writing utilities
    - Manifest generation
    
    Subclasses must implement the `emit()` method to generate code.
    
    Attributes:
        ir_data: The IR data to emit code from.
        out_dir: Output directory for generated files.
        enhancement_context: Optional EnhancementContext for intelligent code gen.
        config: Emitter configuration.
        generated_files: List of files generated during emission.
        epoch: Timestamp of emission start.
    
    Example:
        >>> class MyEmitter(BaseEmitter):
        ...     TARGET = 'python'
        ...     
        ...     def emit(self):
        ...         code = self._emit_module()
        ...         self._write_file('module.py', code)
        ...         return self.generate_manifest()
        >>> 
        >>> emitter = MyEmitter(ir_data, '/output', context=enhancement_context)
        >>> manifest = emitter.emit()
    """
    
    # Override in subclasses
    TARGET: str = 'generic'
    FILE_EXTENSION: str = 'txt'
    
    def __init__(
        self,
        ir_data: Dict[str, Any],
        out_dir: str,
        enhancement_context: Optional['EnhancementContext'] = None,
        config: Optional[EmitterConfig] = None
    ):
        """Initialize the emitter.
        
        Args:
            ir_data: The IR data to emit code from.
            out_dir: Output directory for generated files.
            enhancement_context: Optional EnhancementContext for enhancements.
            config: Optional emitter configuration.
        """
        self.ir_data = ir_data
        self.out_dir = Path(out_dir)
        self.enhancement_context = enhancement_context
        self.config = config or EmitterConfig()
        self.generated_files: List[EmittedFile] = []
        self.epoch = int(time.time())
        
        # Validate enhancement context if provided
        if self.enhancement_context is not None:
            self._validate_enhancement_context()
    
    def _validate_enhancement_context(self) -> None:
        """Validate the enhancement context."""
        if self.enhancement_context is None:
            return
        
        is_valid, errors = self.enhancement_context.validate()
        if not is_valid:
            logger.warning(f"Enhancement context validation warnings: {errors}")
    
    # -------------------------------------------------------------------------
    # Abstract Methods (must be implemented by subclasses)
    # -------------------------------------------------------------------------
    
    @abstractmethod
    def emit(self) -> Dict[str, Any]:
        """Emit code for the target language.
        
        Returns:
            Dictionary containing manifest data for emitted files.
        """
        pass
    
    # -------------------------------------------------------------------------
    # Enhancement Context Accessors (safe, null-checking)
    # -------------------------------------------------------------------------
    
    def has_enhancement_context(self) -> bool:
        """Check if enhancement context is available."""
        return self.enhancement_context is not None
    
    def get_function_cfg(self, func_name: str) -> Optional[Any]:
        """Get CFG for a function.
        
        Args:
            func_name: Name of the function.
            
        Returns:
            ControlFlowGraph or None if not available.
        """
        if self.enhancement_context is None:
            return None
        if not self.config.use_enhancements:
            return None
        return self.enhancement_context.get_function_cfg(func_name)
    
    def get_loops(self, func_name: str) -> List[Any]:
        """Get detected loops for a function.
        
        Args:
            func_name: Name of the function.
            
        Returns:
            List of LoopInfo objects, empty if not available.
        """
        if self.enhancement_context is None:
            return []
        if not self.config.use_enhancements:
            return []
        return self.enhancement_context.get_loops(func_name)
    
    def get_branches(self, func_name: str) -> List[Any]:
        """Get branch information for a function.
        
        Args:
            func_name: Name of the function.
            
        Returns:
            List of BranchInfo objects, empty if not available.
        """
        if self.enhancement_context is None:
            return []
        if not self.config.use_enhancements:
            return []
        return self.enhancement_context.get_branches(func_name)
    
    def lookup_variable(self, name: str) -> Optional[Any]:
        """Look up variable in symbol table.
        
        Args:
            name: Variable name.
            
        Returns:
            VariableInfo or None if not found.
        """
        if self.enhancement_context is None:
            return None
        if not self.config.use_enhancements:
            return None
        return self.enhancement_context.lookup_variable(name)
    
    def lookup_variable_type(self, name: str) -> Optional[str]:
        """Look up variable type from enhancement context.
        
        Args:
            name: Variable name.
            
        Returns:
            Type string or None if not found.
        """
        var_info = self.lookup_variable(name)
        if var_info is None:
            return None
        if hasattr(var_info, 'type'):
            return str(var_info.type)
        if isinstance(var_info, dict):
            return var_info.get('type')
        return None
    
    def lookup_function(self, name: str) -> Optional[Any]:
        """Look up function in symbol table.
        
        Args:
            name: Function name.
            
        Returns:
            FunctionInfo or None if not found.
        """
        if self.enhancement_context is None:
            return None
        if not self.config.use_enhancements:
            return None
        return self.enhancement_context.lookup_function(name)
    
    def get_expression_type(self, expr_id: str) -> Optional[Any]:
        """Get inferred type of an expression.
        
        Args:
            expr_id: Expression identifier.
            
        Returns:
            STUNIRType or None if not available.
        """
        if self.enhancement_context is None:
            return None
        if not self.config.use_enhancements:
            return None
        return self.enhancement_context.get_expression_type(expr_id)
    
    def get_memory_strategy(self) -> str:
        """Get recommended memory strategy for target language.
        
        Returns:
            Memory strategy string (e.g., 'gc', 'manual', 'ownership').
        """
        if self.enhancement_context is None:
            return self._default_memory_strategy()
        if not self.config.use_enhancements:
            return self._default_memory_strategy()
        strategy = self.enhancement_context.get_memory_strategy()
        return strategy if strategy else self._default_memory_strategy()
    
    def _default_memory_strategy(self) -> str:
        """Get default memory strategy based on target."""
        defaults = {
            'python': 'gc',
            'java': 'gc',
            'go': 'gc',
            'rust': 'ownership',
            'c': 'manual',
            'cpp': 'raii',
            'haskell': 'gc',
        }
        return defaults.get(self.TARGET.lower(), 'gc')
    
    def get_memory_pattern(self, var_name: str) -> Optional[str]:
        """Get memory pattern for a variable.
        
        Args:
            var_name: Variable name.
            
        Returns:
            Memory pattern ('stack', 'heap', 'static') or None.
        """
        if self.enhancement_context is None:
            return None
        if not self.config.use_enhancements:
            return None
        return self.enhancement_context.get_memory_pattern(var_name)
    
    def get_optimized_ir(self) -> Dict[str, Any]:
        """Get optimized IR if available, else original.
        
        Returns:
            Best available IR data.
        """
        if self.enhancement_context is None:
            return self.ir_data
        if not self.config.use_enhancements:
            return self.ir_data
        return self.enhancement_context.get_ir()
    
    def get_optimized_function(self, func_name: str) -> Optional[Dict[str, Any]]:
        """Get optimized version of a function.
        
        Args:
            func_name: Function name.
            
        Returns:
            Optimized function dict or None.
        """
        if self.enhancement_context is None:
            return None
        if not self.config.use_enhancements:
            return None
        return self.enhancement_context.get_optimized_function(func_name)
    
    def get_optimization_hint(self, key: str) -> Optional[Any]:
        """Get an optimization hint.
        
        Args:
            key: Hint key.
            
        Returns:
            Hint value or None.
        """
        if self.enhancement_context is None:
            return None
        if not self.config.use_enhancements:
            return None
        return self.enhancement_context.get_optimization_hint(key)
    
    # -------------------------------------------------------------------------
    # IR Accessors
    # -------------------------------------------------------------------------
    
    def get_functions(self) -> List[Dict[str, Any]]:
        """Get all functions from best available IR.
        
        Returns:
            List of function dictionaries.
        """
        ir = self.get_optimized_ir()
        return ir.get('ir_functions', [])
    
    def get_function(self, func_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific function from best available IR.
        
        Args:
            func_name: Function name.
            
        Returns:
            Function dictionary or None.
        """
        for func in self.get_functions():
            if func.get('name') == func_name:
                return func
        return None
    
    def get_module_name(self) -> str:
        """Get the module name from IR."""
        return self.ir_data.get('ir_module', 'module')
    
    def get_imports(self) -> List[str]:
        """Get imports from IR."""
        return self.ir_data.get('ir_imports', [])
    
    def get_exports(self) -> List[str]:
        """Get exports from IR."""
        return self.ir_data.get('ir_exports', [])
    
    # -------------------------------------------------------------------------
    # File Writing Utilities
    # -------------------------------------------------------------------------
    
    def _write_file(self, path: str, content: str) -> Path:
        """Write content to file, creating directories as needed.
        
        Args:
            path: Relative path within output directory.
            content: Content to write.
            
        Returns:
            Full path of written file.
        """
        full_path = self.out_dir / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Normalize line endings
        content = content.replace('\r\n', '\n')
        if self.config.line_ending != '\n':
            content = content.replace('\n', self.config.line_ending)
        
        full_path.write_text(content, encoding='utf-8')
        
        self.generated_files.append(EmittedFile(
            path=str(path),
            sha256=compute_sha256(content),
            size=len(content.encode('utf-8')),
            line_count=content.count('\n') + 1
        ))
        
        logger.debug(f"Wrote {path} ({len(content)} bytes)")
        return full_path
    
    def generate_manifest(self) -> Dict[str, Any]:
        """Generate manifest for emitted files.
        
        Returns:
            Manifest dictionary with file information.
        """
        return {
            'schema': f'stunir.emitter.{self.TARGET}.v1',
            'emitter_epoch': self.epoch,
            'target': self.TARGET,
            'module': self.get_module_name(),
            'files': [
                {
                    'path': f.path,
                    'sha256': f.sha256,
                    'size': f.size,
                    'lines': f.line_count
                }
                for f in self.generated_files
            ],
            'enhancement_context': {
                'available': self.has_enhancement_context(),
                'status': (
                    self.enhancement_context.get_status_summary()
                    if self.enhancement_context else None
                )
            },
            'manifest_hash': compute_sha256(canonical_json({
                'files': [f.sha256 for f in self.generated_files]
            }))
        }
    
    # -------------------------------------------------------------------------
    # Code Generation Helpers
    # -------------------------------------------------------------------------
    
    def _indent(self, code: str, level: int = 1) -> str:
        """Indent code by specified level.
        
        Args:
            code: Code to indent.
            level: Indentation level.
            
        Returns:
            Indented code.
        """
        indent = ' ' * (self.config.indent_size * level)
        lines = code.split('\n')
        return '\n'.join(indent + line if line.strip() else line for line in lines)
    
    def _emit_comment(self, text: str, style: str = 'line') -> str:
        """Emit a comment in appropriate style for target.
        
        Override in subclasses for language-specific comment syntax.
        
        Args:
            text: Comment text.
            style: 'line' or 'block'.
            
        Returns:
            Comment string.
        """
        if style == 'block':
            return f'/* {text} */'
        return f'// {text}'
    
    def _emit_header_comment(self) -> str:
        """Emit standard header comment for generated files.
        
        Returns:
            Header comment string.
        """
        lines = [
            f"Generated by STUNIR {self.TARGET} Emitter",
            f"Module: {self.get_module_name()}",
            f"Epoch: {self.epoch}",
        ]
        
        if self.has_enhancement_context():
            lines.append("Enhancement Context: Available")
            status = self.enhancement_context.get_status_summary()
            for key, val in status.items():
                lines.append(f"  - {key}: {val}")
        else:
            lines.append("Enhancement Context: Not available")
        
        return '\n'.join(self._emit_comment(line) for line in lines)


class EnhancedEmitterMixin:
    """Mixin class adding enhancement-aware code generation helpers.
    
    Use this mixin in emitters that want to generate code differently
    based on enhancement data availability.
    """
    
    def emit_with_cfg(self, func: Dict[str, Any]) -> str:
        """Emit function code using CFG if available.
        
        Args:
            func: Function dictionary.
            
        Returns:
            Generated code string.
        """
        func_name = func.get('name', 'unnamed')
        cfg = self.get_function_cfg(func_name)
        
        if cfg is not None:
            return self._emit_function_from_cfg(func, cfg)
        else:
            return self._emit_function_linear(func)
    
    def _emit_function_from_cfg(
        self,
        func: Dict[str, Any],
        cfg: Any
    ) -> str:
        """Emit function using control flow graph.
        
        Override in subclasses for language-specific CFG-based emission.
        
        Args:
            func: Function dictionary.
            cfg: Control flow graph.
            
        Returns:
            Generated code string.
        """
        # Default: fall back to linear emission
        return self._emit_function_linear(func)
    
    def _emit_function_linear(self, func: Dict[str, Any]) -> str:
        """Emit function linearly (without CFG).
        
        Override in subclasses for language-specific linear emission.
        
        Args:
            func: Function dictionary.
            
        Returns:
            Generated code string.
        """
        raise NotImplementedError("Subclass must implement _emit_function_linear")
    
    def emit_loop_structure(
        self,
        func_name: str,
        loop_info: Any,
        body_emitter: callable
    ) -> str:
        """Emit a loop structure based on loop info.
        
        Args:
            func_name: Function containing the loop.
            loop_info: LoopInfo object.
            body_emitter: Callable to emit loop body.
            
        Returns:
            Generated loop code.
        """
        raise NotImplementedError("Subclass must implement emit_loop_structure")


# Backward compatibility: alias for BaseEmitter
CodeEmitter = BaseEmitter
