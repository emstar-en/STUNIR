#!/usr/bin/env python3
"""STUNIR Normalized Emitter Base Module.

Provides base class for emitters that use normalized IR.
This ensures all emitters receive simplified, canonical IR
before emission, reducing code duplication and complexity.

Part of Phase E: Emitters Refactor.
"""

import json
import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import normalizer
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ir_normalizer import IRNormalizer, NormalizerConfig, ValidationResult


def canonical_json(data: Any) -> str:
    """Generate RFC 8785 / JCS subset canonical JSON."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def compute_sha256(content: Any) -> str:
    """Compute SHA256 hash of content."""
    if isinstance(content, str):
        content = content.encode('utf-8')
    return hashlib.sha256(content).hexdigest()


@dataclass
class EmitterResult:
    """Result of code emission."""
    code: str
    files: Dict[str, str] = field(default_factory=dict)
    manifest: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validation: Optional[ValidationResult] = None
    
    @property
    def success(self) -> bool:
        return len(self.errors) == 0


@dataclass
class NormalizedEmitterConfig:
    """Configuration for normalized emitters."""
    target_dir: Path
    module_prefix: str = "stunir"
    emit_comments: bool = True
    emit_type_hints: bool = True
    line_width: int = 80
    indent_size: int = 2
    # Normalization options
    normalize_before_emit: bool = True
    validate_before_emit: bool = True
    # Pass-specific options
    lower_switch: bool = True
    lower_for: bool = True
    lower_break_continue: bool = True
    lower_try_catch: bool = True
    canonicalize_types: bool = True
    fold_constants: bool = True


class NormalizedEmitterBase(ABC):
    """Base class for emitters using normalized IR.
    
    This class ensures that all emitters receive simplified IR
    by running normalization passes before emission. This reduces
    the complexity of individual emitters.
    
    Benefits:
    - Emitters don't need to handle switch statements (lowered to if/else)
    - Emitters don't need to handle for loops (lowered to while)
    - Emitters don't need to handle break/continue (lowered to flags)
    - Emitters receive canonical type names
    - Emitters receive constant-folded expressions
    
    Usage:
        class MyEmitter(NormalizedEmitterBase):
            DIALECT = "my_language"
            FILE_EXTENSION = ".my"
            
            def emit(self, ir: Dict[str, Any]) -> EmitterResult:
                # IR is already normalized
                return self._emit_module(ir)
    """
    
    DIALECT: str = "unknown"  # Override in subclasses
    FILE_EXTENSION: str = ".txt"  # Override in subclasses
    
    # Common type mappings that most languages need
    BASE_TYPE_MAP: Dict[str, str] = {
        # Integer types
        'i8': 'int8', 'int8': 'int8',
        'i16': 'int16', 'int16': 'int16',
        'i32': 'int', 'int': 'int', 'int32': 'int',
        'i64': 'long', 'long': 'long', 'int64': 'long',
        'u8': 'uint8', 'uint8': 'uint8',
        'u16': 'uint16', 'uint16': 'uint16',
        'u32': 'uint32', 'uint32': 'uint32',
        'u64': 'uint64', 'uint64': 'uint64',
        # Float types
        'f32': 'float', 'float': 'float', 'float32': 'float',
        'f64': 'double', 'double': 'double', 'float64': 'double',
        # Boolean
        'bool': 'bool', 'boolean': 'bool',
        # String
        'string': 'string', 'str': 'string',
        # Void
        'void': 'void',
        # Any
        'any': 'any',
    }
    
    def __init__(self, config: NormalizedEmitterConfig):
        """Initialize the emitter.
        
        Args:
            config: Emitter configuration.
        """
        self.config = config
        self.generated_files: List[Dict[str, Any]] = []
        self.epoch = int(time.time())
        self._indent_level = 0
        self._normalizer: Optional[IRNormalizer] = None
    
    def _get_normalizer(self) -> IRNormalizer:
        """Get or create the IR normalizer."""
        if self._normalizer is None:
            norm_config = NormalizerConfig(
                lower_switch=self.config.lower_switch,
                lower_for=self.config.lower_for,
                lower_break_continue=self.config.lower_break_continue,
                lower_try_catch=self.config.lower_try_catch,
                canonicalize_types=self.config.canonicalize_types,
                fold_constants=self.config.fold_constants,
                validate_before_emit=self.config.validate_before_emit,
            )
            self._normalizer = IRNormalizer(norm_config)
        return self._normalizer
    
    def normalize_ir(self, ir: Dict[str, Any]) -> Tuple[Dict[str, Any], ValidationResult]:
        """Normalize IR before emission.
        
        Args:
            ir: Raw IR data.
            
        Returns:
            Tuple of (normalized_ir, validation_result).
        """
        normalizer = self._get_normalizer()
        
        # Normalize the IR
        normalized = normalizer.normalize_module(ir)
        
        # Validate for emission
        validation = ValidationResult()
        if self.config.validate_before_emit:
            for func in normalized.get("functions", []):
                func_validation = normalizer.validate_for_emission(func)
                validation.errors.extend(func_validation.errors)
                validation.warnings.extend(func_validation.warnings)
                if not func_validation.valid:
                    validation.valid = False
        
        return normalized, validation
    
    @abstractmethod
    def emit(self, ir: Dict[str, Any]) -> EmitterResult:
        """Emit code from IR.
        
        This method receives ALREADY NORMALIZED IR.
        Subclasses should implement emission logic assuming:
        - No switch statements (lowered to if/else)
        - No for loops (lowered to while)
        - No break/continue (lowered to flags)
        - Canonical type names
        - Constant-folded expressions
        
        Args:
            ir: STUNIR IR dictionary (already normalized).
            
        Returns:
            EmitterResult with generated code and metadata.
        """
        pass
    
    def emit_with_normalization(self, ir: Dict[str, Any]) -> EmitterResult:
        """Emit code from IR with normalization.
        
        This is the main entry point for emission.
        It normalizes the IR and then calls emit().
        
        Args:
            ir: Raw IR data.
            
        Returns:
            EmitterResult with generated code and metadata.
        """
        result = EmitterResult(code="", validation=None)
        
        try:
            # Normalize IR
            if self.config.normalize_before_emit:
                normalized, validation = self.normalize_ir(ir)
                result.validation = validation
                
                # Check for validation errors
                if not validation.valid:
                    result.errors.extend(validation.errors)
                    result.warnings.extend(validation.warnings)
                    # Continue with emission even with warnings
            else:
                normalized = ir
            
            # Emit code
            emit_result = self.emit(normalized)
            result.code = emit_result.code
            result.files = emit_result.files
            result.manifest = emit_result.manifest
            result.errors.extend(emit_result.errors)
            result.warnings.extend(emit_result.warnings)
            
        except Exception as e:
            result.errors.append(f"Emission error: {str(e)}")
        
        return result
    
    def _map_type(self, ir_type: str) -> str:
        """Map IR type to target language type.
        
        Override in subclasses for language-specific types.
        """
        return self.BASE_TYPE_MAP.get(ir_type, ir_type)
    
    def _indent(self, text: str = "") -> str:
        """Return indented text."""
        return " " * (self._indent_level * self.config.indent_size) + text
    
    def _push_indent(self):
        """Increase indentation level."""
        self._indent_level += 1
    
    def _pop_indent(self):
        """Decrease indentation level."""
        self._indent_level = max(0, self._indent_level - 1)
    
    def _emit_comment(self, text: str, style: str = "//") -> str:
        """Emit a comment line."""
        if not self.config.emit_comments:
            return ""
        return f"{style} {text}"
    
    def _emit_header(self, module_name: str) -> str:
        """Emit file header comment."""
        lines = [
            f"// Generated by STUNIR {self.DIALECT.title()} Emitter",
            f"// Module: {module_name}",
            f"// Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        return "\n".join(lines)
    
    def _write_file(self, path: str, content: str) -> Path:
        """Write content to file and track it."""
        full_path = self.config.target_dir / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding='utf-8', newline='\n')
        
        self.generated_files.append({
            'path': str(path),
            'sha256': compute_sha256(content),
            'size': len(content.encode('utf-8')),
            'timestamp': self.epoch,
        })
        
        return full_path
    
    def _emit_while(self, data: Dict[str, Any]) -> str:
        """Emit a while loop.
        
        After normalization, this is the only loop type emitters need to handle.
        """
        condition = data.get('condition', 'true')
        body = data.get('body', [])
        
        lines = [f"while ({condition}) {{"]
        self._push_indent()
        for stmt in body:
            lines.append(self._indent(self._emit_statement(stmt)))
        self._pop_indent()
        lines.append("}")
        
        return "\n".join(lines)
    
    def _emit_if(self, data: Dict[str, Any]) -> str:
        """Emit an if statement.
        
        After normalization, switch statements are already lowered to if/else.
        """
        condition = data.get('condition', 'true')
        then_body = data.get('then_body', data.get('then', []))
        else_body = data.get('else_body', data.get('else', []))
        
        lines = [f"if ({condition}) {{"]
        self._push_indent()
        for stmt in then_body:
            lines.append(self._indent(self._emit_statement(stmt)))
        self._pop_indent()
        
        if else_body:
            lines.append("} else {")
            self._push_indent()
            for stmt in else_body:
                lines.append(self._indent(self._emit_statement(stmt)))
            self._pop_indent()
        
        lines.append("}")
        return "\n".join(lines)
    
    @abstractmethod
    def _emit_statement(self, stmt: Dict[str, Any]) -> str:
        """Emit a statement.
        
        Override in subclasses for language-specific statement emission.
        """
        pass
    
    @abstractmethod
    def _emit_expression(self, expr: Any) -> str:
        """Emit an expression.
        
        Override in subclasses for language-specific expression emission.
        """
        pass


def create_emitter_config(target_dir: Path, **kwargs) -> NormalizedEmitterConfig:
    """Create an emitter configuration with defaults.
    
    Args:
        target_dir: Output directory.
        **kwargs: Additional configuration options.
        
    Returns:
        NormalizedEmitterConfig instance.
    """
    return NormalizedEmitterConfig(target_dir=target_dir, **kwargs)


def main():
    """CLI entry point for normalized emitter base."""
    import argparse
    
    parser = argparse.ArgumentParser(description="STUNIR Normalized Emitter Base")
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")
    parser.add_argument("--help-passes", action="store_true", help="Show available normalization passes")
    
    args = parser.parse_args()
    
    if args.help_passes:
        print("Available normalization passes:")
        print("  lower_switch        - Convert switch to if/else")
        print("  lower_for           - Convert for to while")
        print("  lower_break_continue - Convert break/continue to flags")
        print("  lower_try_catch     - Convert try/catch to error flags")
        print("  canonicalize_types  - Normalize type names")
        print("  fold_constants      - Evaluate constant expressions")
        print("  validate_before_emit - Run validation checks")


if __name__ == "__main__":
    main()