#!/usr/bin/env python3
"""STUNIR BEAM VM Emitter Base.

This module provides the shared base class for Erlang and Elixir emitters,
including common utilities for canonical output and pattern emission.

Usage:
    from targets.beam.base import BEAMEmitterBase, EmitterResult
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import hashlib
import json


def canonical_json(data: Any) -> str:
    """Generate canonical JSON (RFC 8785 subset).
    
    Produces deterministic JSON with sorted keys and minimal whitespace.
    """
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def compute_sha256(content: str) -> str:
    """Compute SHA-256 hash of content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


@dataclass
class EmitterResult:
    """Result of code emission.
    
    Attributes:
        code: Generated source code
        manifest: Metadata about generated code
        files: Additional generated files (e.g., project files)
    """
    code: str = ''
    manifest: Dict[str, Any] = field(default_factory=dict)
    files: Dict[str, str] = field(default_factory=dict)


class BEAMEmitterBase(ABC):
    """Base class for BEAM VM language emitters.
    
    Provides shared functionality for Erlang and Elixir emitters,
    including pattern emission, manifest generation, and indentation.
    """
    
    LANGUAGE = 'beam'
    VERSION = '1.0.0'
    
    def __init__(self):
        """Initialize the emitter."""
        self.indent_level = 0
        self.indent_str = '    '
    
    def indent(self) -> str:
        """Get current indentation string."""
        return self.indent_str * self.indent_level
    
    def emit_module(self, module: 'ActorModule') -> EmitterResult:
        """Emit code for an actor module.
        
        Args:
            module: Actor module IR to emit
            
        Returns:
            EmitterResult with generated code and manifest
        """
        code = self._emit_module_impl(module)
        manifest = self._generate_manifest(module, code)
        return EmitterResult(code=code, manifest=manifest)
    
    @abstractmethod
    def _emit_module_impl(self, module: 'ActorModule') -> str:
        """Language-specific module emission.
        
        Must be implemented by subclasses.
        """
        pass
    
    def _generate_manifest(self, module: 'ActorModule', code: str) -> Dict[str, Any]:
        """Generate manifest for emitted code."""
        return {
            'schema': f'stunir.manifest.{self.LANGUAGE}.v1',
            'module': module.name,
            'language': self.LANGUAGE,
            'emitter_version': self.VERSION,
            'code_hash': compute_sha256(code),
            'code_size': len(code),
            'exports': module.exports,
            'behaviors': [b.behavior_type.value for b in module.behaviors] if module.behaviors else [],
        }
    
    # =========================================================================
    # Pattern Emission (shared between Erlang/Elixir)
    # =========================================================================
    
    def emit_pattern(self, pattern: 'Pattern') -> str:
        """Emit pattern for matching.
        
        Dispatches to appropriate emission method based on pattern kind.
        """
        if pattern is None:
            return '_'
        
        kind = pattern.kind
        method = getattr(self, f'_emit_{kind}', None)
        if method:
            return method(pattern)
        return '_'  # Default wildcard
    
    def _emit_wildcard_pattern(self, p: 'WildcardPattern') -> str:
        """Emit wildcard pattern."""
        return '_'
    
    def _emit_var_pattern(self, p: 'VarPattern') -> str:
        """Emit variable pattern."""
        return self._transform_var(p.name)
    
    @abstractmethod
    def _transform_var(self, name: str) -> str:
        """Transform variable name to target syntax.
        
        Erlang: Capitalize first letter
        Elixir: Lowercase/snake_case
        """
        pass
    
    def _emit_literal_pattern(self, p: 'LiteralPattern') -> str:
        """Emit literal pattern."""
        return self.emit_literal(p.value, getattr(p, 'literal_type', 'auto'))
    
    def _emit_atom_pattern(self, p: 'AtomPattern') -> str:
        """Emit atom pattern."""
        return self.emit_literal(p.value, 'atom')
    
    @abstractmethod
    def emit_literal(self, value: Any, literal_type: str = 'auto') -> str:
        """Emit literal value.
        
        Must be implemented by subclasses for language-specific syntax.
        """
        pass
    
    # =========================================================================
    # Expression Emission
    # =========================================================================
    
    def _emit_expr(self, expr: 'Expr') -> str:
        """Emit expression.
        
        Dispatches to appropriate emission method based on expression kind.
        """
        if expr is None:
            return ''
        
        kind = expr.kind
        method = getattr(self, f'_emit_{kind}_expr', None)
        if method:
            return method(expr)
        
        # Fallback for direct kind matching
        method = getattr(self, f'_emit_{kind}', None)
        if method:
            return method(expr)
        
        return f'%% Unknown expression: {kind}'
    
    def _emit_var_expr(self, expr: 'VarExpr') -> str:
        """Emit variable expression."""
        return self._transform_var(expr.name)
    
    def _emit_self_expr(self, expr: 'SelfExpr') -> str:
        """Emit self() expression."""
        return 'self()'
    
    # =========================================================================
    # Guard Emission
    # =========================================================================
    
    def _emit_guards(self, guards: List['Guard']) -> str:
        """Emit guard expressions.
        
        Guards within a group are AND-ed (comma),
        groups are OR-ed (semicolon in Erlang, 'or' in Elixir).
        """
        if not guards:
            return ''
        
        guard_groups = []
        for guard in guards:
            conditions = self._emit_guard_group(guard)
            if conditions:
                guard_groups.append(conditions)
        
        return self._join_guard_groups(guard_groups)
    
    def _emit_guard_group(self, guard: 'Guard') -> str:
        """Emit a single guard group (AND-ed conditions)."""
        conditions = [self._emit_expr(c) for c in guard.conditions if c]
        return self._join_guard_conditions(conditions)
    
    @abstractmethod
    def _join_guard_conditions(self, conditions: List[str]) -> str:
        """Join conditions within a guard group.
        
        Erlang: comma-separated
        Elixir: 'and' separated
        """
        pass
    
    @abstractmethod
    def _join_guard_groups(self, groups: List[str]) -> str:
        """Join guard groups.
        
        Erlang: semicolon-separated
        Elixir: 'or' separated
        """
        pass
