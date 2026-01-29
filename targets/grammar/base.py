#!/usr/bin/env python3
"""Base class for grammar emitters.

Provides common functionality for all grammar format emitters.
"""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from ir.grammar.grammar_ir import Grammar, EmitterResult
from ir.grammar.production import ProductionRule, BodyElement, EBNFOperator
from ir.grammar.production import OptionalOp, Repetition, OneOrMore, Group, Alternation
from ir.grammar.symbol import Symbol


class GrammarEmitterBase(ABC):
    """Abstract base class for grammar emitters.
    
    Subclasses must implement:
    - emit(): Main emission method
    - emit_production(): Emit a single production rule
    
    Attributes:
        FORMAT: The output format name (override in subclass)
        FILE_EXTENSION: The typical file extension (override in subclass)
    """
    
    FORMAT: str = "base"
    FILE_EXTENSION: str = ".grammar"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the emitter.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._warnings: List[str] = []
    
    @abstractmethod
    def emit(self, grammar: Grammar) -> EmitterResult:
        """Emit grammar in target format.
        
        Args:
            grammar: The grammar to emit
        
        Returns:
            EmitterResult with generated code and manifest
        """
        pass
    
    @abstractmethod
    def emit_production(self, rule: ProductionRule) -> str:
        """Emit a single production rule.
        
        Args:
            rule: The production rule to emit
        
        Returns:
            String representation in target format
        """
        pass
    
    def _add_warning(self, message: str) -> None:
        """Add a warning message.
        
        Args:
            message: The warning message
        """
        self._warnings.append(message)
    
    def _generate_manifest(self, grammar: Grammar, code: str) -> Dict[str, Any]:
        """Generate deterministic manifest for the emitted grammar.
        
        Args:
            grammar: The source grammar
            code: The generated code
        
        Returns:
            Manifest dictionary
        """
        def compute_sha256(data: bytes) -> str:
            return hashlib.sha256(data).hexdigest()
        
        def canonical_json(obj: Any) -> str:
            return json.dumps(obj, sort_keys=True, separators=(',', ':'))
        
        manifest = {
            "schema": f"stunir.grammar.{self.FORMAT}.v1",
            "generator": f"stunir.grammar.{self.FORMAT}_emitter",
            "epoch": int(time.time()),
            "grammar_name": grammar.name,
            "grammar_type": grammar.grammar_type.name,
            "output_format": self.FORMAT,
            "production_count": grammar.production_count(),
            "terminal_count": len(grammar.terminals),
            "nonterminal_count": len(grammar.nonterminals),
            "output_hash": compute_sha256(code.encode('utf-8')),
            "output_size": len(code),
        }
        
        # Compute manifest hash
        manifest_for_hash = {k: v for k, v in manifest.items() if k != "manifest_hash"}
        manifest["manifest_hash"] = compute_sha256(
            canonical_json(manifest_for_hash).encode('utf-8')
        )
        
        return manifest
    
    def _emit_header(self, grammar: Grammar) -> str:
        """Generate header comment for the output.
        
        Args:
            grammar: The source grammar
        
        Returns:
            Header comment string
        """
        import datetime
        
        comment_char = self._get_comment_char()
        lines = [
            f"{comment_char} Grammar: {grammar.name}",
            f"{comment_char} Type: {grammar.grammar_type.name}",
            f"{comment_char} Format: {self.FORMAT}",
            f"{comment_char} Generated: {datetime.datetime.now().isoformat()}",
            f"{comment_char} Generator: STUNIR Grammar Emitter",
            "",
        ]
        return "\n".join(lines)
    
    def _get_comment_char(self) -> str:
        """Get the comment character for this format."""
        return ";"
    
    def _emit_symbol(self, sym: Symbol) -> str:
        """Emit a symbol in the appropriate format.
        
        Args:
            sym: The symbol to emit
        
        Returns:
            String representation
        """
        if sym.is_epsilon():
            return self._get_epsilon_repr()
        return sym.name
    
    def _get_epsilon_repr(self) -> str:
        """Get the representation of epsilon in this format."""
        return "ε"
    
    def _emit_element(self, element: BodyElement) -> str:
        """Emit a body element (symbol or EBNF operator).
        
        Args:
            element: The element to emit
        
        Returns:
            String representation
        """
        if isinstance(element, Symbol):
            return self._emit_symbol(element)
        elif isinstance(element, OptionalOp):
            inner = self._emit_element(element.element)
            return f"[{inner}]"
        elif isinstance(element, Repetition):
            inner = self._emit_element(element.element)
            return f"{{{inner}}}"
        elif isinstance(element, OneOrMore):
            inner = self._emit_element(element.element)
            return f"{inner}+"
        elif isinstance(element, Group):
            inner = " ".join(self._emit_element(e) for e in element.elements)
            return f"({inner})"
        elif isinstance(element, Alternation):
            parts = []
            for alt in element.alternatives:
                if isinstance(alt, tuple):
                    parts.append(" ".join(self._emit_element(e) for e in alt))
                else:
                    parts.append(self._emit_element(alt))
            return f"({' | '.join(parts)})"
        else:
            return str(element)
