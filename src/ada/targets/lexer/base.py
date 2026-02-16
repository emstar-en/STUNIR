"""
Base Lexer Emitter for STUNIR.

Provides abstract base class and common utilities for lexer code emitters.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
import hashlib
import json

from ir.lexer.token_spec import LexerSpec, TokenSpec
from ir.lexer.dfa import MinimizedDFA, TransitionTable


def canonical_json(data: Any) -> str:
    """Generate canonical JSON output (RFC 8785 subset)."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def compute_sha256(data: str) -> str:
    """Compute SHA-256 hash of string."""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


class BaseLexerEmitter(ABC):
    """
    Abstract base class for lexer emitters.
    
    Subclasses implement target-specific code generation.
    """
    
    def __init__(self, indent: str = "    "):
        """
        Initialize emitter.
        
        Args:
            indent: Indentation string (default: 4 spaces)
        """
        self.indent = indent
    
    @abstractmethod
    def emit(self, spec: LexerSpec, dfa: MinimizedDFA, table: TransitionTable) -> str:
        """
        Emit lexer code for the target language.
        
        Args:
            spec: Lexer specification
            dfa: Minimized DFA
            table: Transition table
            
        Returns:
            Generated lexer code
        """
        pass
    
    @abstractmethod
    def emit_transition_table(self, table: TransitionTable) -> str:
        """
        Emit transition table code.
        
        Args:
            table: Transition table
            
        Returns:
            Generated table code
        """
        pass
    
    @abstractmethod
    def emit_token_class(self, spec: LexerSpec) -> str:
        """
        Emit token class/struct definition.
        
        Args:
            spec: Lexer specification
            
        Returns:
            Generated token class code
        """
        pass
    
    @abstractmethod
    def emit_lexer_class(self, spec: LexerSpec, dfa: MinimizedDFA) -> str:
        """
        Emit lexer class implementation.
        
        Args:
            spec: Lexer specification
            dfa: Minimized DFA
            
        Returns:
            Generated lexer class code
        """
        pass
    
    def get_manifest(self, spec: LexerSpec, dfa: MinimizedDFA) -> Dict[str, Any]:
        """
        Generate manifest for emitted lexer.
        
        Args:
            spec: Lexer specification
            dfa: Minimized DFA
            
        Returns:
            Manifest dictionary
        """
        return {
            "schema": "stunir.lexer.v1",
            "lexer_name": spec.name,
            "num_tokens": len(spec.tokens),
            "num_states": dfa.num_states,
            "alphabet_size": len(dfa.alphabet),
            "skip_tokens": [t.name for t in spec.tokens if t.skip],
            "emitter": self.__class__.__name__,
            "tokens": [
                {
                    "name": t.name,
                    "pattern": t.pattern,
                    "priority": t.priority,
                    "skip": t.skip
                }
                for t in spec.tokens
            ]
        }
    
    def _indent_lines(self, code: str, level: int = 1) -> str:
        """Indent all lines of code."""
        prefix = self.indent * level
        return '\n'.join(prefix + line if line else line for line in code.split('\n'))
    
    def _escape_string(self, s: str, quote: str = '"') -> str:
        """Escape string for target language."""
        result = []
        for c in s:
            if c == '\\':
                result.append('\\\\')
            elif c == quote:
                result.append('\\' + quote)
            elif c == '\n':
                result.append('\\n')
            elif c == '\r':
                result.append('\\r')
            elif c == '\t':
                result.append('\\t')
            elif ord(c) < 32 or ord(c) > 126:
                result.append(f'\\x{ord(c):02x}')
            else:
                result.append(c)
        return ''.join(result)


class EmitterUtils:
    """Utility functions for lexer emitters."""
    
    @staticmethod
    def format_table_array(values: List[int], items_per_line: int = 16) -> str:
        """Format a list of integers as array initializer."""
        lines = []
        for i in range(0, len(values), items_per_line):
            chunk = values[i:i + items_per_line]
            lines.append(', '.join(str(v) for v in chunk))
        return ',\n'.join(lines)
    
    @staticmethod
    def format_symbol_map(symbols: Dict[str, int]) -> str:
        """Format symbol-to-index mapping."""
        entries = []
        for symbol, idx in sorted(symbols.items(), key=lambda x: x[1]):
            escaped = EmitterUtils.escape_char(symbol)
            entries.append(f"'{escaped}': {idx}")
        return '{' + ', '.join(entries) + '}'
    
    @staticmethod
    def escape_char(c: str) -> str:
        """Escape a single character."""
        if c == '\\':
            return '\\\\'
        elif c == "'":
            return "\\'"
        elif c == '\n':
            return '\\n'
        elif c == '\r':
            return '\\r'
        elif c == '\t':
            return '\\t'
        elif ord(c) < 32 or ord(c) > 126:
            return f'\\x{ord(c):02x}'
        return c
