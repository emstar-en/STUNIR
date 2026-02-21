#!/usr/bin/env python3
"""Grammar symbol definitions.

This module defines the fundamental symbol types used in grammar representations:
- SymbolType enum for categorizing symbols
- Symbol dataclass for representing terminals and non-terminals
- Constants for epsilon and end-of-file markers
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional


class SymbolType(Enum):
    """Types of grammar symbols."""
    TERMINAL = auto()        # Literal token (e.g., '+', 'num', 'if')
    NONTERMINAL = auto()     # Rule reference (e.g., expr, stmt)
    EPSILON = auto()         # Empty production (ε)
    EOF = auto()             # End of input ($)


@dataclass(frozen=True)
class Symbol:
    """Grammar symbol (terminal or non-terminal).
    
    Attributes:
        name: The symbol name (e.g., 'expr', 'num', '+')
        symbol_type: The type of this symbol
        pattern: Optional regex pattern for terminals (used in lexer generation)
    
    Examples:
        >>> expr = Symbol("expr", SymbolType.NONTERMINAL)
        >>> num = Symbol("num", SymbolType.TERMINAL, pattern=r"[0-9]+")
        >>> plus = Symbol("+", SymbolType.TERMINAL)
    """
    name: str
    symbol_type: SymbolType
    pattern: Optional[str] = None
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal symbol."""
        return self.symbol_type == SymbolType.TERMINAL
    
    def is_nonterminal(self) -> bool:
        """Check if this is a non-terminal symbol."""
        return self.symbol_type == SymbolType.NONTERMINAL
    
    def is_epsilon(self) -> bool:
        """Check if this is the epsilon (empty) symbol."""
        return self.symbol_type == SymbolType.EPSILON
    
    def is_eof(self) -> bool:
        """Check if this is the end-of-file symbol."""
        return self.symbol_type == SymbolType.EOF
    
    def __str__(self) -> str:
        """Return string representation of the symbol."""
        if self.is_epsilon():
            return "ε"
        if self.is_eof():
            return "$"
        return self.name
    
    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        return f"Symbol({self.name!r}, {self.symbol_type.name})"


# Predefined special symbols
EPSILON = Symbol("ε", SymbolType.EPSILON)
"""The epsilon symbol representing empty productions."""

EOF = Symbol("$", SymbolType.EOF)
"""The end-of-file symbol marking input termination."""


def terminal(name: str, pattern: Optional[str] = None) -> Symbol:
    """Create a terminal symbol.
    
    Args:
        name: The terminal name
        pattern: Optional regex pattern for lexer generation
    
    Returns:
        A terminal Symbol
    
    Example:
        >>> num = terminal("num", r"[0-9]+")
        >>> plus = terminal("+")
    """
    return Symbol(name, SymbolType.TERMINAL, pattern)


def nonterminal(name: str) -> Symbol:
    """Create a non-terminal symbol.
    
    Args:
        name: The non-terminal name
    
    Returns:
        A non-terminal Symbol
    
    Example:
        >>> expr = nonterminal("expr")
        >>> stmt = nonterminal("stmt")
    """
    return Symbol(name, SymbolType.NONTERMINAL)
