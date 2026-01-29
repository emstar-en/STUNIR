#!/usr/bin/env python3
"""STUNIR Intermediate Representation Package.

Provides intermediate representation modules for various language families:
- grammar: Grammar IR for BNF/EBNF grammars
- parser: Parser generator for LR/LALR/LL parsers
- lexer: Lexer generator with DFA construction
"""

# Import available IR modules
try:
    from . import grammar
except ImportError:
    grammar = None

try:
    from . import parser
except ImportError:
    parser = None

try:
    from . import lexer
except ImportError:
    lexer = None

__all__ = [
    'grammar',
    'parser',
    'lexer',
]
