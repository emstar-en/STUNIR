#!/usr/bin/env python3
"""STUNIR Intermediate Representation Package.

Provides intermediate representation modules for various language families:
- grammar: Grammar IR for BNF/EBNF grammars
- parser: Parser generator for LR/LALR/LL parsers
- lexer: Lexer generator with DFA construction
- rules: Rule-based IR for expert systems (CLIPS/Jess)
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

try:
    from . import rules
except ImportError:
    rules = None

try:
    from . import constraints
except ImportError:
    constraints = None

try:
    from . import planning
except ImportError:
    planning = None

try:
    from . import asp
except ImportError:
    asp = None

__all__ = [
    'grammar',
    'parser',
    'lexer',
    'rules',
    'constraints',
    'planning',
    'asp',
]
