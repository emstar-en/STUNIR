"""STUNIR Lexer Emitter Package"""

from .emitter import LexerEmitter

# Aliases for backward compatibility
PythonLexerEmitter = LexerEmitter
RustLexerEmitter = LexerEmitter
CLexerEmitter = LexerEmitter
TableDrivenEmitter = LexerEmitter
CompactTableEmitter = LexerEmitter

__all__ = [
    "LexerEmitter",
    "PythonLexerEmitter",
    "RustLexerEmitter",
    "CLexerEmitter",
    "TableDrivenEmitter",
    "CompactTableEmitter"
]
