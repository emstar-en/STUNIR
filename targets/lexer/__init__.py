"""
STUNIR Lexer Emitters Package.

Provides code generation for lexers in multiple target languages:
- Python: Self-contained Python lexer module
- Rust: Rust lexer with TokenType enum
- C: Header and source files for C lexer
- Table-driven: Portable JSON representation

Usage:
    from ir.lexer import LexerSpec, TokenSpec, LexerGenerator
    from targets.lexer import PythonLexerEmitter
    
    spec = LexerSpec("MyLexer", [
        TokenSpec("INT", "[0-9]+"),
        TokenSpec("ID", "[a-z]+"),
        TokenSpec("WS", "[ \\t\\n]+", skip=True)
    ])
    
    gen = LexerGenerator(spec)
    gen.generate()
    
    emitter = PythonLexerEmitter()
    code = gen.emit(emitter)
"""

from .base import (
    BaseLexerEmitter,
    EmitterUtils,
    canonical_json,
    compute_sha256,
)

from .python_lexer import PythonLexerEmitter
from .rust_lexer import RustLexerEmitter
from .c_lexer import CLexerEmitter
from .table_driven import TableDrivenEmitter, CompactTableEmitter

__all__ = [
    # Base
    'BaseLexerEmitter',
    'EmitterUtils',
    'canonical_json',
    'compute_sha256',
    
    # Emitters
    'PythonLexerEmitter',
    'RustLexerEmitter',
    'CLexerEmitter',
    'TableDrivenEmitter',
    'CompactTableEmitter',
]
