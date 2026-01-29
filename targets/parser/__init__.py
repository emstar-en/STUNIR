#!/usr/bin/env python3
"""STUNIR Parser Emitters Package.

Provides code emitters for generating parsers in various target languages
from parser tables produced by the parser generator.

Emitters:
- PythonParserEmitter: Generate Python parser code
- RustParserEmitter: Generate Rust parser code
- CParserEmitter: Generate C parser code
- TableDrivenEmitter: Generate portable JSON tables

Base Classes:
- ParserEmitterBase: Abstract base for parser emitters
- ParserEmitterResult: Result of parser emission

Example:
    from ir.parser import LRParserGenerator, ParserType
    from targets.parser import PythonParserEmitter
    
    # Generate parser tables
    generator = LRParserGenerator(ParserType.LALR1)
    result = generator.generate(grammar)
    
    # Emit Python code
    emitter = PythonParserEmitter()
    emit_result = emitter.emit(result, grammar)
    
    # Write to files
    with open("parser.py", "w") as f:
        f.write(emit_result.code)
    
    with open("ast_nodes.py", "w") as f:
        f.write(emit_result.ast_code)
"""

from targets.parser.base import (
    ParserEmitterBase,
    ParserEmitterResult,
)

from targets.parser.python_parser import PythonParserEmitter
from targets.parser.rust_parser import RustParserEmitter
from targets.parser.c_parser import CParserEmitter
from targets.parser.table_driven import TableDrivenEmitter

__all__ = [
    # Base classes
    'ParserEmitterBase',
    'ParserEmitterResult',
    
    # Emitters
    'PythonParserEmitter',
    'RustParserEmitter',
    'CParserEmitter',
    'TableDrivenEmitter',
]
