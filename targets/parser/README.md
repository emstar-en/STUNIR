# STUNIR Parser Emitters

Parser code emitters for generating parsers in various target languages.

## Overview

The Parser Emitters module generates parser source code from parse tables produced by the parser generator. It supports multiple target languages and output formats.

## Supported Targets

| Target | Language | Output Files |
|--------|----------|-------------|
| Python | Python 3 | `parser.py`, `ast_nodes.py` |
| Rust | Rust | `parser.rs`, `ast.rs`, `Cargo.toml` |
| C | C89/C99 | `parser.c`, `parser.h`, `ast.h`, `Makefile` |
| Table-Driven | JSON | `parser_tables.json`, `parser_runtime.py` |

## Quick Start

### Python Parser

```python
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
```

### Rust Parser

```python
from targets.parser import RustParserEmitter

emitter = RustParserEmitter()
emit_result = emitter.emit(result, grammar)

# Main parser code
with open("src/parser.rs", "w") as f:
    f.write(emit_result.code)

# Cargo.toml
with open("Cargo.toml", "w") as f:
    f.write(emit_result.auxiliary_files["Cargo.toml"])
```

### C Parser

```python
from targets.parser import CParserEmitter

# Use C89 standard
emitter = CParserEmitter(config={'c_standard': 'c89'})
emit_result = emitter.emit(result, grammar)

# Implementation
with open("parser.c", "w") as f:
    f.write(emit_result.code)

# Header
with open("parser.h", "w") as f:
    f.write(emit_result.auxiliary_files["parser.h"])

# Makefile
with open("Makefile", "w") as f:
    f.write(emit_result.auxiliary_files["Makefile"])
```

### Table-Driven Parser

```python
from targets.parser import TableDrivenEmitter

# Compact JSON output
emitter = TableDrivenEmitter(config={'compact': True})
emit_result = emitter.emit(result, grammar)

# Parse tables as JSON
with open("parser_tables.json", "w") as f:
    f.write(emit_result.code)

# Generic runtime
with open("parser_runtime.py", "w") as f:
    f.write(emit_result.auxiliary_files["parser_runtime.py"])
```

## Module Structure

```
targets/parser/
├── __init__.py         # Package exports
├── base.py             # Base emitter class
├── python_parser.py    # Python emitter
├── rust_parser.py      # Rust emitter
├── c_parser.py         # C emitter
├── table_driven.py     # Table-driven/JSON emitter
└── README.md           # This file
```

## Emitter Classes

### ParserEmitterBase (Abstract)

Base class for all parser emitters. Provides:

- Manifest generation with SHA256 hashes
- Warning collection
- Configuration handling
- Symbol name formatting

### PythonParserEmitter

Generates Python 3 parser code:

- Dataclass-based Token and AST nodes
- LR or LL parser implementation
- Full parse table as Python dicts
- Type hints throughout

### RustParserEmitter

Generates Rust parser code:

- Enum-based TokenType and AST
- LR or LL parser implementation
- Match-based action dispatch
- Cargo.toml for building

### CParserEmitter

Generates C parser code:

- Typedef-based types
- Header file with declarations
- Static parse tables
- Makefile for building
- C89 or C99 standard compliance

### TableDrivenEmitter

Generates portable JSON parse tables:

- Complete parse table serialization
- AST schema in JSON format
- Generic Python runtime interpreter
- Compact or pretty-printed output

## Configuration Options

### Common Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `include_comments` | bool | True | Include comments in output |
| `error_recovery` | bool | False | Include error recovery code |
| `optimize` | bool | False | Optimize generated code |

### C-Specific Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `c_standard` | str | "c99" | C standard ("c89" or "c99") |

### Table-Driven Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `compact` | bool | False | Compact JSON output |
| `include_debug` | bool | True | Include debug info |

## Output Structure

### ParserEmitterResult

```python
@dataclass
class ParserEmitterResult:
    code: str                        # Main parser code
    ast_code: str                    # AST node definitions
    manifest: Dict[str, Any]         # Build manifest
    warnings: List[str]              # Emission warnings
    auxiliary_files: Dict[str, str]  # Additional files
```

### Manifest Format

```json
{
    "schema": "stunir.parser.python.v1",
    "generator": "stunir.parser.python_emitter",
    "epoch": 1706500000,
    "grammar_name": "expr",
    "parser_code_hash": "abc123...",
    "ast_code_hash": "def456...",
    "total_size": 12345,
    "manifest_hash": "789xyz..."
}
```

## Generated Parser Features

### LR Parser

- Stack-based parsing algorithm
- ACTION/GOTO table lookup
- Shift, reduce, accept, error actions
- Production-based AST construction

### LL Parser

- Predictive parsing algorithm
- Parse table lookup by (nonterminal, lookahead)
- Recursive-style with explicit stack

## AST Generation

All emitters support generating AST node definitions:

- One node class/struct per grammar production
- Abstract base nodes for alternatives
- Type-safe field definitions
- Visitor pattern support (where applicable)

## Integration

### With Parser Generator

```python
from ir.parser import LRParserGenerator, ParserType
from targets.parser import PythonParserEmitter

# Pipeline: Grammar → Tables → Code
generator = LRParserGenerator(ParserType.LALR1)
tables = generator.generate(grammar)

emitter = PythonParserEmitter()
code = emitter.emit(tables, grammar)
```

### With Build System

The generated Makefiles and Cargo.toml files integrate with standard build systems:

```bash
# C
make clean && make

# Rust
cargo build --release
```

## See Also

- [Parser Generation Documentation](../../ir/parser/README.md)
- [Grammar IR Documentation](../../ir/grammar/README.md)
