# STUNIR Lexer Emitters

Code generators for producing lexers in multiple target languages from DFA specifications.

## Available Emitters

| Emitter | Language | Output |
|---------|----------|--------|
| `PythonLexerEmitter` | Python | Single `.py` module |
| `RustLexerEmitter` | Rust | Single `.rs` module |
| `CLexerEmitter` | C | Header + source files |
| `TableDrivenEmitter` | JSON | Portable table format |
| `CompactTableEmitter` | JSON | Minimal table format |

## Quick Start

```python
from ir.lexer import LexerSpec, TokenSpec, LexerGenerator
from targets.lexer import PythonLexerEmitter

# Define lexer
spec = LexerSpec("Calculator", [
    TokenSpec("NUM", "[0-9]+"),
    TokenSpec("PLUS", "\\+"),
    TokenSpec("MINUS", "-"),
    TokenSpec("MUL", "\\*"),
    TokenSpec("DIV", "/"),
    TokenSpec("WS", "[ \\t]+", skip=True)
])

# Generate DFA
gen = LexerGenerator(spec)
gen.generate()

# Emit Python code
emitter = PythonLexerEmitter()
code = emitter.emit(spec, gen.minimized_dfa, gen.table)

# Save to file
with open("calculator_lexer.py", "w") as f:
    f.write(code)
```

## Python Emitter

Generates a self-contained Python module.

```python
from targets.lexer import PythonLexerEmitter

emitter = PythonLexerEmitter()
code = gen.emit(emitter)
```

**Generated Code Structure:**
```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Token:
    type: str
    value: str
    line: int
    column: int

class TokenType:
    NUM = "NUM"
    PLUS = "PLUS"
    # ...

# DFA tables
_START_STATE = 0
_TRANSITIONS = [...]
_ACCEPT = [...]

class CalculatorLexer:
    def __init__(self, input_str: str): ...
    def tokenize(self) -> List[Token]: ...
    def _next_token(self) -> Optional[Token]: ...

def tokenize(input_str: str) -> List[Token]:
    return CalculatorLexer(input_str).tokenize()
```

**Usage of Generated Lexer:**
```python
from calculator_lexer import CalculatorLexer, tokenize

# Using class
lexer = CalculatorLexer("1 + 2 * 3")
tokens = lexer.tokenize()

# Using convenience function
tokens = tokenize("1 + 2 * 3")
```

## Rust Emitter

Generates a Rust module with proper types.

```python
from targets.lexer import RustLexerEmitter

emitter = RustLexerEmitter()
code = gen.emit(emitter)
```

**Generated Code Structure:**
```rust
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    NUM,
    PLUS,
    // ...
}

#[derive(Debug, Clone)]
pub struct Token {
    pub token_type: TokenType,
    pub value: String,
    pub line: usize,
    pub column: usize,
}

pub struct CalculatorLexer { ... }

impl CalculatorLexer {
    pub fn new(input: &str) -> Self;
    pub fn tokenize(&mut self) -> Result<Vec<Token>, LexerError>;
}

pub fn tokenize(input: &str) -> Result<Vec<Token>, LexerError>;
```

## C Emitter

Generates header and source files.

```python
from targets.lexer import CLexerEmitter

emitter = CLexerEmitter()
code = gen.emit(emitter)
# Contains both .h and .c content separated by comments
```

**Header (calculator_lexer.h):**
```c
#ifndef CALCULATOR_LEXER_H
#define CALCULATOR_LEXER_H

typedef enum {
    TOKEN_NUM,
    TOKEN_PLUS,
    // ...
} CalculatorTokenType;

typedef struct {
    CalculatorTokenType type;
    const char* value;
    size_t length;
    size_t line;
    size_t column;
} CalculatorToken;

typedef struct {
    const char* input;
    size_t pos;
    // ...
} CalculatorLexer;

void calculator_lexer_init(CalculatorLexer* lexer, const char* input, size_t len);
CalculatorToken calculator_lexer_next_token(CalculatorLexer* lexer);

#endif
```

**Usage:**
```c
#include "calculator_lexer.h"

int main() {
    CalculatorLexer lexer;
    calculator_lexer_init(&lexer, "1 + 2", 5);
    
    CalculatorToken token;
    while ((token = calculator_lexer_next_token(&lexer)).type != TOKEN_EOF) {
        printf("%d: %.*s\n", token.type, (int)token.length, token.value);
    }
}
```

## Table-Driven Emitter

Generates portable JSON for runtime loading.

```python
from targets.lexer import TableDrivenEmitter

emitter = TableDrivenEmitter()
json_output = gen.emit(emitter)
```

**JSON Structure:**
```json
{
  "schema": "stunir.lexer.table.v1",
  "version": "1.0.0",
  "lexer_name": "Calculator",
  "tokens": [
    {"name": "NUM", "pattern": "[0-9]+", "priority": 0, "skip": false},
    ...
  ],
  "dfa": {
    "start_state": 0,
    "num_states": 10,
    "alphabet": ["0", "1", ..., "+", "-"],
    "transitions": {"0": {"0": 1, ...}},
    "accept_states": {"1": {"token": "NUM", "priority": 0}}
  },
  "table": {
    "num_states": 10,
    "transitions": [0, 1, ...],
    "accept_table": [{"token": "NUM", "priority": 0}, null, ...]
  },
  "content_hash": "abc123..."
}
```

**Pretty Print:**
```python
emitter = TableDrivenEmitter()
pretty_json = emitter.emit_pretty(spec, dfa, table)
```

## Compact Table Emitter

Generates minimal JSON for size-constrained environments.

```python
from targets.lexer import CompactTableEmitter

emitter = CompactTableEmitter()
compact_json = gen.emit(emitter)
```

**Compact Structure:**
```json
{
  "v": 1,
  "n": "Calculator",
  "t": ["NUM", "PLUS", ...],
  "s": [5],
  "d": {
    "ss": 0,
    "ns": 10,
    "a": ["0", "1", ...],
    "tr": [0, 1, ...],
    "ac": [[0, 0], null, ...]
  }
}
```

## Custom Emitters

Create custom emitters by extending `BaseLexerEmitter`:

```python
from targets.lexer import BaseLexerEmitter

class MyCustomEmitter(BaseLexerEmitter):
    def emit(self, spec, dfa, table):
        lines = []
        lines.append(f"// Custom lexer for {spec.name}")
        lines.append(self.emit_token_class(spec))
        lines.append(self.emit_transition_table(table))
        lines.append(self.emit_lexer_class(spec, dfa))
        return '\n'.join(lines)
    
    def emit_token_class(self, spec):
        # Generate token definitions
        pass
    
    def emit_transition_table(self, table):
        # Generate DFA tables
        pass
    
    def emit_lexer_class(self, spec, dfa):
        # Generate lexer implementation
        pass
```

## Manifest Generation

All emitters can generate manifests:

```python
emitter = PythonLexerEmitter()
manifest = emitter.get_manifest(spec, dfa)

print(manifest)
# {
#   "schema": "stunir.lexer.v1",
#   "lexer_name": "Calculator",
#   "num_tokens": 6,
#   "num_states": 10,
#   "skip_tokens": ["WS"],
#   "emitter": "PythonLexerEmitter"
# }
```

## Emitter Comparison

| Feature | Python | Rust | C | Table-Driven | Compact |
|---------|--------|------|---|--------------|---------|
| Self-contained | ✓ | ✓ | ✓ | - | - |
| Runtime needed | - | - | - | ✓ | ✓ |
| Type safety | - | ✓ | Partial | - | - |
| Human readable | ✓ | ✓ | ✓ | ✓ | - |
| Size efficient | - | - | - | - | ✓ |
| Tests included | - | ✓ | - | - | - |

## Best Practices

1. **Use appropriate emitter** - Python for prototyping, C for performance
2. **Validate specs first** - `gen.validate()` before generating
3. **Test generated code** - Verify with representative input
4. **Include skip tokens** - Handle whitespace/comments properly
5. **Set priorities** - Keywords should have higher priority than identifiers

## See Also

- [ir/lexer/](../../ir/lexer/README.md) - Lexer generator core
- [targets/parser/](../parser/README.md) - Parser emitters
- [targets/grammar/](../grammar/README.md) - Grammar emitters
