# STUNIR Lexer Generator

The Lexer Generator provides infrastructure for generating efficient lexers from token specifications using finite automata theory.

## Overview

The lexer generator implements:

1. **Token Specification** - Declarative token definitions with regex patterns
2. **Regular Expression Parsing** - Convert regex patterns to AST
3. **Thompson's Construction** - Build NFA from regex AST
4. **Subset Construction** - Convert NFA to DFA
5. **Hopcroft's Algorithm** - Minimize DFA for optimal performance
6. **Longest Match Rule** - Standard lexical analysis semantics
7. **Priority Resolution** - Handle ambiguous matches

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Lexer Generator                          │
├─────────────┬─────────────┬─────────────┬───────────────────┤
│ token_spec  │   regex     │    nfa      │       dfa         │
│ Token       │   Parser    │   Builder   │   Construction    │
│ definitions │   & AST     │ Thompson's  │  & Minimization   │
└─────────────┴─────────────┴─────────────┴───────────────────┘
                              │
                              ▼
                        ┌─────────────┐
                        │   Lexer     │
                        │ Simulator   │
                        └─────────────┘
```

## Quick Start

```python
from ir.lexer import LexerSpec, TokenSpec, LexerGenerator, create_lexer

# Define token specifications
spec = LexerSpec("Calculator", [
    # Keywords (higher priority)
    TokenSpec("IF", "if", priority=10),
    TokenSpec("ELSE", "else", priority=10),
    
    # Literals
    TokenSpec("INT", "[0-9]+"),
    TokenSpec("FLOAT", "[0-9]+\\.[0-9]+", priority=5),
    
    # Identifiers
    TokenSpec("ID", "[a-zA-Z_][a-zA-Z0-9_]*"),
    
    # Operators
    TokenSpec("PLUS", "\\+"),
    TokenSpec("MINUS", "-"),
    TokenSpec("MUL", "\\*"),
    TokenSpec("DIV", "/"),
    
    # Skip tokens
    TokenSpec("WS", "[ \\t\\n]+", skip=True)
])

# Create lexer and tokenize
lexer = create_lexer(spec)
tokens = lexer.tokenize("if x + 123")

for token in tokens:
    print(f"{token.type}: {token.value!r}")
```

Output:
```
IF: 'if'
ID: 'x'
PLUS: '+'
INT: '123'
```

## Module Reference

### token_spec.py

Token and lexer specification classes.

```python
class TokenType(Enum):
    """Standard token categories."""
    KEYWORD = auto()
    IDENTIFIER = auto()
    LITERAL = auto()
    OPERATOR = auto()
    PUNCTUATION = auto()
    COMMENT = auto()
    WHITESPACE = auto()
    ERROR = auto()
    EOF = auto()

@dataclass
class TokenSpec:
    """Single token specification."""
    name: str           # Token name (e.g., "INTEGER")
    pattern: str        # Regex pattern
    priority: int = 0   # Higher = higher priority
    skip: bool = False  # Skip in output

@dataclass
class LexerSpec:
    """Complete lexer specification."""
    name: str                     # Lexer name
    tokens: List[TokenSpec]       # Token definitions
    keywords: Dict[str, str] = {} # Keyword map
    case_sensitive: bool = True
```

### regex.py

Regular expression parser and AST.

**Supported Syntax:**
- Literal characters: `a`, `b`, `c`
- Any character: `.`
- Character classes: `[abc]`, `[a-z]`, `[^abc]`
- Escapes: `\d`, `\w`, `\s`, `\n`, `\t`, `\\`
- Quantifiers: `*`, `+`, `?`
- Grouping: `(ab)`
- Alternation: `a|b`

**AST Nodes:**
- `CharNode` - Single character
- `CharClassNode` - Character class
- `ConcatNode` - Concatenation
- `UnionNode` - Alternation
- `StarNode` - Zero or more
- `PlusNode` - One or more
- `OptionalNode` - Zero or one
- `AnyCharNode` - Any character

### nfa.py

NFA construction using Thompson's algorithm.

```python
from ir.lexer import build_nfa_from_pattern

# Build NFA from pattern
nfa = build_nfa_from_pattern("[a-z]+", "ID", priority=1)

# Test NFA
assert nfa.simulate("hello")  # True
assert not nfa.simulate("123")  # False
```

**Key Classes:**
- `NFAState` - State with transitions
- `NFA` - Complete NFA
- `NFABuilder` - Thompson's construction

### dfa.py

DFA construction and minimization.

```python
from ir.lexer import nfa_to_dfa, minimize_dfa

# Convert NFA to DFA
dfa = nfa_to_dfa(nfa)

# Minimize DFA
minimized = minimize_dfa(dfa)

# Get transition table
table = minimized.to_table()
```

**Algorithms:**
- **Subset Construction** - NFA to DFA conversion
- **Hopcroft's Algorithm** - DFA minimization

**Key Classes:**
- `DFAState` - DFA state
- `DFA` - Deterministic FA
- `MinimizedDFA` - Minimized DFA
- `TransitionTable` - Flat transition array

### lexer_generator.py

Main generator interface.

```python
from ir.lexer import LexerGenerator, LexerSimulator

# Generate lexer
gen = LexerGenerator(spec)
dfa = gen.generate()

# Get statistics
stats = gen.get_statistics()
print(f"States: {stats['minimized_states']}")
print(f"Alphabet: {stats['alphabet_size']}")

# Create simulator
lexer = LexerSimulator(gen.table, gen.get_skip_tokens())
tokens = lexer.tokenize("input string")
```

## Token Priority

When multiple patterns match the same prefix:

1. **Longest Match** - Longer match wins
2. **Priority** - Higher priority wins on equal length
3. **Definition Order** - Earlier definition wins on equal priority

Example:
```python
spec = LexerSpec("Test", [
    TokenSpec("IF", "if", priority=10),
    TokenSpec("ID", "[a-z]+", priority=0)
])

# "if" matches IF (higher priority)
# "iff" matches ID (longest match)
```

## Regular Expression Reference

| Pattern | Meaning | Example |
|---------|---------|---------|
| `a` | Literal character | `a` matches "a" |
| `.` | Any character | `.` matches "a", "b", etc. |
| `[abc]` | Character class | `[abc]` matches "a", "b", or "c" |
| `[a-z]` | Character range | `[a-z]` matches lowercase letters |
| `[^abc]` | Negated class | `[^abc]` matches anything except "a", "b", "c" |
| `\d` | Digit | `\d` = `[0-9]` |
| `\w` | Word character | `\w` = `[a-zA-Z0-9_]` |
| `\s` | Whitespace | `\s` = `[ \t\n\r\f\v]` |
| `a*` | Zero or more | `a*` matches "", "a", "aa", ... |
| `a+` | One or more | `a+` matches "a", "aa", ... |
| `a?` | Optional | `a?` matches "" or "a" |
| `ab` | Concatenation | `ab` matches "ab" |
| `a\|b` | Alternation | `a\|b` matches "a" or "b" |
| `(ab)` | Grouping | `(ab)+` matches "ab", "abab", ... |

## Error Handling

```python
from ir.lexer import LexerError

try:
    tokens = lexer.tokenize("valid @ invalid")
except LexerError as e:
    print(f"Error: {e}")
    print(f"Line: {e.line}, Column: {e.column}")
```

## Integration with Parser Generator

```python
from ir.lexer import LexerGenerator, TokenSpec, LexerSpec
from ir.parser import ParserGenerator

# Create lexer spec
lexer_spec = LexerSpec("Expr", [
    TokenSpec("NUM", "[0-9]+"),
    TokenSpec("PLUS", "\\+"),
    TokenSpec("MUL", "\\*"),
    TokenSpec("LPAREN", "\\("),
    TokenSpec("RPAREN", "\\)"),
    TokenSpec("WS", "[ ]+", skip=True)
])

# Generate lexer
lexer_gen = LexerGenerator(lexer_spec)
lexer_gen.generate()

# Lexer tokens feed into parser
```

## Performance

The generated lexer uses:
- **O(n) time** - Single pass over input
- **O(1) space** - Constant memory per token
- **Table-driven** - Efficient transition lookup

For a typical programming language lexer:
- DFA states: 50-200
- Alphabet size: ~100 characters
- Table size: 5,000-20,000 entries

## See Also

- [targets/lexer/](../../targets/lexer/README.md) - Lexer code emitters
- [ir/grammar/](../grammar/README.md) - Grammar IR
- [ir/parser/](../parser/README.md) - Parser generator
