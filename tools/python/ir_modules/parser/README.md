# STUNIR Parser Generation

Parser generation infrastructure supporting LR, LALR, LL, and Recursive Descent parsing.

## Overview

The Parser Generation module provides comprehensive tools for generating parsers from grammar specifications. Building on the Grammar IR from Phase 6A, this module enables automatic generation of efficient parsers in multiple target languages.

## Features

- **LR/LALR/SLR Parser Generation**: Full support for bottom-up LR-family parsers
- **LL(1) Parser Generation**: Predictive top-down parsing
- **Conflict Detection**: Automatic detection of shift-reduce and reduce-reduce conflicts
- **AST Generation**: Automatic AST node schema generation from grammars
- **Error Recovery**: Configurable error recovery strategies

## Parser Types

| Type | Description | Use Case |
|------|-------------|----------|
| LR(0) | Simple LR, no lookahead | Educational, simple grammars |
| SLR(1) | Simple LR with FOLLOW sets | Simple unambiguous grammars |
| LALR(1) | Look-Ahead LR | Most practical grammars |
| LR(1) | Canonical LR | Maximum power, large tables |
| LL(1) | Predictive parsing | Recursive descent compatible |

## Quick Start

### LR Parser Generation

```python
from ir.grammar import Grammar, Symbol, SymbolType, ProductionRule, GrammarType
from ir.parser import LRParserGenerator, ParserType

# Create grammar
E = Symbol("E", SymbolType.NONTERMINAL)
num = Symbol("num", SymbolType.TERMINAL)

grammar = Grammar("expr", GrammarType.BNF, E)
grammar.add_production(ProductionRule(E, (num,)))

# Generate LALR(1) parser
generator = LRParserGenerator(ParserType.LALR1)
result = generator.generate(grammar)

if result.is_successful():
    print(f"Parser generated with {result.info['state_count']} states")
else:
    for conflict in result.conflicts:
        print(f"Conflict: {conflict}")
```

### LL(1) Parser Generation

```python
from ir.parser import LLParserGenerator

# Generate LL(1) parser (grammar must be LL(1) compatible)
generator = LLParserGenerator()
result = generator.generate(grammar)

if result.is_successful():
    print(f"LL(1) table generated with {result.info['table_entries']} entries")
```

### Checking Grammar Compatibility

```python
# Check if grammar is suitable for LL(1)
generator = LLParserGenerator()
supported, issues = generator.supports_grammar(grammar)

if not supported:
    for issue in issues:
        print(f"Issue: {issue}")
```

## Module Structure

```
ir/parser/
├── __init__.py           # Package exports
├── ast_node.py           # AST node specifications
├── parse_table.py        # Parse table data structures
├── parser_generator.py   # Generator interface
├── lr_parser.py          # LR/LALR implementation
├── ll_parser.py          # LL(1) implementation
└── README.md             # This file
```

## Key Classes

### Data Structures

- **LRItem**: LR(0)/LR(1) item representation (A → α • β)
- **LRItemSet**: Set of LR items forming a parser state
- **ParseTable**: LR parse table with ACTION and GOTO tables
- **LL1Table**: LL(1) parse table
- **Action**: Parser action (shift, reduce, accept, error)
- **Conflict**: Parse table conflict representation

### Generators

- **LRParserGenerator**: Generates LR/LALR/SLR parsers
- **LLParserGenerator**: Generates LL(1) parsers
- **ParserGeneratorResult**: Result of parser generation

### AST Support

- **ASTNodeSpec**: Specification for an AST node type
- **ASTSchema**: Complete AST schema for a grammar

## Algorithms

### LR Item Set Construction

The module implements standard LR algorithms:

1. **Closure**: Expand item sets with new items for nonterminals after the dot
2. **GOTO**: Compute state transitions on grammar symbols
3. **Item Set Construction**: Build the canonical collection of LR item sets

### LALR(1) Lookahead Computation

LALR(1) lookaheads are computed using:

1. Merging LR(0) states with same kernel
2. Lookahead propagation from kernel items
3. Spontaneous lookahead generation

### LL(1) Table Construction

LL(1) tables are built using:

1. FIRST set computation (reused from Grammar IR)
2. FOLLOW set computation (reused from Grammar IR)
3. Table population: M[A, a] = A → α if a ∈ FIRST(α) or (ε ∈ FIRST(α) and a ∈ FOLLOW(A))

## Conflict Handling

### Detection

Conflicts are automatically detected during table construction:

- **Shift-Reduce**: When both shift and reduce are valid for (state, symbol)
- **Reduce-Reduce**: When multiple reductions are valid for (state, symbol)
- **LL(1) Conflicts**: When multiple productions apply for (nonterminal, lookahead)

### Resolution

```python
from ir.parser.lr_parser import resolve_conflicts

# Resolve conflicts (e.g., prefer shift for dangling else)
resolved_table = resolve_conflicts(table, strategy="shift")
```

## Error Recovery

```python
from ir.parser import generate_error_recovery, ErrorRecoveryStrategy

# Generate panic mode recovery
recovery = generate_error_recovery(grammar, ErrorRecoveryStrategy.PANIC_MODE)
print(f"Sync tokens: {recovery['sync_tokens']}")
```

## Integration with Grammar IR

The parser generator integrates seamlessly with the Grammar IR from Phase 6A:

```python
from ir.grammar import Grammar, validate_grammar
from ir.parser import LRParserGenerator

# Validate grammar first
validation = validate_grammar(grammar)
if not validation.valid:
    print(f"Invalid grammar: {validation.errors}")
    exit(1)

# Generate parser
generator = LRParserGenerator()
result = generator.generate(grammar)
```

## See Also

- [Grammar IR Documentation](../grammar/README.md)
- [Parser Emitters Documentation](../../targets/parser/README.md)
