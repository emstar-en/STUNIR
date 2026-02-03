# STUNIR Grammar IR

The Grammar IR module provides a unified intermediate representation for formal grammars, supporting BNF, EBNF, and PEG notations.

## Overview

Grammar IR enables STUNIR to:
- Represent formal grammars programmatically
- Validate grammars (left recursion, ambiguity, reachability)
- Transform grammars (left recursion elimination, left factoring)
- Emit grammars to multiple target formats (BNF, EBNF, PEG, ANTLR, Yacc)

## Quick Start

```python
from ir.grammar import (
    Grammar, GrammarType, Symbol, SymbolType, ProductionRule,
    terminal, nonterminal,
    validate_grammar,
    eliminate_left_recursion
)

# Create symbols
E = nonterminal("expr")
T = nonterminal("term")
num = terminal("num")
plus = terminal("+")

# Create grammar
grammar = Grammar("calculator", GrammarType.BNF, E)
grammar.add_production(ProductionRule(E, (E, plus, T)))
grammar.add_production(ProductionRule(E, (T,)))
grammar.add_production(ProductionRule(T, (num,)))

# Validate grammar
result = validate_grammar(grammar)
print(f"Valid: {result.valid}")
print(f"Warnings: {result.warnings}")

# Eliminate left recursion
new_grammar = eliminate_left_recursion(grammar)
```

## Modules

### symbol.py

Defines grammar symbols:
- `Symbol`: Represents terminals and non-terminals
- `SymbolType`: Enum (TERMINAL, NONTERMINAL, EPSILON, EOF)
- `EPSILON`, `EOF`: Predefined special symbols
- `terminal()`, `nonterminal()`: Helper functions

### production.py

Defines production rules:
- `ProductionRule`: A rule like A → α
- EBNF operators:
  - `OptionalOp`: [x] or x?
  - `Repetition`: {x} or x*
  - `OneOrMore`: x+
  - `Group`: (x y z)
  - `Alternation`: x | y | z

### grammar_ir.py

Core grammar class:
- `Grammar`: Complete grammar representation
- `GrammarType`: BNF, EBNF, PEG
- `ValidationResult`: Result of validation
- `EmitterResult`: Result of emission
- `BaseGrammarEmitter`: Abstract base for emitters

### validation.py

Validation and analysis:
- `validate_grammar()`: Comprehensive validation
- `compute_first_sets()`: FIRST set computation
- `compute_follow_sets()`: FOLLOW set computation
- `compute_nullable()`: Nullable non-terminals
- `detect_left_recursion()`: Find LR cycles
- `detect_ambiguity()`: Find LL(1) conflicts
- `find_unreachable_nonterminals()`: Find unreachable symbols

### transformation.py

Grammar transformations:
- `eliminate_left_recursion()`: Remove left recursion
- `left_factor()`: Apply left factoring
- `convert_ebnf_to_bnf()`: Convert EBNF to BNF
- `normalize_grammar()`: Normalize grammar

## Grammar Types

### BNF (Backus-Naur Form)

Classic context-free grammar notation:
```
<expr> ::= <expr> "+" <term> | <term>
<term> ::= "num"
```

### EBNF (Extended BNF)

Extended with repetition operators:
```
expr = term { "+" term } ;
term = "num" ;
```

### PEG (Parsing Expression Grammar)

Ordered choice semantics:
```
expr <- term ("+" term)*
term <- 'num'
```

## Validation

Grammar validation checks for:

1. **Start Symbol**: Must be defined
2. **Productions**: All non-terminals must have at least one production
3. **Undefined References**: No undefined non-terminals in bodies
4. **Reachability**: Warn about unreachable non-terminals
5. **Left Recursion**: Warn for BNF/EBNF, error for PEG
6. **Ambiguity**: Check for LL(1) conflicts

```python
result = validate_grammar(grammar)
if not result.valid:
    for error in result.errors:
        print(f"Error: {error}")
for warning in result.warnings:
    print(f"Warning: {warning}")

# Analysis info
print(result.info["first_sets"])
print(result.info["follow_sets"])
print(result.info["nullable"])
```

## Transformations

### Left Recursion Elimination

Transforms A → Aα | β into:
- A → βA'
- A' → αA' | ε

```python
new_grammar = eliminate_left_recursion(grammar)
```

### Left Factoring

Transforms A → αβ1 | αβ2 into:
- A → αA'
- A' → β1 | β2

```python
new_grammar = left_factor(grammar)
```

### EBNF to BNF

Converts EBNF operators to pure BNF:
- [x] → x_opt → x | ε
- {x} → x_rep → x x_rep | ε
- x+ → x x_rep

```python
new_grammar = convert_ebnf_to_bnf(grammar)
```

## See Also

- [Grammar Emitters](../../targets/grammar/README.md)
- [HLI Document](../../../stunir_implementation_framework/phase6/HLI_GRAMMAR_IR.md)
