# STUNIR Grammar Emitters

Grammar emitters generate formal grammar specifications in various target formats.

## Overview

The grammar emitters package provides emitters for:
- **BNF**: Backus-Naur Form
- **EBNF**: Extended Backus-Naur Form (ISO style)
- **PEG**: Parsing Expression Grammar
- **ANTLR**: ANTLR4 grammar format
- **Yacc**: Yacc/Bison grammar format

## Quick Start

```python
from ir.grammar import Grammar, GrammarType, ProductionRule, terminal, nonterminal
from targets.grammar import BNFEmitter, ANTLREmitter, YaccEmitter

# Create a simple grammar
E = nonterminal("expr")
num = terminal("num")
plus = terminal("+")

grammar = Grammar("calculator", GrammarType.BNF, E)
grammar.add_production(ProductionRule(E, (E, plus, num)))
grammar.add_production(ProductionRule(E, (num,)))

# Emit as BNF
bnf_emitter = BNFEmitter()
result = bnf_emitter.emit(grammar)
print(result.code)

# Emit as ANTLR
antlr_emitter = ANTLREmitter()
result = antlr_emitter.emit(grammar)
print(result.code)
```

## Emitters

### BNFEmitter

Emits grammars in standard BNF notation:

```
; Grammar: calculator
<expr> ::= <expr> "+" <num>
       | <num>
```

**Config options:**
- `wrap_terminals` (bool): Wrap terminals in quotes (default: True)
- `compact` (bool): Use compact format with | on same line (default: True)

### EBNFEmitter

Emits grammars in ISO EBNF notation:

```
(* Grammar: calculator *)
expr = expr "+" num | num ;
```

**Config options:**
- `iso_style` (bool): Use ISO style with = and ; (default: True)
- `wrap_terminals` (bool): Wrap terminals in quotes (default: True)

### PEGEmitter

Emits grammars in PEG notation:

```
# Grammar: calculator
expr <- expr '+' num / num
```

**Config options:**
- `arrow_style` (str): Arrow symbol (default: "<-")
- `check_left_recursion` (bool): Warn about left recursion (default: True)

### ANTLREmitter

Emits grammars in ANTLR4 format:

```
grammar Calculator;

// Parser Rules
expr
    : expr '+' term
    | term
    ;

// Lexer Rules
NUM : [0-9]+ ;
WS : [ \t\r\n]+ -> skip ;
```

**Config options:**
- `grammar_type` (str): "parser", "lexer", or "combined" (default: "combined")
- `generate_lexer_rules` (bool): Auto-generate lexer rules (default: True)
- `skip_whitespace` (bool): Add whitespace skip rule (default: True)

### YaccEmitter

Emits grammars in Yacc/Bison format:

```c
%{
#include <stdio.h>
%}

%token NUM PLUS
%start expr

%%

expr
    : expr PLUS term
    | term
    ;

%%

void yyerror(const char *s) {
    fprintf(stderr, "Error: %s\n", s);
}
```

**Config options:**
- `include_prolog` (bool): Include C prolog section (default: True)
- `include_epilog` (bool): Include C epilog section (default: True)
- `generate_actions` (bool): Generate placeholder semantic actions (default: False)

## EmitterResult

All emitters return an `EmitterResult` with:

```python
result = emitter.emit(grammar)

# Generated code
print(result.code)

# Build manifest (deterministic)
print(result.manifest)
# {
#   "schema": "stunir.grammar.bnf.v1",
#   "generator": "stunir.grammar.bnf_emitter",
#   "epoch": 1706500000,
#   "grammar_name": "calculator",
#   "production_count": 2,
#   "output_hash": "abc123...",
#   ...
# }

# Output format
print(result.format)  # "bnf"

# Warnings (if any)
for warning in result.warnings:
    print(warning)
```

## Custom Emitters

To create a custom emitter, inherit from `GrammarEmitterBase`:

```python
from targets.grammar.base import GrammarEmitterBase
from ir.grammar import Grammar, EmitterResult, ProductionRule

class MyEmitter(GrammarEmitterBase):
    FORMAT = "myformat"
    FILE_EXTENSION = ".my"
    
    def emit(self, grammar: Grammar) -> EmitterResult:
        lines = []
        for prod in grammar.all_productions():
            lines.append(self.emit_production(prod))
        code = "\n".join(lines)
        return EmitterResult(
            code=code,
            manifest=self._generate_manifest(grammar, code),
            format=self.FORMAT
        )
    
    def emit_production(self, rule: ProductionRule) -> str:
        # Custom emission logic
        return f"{rule.head.name} -> ..."
```

## See Also

- [Grammar IR](../../ir/grammar/README.md)
- [HLI Document](../../../stunir_implementation_framework/phase6/HLI_GRAMMAR_IR.md)
