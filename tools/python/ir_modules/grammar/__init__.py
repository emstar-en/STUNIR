#!/usr/bin/env python3
"""STUNIR Grammar IR Package.

Provides intermediate representation for formal grammars supporting
BNF, EBNF, and PEG notations.

Core Classes:
- Symbol: Grammar symbol (terminal or non-terminal)
- ProductionRule: A single production rule A → α
- Grammar: Complete grammar representation

Validation Functions:
- validate_grammar: Comprehensive grammar validation
- detect_left_recursion: Find left-recursive cycles
- compute_first_sets: Compute FIRST sets
- compute_follow_sets: Compute FOLLOW sets

Transformation Functions:
- eliminate_left_recursion: Remove left recursion
- left_factor: Apply left factoring
- convert_ebnf_to_bnf: Convert EBNF to BNF
- normalize_grammar: Normalize grammar

Example:
    from ir.grammar import Grammar, Symbol, SymbolType, ProductionRule
    from ir.grammar import validate_grammar, eliminate_left_recursion
    
    E = Symbol("E", SymbolType.NONTERMINAL)
    num = Symbol("num", SymbolType.TERMINAL)
    
    grammar = Grammar("expr", GrammarType.BNF, E)
    grammar.add_production(ProductionRule(E, (num,)))
    
    result = validate_grammar(grammar)
"""

from ir.grammar.symbol import (
    SymbolType,
    Symbol,
    EPSILON,
    EOF,
)

from ir.grammar.production import (
    ProductionRule,
    EBNFOperator,
    OptionalOp,
    Optional,  # Alias for OptionalOp
    Repetition,
    OneOrMore,
    Group,
    Alternation,
)

from ir.grammar.grammar_ir import (
    GrammarType,
    Grammar,
    ValidationResult,
    EmitterResult,
    BaseGrammarEmitter,
)

from ir.grammar.validation import (
    validate_grammar,
    compute_first_sets,
    compute_follow_sets,
    compute_nullable,
    compute_first_of_string,
    compute_first_of_production,
    detect_left_recursion,
    detect_ambiguity,
    find_unreachable_nonterminals,
)

from ir.grammar.transformation import (
    eliminate_left_recursion,
    left_factor,
    convert_ebnf_to_bnf,
    normalize_grammar,
)

__all__ = [
    # Symbol module
    'SymbolType',
    'Symbol',
    'EPSILON',
    'EOF',
    # Production module
    'ProductionRule',
    'EBNFOperator',
    'OptionalOp',
    'Optional',  # Alias for OptionalOp
    'Repetition',
    'OneOrMore',
    'Group',
    'Alternation',
    # Grammar IR module
    'GrammarType',
    'Grammar',
    'ValidationResult',
    'EmitterResult',
    'BaseGrammarEmitter',
    # Validation module
    'validate_grammar',
    'compute_first_sets',
    'compute_follow_sets',
    'compute_nullable',
    'compute_first_of_string',
    'compute_first_of_production',
    'detect_left_recursion',
    'detect_ambiguity',
    'find_unreachable_nonterminals',
    # Transformation module
    'eliminate_left_recursion',
    'left_factor',
    'convert_ebnf_to_bnf',
    'normalize_grammar',
]
