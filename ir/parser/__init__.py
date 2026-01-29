#!/usr/bin/env python3
"""STUNIR Parser Generation Package.

Provides parser generation infrastructure supporting LR, LALR, LL,
and Recursive Descent parsing.

Parser Types:
- LR(0): Simple LR parsing
- SLR(1): Simple LR with lookahead
- LALR(1): Look-Ahead LR (most common)
- LR(1): Canonical LR
- LL(1): Predictive parsing

Core Classes:
- ParserGenerator: Abstract base for parser generators
- LRParserGenerator: LR/LALR/SLR parser generator
- LLParserGenerator: LL(1) parser generator
- ParserGeneratorResult: Result of parser generation

Data Structures:
- LRItem: LR item (production with dot position)
- LRItemSet: Set of LR items (parser state)
- ParseTable: LR parse table (ACTION/GOTO)
- LL1Table: LL(1) parse table
- ASTNodeSpec: AST node specification
- ASTSchema: Complete AST schema

Actions and Conflicts:
- Action: Parser action (shift, reduce, accept, error)
- Conflict: Parse table conflict
- LL1Conflict: LL(1) conflict

Algorithms:
- closure(): Compute closure of LR item sets
- goto(): Compute GOTO function
- build_parse_table(): Build LR parse table
- build_ll1_table(): Build LL(1) parse table

Example:
    from ir.grammar import Grammar, Symbol, SymbolType, ProductionRule
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
"""

# Parse table data structures
from ir.parser.parse_table import (
    ParserType,
    ActionType,
    Action,
    LRItem,
    LRItemSet,
    Conflict,
    ParseTable,
    LL1Conflict,
    LL1Table,
)

# AST node types
from ir.parser.ast_node import (
    ASTNodeSpec,
    ASTSchema,
    generate_ast_schema,
    to_pascal_case,
    to_snake_case,
)

# Parser generator interface
from ir.parser.parser_generator import (
    ErrorRecoveryStrategy,
    ParserGeneratorResult,
    ParserGenerator,
    generate_error_recovery,
    validate_grammar_for_parsing,
)

# LR parser generator
from ir.parser.lr_parser import (
    LRParserGenerator,
    closure,
    closure_lr1,
    goto,
    build_lr0_items,
    build_lalr_items,
    build_parse_table,
    detect_conflicts,
    resolve_conflicts,
)

# LL parser generator
from ir.parser.ll_parser import (
    LLParserGenerator,
    build_ll1_table,
    check_ll1_conditions,
    compute_first_of_string,
    generate_recursive_descent,
)

__all__ = [
    # Enums
    'ParserType',
    'ActionType',
    'ErrorRecoveryStrategy',
    
    # LR data structures
    'Action',
    'LRItem',
    'LRItemSet',
    'Conflict',
    'ParseTable',
    
    # LL data structures
    'LL1Conflict',
    'LL1Table',
    
    # AST
    'ASTNodeSpec',
    'ASTSchema',
    'generate_ast_schema',
    'to_pascal_case',
    'to_snake_case',
    
    # Generator interface
    'ParserGeneratorResult',
    'ParserGenerator',
    'generate_error_recovery',
    'validate_grammar_for_parsing',
    
    # LR generator
    'LRParserGenerator',
    'closure',
    'closure_lr1',
    'goto',
    'build_lr0_items',
    'build_lalr_items',
    'build_parse_table',
    'detect_conflicts',
    'resolve_conflicts',
    
    # LL generator
    'LLParserGenerator',
    'build_ll1_table',
    'check_ll1_conditions',
    'compute_first_of_string',
    'generate_recursive_descent',
]
