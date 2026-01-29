#!/usr/bin/env python3
"""STUNIR Grammar Emitters Package.

Provides emitters for generating grammars in various formats:
- BNF (Backus-Naur Form)
- EBNF (Extended Backus-Naur Form)
- PEG (Parsing Expression Grammar)
- ANTLR (ANother Tool for Language Recognition)
- Yacc/Bison

Example:
    from ir.grammar import Grammar, Symbol, SymbolType, ProductionRule, GrammarType
    from targets.grammar import BNFEmitter, ANTLREmitter
    
    E = Symbol("expr", SymbolType.NONTERMINAL)
    num = Symbol("NUM", SymbolType.TERMINAL)
    
    grammar = Grammar("calculator", GrammarType.BNF, E)
    grammar.add_production(ProductionRule(E, (num,)))
    
    # Emit as BNF
    bnf_emitter = BNFEmitter()
    result = bnf_emitter.emit(grammar)
    print(result.code)
    
    # Emit as ANTLR
    antlr_emitter = ANTLREmitter()
    result = antlr_emitter.emit(grammar)
    print(result.code)
"""

from targets.grammar.base import GrammarEmitterBase
from targets.grammar.bnf_emitter import BNFEmitter
from targets.grammar.ebnf_emitter import EBNFEmitter
from targets.grammar.peg_emitter import PEGEmitter
from targets.grammar.antlr_emitter import ANTLREmitter
from targets.grammar.yacc_emitter import YaccEmitter

__all__ = [
    'GrammarEmitterBase',
    'BNFEmitter',
    'EBNFEmitter',
    'PEGEmitter',
    'ANTLREmitter',
    'YaccEmitter',
]
