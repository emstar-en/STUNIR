#!/usr/bin/env python3
"""Tests for Grammar emitters.

Tests:
- BNF emitter
- EBNF emitter
- PEG emitter
- ANTLR emitter
- Yacc/Bison emitter
"""

import pytest
import sys
import os

# Add the repository root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ir.grammar.symbol import Symbol, SymbolType, terminal, nonterminal
from ir.grammar.production import ProductionRule, Optional, Repetition
from ir.grammar.grammar_ir import Grammar, GrammarType, EmitterResult
from targets.grammar.bnf_emitter import BNFEmitter
from targets.grammar.ebnf_emitter import EBNFEmitter
from targets.grammar.peg_emitter import PEGEmitter
from targets.grammar.antlr_emitter import ANTLREmitter
from targets.grammar.yacc_emitter import YaccEmitter


def create_simple_grammar():
    """Create a simple expression grammar for testing."""
    E = nonterminal("expr")
    T = nonterminal("term")
    num = terminal("num")
    plus = terminal("+")
    
    grammar = Grammar("calculator", GrammarType.BNF, E)
    grammar.add_production(ProductionRule(E, (E, plus, T)))
    grammar.add_production(ProductionRule(E, (T,)))
    grammar.add_production(ProductionRule(T, (num,)))
    
    return grammar


def create_simple_grammar_no_lr():
    """Create a simple grammar without left recursion."""
    E = nonterminal("expr")
    T = nonterminal("term")
    num = terminal("num")
    plus = terminal("+")
    
    grammar = Grammar("calculator", GrammarType.BNF, E)
    grammar.add_production(ProductionRule(E, (T,)))
    grammar.add_production(ProductionRule(E, (T, plus, E)))
    grammar.add_production(ProductionRule(T, (num,)))
    
    return grammar


class TestBNFEmitter:
    """Test BNF emitter."""
    
    def test_basic_emission(self):
        """Test basic BNF emission."""
        grammar = create_simple_grammar()
        emitter = BNFEmitter()
        
        result = emitter.emit(grammar)
        
        assert isinstance(result, EmitterResult)
        assert result.format == "bnf"
        assert "<expr>" in result.code
        assert "::=" in result.code
    
    def test_terminals_quoted(self):
        """Test that terminals are quoted."""
        grammar = create_simple_grammar()
        emitter = BNFEmitter({'wrap_terminals': True})
        
        result = emitter.emit(grammar)
        
        assert '"num"' in result.code or '"+"' in result.code
    
    def test_nonterminals_bracketed(self):
        """Test that non-terminals are in angle brackets."""
        grammar = create_simple_grammar()
        emitter = BNFEmitter()
        
        result = emitter.emit(grammar)
        
        assert "<expr>" in result.code
        assert "<term>" in result.code
    
    def test_manifest_generated(self):
        """Test that manifest is generated."""
        grammar = create_simple_grammar()
        emitter = BNFEmitter()
        
        result = emitter.emit(grammar)
        
        assert result.manifest is not None
        assert "schema" in result.manifest
        assert "bnf" in result.manifest["schema"]
        assert result.manifest["output_hash"] is not None
    
    def test_epsilon_production(self):
        """Test epsilon production handling."""
        E = nonterminal("E")
        
        grammar = Grammar("test", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, ()))
        
        emitter = BNFEmitter()
        result = emitter.emit(grammar)
        
        assert "ε" in result.code or "empty" in result.code.lower()
    
    def test_emit_production(self):
        """Test emitting a single production."""
        E = nonterminal("E")
        num = terminal("num")
        rule = ProductionRule(E, (num,))
        
        emitter = BNFEmitter()
        prod_str = emitter.emit_production(rule)
        
        assert "<E>" in prod_str
        assert "::=" in prod_str


class TestEBNFEmitter:
    """Test EBNF emitter."""
    
    def test_basic_emission(self):
        """Test basic EBNF emission."""
        grammar = create_simple_grammar()
        emitter = EBNFEmitter()
        
        result = emitter.emit(grammar)
        
        assert isinstance(result, EmitterResult)
        assert result.format == "ebnf"
    
    def test_iso_style(self):
        """Test ISO EBNF style (= and ;)."""
        grammar = create_simple_grammar()
        emitter = EBNFEmitter({'iso_style': True})
        
        result = emitter.emit(grammar)
        
        # Should use = for definition and ; at end
        assert " = " in result.code
        assert ";" in result.code
    
    def test_ebnf_operators(self):
        """Test EBNF operator emission."""
        E = nonterminal("E")
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.EBNF, E)
        grammar.add_production(ProductionRule(E, (num, Optional(num), Repetition(num))))
        
        emitter = EBNFEmitter()
        result = emitter.emit(grammar)
        
        # Should have [ ] for optional and { } for repetition
        assert "[" in result.code or "]" in result.code
        assert "{" in result.code or "}" in result.code
    
    def test_comment_style(self):
        """Test EBNF comment style."""
        grammar = create_simple_grammar()
        emitter = EBNFEmitter()
        
        result = emitter.emit(grammar)
        
        # EBNF uses (* *) for comments
        assert "(*" in result.code


class TestPEGEmitter:
    """Test PEG emitter."""
    
    def test_basic_emission(self):
        """Test basic PEG emission."""
        grammar = create_simple_grammar_no_lr()  # PEG doesn't allow LR
        emitter = PEGEmitter()
        
        result = emitter.emit(grammar)
        
        assert isinstance(result, EmitterResult)
        assert result.format == "peg"
    
    def test_arrow_style(self):
        """Test PEG arrow style."""
        grammar = create_simple_grammar_no_lr()
        emitter = PEGEmitter({'arrow_style': '<-'})
        
        result = emitter.emit(grammar)
        
        assert "<-" in result.code
    
    def test_ordered_choice(self):
        """Test PEG ordered choice (/)."""
        E = nonterminal("E")
        num = terminal("num")
        id_tok = terminal("id")
        
        grammar = Grammar("test", GrammarType.PEG, E)
        grammar.add_production(ProductionRule(E, (num,)))
        grammar.add_production(ProductionRule(E, (id_tok,)))
        
        emitter = PEGEmitter()
        result = emitter.emit(grammar)
        
        # PEG uses / for ordered choice, not |
        assert "/" in result.code
    
    def test_left_recursion_warning(self):
        """Test that left recursion generates warning for PEG."""
        E = nonterminal("E")
        plus = terminal("+")
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.PEG, E)
        grammar.add_production(ProductionRule(E, (E, plus, num)))
        grammar.add_production(ProductionRule(E, (num,)))
        
        emitter = PEGEmitter({'check_left_recursion': True})
        result = emitter.emit(grammar)
        
        # Should have warning about left recursion
        assert len(result.warnings) > 0


class TestANTLREmitter:
    """Test ANTLR emitter."""
    
    def test_basic_emission(self):
        """Test basic ANTLR emission."""
        grammar = create_simple_grammar()
        emitter = ANTLREmitter()
        
        result = emitter.emit(grammar)
        
        assert isinstance(result, EmitterResult)
        assert result.format == "antlr"
    
    def test_grammar_declaration(self):
        """Test grammar declaration."""
        grammar = create_simple_grammar()
        emitter = ANTLREmitter()
        
        result = emitter.emit(grammar)
        
        assert "grammar" in result.code
        assert "Calculator" in result.code  # PascalCase of "calculator"
    
    def test_parser_rules_lowercase(self):
        """Test that parser rules are lowercase."""
        grammar = create_simple_grammar()
        emitter = ANTLREmitter()
        
        result = emitter.emit(grammar)
        
        # Parser rules should be lowercase
        assert "expr" in result.code.lower()
        assert "term" in result.code.lower()
    
    def test_lexer_rules_uppercase(self):
        """Test that lexer rules are uppercase."""
        grammar = create_simple_grammar()
        emitter = ANTLREmitter({'generate_lexer_rules': True})
        
        result = emitter.emit(grammar)
        
        # Lexer rules should be uppercase
        assert "NUM" in result.code
    
    def test_whitespace_skip(self):
        """Test whitespace skip rule."""
        grammar = create_simple_grammar()
        emitter = ANTLREmitter({'skip_whitespace': True})
        
        result = emitter.emit(grammar)
        
        assert "WS" in result.code
        assert "skip" in result.code
    
    def test_file_extension(self):
        """Test file extension."""
        emitter = ANTLREmitter()
        assert emitter.FILE_EXTENSION == ".g4"


class TestYaccEmitter:
    """Test Yacc/Bison emitter."""
    
    def test_basic_emission(self):
        """Test basic Yacc emission."""
        grammar = create_simple_grammar()
        emitter = YaccEmitter()
        
        result = emitter.emit(grammar)
        
        assert isinstance(result, EmitterResult)
        assert result.format == "yacc"
    
    def test_section_separators(self):
        """Test %% section separators."""
        grammar = create_simple_grammar()
        emitter = YaccEmitter()
        
        result = emitter.emit(grammar)
        
        # Should have two %% separators
        assert result.code.count("%%") >= 2
    
    def test_token_declarations(self):
        """Test %token declarations."""
        grammar = create_simple_grammar()
        emitter = YaccEmitter()
        
        result = emitter.emit(grammar)
        
        assert "%token" in result.code
    
    def test_start_declaration(self):
        """Test %start declaration."""
        grammar = create_simple_grammar()
        emitter = YaccEmitter()
        
        result = emitter.emit(grammar)
        
        assert "%start" in result.code
        assert "expr" in result.code.lower()
    
    def test_prolog_section(self):
        """Test C prolog section."""
        grammar = create_simple_grammar()
        emitter = YaccEmitter({'include_prolog': True})
        
        result = emitter.emit(grammar)
        
        assert "%{" in result.code
        assert "%}" in result.code
        assert "#include" in result.code
    
    def test_epilog_section(self):
        """Test C epilog section."""
        grammar = create_simple_grammar()
        emitter = YaccEmitter({'include_epilog': True})
        
        result = emitter.emit(grammar)
        
        assert "yyerror" in result.code
        assert "main" in result.code
    
    def test_rule_format(self):
        """Test production rule format."""
        grammar = create_simple_grammar()
        emitter = YaccEmitter()
        
        result = emitter.emit(grammar)
        
        # Rules should have : and ;
        assert " :" in result.code or "\n    :" in result.code
        assert ";" in result.code
    
    def test_file_extension(self):
        """Test file extension."""
        emitter = YaccEmitter()
        assert emitter.FILE_EXTENSION == ".y"


class TestEmitterManifests:
    """Test manifest generation across emitters."""
    
    @pytest.mark.parametrize("emitter_class,format_name", [
        (BNFEmitter, "bnf"),
        (EBNFEmitter, "ebnf"),
        (PEGEmitter, "peg"),
        (ANTLREmitter, "antlr"),
        (YaccEmitter, "yacc"),
    ])
    def test_manifest_schema(self, emitter_class, format_name):
        """Test manifest has correct schema."""
        grammar = create_simple_grammar_no_lr()
        emitter = emitter_class()
        
        result = emitter.emit(grammar)
        
        assert f"stunir.grammar.{format_name}.v1" in result.manifest["schema"]
    
    @pytest.mark.parametrize("emitter_class", [
        BNFEmitter, EBNFEmitter, PEGEmitter, ANTLREmitter, YaccEmitter
    ])
    def test_manifest_has_hash(self, emitter_class):
        """Test manifest has output hash."""
        grammar = create_simple_grammar_no_lr()
        emitter = emitter_class()
        
        result = emitter.emit(grammar)
        
        assert "output_hash" in result.manifest
        assert len(result.manifest["output_hash"]) == 64  # SHA256 hex
    
    @pytest.mark.parametrize("emitter_class", [
        BNFEmitter, EBNFEmitter, PEGEmitter, ANTLREmitter, YaccEmitter
    ])
    def test_manifest_counts(self, emitter_class):
        """Test manifest has correct counts."""
        grammar = create_simple_grammar_no_lr()
        emitter = emitter_class()
        
        result = emitter.emit(grammar)
        
        assert result.manifest["production_count"] == grammar.production_count()
        assert result.manifest["terminal_count"] == len(grammar.terminals)
        assert result.manifest["nonterminal_count"] == len(grammar.nonterminals)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
