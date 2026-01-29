#!/usr/bin/env python3
"""Tests for Grammar IR core functionality.

Tests:
- Symbol creation and properties
- Production rule creation
- Grammar construction
- Basic grammar operations
"""

import pytest
import sys
import os

# Add the repository root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ir.grammar.symbol import Symbol, SymbolType, EPSILON, EOF, terminal, nonterminal
from ir.grammar.production import ProductionRule, Optional, Repetition, OneOrMore, Group, Alternation
from ir.grammar.grammar_ir import Grammar, GrammarType, ValidationResult, EmitterResult


class TestSymbol:
    """Test Symbol class."""
    
    def test_terminal_creation(self):
        """Test creating terminal symbols."""
        num = Symbol("num", SymbolType.TERMINAL)
        assert num.name == "num"
        assert num.is_terminal()
        assert not num.is_nonterminal()
        assert not num.is_epsilon()
    
    def test_nonterminal_creation(self):
        """Test creating non-terminal symbols."""
        expr = Symbol("expr", SymbolType.NONTERMINAL)
        assert expr.name == "expr"
        assert expr.is_nonterminal()
        assert not expr.is_terminal()
    
    def test_terminal_with_pattern(self):
        """Test creating terminal with regex pattern."""
        num = Symbol("num", SymbolType.TERMINAL, pattern=r"[0-9]+")
        assert num.pattern == r"[0-9]+"
    
    def test_epsilon_symbol(self):
        """Test epsilon symbol."""
        assert EPSILON.is_epsilon()
        assert str(EPSILON) == "ε"
    
    def test_eof_symbol(self):
        """Test EOF symbol."""
        assert EOF.is_eof()
        assert str(EOF) == "$"
    
    def test_symbol_equality(self):
        """Test symbol equality (frozen dataclass)."""
        sym1 = Symbol("x", SymbolType.TERMINAL)
        sym2 = Symbol("x", SymbolType.TERMINAL)
        assert sym1 == sym2
        assert hash(sym1) == hash(sym2)
    
    def test_helper_functions(self):
        """Test terminal() and nonterminal() helper functions."""
        t = terminal("num", r"[0-9]+")
        assert t.is_terminal()
        assert t.pattern == r"[0-9]+"
        
        nt = nonterminal("expr")
        assert nt.is_nonterminal()


class TestProductionRule:
    """Test ProductionRule class."""
    
    def test_simple_production(self):
        """Test creating a simple production rule."""
        E = nonterminal("E")
        num = terminal("num")
        
        rule = ProductionRule(E, (num,))
        
        assert rule.head == E
        assert rule.body == (num,)
        assert not rule.is_epsilon_production()
    
    def test_epsilon_production(self):
        """Test epsilon production detection."""
        E = nonterminal("E")
        
        # Empty body
        rule1 = ProductionRule(E, ())
        assert rule1.is_epsilon_production()
        
        # Explicit epsilon
        rule2 = ProductionRule(E, (EPSILON,))
        assert rule2.is_epsilon_production()
    
    def test_production_with_label(self):
        """Test production with label."""
        E = nonterminal("E")
        num = terminal("num")
        
        rule = ProductionRule(E, (num,), label="literal")
        assert rule.label == "literal"
    
    def test_production_with_action(self):
        """Test production with semantic action."""
        E = nonterminal("E")
        num = terminal("num")
        
        rule = ProductionRule(E, (num,), action="$$ = $1;")
        assert rule.action == "$$ = $1;"
    
    def test_body_symbols(self):
        """Test extracting symbols from body."""
        E = nonterminal("E")
        T = nonterminal("T")
        plus = terminal("+")
        
        rule = ProductionRule(E, (E, plus, T))
        symbols = rule.body_symbols()
        
        assert E in symbols
        assert T in symbols
        assert plus in symbols
    
    def test_terminals_and_nonterminals(self):
        """Test separating terminals and non-terminals."""
        E = nonterminal("E")
        T = nonterminal("T")
        plus = terminal("+")
        
        rule = ProductionRule(E, (E, plus, T))
        
        assert rule.terminals() == [plus]
        assert E in rule.nonterminals()
        assert T in rule.nonterminals()
    
    def test_production_str(self):
        """Test string representation."""
        E = nonterminal("E")
        num = terminal("num")
        
        rule = ProductionRule(E, (num,))
        assert "E" in str(rule)
        assert "→" in str(rule)
        assert "num" in str(rule)
    
    def test_invalid_head(self):
        """Test that non-terminal head is required."""
        num = terminal("num")
        
        with pytest.raises(ValueError):
            ProductionRule(num, (num,))


class TestEBNFOperators:
    """Test EBNF operator classes."""
    
    def test_optional(self):
        """Test Optional operator."""
        num = terminal("num")
        opt = Optional(num)
        assert opt.element == num
    
    def test_repetition(self):
        """Test Repetition operator."""
        num = terminal("num")
        rep = Repetition(num)
        assert rep.element == num
    
    def test_one_or_more(self):
        """Test OneOrMore operator."""
        num = terminal("num")
        plus = OneOrMore(num)
        assert plus.element == num
    
    def test_group(self):
        """Test Group operator."""
        a = terminal("a")
        b = terminal("b")
        grp = Group((a, b))
        assert grp.elements == (a, b)
    
    def test_alternation(self):
        """Test Alternation operator."""
        a = terminal("a")
        b = terminal("b")
        alt = Alternation((a, b))
        assert alt.alternatives == (a, b)
    
    def test_contains_ebnf(self):
        """Test detecting EBNF operators in production."""
        E = nonterminal("E")
        num = terminal("num")
        
        # Without EBNF
        rule1 = ProductionRule(E, (num,))
        assert not rule1.contains_ebnf()
        
        # With EBNF
        rule2 = ProductionRule(E, (Optional(num),))
        assert rule2.contains_ebnf()


class TestGrammar:
    """Test Grammar class."""
    
    def test_grammar_creation(self):
        """Test creating a grammar."""
        E = nonterminal("E")
        
        grammar = Grammar("expr", GrammarType.BNF, E)
        
        assert grammar.name == "expr"
        assert grammar.grammar_type == GrammarType.BNF
        assert grammar.start_symbol == E
        assert E in grammar.nonterminals
    
    def test_add_production(self):
        """Test adding production rules."""
        E = nonterminal("E")
        T = nonterminal("T")
        num = terminal("num")
        plus = terminal("+")
        
        grammar = Grammar("expr", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (E, plus, T)))
        grammar.add_production(ProductionRule(E, (T,)))
        grammar.add_production(ProductionRule(T, (num,)))
        
        assert grammar.production_count() == 3
        assert E in grammar.nonterminals
        assert T in grammar.nonterminals
        assert plus in grammar.terminals
        assert num in grammar.terminals
    
    def test_get_productions(self):
        """Test getting productions for a non-terminal."""
        E = nonterminal("E")
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.BNF, E)
        rule1 = ProductionRule(E, (num,))
        rule2 = ProductionRule(E, ())
        
        grammar.add_production(rule1)
        grammar.add_production(rule2)
        
        prods = grammar.get_productions(E)
        assert len(prods) == 2
        assert rule1 in prods
        assert rule2 in prods
    
    def test_all_productions(self):
        """Test getting all productions."""
        E = nonterminal("E")
        T = nonterminal("T")
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (T,)))
        grammar.add_production(ProductionRule(T, (num,)))
        
        all_prods = grammar.all_productions()
        assert len(all_prods) == 2
    
    def test_all_symbols(self):
        """Test getting all symbols."""
        E = nonterminal("E")
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (num,)))
        
        symbols = grammar.all_symbols()
        assert E in symbols
        assert num in symbols
    
    def test_grammar_copy(self):
        """Test copying a grammar."""
        E = nonterminal("E")
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (num,)))
        
        copy = grammar.copy()
        
        assert copy.name == grammar.name
        assert copy.production_count() == grammar.production_count()
        assert copy is not grammar
    
    def test_grammar_str(self):
        """Test string representation."""
        E = nonterminal("E")
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (num,)))
        
        s = str(grammar)
        assert "test" in s
        assert "BNF" in s
    
    def test_invalid_start_symbol(self):
        """Test that start symbol must be a non-terminal."""
        num = terminal("num")
        
        with pytest.raises(ValueError):
            Grammar("test", GrammarType.BNF, num)


class TestGrammarTypes:
    """Test different grammar types."""
    
    def test_bnf_type(self):
        """Test BNF grammar type."""
        E = nonterminal("E")
        grammar = Grammar("test", GrammarType.BNF, E)
        assert grammar.grammar_type == GrammarType.BNF
    
    def test_ebnf_type(self):
        """Test EBNF grammar type."""
        E = nonterminal("E")
        grammar = Grammar("test", GrammarType.EBNF, E)
        assert grammar.grammar_type == GrammarType.EBNF
    
    def test_peg_type(self):
        """Test PEG grammar type."""
        E = nonterminal("E")
        grammar = Grammar("test", GrammarType.PEG, E)
        assert grammar.grammar_type == GrammarType.PEG


class TestResultClasses:
    """Test result classes."""
    
    def test_validation_result(self):
        """Test ValidationResult."""
        result = ValidationResult(valid=True, errors=[], warnings=["warning"])
        assert result.valid
        assert bool(result)  # Test __bool__
        assert len(result.warnings) == 1
    
    def test_validation_result_invalid(self):
        """Test invalid ValidationResult."""
        result = ValidationResult(valid=False, errors=["error"])
        assert not result.valid
        assert not bool(result)
    
    def test_emitter_result(self):
        """Test EmitterResult."""
        result = EmitterResult(
            code="test code",
            manifest={"key": "value"},
            format="bnf"
        )
        assert result.code == "test code"
        assert result.format == "bnf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
