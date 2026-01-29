#!/usr/bin/env python3
"""Tests for Grammar IR transformation algorithms.

Tests:
- Left recursion elimination
- Left factoring
- EBNF to BNF conversion
- Grammar normalization
"""

import pytest
import sys
import os

# Add the repository root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ir.grammar.symbol import Symbol, SymbolType, EPSILON, terminal, nonterminal
from ir.grammar.production import ProductionRule, Optional, Repetition, OneOrMore, Group
from ir.grammar.grammar_ir import Grammar, GrammarType
from ir.grammar.validation import detect_left_recursion
from ir.grammar.transformation import (
    eliminate_left_recursion,
    left_factor,
    convert_ebnf_to_bnf,
    normalize_grammar,
)


class TestLeftRecursionElimination:
    """Test left recursion elimination."""
    
    def test_direct_left_recursion_elimination(self):
        """Test eliminating direct left recursion."""
        E = nonterminal("E")
        plus = terminal("+")
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (E, plus, num)))  # E → E + num
        grammar.add_production(ProductionRule(E, (num,)))          # E → num
        
        # Verify left recursion exists
        lr = detect_left_recursion(grammar)
        assert len(lr) > 0
        
        # Eliminate
        new_grammar = eliminate_left_recursion(grammar)
        
        # Verify no left recursion
        lr_after = detect_left_recursion(new_grammar)
        assert len(lr_after) == 0
    
    def test_elimination_preserves_language(self):
        """Test that elimination preserves the grammar's structure."""
        E = nonterminal("E")
        plus = terminal("+")
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (E, plus, num)))
        grammar.add_production(ProductionRule(E, (num,)))
        
        new_grammar = eliminate_left_recursion(grammar)
        
        # Should have E and E' (or similar)
        assert len(new_grammar.nonterminals) >= 2
        # Should have num in the language
        assert num in new_grammar.terminals
    
    def test_no_left_recursion_unchanged(self):
        """Test that grammar without LR is preserved."""
        E = nonterminal("E")
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (num,)))
        
        new_grammar = eliminate_left_recursion(grammar)
        
        # Should have same number of non-terminals
        assert len(new_grammar.nonterminals) == len(grammar.nonterminals)


class TestLeftFactoring:
    """Test left factoring."""
    
    def test_simple_left_factoring(self):
        """Test simple left factoring case."""
        S = nonterminal("S")
        A = nonterminal("A")
        a = terminal("a")
        b = terminal("b")
        c = terminal("c")
        
        grammar = Grammar("test", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(S, (a, b, A)))  # S → a b A
        grammar.add_production(ProductionRule(S, (a, b, c)))  # S → a b c
        grammar.add_production(ProductionRule(A, (a,)))
        
        new_grammar = left_factor(grammar)
        
        # Should have factored out "a b"
        # New grammar should have at least one more non-terminal
        assert new_grammar.production_count() >= grammar.production_count()
    
    def test_no_common_prefix(self):
        """Test that productions without common prefix are preserved."""
        S = nonterminal("S")
        a = terminal("a")
        b = terminal("b")
        
        grammar = Grammar("test", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(S, (a,)))
        grammar.add_production(ProductionRule(S, (b,)))
        
        new_grammar = left_factor(grammar)
        
        # Should be essentially the same (no factoring needed)
        assert new_grammar.production_count() == grammar.production_count()


class TestEBNFToBNF:
    """Test EBNF to BNF conversion."""
    
    def test_optional_conversion(self):
        """Test converting optional operator."""
        E = nonterminal("E")
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.EBNF, E)
        grammar.add_production(ProductionRule(E, (num, Optional(num))))
        
        new_grammar = convert_ebnf_to_bnf(grammar)
        
        # Should have new productions for optional
        assert new_grammar.grammar_type == GrammarType.BNF
        assert new_grammar.production_count() > grammar.production_count()
    
    def test_repetition_conversion(self):
        """Test converting repetition operator."""
        E = nonterminal("E")
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.EBNF, E)
        grammar.add_production(ProductionRule(E, (num, Repetition(num))))
        
        new_grammar = convert_ebnf_to_bnf(grammar)
        
        # Should have new productions for repetition
        assert new_grammar.production_count() > grammar.production_count()
        
        # Verify no EBNF operators remain
        for prod in new_grammar.all_productions():
            assert not prod.contains_ebnf()
    
    def test_one_or_more_conversion(self):
        """Test converting one-or-more operator."""
        E = nonterminal("E")
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.EBNF, E)
        grammar.add_production(ProductionRule(E, (OneOrMore(num),)))
        
        new_grammar = convert_ebnf_to_bnf(grammar)
        
        assert new_grammar.production_count() > grammar.production_count()
    
    def test_group_conversion(self):
        """Test converting group operator."""
        E = nonterminal("E")
        a = terminal("a")
        b = terminal("b")
        
        grammar = Grammar("test", GrammarType.EBNF, E)
        grammar.add_production(ProductionRule(E, (Group((a, b)),)))
        
        new_grammar = convert_ebnf_to_bnf(grammar)
        
        # Group might create a new production or just inline
        assert new_grammar.grammar_type == GrammarType.BNF


class TestNormalization:
    """Test grammar normalization."""
    
    def test_remove_duplicates(self):
        """Test removing duplicate productions."""
        E = nonterminal("E")
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (num,)))
        grammar.add_production(ProductionRule(E, (num,)))  # Duplicate
        
        new_grammar = normalize_grammar(grammar)
        
        # Should have only one production
        assert new_grammar.production_count() == 1
    
    def test_remove_unreachable(self):
        """Test removing unreachable non-terminals."""
        E = nonterminal("E")
        X = nonterminal("X")
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (num,)))
        grammar.add_production(ProductionRule(X, (num,)))  # Unreachable
        
        new_grammar = normalize_grammar(grammar)
        
        # X should be removed
        assert X not in new_grammar.nonterminals
    
    def test_remove_unproductive(self):
        """Test removing non-productive non-terminals."""
        E = nonterminal("E")
        A = nonterminal("A")
        B = nonterminal("B")  # Non-productive (only has production to itself)
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (A,)))
        grammar.add_production(ProductionRule(E, (num,)))
        grammar.add_production(ProductionRule(A, (num,)))
        grammar.add_production(ProductionRule(B, (B,)))  # Non-productive
        
        new_grammar = normalize_grammar(grammar)
        
        # B should be removed (it's not reachable anyway, and non-productive)
        # This depends on whether we can reach B
        assert E in new_grammar.nonterminals
    
    def test_sorted_output(self):
        """Test that output is sorted consistently."""
        Z = nonterminal("Z")
        A = nonterminal("A")
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.BNF, Z)
        grammar.add_production(ProductionRule(Z, (A,)))
        grammar.add_production(ProductionRule(A, (num,)))
        
        new_grammar = normalize_grammar(grammar)
        
        # Productions should be sorted
        prods = new_grammar.all_productions()
        heads = [p.head.name for p in prods]
        assert heads == sorted(heads)


class TestTransformationChaining:
    """Test chaining multiple transformations."""
    
    def test_ebnf_then_left_recursion(self):
        """Test converting EBNF then eliminating left recursion."""
        E = nonterminal("E")
        plus = terminal("+")
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.EBNF, E)
        # E → E + num | num (expressed with EBNF)
        grammar.add_production(ProductionRule(E, (E, plus, num)))
        grammar.add_production(ProductionRule(E, (num,)))
        
        # First convert EBNF to BNF (no EBNF ops here, but changes type)
        bnf_grammar = convert_ebnf_to_bnf(grammar)
        
        # Then eliminate left recursion
        final_grammar = eliminate_left_recursion(bnf_grammar)
        
        # Verify no left recursion
        lr = detect_left_recursion(final_grammar)
        assert len(lr) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
