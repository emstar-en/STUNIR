#!/usr/bin/env python3
"""Tests for Grammar IR validation algorithms.

Tests:
- FIRST set computation
- FOLLOW set computation
- Nullable computation
- Left recursion detection
- Ambiguity detection
- Reachability analysis
- Full grammar validation
"""

import pytest
import sys
import os

# Add the repository root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ir.grammar.symbol import Symbol, SymbolType, EPSILON, EOF, terminal, nonterminal
from ir.grammar.production import ProductionRule
from ir.grammar.grammar_ir import Grammar, GrammarType
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


class TestNullable:
    """Test nullable computation."""
    
    def test_epsilon_production_nullable(self):
        """Test that non-terminal with epsilon production is nullable."""
        E = nonterminal("E")
        
        grammar = Grammar("test", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, ()))
        
        nullable = compute_nullable(grammar)
        assert E in nullable
    
    def test_terminal_not_nullable(self):
        """Test that terminal production is not nullable."""
        E = nonterminal("E")
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (num,)))
        
        nullable = compute_nullable(grammar)
        assert E not in nullable
    
    def test_transitive_nullable(self):
        """Test transitive nullable computation."""
        E = nonterminal("E")
        T = nonterminal("T")
        
        grammar = Grammar("test", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (T,)))  # E → T
        grammar.add_production(ProductionRule(T, ()))    # T → ε
        
        nullable = compute_nullable(grammar)
        assert T in nullable
        assert E in nullable


class TestFirstSets:
    """Test FIRST set computation."""
    
    def test_terminal_first(self):
        """Test FIRST of terminal-only production."""
        E = nonterminal("E")
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (num,)))
        
        first = compute_first_sets(grammar)
        assert num in first[E]
    
    def test_nonterminal_first(self):
        """Test FIRST set propagation through non-terminals."""
        E = nonterminal("E")
        T = nonterminal("T")
        num = terminal("num")
        lparen = terminal("(")
        
        grammar = Grammar("test", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (T,)))
        grammar.add_production(ProductionRule(T, (num,)))
        grammar.add_production(ProductionRule(T, (lparen, E)))
        
        first = compute_first_sets(grammar)
        
        assert num in first[T]
        assert lparen in first[T]
        assert num in first[E]
        assert lparen in first[E]
    
    def test_epsilon_in_first(self):
        """Test epsilon in FIRST set."""
        E = nonterminal("E")
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (num,)))
        grammar.add_production(ProductionRule(E, ()))
        
        first = compute_first_sets(grammar)
        
        assert num in first[E]
        assert EPSILON in first[E]
    
    def test_first_of_string(self):
        """Test FIRST of string computation."""
        E = nonterminal("E")
        T = nonterminal("T")
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (num,)))
        grammar.add_production(ProductionRule(T, ()))
        
        first = compute_first_sets(grammar)
        
        # FIRST(T E) where T is nullable
        result = compute_first_of_string([T, E], first)
        assert num in result


class TestFollowSets:
    """Test FOLLOW set computation."""
    
    def test_start_symbol_has_eof(self):
        """Test that start symbol has EOF in FOLLOW."""
        E = nonterminal("E")
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (num,)))
        
        first = compute_first_sets(grammar)
        follow = compute_follow_sets(grammar, first)
        
        assert EOF in follow[E]
    
    def test_follow_from_sibling(self):
        """Test FOLLOW from sibling in production."""
        E = nonterminal("E")
        T = nonterminal("T")
        num = terminal("num")
        plus = terminal("+")
        
        grammar = Grammar("test", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (T, plus, E)))  # E → T + E
        grammar.add_production(ProductionRule(T, (num,)))
        
        first = compute_first_sets(grammar)
        follow = compute_follow_sets(grammar, first)
        
        # FOLLOW(T) should include + (from E → T + E)
        assert plus in follow[T]
    
    def test_follow_propagation(self):
        """Test FOLLOW set propagation."""
        E = nonterminal("E")
        T = nonterminal("T")
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (T,)))  # E → T
        grammar.add_production(ProductionRule(T, (num,)))
        
        first = compute_first_sets(grammar)
        follow = compute_follow_sets(grammar, first)
        
        # FOLLOW(T) should include FOLLOW(E) since E → T
        assert EOF in follow[T]


class TestLeftRecursion:
    """Test left recursion detection."""
    
    def test_direct_left_recursion(self):
        """Test detecting direct left recursion."""
        E = nonterminal("E")
        plus = terminal("+")
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (E, plus, num)))  # E → E + num
        grammar.add_production(ProductionRule(E, (num,)))          # E → num
        
        result = detect_left_recursion(grammar)
        
        assert len(result) > 0
        # Should detect E as left-recursive
        nts = [nt for nt, _ in result]
        assert E in nts
    
    def test_indirect_left_recursion(self):
        """Test detecting indirect left recursion."""
        A = nonterminal("A")
        B = nonterminal("B")
        a = terminal("a")
        b = terminal("b")
        
        grammar = Grammar("test", GrammarType.BNF, A)
        grammar.add_production(ProductionRule(A, (B, a)))  # A → B a
        grammar.add_production(ProductionRule(B, (A, b)))  # B → A b
        grammar.add_production(ProductionRule(B, (b,)))    # B → b
        
        result = detect_left_recursion(grammar)
        
        # Should detect A or B as left-recursive
        assert len(result) > 0
    
    def test_no_left_recursion(self):
        """Test grammar without left recursion."""
        E = nonterminal("E")
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (num,)))
        
        result = detect_left_recursion(grammar)
        assert len(result) == 0


class TestAmbiguity:
    """Test ambiguity detection."""
    
    def test_first_first_conflict(self):
        """Test detecting FIRST/FIRST conflict."""
        E = nonterminal("E")
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.BNF, E)
        # Two productions both starting with num
        grammar.add_production(ProductionRule(E, (num,), label="prod1"))
        grammar.add_production(ProductionRule(E, (num, num), label="prod2"))
        
        # Need to compute first/follow sets first
        first = compute_first_sets(grammar)
        grammar._first_sets = first
        follow = compute_follow_sets(grammar, first)
        grammar._follow_sets = follow
        
        result = detect_ambiguity(grammar)
        
        # Should detect conflict on 'num'
        assert len(result) > 0
    
    def test_no_ambiguity(self):
        """Test grammar without ambiguity."""
        E = nonterminal("E")
        num = terminal("num")
        id_tok = terminal("id")
        
        grammar = Grammar("test", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (num,)))
        grammar.add_production(ProductionRule(E, (id_tok,)))
        
        # Compute first/follow sets
        first = compute_first_sets(grammar)
        grammar._first_sets = first
        follow = compute_follow_sets(grammar, first)
        grammar._follow_sets = follow
        
        result = detect_ambiguity(grammar)
        
        assert len(result) == 0


class TestReachability:
    """Test reachability analysis."""
    
    def test_unreachable_nonterminal(self):
        """Test detecting unreachable non-terminals."""
        E = nonterminal("E")
        T = nonterminal("T")
        X = nonterminal("X")  # Unreachable
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (T,)))
        grammar.add_production(ProductionRule(T, (num,)))
        grammar.add_production(ProductionRule(X, (num,)))  # X not reachable from E
        
        unreachable = find_unreachable_nonterminals(grammar)
        
        assert X in unreachable
        assert E not in unreachable
        assert T not in unreachable
    
    def test_all_reachable(self):
        """Test grammar where all non-terminals are reachable."""
        E = nonterminal("E")
        T = nonterminal("T")
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (T,)))
        grammar.add_production(ProductionRule(T, (num,)))
        
        unreachable = find_unreachable_nonterminals(grammar)
        
        assert len(unreachable) == 0


class TestFullValidation:
    """Test complete grammar validation."""
    
    def test_valid_grammar(self):
        """Test validating a valid grammar."""
        E = nonterminal("E")
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (num,)))
        
        result = validate_grammar(grammar)
        
        assert result.valid
        assert len(result.errors) == 0
    
    def test_undefined_nonterminal(self):
        """Test detecting undefined non-terminal."""
        E = nonterminal("E")
        T = nonterminal("T")  # Never defined
        
        grammar = Grammar("test", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (T,)))  # References undefined T
        
        result = validate_grammar(grammar)
        
        assert not result.valid
        assert any("T" in e for e in result.errors)
    
    def test_peg_left_recursion_error(self):
        """Test that PEG grammar with left recursion is invalid."""
        E = nonterminal("E")
        plus = terminal("+")
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.PEG, E)  # PEG type
        grammar.add_production(ProductionRule(E, (E, plus, num)))
        grammar.add_production(ProductionRule(E, (num,)))
        
        result = validate_grammar(grammar)
        
        # Left recursion should be an error for PEG
        assert not result.valid
    
    def test_bnf_left_recursion_warning(self):
        """Test that BNF grammar with left recursion is valid but warns."""
        E = nonterminal("E")
        plus = terminal("+")
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.BNF, E)  # BNF type
        grammar.add_production(ProductionRule(E, (E, plus, num)))
        grammar.add_production(ProductionRule(E, (num,)))
        
        result = validate_grammar(grammar)
        
        # Should be valid but with warnings
        assert result.valid
        assert len(result.warnings) > 0
    
    def test_validation_info(self):
        """Test that validation provides analysis info."""
        E = nonterminal("E")
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (num,)))
        
        result = validate_grammar(grammar)
        
        # Should have first/follow/nullable info
        assert "first_sets" in result.info
        assert "follow_sets" in result.info
        assert "nullable" in result.info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
