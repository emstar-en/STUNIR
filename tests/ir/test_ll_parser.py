#!/usr/bin/env python3
"""Tests for LL(1) Parser Generator.

Tests:
- LL(1) table construction
- LL(1) condition checking
- Conflict detection
- Recursive descent generation
"""

import pytest
import sys
import os

# Add the repository root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ir.grammar.grammar_ir import Grammar, GrammarType
from ir.grammar.symbol import Symbol, SymbolType, terminal, nonterminal, EPSILON, EOF
from ir.grammar.production import ProductionRule

from ir.parser.parse_table import ParserType, LL1Table, LL1Conflict
from ir.parser.ll_parser import (
    LLParserGenerator, build_ll1_table, check_ll1_conditions,
    compute_first_of_string, generate_recursive_descent
)


class TestLL1Table:
    """Test LL(1) table construction."""
    
    def test_simple_ll1_table(self):
        """Test building LL(1) table for simple grammar."""
        S = nonterminal("S")
        a = terminal("a")
        
        grammar = Grammar("simple", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(S, (a,)))
        
        table = build_ll1_table(grammar)
        
        assert not table.has_conflicts()
        prod = table.get_production(S, a)
        assert prod is not None
    
    def test_ll1_table_with_epsilon(self):
        """Test LL(1) table with epsilon production."""
        S = nonterminal("S")
        A = nonterminal("A")
        a = terminal("a")
        b = terminal("b")
        
        grammar = Grammar("epsilon", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(S, (a, A)))
        grammar.add_production(ProductionRule(A, (b,)))
        grammar.add_production(ProductionRule(A, ()))  # A → ε
        
        table = build_ll1_table(grammar)
        
        # Should have entries in table
        assert len(table.table) > 0
    
    def test_ll1_table_multiple_productions(self):
        """Test LL(1) table with multiple alternatives."""
        S = nonterminal("S")
        a = terminal("a")
        b = terminal("b")
        
        grammar = Grammar("alternatives", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(S, (a,)))
        grammar.add_production(ProductionRule(S, (b,)))
        
        table = build_ll1_table(grammar)
        
        # Should have entry for both 'a' and 'b'
        assert table.get_production(S, a) is not None
        assert table.get_production(S, b) is not None
        assert not table.has_conflicts()
    
    def test_ll1_conflict_detection(self):
        """Test detecting LL(1) conflicts."""
        # Grammar with FIRST/FIRST conflict: S → a | a b
        S = nonterminal("S")
        a = terminal("a")
        b = terminal("b")
        
        grammar = Grammar("conflict", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(S, (a,)))
        grammar.add_production(ProductionRule(S, (a, b)))
        
        table = build_ll1_table(grammar)
        
        # Both productions start with 'a', so there's a conflict
        assert table.has_conflicts()
        assert len(table.conflicts) > 0


class TestLL1Conditions:
    """Test LL(1) condition checking."""
    
    def test_ll1_valid_grammar(self):
        """Test valid LL(1) grammar."""
        S = nonterminal("S")
        a = terminal("a")
        b = terminal("b")
        
        grammar = Grammar("ll1", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(S, (a,)))
        grammar.add_production(ProductionRule(S, (b,)))
        
        is_ll1, issues = check_ll1_conditions(grammar)
        
        assert is_ll1
        assert len(issues) == 0
    
    def test_left_recursion_detection(self):
        """Test detecting left recursion."""
        E = nonterminal("E")
        plus = terminal("+")
        num = terminal("num")
        
        grammar = Grammar("left_recursive", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (E, plus, num)))
        grammar.add_production(ProductionRule(E, (num,)))
        
        is_ll1, issues = check_ll1_conditions(grammar)
        
        assert not is_ll1
        assert any("left recursion" in issue.lower() for issue in issues)
    
    def test_first_first_conflict(self):
        """Test detecting FIRST/FIRST conflict."""
        S = nonterminal("S")
        a = terminal("a")
        
        grammar = Grammar("first_conflict", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(S, (a,)))
        grammar.add_production(ProductionRule(S, (a,)))  # Duplicate
        
        is_ll1, issues = check_ll1_conditions(grammar)
        
        # Same production twice - conflict
        assert len(issues) > 0 or not is_ll1


class TestFirstOfString:
    """Test FIRST set computation for strings."""
    
    def test_first_of_terminal(self):
        """Test FIRST of single terminal."""
        from ir.grammar.validation import compute_first_sets, compute_nullable
        
        S = nonterminal("S")
        a = terminal("a")
        
        grammar = Grammar("test", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(S, (a,)))
        
        first_sets = compute_first_sets(grammar)
        nullable = compute_nullable(grammar)
        
        result = compute_first_of_string((a,), first_sets, nullable)
        
        assert a in result
    
    def test_first_of_nonterminal(self):
        """Test FIRST of nonterminal."""
        from ir.grammar.validation import compute_first_sets, compute_nullable
        
        S = nonterminal("S")
        A = nonterminal("A")
        a = terminal("a")
        
        grammar = Grammar("test", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(S, (A,)))
        grammar.add_production(ProductionRule(A, (a,)))
        
        first_sets = compute_first_sets(grammar)
        nullable = compute_nullable(grammar)
        
        result = compute_first_of_string((A,), first_sets, nullable)
        
        assert a in result
    
    def test_first_of_empty(self):
        """Test FIRST of empty string."""
        from ir.grammar.validation import compute_first_sets, compute_nullable
        
        S = nonterminal("S")
        a = terminal("a")
        
        grammar = Grammar("test", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(S, (a,)))
        
        first_sets = compute_first_sets(grammar)
        nullable = compute_nullable(grammar)
        
        result = compute_first_of_string((), first_sets, nullable)
        
        # Empty string should have epsilon in FIRST
        assert EPSILON in result or len(result) == 1


class TestLLParserGenerator:
    """Test LLParserGenerator class."""
    
    def test_generator_creation(self):
        """Test creating an LL parser generator."""
        gen = LLParserGenerator()
        assert gen.get_parser_type() == ParserType.LL1
    
    def test_generate_parser(self):
        """Test generating an LL(1) parser."""
        S = nonterminal("S")
        a = terminal("a")
        b = terminal("b")
        
        grammar = Grammar("simple", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(S, (a,)))
        grammar.add_production(ProductionRule(S, (b,)))
        
        gen = LLParserGenerator()
        result = gen.generate(grammar)
        
        assert result.parser_type == ParserType.LL1
        assert "table_entries" in result.info
    
    def test_supports_grammar_valid(self):
        """Test checking support for valid LL(1) grammar."""
        S = nonterminal("S")
        a = terminal("a")
        
        grammar = Grammar("simple", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(S, (a,)))
        
        gen = LLParserGenerator()
        supported, issues = gen.supports_grammar(grammar)
        
        assert supported
    
    def test_supports_grammar_invalid(self):
        """Test checking support for invalid LL(1) grammar."""
        E = nonterminal("E")
        plus = terminal("+")
        num = terminal("num")
        
        grammar = Grammar("left_recursive", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (E, plus, num)))
        grammar.add_production(ProductionRule(E, (num,)))
        
        gen = LLParserGenerator()
        supported, issues = gen.supports_grammar(grammar)
        
        assert not supported
        assert len(issues) > 0
    
    def test_generate_with_ast(self):
        """Test generating parser with AST schema."""
        S = nonterminal("S")
        a = terminal("a")
        
        grammar = Grammar("simple", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(S, (a,)))
        
        gen = LLParserGenerator(generate_ast=True)
        result = gen.generate(grammar)
        
        assert result.ast_schema is not None


class TestRecursiveDescentGeneration:
    """Test recursive descent parser generation."""
    
    def test_generate_rd_skeleton(self):
        """Test generating recursive descent parser skeleton."""
        S = nonterminal("S")
        A = nonterminal("A")
        a = terminal("a")
        b = terminal("b")
        
        grammar = Grammar("rd_test", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(S, (A,)))
        grammar.add_production(ProductionRule(A, (a,)))
        grammar.add_production(ProductionRule(A, (b,)))
        
        code = generate_recursive_descent(grammar)
        
        assert "class Parser" in code
        assert "parse_s" in code.lower()
        assert "parse_a" in code.lower()
    
    def test_rd_single_production(self):
        """Test RD generation with single production per nonterminal."""
        S = nonterminal("S")
        a = terminal("a")
        
        grammar = Grammar("single", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(S, (a,)))
        
        code = generate_recursive_descent(grammar)
        
        assert "def parse_s" in code.lower()


class TestLL1TableMethods:
    """Test LL1Table class methods."""
    
    def test_table_getters(self):
        """Test table getter methods."""
        S = nonterminal("S")
        a = terminal("a")
        b = terminal("b")
        
        grammar = Grammar("test", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(S, (a,)))
        grammar.add_production(ProductionRule(S, (b,)))
        
        table = build_ll1_table(grammar)
        
        nonterminals = table.get_nonterminals()
        terminals = table.get_terminals()
        
        assert S in nonterminals
        assert a in terminals
        assert b in terminals
    
    def test_table_to_dict(self):
        """Test converting table to dictionary."""
        S = nonterminal("S")
        a = terminal("a")
        
        grammar = Grammar("test", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(S, (a,)))
        
        table = build_ll1_table(grammar)
        data = table.to_dict()
        
        assert "table" in data
        assert "conflict_count" in data
    
    def test_table_string_representation(self):
        """Test table string representation."""
        S = nonterminal("S")
        a = terminal("a")
        
        grammar = Grammar("test", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(S, (a,)))
        
        table = build_ll1_table(grammar)
        str_repr = str(table)
        
        assert "LL(1)" in str_repr
        assert "Entries" in str_repr


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
