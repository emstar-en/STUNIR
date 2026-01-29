#!/usr/bin/env python3
"""Tests for LR Parser Generator.

Tests:
- LR item construction
- Closure and GOTO operations
- LR(0) item set construction
- Parse table construction
- Conflict detection
"""

import pytest
import sys
import os

# Add the repository root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ir.grammar.grammar_ir import Grammar, GrammarType
from ir.grammar.symbol import Symbol, SymbolType, terminal, nonterminal, EPSILON, EOF
from ir.grammar.production import ProductionRule

from ir.parser.parse_table import (
    ParserType, LRItem, LRItemSet, Action, ActionType,
    ParseTable, Conflict
)
from ir.parser.lr_parser import (
    LRParserGenerator, closure, goto, 
    build_lr0_items, build_parse_table,
    detect_conflicts, resolve_conflicts
)


class TestLRItem:
    """Test LR item creation and operations."""
    
    def test_item_creation(self):
        """Test creating an LR item."""
        E = nonterminal("E")
        T = nonterminal("T")
        plus = terminal("+")
        
        prod = ProductionRule(E, (E, plus, T))
        item = LRItem(prod, 0)
        
        assert item.production == prod
        assert item.dot_position == 0
        assert not item.is_complete()
    
    def test_next_symbol(self):
        """Test getting next symbol after dot."""
        E = nonterminal("E")
        T = nonterminal("T")
        plus = terminal("+")
        
        prod = ProductionRule(E, (E, plus, T))
        
        item0 = LRItem(prod, 0)
        assert item0.next_symbol() == E
        
        item1 = LRItem(prod, 1)
        assert item1.next_symbol() == plus
        
        item2 = LRItem(prod, 2)
        assert item2.next_symbol() == T
        
        item3 = LRItem(prod, 3)
        assert item3.next_symbol() is None
    
    def test_is_complete(self):
        """Test checking if item is complete."""
        E = nonterminal("E")
        T = nonterminal("T")
        plus = terminal("+")
        
        prod = ProductionRule(E, (E, plus, T))
        
        assert not LRItem(prod, 0).is_complete()
        assert not LRItem(prod, 1).is_complete()
        assert not LRItem(prod, 2).is_complete()
        assert LRItem(prod, 3).is_complete()
    
    def test_advance(self):
        """Test advancing an item."""
        E = nonterminal("E")
        num = terminal("num")
        
        prod = ProductionRule(E, (num,))
        
        item0 = LRItem(prod, 0)
        item1 = item0.advance()
        
        assert item1.dot_position == 1
        assert item1.is_complete()
    
    def test_epsilon_production_item(self):
        """Test item for epsilon production."""
        E = nonterminal("E")
        
        prod = ProductionRule(E, ())  # E -> epsilon
        item = LRItem(prod, 0)
        
        assert item.is_complete()
        assert item.next_symbol() is None
    
    def test_item_with_lookahead(self):
        """Test LR(1) item with lookahead."""
        E = nonterminal("E")
        num = terminal("num")
        plus = terminal("+")
        
        prod = ProductionRule(E, (num,))
        item = LRItem(prod, 0, lookahead=plus)
        
        assert item.lookahead == plus
        
        # Core should not have lookahead
        core = item.core()
        assert core.lookahead is None
    
    def test_item_string_representation(self):
        """Test item string representation."""
        E = nonterminal("E")
        num = terminal("num")
        
        prod = ProductionRule(E, (num,))
        item = LRItem(prod, 0)
        
        str_repr = str(item)
        assert "E" in str_repr
        assert "•" in str_repr


class TestLRItemSet:
    """Test LR item set operations."""
    
    def test_item_set_creation(self):
        """Test creating an item set."""
        E = nonterminal("E")
        num = terminal("num")
        
        prod = ProductionRule(E, (num,))
        item = LRItem(prod, 0)
        
        item_set = LRItemSet(frozenset([item]), state_id=0)
        
        assert len(item_set.items) == 1
        assert item_set.state_id == 0
    
    def test_kernel_items(self):
        """Test getting kernel items."""
        E = nonterminal("E")
        num = terminal("num")
        
        prod = ProductionRule(E, (num,))
        
        # Non-kernel item (dot at start)
        item0 = LRItem(prod, 0)
        # Kernel item (dot not at start)
        item1 = LRItem(prod, 1)
        
        item_set = LRItemSet(frozenset([item0, item1]), state_id=0)
        kernel = item_set.kernel_items()
        
        assert item1 in kernel
        # item0 might or might not be kernel depending on if it's augmented start
    
    def test_get_shift_symbols(self):
        """Test getting shiftable symbols."""
        E = nonterminal("E")
        T = nonterminal("T")
        num = terminal("num")
        plus = terminal("+")
        
        prod1 = ProductionRule(E, (E, plus, T))
        prod2 = ProductionRule(E, (num,))
        
        items = frozenset([
            LRItem(prod1, 0),  # E → • E + T (can shift E)
            LRItem(prod2, 0),  # E → • num (can shift num)
        ])
        
        item_set = LRItemSet(items, state_id=0)
        symbols = item_set.get_shift_symbols()
        
        assert E in symbols
        assert num in symbols


class TestClosureAndGoto:
    """Test closure and GOTO operations."""
    
    def test_closure_simple(self):
        """Test closure of simple item set."""
        E = nonterminal("E")
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (num,)))
        
        # Initial item: E → • num
        initial = LRItem(grammar.all_productions()[0], 0)
        result = closure({initial}, grammar)
        
        # Closure should contain just the initial item (no nonterminal after dot)
        assert initial in result
    
    def test_closure_with_nonterminal(self):
        """Test closure when dot is before nonterminal."""
        E = nonterminal("E")
        T = nonterminal("T")
        num = terminal("num")
        
        grammar = Grammar("expr", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (T,)))
        grammar.add_production(ProductionRule(T, (num,)))
        
        # Initial item: E → • T
        initial = LRItem(grammar.all_productions()[0], 0)
        result = closure({initial}, grammar)
        
        # Closure should also include T → • num
        assert len(result) >= 2
    
    def test_goto_terminal(self):
        """Test GOTO on terminal."""
        E = nonterminal("E")
        num = terminal("num")
        
        grammar = Grammar("test", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (num,)))
        
        items = frozenset([LRItem(grammar.all_productions()[0], 0)])
        result = goto(items, num, grammar)
        
        # Should have E → num •
        assert len(result) == 1
        item = list(result)[0]
        assert item.dot_position == 1
        assert item.is_complete()
    
    def test_goto_nonterminal(self):
        """Test GOTO on nonterminal."""
        E = nonterminal("E")
        T = nonterminal("T")
        num = terminal("num")
        plus = terminal("+")
        
        grammar = Grammar("expr", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (E, plus, T)))
        grammar.add_production(ProductionRule(E, (T,)))
        grammar.add_production(ProductionRule(T, (num,)))
        
        # Start with closure of E' → • E
        augmented = Symbol("S'", SymbolType.NONTERMINAL)
        start_prod = ProductionRule(augmented, (E,))
        initial = LRItem(start_prod, 0)
        
        items = closure({initial}, grammar)
        result = goto(items, E, grammar)
        
        # Should move dot past E
        assert len(result) > 0
    
    def test_goto_empty_result(self):
        """Test GOTO with no matching items."""
        E = nonterminal("E")
        num = terminal("num")
        plus = terminal("+")
        
        grammar = Grammar("test", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (num,)))
        
        items = frozenset([LRItem(grammar.all_productions()[0], 0)])
        
        # GOTO on '+' should be empty (num expected, not +)
        result = goto(items, plus, grammar)
        assert len(result) == 0


class TestLR0ItemSets:
    """Test LR(0) item set construction."""
    
    def test_simple_grammar_items(self):
        """Test item set construction for simple grammar."""
        S = nonterminal("S")
        num = terminal("num")
        
        grammar = Grammar("simple", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(S, (num,)))
        
        states, start_prod = build_lr0_items(grammar)
        
        # Should have at least 2 states: initial and after shifting num
        assert len(states) >= 2
        assert start_prod.head.name == "S'"
    
    def test_expression_grammar_items(self):
        """Test item set construction for expression grammar."""
        E = nonterminal("E")
        T = nonterminal("T")
        num = terminal("num")
        plus = terminal("+")
        
        grammar = Grammar("expr", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (E, plus, T)))
        grammar.add_production(ProductionRule(E, (T,)))
        grammar.add_production(ProductionRule(T, (num,)))
        
        states, start_prod = build_lr0_items(grammar)
        
        # Expression grammar should have several states
        assert len(states) >= 5
    
    def test_state_ids_unique(self):
        """Test that state IDs are unique."""
        S = nonterminal("S")
        A = nonterminal("A")
        a = terminal("a")
        b = terminal("b")
        
        grammar = Grammar("test", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(S, (A,)))
        grammar.add_production(ProductionRule(A, (a,)))
        grammar.add_production(ProductionRule(A, (b,)))
        
        states, _ = build_lr0_items(grammar)
        
        ids = [s.state_id for s in states]
        assert len(ids) == len(set(ids))  # All unique


class TestParseTableConstruction:
    """Test parse table construction."""
    
    def test_simple_grammar_table(self):
        """Test table construction for simple grammar."""
        S = nonterminal("S")
        num = terminal("num")
        
        grammar = Grammar("simple", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(S, (num,)))
        
        table = build_parse_table(grammar, ParserType.LALR1)
        
        assert table.state_count() > 0
        assert len(table.productions) > 0
        assert not table.has_conflicts()
    
    def test_table_has_accept(self):
        """Test that table has accept action."""
        S = nonterminal("S")
        num = terminal("num")
        
        grammar = Grammar("simple", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(S, (num,)))
        
        table = build_parse_table(grammar, ParserType.LALR1)
        
        # Should have at least one accept action
        has_accept = any(
            action.is_accept() 
            for action in table.action.values()
        )
        assert has_accept
    
    def test_slr1_table(self):
        """Test SLR(1) table construction."""
        S = nonterminal("S")
        num = terminal("num")
        
        grammar = Grammar("simple", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(S, (num,)))
        
        table = build_parse_table(grammar, ParserType.SLR1)
        
        assert table.parser_type == ParserType.SLR1
        assert table.state_count() > 0
    
    def test_lr0_table(self):
        """Test LR(0) table construction."""
        S = nonterminal("S")
        num = terminal("num")
        
        grammar = Grammar("simple", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(S, (num,)))
        
        table = build_parse_table(grammar, ParserType.LR0)
        
        assert table.parser_type == ParserType.LR0


class TestConflictDetection:
    """Test conflict detection."""
    
    def test_shift_reduce_conflict(self):
        """Test detecting shift-reduce conflict."""
        # Classic ambiguous grammar: E → E + E | num
        E = nonterminal("E")
        plus = terminal("+")
        num = terminal("num")
        
        grammar = Grammar("ambiguous", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (E, plus, E)))
        grammar.add_production(ProductionRule(E, (num,)))
        
        table = build_parse_table(grammar, ParserType.LALR1)
        
        # This grammar has shift-reduce conflicts
        conflicts = detect_conflicts(table)
        assert len(conflicts) > 0
        assert any(c.is_shift_reduce() for c in conflicts)
    
    def test_no_conflict_simple_grammar(self):
        """Test no conflicts for simple unambiguous grammar."""
        S = nonterminal("S")
        num = terminal("num")
        
        grammar = Grammar("simple", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(S, (num,)))
        
        table = build_parse_table(grammar, ParserType.LALR1)
        
        conflicts = detect_conflicts(table)
        assert len(conflicts) == 0


class TestConflictResolution:
    """Test conflict resolution."""
    
    def test_resolve_shift_prefer(self):
        """Test resolving conflicts by preferring shift."""
        E = nonterminal("E")
        plus = terminal("+")
        num = terminal("num")
        
        grammar = Grammar("ambiguous", GrammarType.BNF, E)
        grammar.add_production(ProductionRule(E, (E, plus, E)))
        grammar.add_production(ProductionRule(E, (num,)))
        
        table = build_parse_table(grammar, ParserType.LALR1)
        resolved = resolve_conflicts(table, strategy="shift")
        
        # Resolved table should still work
        assert resolved.state_count() == table.state_count()


class TestLRParserGenerator:
    """Test LRParserGenerator class."""
    
    def test_generator_creation(self):
        """Test creating a parser generator."""
        gen = LRParserGenerator(ParserType.LALR1)
        assert gen.parser_type == ParserType.LALR1
        assert gen.get_parser_type() == ParserType.LALR1
    
    def test_generator_invalid_type(self):
        """Test creating generator with invalid type."""
        with pytest.raises(ValueError):
            LRParserGenerator(ParserType.LL1)  # LL1 not supported by LR generator
    
    def test_generate_parser(self):
        """Test generating a parser."""
        S = nonterminal("S")
        num = terminal("num")
        
        grammar = Grammar("simple", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(S, (num,)))
        
        gen = LRParserGenerator(ParserType.LALR1)
        result = gen.generate(grammar)
        
        assert result.is_successful()
        assert result.parser_type == ParserType.LALR1
        assert "state_count" in result.info
    
    def test_supports_grammar(self):
        """Test checking grammar support."""
        S = nonterminal("S")
        num = terminal("num")
        
        grammar = Grammar("simple", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(S, (num,)))
        
        gen = LRParserGenerator(ParserType.LALR1)
        supported, issues = gen.supports_grammar(grammar)
        
        assert supported
        assert len(issues) == 0
    
    def test_generate_with_ast(self):
        """Test generating parser with AST schema."""
        S = nonterminal("S")
        num = terminal("num")
        
        grammar = Grammar("simple", GrammarType.BNF, S)
        grammar.add_production(ProductionRule(S, (num,)))
        
        gen = LRParserGenerator(ParserType.LALR1, generate_ast=True)
        result = gen.generate(grammar)
        
        assert result.ast_schema is not None
        assert result.ast_schema.node_count() > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
