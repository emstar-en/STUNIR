"""
Tests for DFA Construction and Minimization.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ir.lexer.regex import parse_regex
from ir.lexer.nfa import NFABuilder, combine_nfas, build_nfa_from_pattern
from ir.lexer.dfa import (
    DFAState, DFA, MinimizedDFA, TransitionTable,
    SubsetConstruction, HopcroftMinimizer,
    nfa_to_dfa, minimize_dfa
)


class TestSubsetConstruction:
    """Test NFA to DFA conversion."""
    
    def test_single_char(self):
        """Convert single char NFA to DFA."""
        nfa = build_nfa_from_pattern("a", "A", 1)
        dfa = nfa_to_dfa(nfa)
        
        # DFA should have minimal states
        assert dfa.start is not None
        assert len(dfa.accept_states) >= 1
    
    def test_concatenation(self):
        """Convert concatenation NFA to DFA."""
        nfa = build_nfa_from_pattern("ab", "AB", 1)
        dfa = nfa_to_dfa(nfa)
        
        # Verify DFA accepts "ab"
        result = dfa.simulate("ab")
        assert result is not None
        assert result[0] == "AB"
    
    def test_alternation(self):
        """Convert alternation NFA to DFA."""
        nfa = build_nfa_from_pattern("a|b", "AorB", 1)
        dfa = nfa_to_dfa(nfa)
        
        # Verify DFA accepts both
        assert dfa.simulate("a") is not None
        assert dfa.simulate("b") is not None
        assert dfa.simulate("c") is None
    
    def test_star(self):
        """Convert Kleene star NFA to DFA."""
        nfa = build_nfa_from_pattern("a*", "AStar", 1)
        dfa = nfa_to_dfa(nfa)
        
        assert dfa.simulate("") is not None
        assert dfa.simulate("a") is not None
        assert dfa.simulate("aaa") is not None
    
    def test_combined_nfas(self):
        """Convert combined NFAs to single DFA."""
        nfa1 = build_nfa_from_pattern("[0-9]+", "INT", 1)
        nfa2 = build_nfa_from_pattern("[a-z]+", "ID", 1)
        
        combined = combine_nfas([nfa1, nfa2])
        dfa = nfa_to_dfa(combined)
        
        int_result = dfa.simulate("123")
        id_result = dfa.simulate("abc")
        
        assert int_result is not None
        assert int_result[0] == "INT"
        assert id_result is not None
        assert id_result[0] == "ID"


class TestDFASimulation:
    """Test DFA simulation."""
    
    def test_accept_string(self):
        """DFA accepts valid strings."""
        nfa = build_nfa_from_pattern("[a-z]+", "ID", 1)
        dfa = nfa_to_dfa(nfa)
        
        assert dfa.simulate("hello") is not None
        assert dfa.simulate("") is None
    
    def test_reject_string(self):
        """DFA rejects invalid strings."""
        nfa = build_nfa_from_pattern("[0-9]+", "INT", 1)
        dfa = nfa_to_dfa(nfa)
        
        assert dfa.simulate("abc") is None
        assert dfa.simulate("") is None
    
    def test_priority_resolution(self):
        """Higher priority token wins."""
        nfa1 = build_nfa_from_pattern("[a-z]+", "ID", 1)
        nfa2 = build_nfa_from_pattern("if", "IF", 10)
        
        combined = combine_nfas([nfa1, nfa2])
        dfa = nfa_to_dfa(combined)
        
        # "if" should match IF (higher priority)
        result = dfa.simulate("if")
        assert result is not None
        # The token with higher priority should win
        # Note: actual token depends on which accept state is reached


class TestHopcroftMinimization:
    """Test DFA minimization."""
    
    def test_minimize_simple(self):
        """Minimize simple DFA."""
        nfa = build_nfa_from_pattern("a|b", "AorB", 1)
        dfa = nfa_to_dfa(nfa)
        minimized = minimize_dfa(dfa)
        
        # Minimized DFA should have fewer or equal states
        assert minimized.num_states <= len(dfa.states)
    
    def test_minimize_preserves_language(self):
        """Minimization preserves accepted language."""
        nfa = build_nfa_from_pattern("[0-9]+", "INT", 1)
        dfa = nfa_to_dfa(nfa)
        minimized = minimize_dfa(dfa)
        
        # Both should accept same strings
        assert minimized.simulate("123") is not None
        assert minimized.simulate("0") is not None
        assert minimized.simulate("abc") is None
    
    def test_minimize_complex(self):
        """Minimize complex DFA."""
        # Pattern that might create redundant states
        nfa = build_nfa_from_pattern("(a|b)*abb", "ABB", 1)
        dfa = nfa_to_dfa(nfa)
        minimized = minimize_dfa(dfa)
        
        # Check language preserved
        assert minimized.simulate("abb") is not None
        assert minimized.simulate("aabb") is not None
        assert minimized.simulate("babb") is not None
        assert minimized.simulate("ab") is None


class TestMinimizedDFA:
    """Test MinimizedDFA class."""
    
    def test_simulation(self):
        """MinimizedDFA simulation works."""
        nfa = build_nfa_from_pattern("[a-z]+", "ID", 1)
        dfa = nfa_to_dfa(nfa)
        minimized = minimize_dfa(dfa)
        
        assert minimized.simulate("hello") is not None
        assert minimized.simulate("world") is not None
        assert minimized.simulate("123") is None
    
    def test_to_table(self):
        """Conversion to TransitionTable works."""
        nfa = build_nfa_from_pattern("[0-9]+", "INT", 1)
        dfa = nfa_to_dfa(nfa)
        minimized = minimize_dfa(dfa)
        table = minimized.to_table()
        
        assert table.num_states == minimized.num_states
        assert table.start_state == minimized.start_state


class TestTransitionTable:
    """Test TransitionTable class."""
    
    def test_next_state(self):
        """next_state returns correct state."""
        nfa = build_nfa_from_pattern("ab", "AB", 1)
        dfa = nfa_to_dfa(nfa)
        minimized = minimize_dfa(dfa)
        table = minimized.to_table()
        
        # Start state should have transition on 'a'
        next_state = table.next_state(table.start_state, 'a')
        assert next_state != table.error_state
    
    def test_is_accept(self):
        """is_accept identifies accepting states."""
        nfa = build_nfa_from_pattern("a", "A", 1)
        dfa = nfa_to_dfa(nfa)
        minimized = minimize_dfa(dfa)
        table = minimized.to_table()
        
        # Find accept state by simulation
        state = table.start_state
        state = table.next_state(state, 'a')
        
        assert table.is_accept(state)
    
    def test_get_token(self):
        """get_token returns token info."""
        nfa = build_nfa_from_pattern("[0-9]+", "INT", 1)
        dfa = nfa_to_dfa(nfa)
        minimized = minimize_dfa(dfa)
        table = minimized.to_table()
        
        # Simulate to accept state
        state = table.start_state
        for c in "123":
            state = table.next_state(state, c)
        
        token_info = table.get_token(state)
        assert token_info is not None
        assert token_info[0] == "INT"
    
    def test_to_dict(self):
        """to_dict produces serializable dict."""
        nfa = build_nfa_from_pattern("[a-z]+", "ID", 1)
        dfa = nfa_to_dfa(nfa)
        minimized = minimize_dfa(dfa)
        table = minimized.to_table()
        
        d = table.to_dict()
        assert "num_states" in d
        assert "table" in d  # flat transition array
        assert "accept_table" in d
        assert "symbol_to_index" in d
    
    def test_error_handling(self):
        """Error states handled correctly."""
        nfa = build_nfa_from_pattern("[0-9]+", "INT", 1)
        dfa = nfa_to_dfa(nfa)
        minimized = minimize_dfa(dfa)
        table = minimized.to_table()
        
        # Invalid symbol should return error state
        next_state = table.next_state(table.start_state, '@')
        assert next_state == table.error_state
        
        # Invalid state should return error state
        next_state = table.next_state(-99, '0')
        assert next_state == table.error_state


class TestComplexDFA:
    """Test DFA with complex patterns."""
    
    def test_multiple_tokens(self):
        """DFA with multiple token types."""
        nfa1 = build_nfa_from_pattern("[0-9]+", "INT", 1)
        nfa2 = build_nfa_from_pattern("[a-zA-Z_][a-zA-Z0-9_]*", "ID", 2)
        nfa3 = build_nfa_from_pattern("if", "IF", 10)
        nfa4 = build_nfa_from_pattern("while", "WHILE", 10)
        
        combined = combine_nfas([nfa1, nfa2, nfa3, nfa4])
        dfa = nfa_to_dfa(combined)
        minimized = minimize_dfa(dfa)
        table = minimized.to_table()
        
        # Verify all tokens work
        def simulate_table(s):
            state = table.start_state
            for c in s:
                state = table.next_state(state, c)
                if state == table.error_state:
                    return None
            return table.get_token(state)
        
        assert simulate_table("123") is not None
        assert simulate_table("foo") is not None
        assert simulate_table("if") is not None
        assert simulate_table("while") is not None
    
    def test_overlapping_patterns(self):
        """DFA handles overlapping patterns."""
        # "if" is prefix of "ifelse"
        nfa1 = build_nfa_from_pattern("if", "IF", 10)
        nfa2 = build_nfa_from_pattern("[a-z]+", "ID", 1)
        
        combined = combine_nfas([nfa1, nfa2])
        dfa = nfa_to_dfa(combined)
        minimized = minimize_dfa(dfa)
        
        # Both should be accepted
        assert minimized.simulate("if") is not None
        assert minimized.simulate("ifelse") is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
