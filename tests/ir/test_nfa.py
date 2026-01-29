"""
Tests for NFA Construction (Thompson's Algorithm).
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ir.lexer.regex import parse_regex
from ir.lexer.nfa import (
    NFAState, NFA, NFABuilder, EPSILON,
    combine_nfas, build_nfa_from_pattern
)


class TestNFAState:
    """Test NFAState class."""
    
    def test_state_creation(self):
        """Test basic state creation."""
        state = NFAState(id=0)
        assert state.id == 0
        assert not state.is_accept
        assert state.accept_token is None
    
    def test_add_transition(self):
        """Test adding transitions."""
        s1 = NFAState(id=0)
        s2 = NFAState(id=1)
        
        s1.add_transition('a', s2)
        assert s2 in s1.get_transitions('a')
    
    def test_add_epsilon(self):
        """Test adding epsilon transitions."""
        s1 = NFAState(id=0)
        s2 = NFAState(id=1)
        
        s1.add_epsilon(s2)
        assert s2 in s1.get_epsilon_transitions()
    
    def test_accept_state(self):
        """Test accept state properties."""
        state = NFAState(id=0, is_accept=True, accept_token="INT", accept_priority=10)
        assert state.is_accept
        assert state.accept_token == "INT"
        assert state.accept_priority == 10


class TestNFABuilder:
    """Test NFA building using Thompson's construction."""
    
    def test_single_char(self):
        """Single character produces 2-state NFA."""
        ast = parse_regex("a")
        builder = NFABuilder()
        nfa = builder.build(ast, "A", 1)
        
        assert len(nfa.states) == 2
        assert nfa.accept.is_accept
        assert nfa.accept.accept_token == "A"
    
    def test_concatenation(self):
        """Concatenation chains NFAs."""
        ast = parse_regex("ab")
        builder = NFABuilder()
        nfa = builder.build(ast, "AB", 1)
        
        # Thompson's: a(2) + b(2) + epsilon = 4 states
        assert len(nfa.states) == 4
        assert nfa.accept.is_accept
    
    def test_alternation(self):
        """Alternation creates parallel paths."""
        ast = parse_regex("a|b")
        builder = NFABuilder()
        nfa = builder.build(ast, "AorB", 1)
        
        # Thompson's: a(2) + b(2) + new start/end = 6 states
        assert len(nfa.states) == 6
    
    def test_star(self):
        """Kleene star adds loop."""
        ast = parse_regex("a*")
        builder = NFABuilder()
        nfa = builder.build(ast, "AStar", 1)
        
        # Thompson's: a(2) + new start/end = 4 states
        assert len(nfa.states) == 4
    
    def test_plus(self):
        """Plus adds one-or-more loop."""
        ast = parse_regex("a+")
        builder = NFABuilder()
        nfa = builder.build(ast, "APlus", 1)
        
        assert len(nfa.states) == 4
    
    def test_optional(self):
        """Optional adds bypass."""
        ast = parse_regex("a?")
        builder = NFABuilder()
        nfa = builder.build(ast, "AOpt", 1)
        
        assert len(nfa.states) == 4


class TestNFAEpsilonClosure:
    """Test epsilon closure computation."""
    
    def test_no_epsilon(self):
        """State with no epsilon transitions."""
        s1 = NFAState(id=0)
        s2 = NFAState(id=1)
        s1.add_transition('a', s2)
        
        nfa = NFA(start=s1, accept=s2, states=[s1, s2], alphabet={'a'})
        closure = nfa.epsilon_closure({s1})
        
        assert closure == frozenset({s1})
    
    def test_single_epsilon(self):
        """State with one epsilon transition."""
        s1 = NFAState(id=0)
        s2 = NFAState(id=1)
        s1.add_epsilon(s2)
        
        nfa = NFA(start=s1, accept=s2, states=[s1, s2], alphabet=set())
        closure = nfa.epsilon_closure({s1})
        
        assert closure == frozenset({s1, s2})
    
    def test_chain_epsilon(self):
        """Chain of epsilon transitions."""
        s1 = NFAState(id=0)
        s2 = NFAState(id=1)
        s3 = NFAState(id=2)
        s1.add_epsilon(s2)
        s2.add_epsilon(s3)
        
        nfa = NFA(start=s1, accept=s3, states=[s1, s2, s3], alphabet=set())
        closure = nfa.epsilon_closure({s1})
        
        assert closure == frozenset({s1, s2, s3})
    
    def test_branching_epsilon(self):
        """Branching epsilon transitions."""
        s1 = NFAState(id=0)
        s2 = NFAState(id=1)
        s3 = NFAState(id=2)
        s1.add_epsilon(s2)
        s1.add_epsilon(s3)
        
        nfa = NFA(start=s1, accept=s3, states=[s1, s2, s3], alphabet=set())
        closure = nfa.epsilon_closure({s1})
        
        assert closure == frozenset({s1, s2, s3})


class TestNFAMove:
    """Test move operation."""
    
    def test_simple_move(self):
        """Move on single transition."""
        s1 = NFAState(id=0)
        s2 = NFAState(id=1)
        s1.add_transition('a', s2)
        
        nfa = NFA(start=s1, accept=s2, states=[s1, s2], alphabet={'a'})
        result = nfa.move({s1}, 'a')
        
        assert result == {s2}
    
    def test_no_transition(self):
        """Move with no matching transition."""
        s1 = NFAState(id=0)
        s2 = NFAState(id=1)
        s1.add_transition('a', s2)
        
        nfa = NFA(start=s1, accept=s2, states=[s1, s2], alphabet={'a', 'b'})
        result = nfa.move({s1}, 'b')
        
        assert result == set()
    
    def test_multiple_targets(self):
        """Move with multiple target states."""
        s1 = NFAState(id=0)
        s2 = NFAState(id=1)
        s3 = NFAState(id=2)
        s1.add_transition('a', s2)
        s1.add_transition('a', s3)
        
        nfa = NFA(start=s1, accept=s3, states=[s1, s2, s3], alphabet={'a'})
        result = nfa.move({s1}, 'a')
        
        assert result == {s2, s3}


class TestNFASimulation:
    """Test NFA simulation."""
    
    def test_accept_single_char(self):
        """NFA accepts single character."""
        nfa = build_nfa_from_pattern("a", "A", 1)
        assert nfa.simulate("a")
        assert not nfa.simulate("b")
        assert not nfa.simulate("aa")
    
    def test_accept_concatenation(self):
        """NFA accepts concatenation."""
        nfa = build_nfa_from_pattern("ab", "AB", 1)
        assert nfa.simulate("ab")
        assert not nfa.simulate("a")
        assert not nfa.simulate("b")
        assert not nfa.simulate("abc")
    
    def test_accept_alternation(self):
        """NFA accepts alternation."""
        nfa = build_nfa_from_pattern("a|b", "AorB", 1)
        assert nfa.simulate("a")
        assert nfa.simulate("b")
        assert not nfa.simulate("c")
        assert not nfa.simulate("ab")
    
    def test_accept_star(self):
        """NFA accepts Kleene star."""
        nfa = build_nfa_from_pattern("a*", "AStar", 1)
        assert nfa.simulate("")
        assert nfa.simulate("a")
        assert nfa.simulate("aa")
        assert nfa.simulate("aaa")
        assert not nfa.simulate("b")
    
    def test_accept_plus(self):
        """NFA accepts plus."""
        nfa = build_nfa_from_pattern("a+", "APlus", 1)
        assert not nfa.simulate("")
        assert nfa.simulate("a")
        assert nfa.simulate("aa")
        assert nfa.simulate("aaa")
    
    def test_accept_optional(self):
        """NFA accepts optional."""
        nfa = build_nfa_from_pattern("a?", "AOpt", 1)
        assert nfa.simulate("")
        assert nfa.simulate("a")
        assert not nfa.simulate("aa")


class TestCombineNFAs:
    """Test combining multiple NFAs."""
    
    def test_combine_two(self):
        """Combine two NFAs."""
        nfa1 = build_nfa_from_pattern("a", "A", 2)
        nfa2 = build_nfa_from_pattern("b", "B", 1)
        
        combined = combine_nfas([nfa1, nfa2])
        
        # Combined NFA accepts both patterns
        assert combined.simulate("a")
        assert combined.simulate("b")
        assert not combined.simulate("c")
    
    def test_combine_preserves_priority(self):
        """Combined NFA preserves token priorities."""
        nfa1 = build_nfa_from_pattern("[a-z]+", "ID", 1)
        nfa2 = build_nfa_from_pattern("if", "IF", 10)
        
        combined = combine_nfas([nfa1, nfa2])
        
        # Both patterns match "if", but check accept states exist
        accept_states = combined.get_accept_states()
        assert len(accept_states) >= 2


class TestComplexPatterns:
    """Test complex pattern compilation."""
    
    def test_identifier(self):
        """Identifier pattern works."""
        nfa = build_nfa_from_pattern("[a-zA-Z_][a-zA-Z0-9_]*", "ID", 1)
        assert nfa.simulate("foo")
        assert nfa.simulate("_bar")
        assert nfa.simulate("x123")
        assert not nfa.simulate("123")
        assert not nfa.simulate("")
    
    def test_integer(self):
        """Integer pattern works."""
        nfa = build_nfa_from_pattern("[0-9]+", "INT", 1)
        assert nfa.simulate("0")
        assert nfa.simulate("123")
        assert nfa.simulate("456789")
        assert not nfa.simulate("")
        assert not nfa.simulate("12a")
    
    def test_float(self):
        """Float pattern works."""
        nfa = build_nfa_from_pattern("[0-9]+\\.[0-9]+", "FLOAT", 1)
        assert nfa.simulate("1.0")
        assert nfa.simulate("123.456")
        assert not nfa.simulate("1")
        assert not nfa.simulate(".5")
        assert not nfa.simulate("1.")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
