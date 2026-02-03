"""
NFA (Non-deterministic Finite Automaton) Module for STUNIR Lexer Generator.

Implements Thompson's construction for converting regex AST to NFA.
Provides NFA data structures and operations including epsilon closure.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from .regex import RegexNode


# Epsilon transition symbol (empty string)
EPSILON = ''


@dataclass
class NFAState:
    """
    NFA state with transitions.
    
    Attributes:
        id: Unique state identifier
        transitions: Mapping from symbol to set of target states
        is_accept: Whether this is an accepting state
        accept_token: Token name if accepting
        accept_priority: Token priority (higher = higher priority)
    """
    id: int
    transitions: Dict[str, Set['NFAState']] = field(default_factory=lambda: defaultdict(set))
    is_accept: bool = False
    accept_token: Optional[str] = None
    accept_priority: int = 0
    
    def add_transition(self, symbol: str, target: 'NFAState') -> None:
        """Add a transition on symbol to target state."""
        self.transitions[symbol].add(target)
    
    def add_epsilon(self, target: 'NFAState') -> None:
        """Add an epsilon transition to target state."""
        self.transitions[EPSILON].add(target)
    
    def get_transitions(self, symbol: str) -> Set['NFAState']:
        """Get states reachable on symbol."""
        return self.transitions.get(symbol, set())
    
    def get_epsilon_transitions(self) -> Set['NFAState']:
        """Get states reachable on epsilon."""
        return self.transitions.get(EPSILON, set())
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, NFAState):
            return False
        return self.id == other.id
    
    def __repr__(self):
        accept_str = f", accept={self.accept_token}" if self.is_accept else ""
        return f"NFAState(id={self.id}{accept_str})"


@dataclass
class NFA:
    """
    Non-deterministic Finite Automaton.
    
    Attributes:
        start: Start state
        accept: Accept state (may be None for combined NFAs)
        states: List of all states
        alphabet: Set of input symbols
    """
    start: NFAState
    accept: Optional[NFAState]
    states: List[NFAState]
    alphabet: Set[str]
    
    def epsilon_closure(self, states: Set[NFAState]) -> FrozenSet[NFAState]:
        """
        Compute epsilon closure of a set of states.
        
        The epsilon closure is the set of all states reachable from
        the given states through epsilon transitions.
        
        Args:
            states: Initial set of states
            
        Returns:
            Frozen set of all reachable states
        """
        closure = set(states)
        worklist = list(states)
        
        while worklist:
            state = worklist.pop()
            for target in state.get_epsilon_transitions():
                if target not in closure:
                    closure.add(target)
                    worklist.append(target)
        
        return frozenset(closure)
    
    def move(self, states: Set[NFAState], symbol: str) -> Set[NFAState]:
        """
        Compute states reachable on symbol from given states.
        
        Args:
            states: Set of source states
            symbol: Input symbol
            
        Returns:
            Set of target states
        """
        result = set()
        for state in states:
            result.update(state.get_transitions(symbol))
        return result
    
    def simulate(self, input_str: str) -> bool:
        """
        Simulate NFA on input string.
        
        Args:
            input_str: Input to process
            
        Returns:
            True if input is accepted
        """
        current = self.epsilon_closure({self.start})
        
        for char in input_str:
            next_states = self.move(current, char)
            if not next_states:
                return False
            current = self.epsilon_closure(next_states)
        
        # Check if any current state is an accept state
        return any(s.is_accept for s in current)
    
    def get_accept_states(self) -> List[NFAState]:
        """Get all accepting states."""
        return [s for s in self.states if s.is_accept]
    
    def __repr__(self):
        return f"NFA(states={len(self.states)}, alphabet_size={len(self.alphabet)})"


class NFABuilder:
    """
    Build NFA from regex using Thompson's construction.
    
    Thompson's construction produces an NFA with:
    - Exactly one start state and one accept state per fragment
    - O(n) states for regex of size n
    - No transitions go back to the start state
    """
    
    def __init__(self):
        self._state_counter = 0
        self._states: List[NFAState] = []
        self._alphabet: Set[str] = set()
    
    def new_state(self) -> NFAState:
        """Create a new NFA state."""
        state = NFAState(id=self._state_counter)
        self._state_counter += 1
        self._states.append(state)
        return state
    
    def add_to_alphabet(self, symbol: str) -> None:
        """Add symbol to alphabet."""
        if symbol != EPSILON:
            self._alphabet.add(symbol)
    
    def build(self, regex_ast: RegexNode, token_name: str, priority: int) -> NFA:
        """
        Build NFA from regex AST using Thompson's construction.
        
        Args:
            regex_ast: Parsed regex AST
            token_name: Name of token this NFA recognizes
            priority: Token priority
            
        Returns:
            NFA for the regex
        """
        start, end = regex_ast.to_nfa(self)
        
        # Mark accept state
        end.is_accept = True
        end.accept_token = token_name
        end.accept_priority = priority
        
        return NFA(
            start=start,
            accept=end,
            states=self._states,
            alphabet=self._alphabet
        )
    
    def reset(self) -> None:
        """Reset builder for new NFA construction."""
        self._state_counter = 0
        self._states = []
        self._alphabet = set()


def combine_nfas(nfas: List[NFA]) -> NFA:
    """
    Combine multiple NFAs into one with a common start state.
    
    The combined NFA has:
    - A new start state with epsilon transitions to each NFA's start
    - Multiple accept states (one per original NFA)
    - Union of all alphabets
    
    Args:
        nfas: List of NFAs to combine
        
    Returns:
        Combined NFA
    """
    if not nfas:
        raise ValueError("Cannot combine empty list of NFAs")
    
    if len(nfas) == 1:
        return nfas[0]
    
    # Collect all states and renumber them
    all_states: List[NFAState] = []
    alphabet: Set[str] = set()
    state_id = 0
    
    # Create new combined start state
    combined_start = NFAState(id=state_id)
    state_id += 1
    all_states.append(combined_start)
    
    # Add states from each NFA
    for nfa in nfas:
        # Add epsilon transition from combined start to NFA start
        combined_start.add_epsilon(nfa.start)
        
        # Renumber and add all states
        for state in nfa.states:
            state.id = state_id
            state_id += 1
            all_states.append(state)
        
        # Merge alphabet
        alphabet.update(nfa.alphabet)
    
    return NFA(
        start=combined_start,
        accept=None,  # Multiple accept states
        states=all_states,
        alphabet=alphabet
    )


def build_nfa_from_pattern(pattern: str, token_name: str, priority: int = 0) -> NFA:
    """
    Convenience function to build NFA from regex pattern.
    
    Args:
        pattern: Regular expression pattern
        token_name: Token name
        priority: Token priority
        
    Returns:
        NFA for the pattern
    """
    from .regex import parse_regex
    
    ast = parse_regex(pattern)
    builder = NFABuilder()
    return builder.build(ast, token_name, priority)
