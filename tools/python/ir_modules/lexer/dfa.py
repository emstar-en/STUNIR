"""
DFA (Deterministic Finite Automaton) Module for STUNIR Lexer Generator.

Implements:
- Subset construction (NFA to DFA conversion)
- Hopcroft's algorithm for DFA minimization
- Transition table generation
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from .nfa import NFA, NFAState, EPSILON


@dataclass
class DFAState:
    """
    DFA state with transitions.
    
    Attributes:
        id: Unique state identifier
        nfa_states: Set of NFA states this DFA state corresponds to
        transitions: Mapping from symbol to target DFA state
        is_accept: Whether this is an accepting state
        accept_token: Token name if accepting
        accept_priority: Token priority (for conflict resolution)
    """
    id: int
    nfa_states: FrozenSet[NFAState]
    transitions: Dict[str, 'DFAState'] = field(default_factory=dict)
    is_accept: bool = False
    accept_token: Optional[str] = None
    accept_priority: int = -1
    
    def add_transition(self, symbol: str, target: 'DFAState') -> None:
        """Add transition on symbol to target state."""
        self.transitions[symbol] = target
    
    def get_transition(self, symbol: str) -> Optional['DFAState']:
        """Get target state for symbol, or None."""
        return self.transitions.get(symbol)
    
    def __hash__(self):
        return hash(self.nfa_states)
    
    def __eq__(self, other):
        if not isinstance(other, DFAState):
            return False
        return self.nfa_states == other.nfa_states
    
    def __repr__(self):
        accept_str = f", accept={self.accept_token}" if self.is_accept else ""
        return f"DFAState(id={self.id}{accept_str})"


@dataclass
class DFA:
    """
    Deterministic Finite Automaton.
    
    Attributes:
        start: Start state
        states: List of all states
        alphabet: Set of input symbols
        accept_states: List of accepting states
    """
    start: DFAState
    states: List[DFAState]
    alphabet: Set[str]
    accept_states: List[DFAState]
    
    def simulate(self, input_str: str) -> Optional[Tuple[str, int]]:
        """
        Simulate DFA on input string.
        
        Args:
            input_str: Input to process
            
        Returns:
            (token_name, priority) if accepted, None otherwise
        """
        state = self.start
        
        for char in input_str:
            next_state = state.get_transition(char)
            if next_state is None:
                return None
            state = next_state
        
        if state.is_accept:
            return (state.accept_token, state.accept_priority)
        return None
    
    def minimize(self) -> 'MinimizedDFA':
        """Minimize this DFA using Hopcroft's algorithm."""
        return HopcroftMinimizer(self).minimize()
    
    def __repr__(self):
        return f"DFA(states={len(self.states)}, alphabet_size={len(self.alphabet)})"


@dataclass
class MinimizedDFA:
    """
    Minimized DFA with canonical state numbering.
    
    Attributes:
        start_state: Index of start state
        num_states: Total number of states
        transitions: state -> symbol -> state mapping
        accept_states: state -> (token_name, priority) mapping
        alphabet: Sorted list of alphabet symbols
    """
    start_state: int
    num_states: int
    transitions: Dict[int, Dict[str, int]]
    accept_states: Dict[int, Tuple[str, int]]
    alphabet: List[str]
    
    def simulate(self, input_str: str) -> Optional[Tuple[str, int]]:
        """
        Simulate minimized DFA on input string.
        
        Args:
            input_str: Input to process
            
        Returns:
            (token_name, priority) if accepted, None otherwise
        """
        state = self.start_state
        
        for char in input_str:
            next_state = self.transitions.get(state, {}).get(char, -1)
            if next_state < 0:
                return None
            state = next_state
        
        if state in self.accept_states:
            return self.accept_states[state]
        return None
    
    def to_table(self) -> 'TransitionTable':
        """Convert to transition table representation."""
        return TransitionTable.from_minimized_dfa(self)
    
    def __repr__(self):
        return f"MinimizedDFA(states={self.num_states}, alphabet_size={len(self.alphabet)})"


@dataclass
class TransitionTable:
    """
    Compressed transition table for efficient lexing.
    
    The table is stored as a flat array where:
    - table[state * num_symbols + symbol_index] = next_state
    - accept_table[state] = (token_name, priority) or None
    """
    table: List[int]
    num_states: int
    num_symbols: int
    symbol_to_index: Dict[str, int]
    index_to_symbol: List[str]
    accept_table: List[Optional[Tuple[str, int]]]
    start_state: int = 0
    error_state: int = -1
    
    def next_state(self, state: int, symbol: str) -> int:
        """Get next state for given state and symbol."""
        idx = self.symbol_to_index.get(symbol, -1)
        if idx < 0 or state < 0 or state >= self.num_states:
            return self.error_state
        return self.table[state * self.num_symbols + idx]
    
    def is_accept(self, state: int) -> bool:
        """Check if state is an accept state."""
        return 0 <= state < self.num_states and self.accept_table[state] is not None
    
    def get_token(self, state: int) -> Optional[Tuple[str, int]]:
        """Get token info for accept state."""
        if 0 <= state < self.num_states:
            return self.accept_table[state]
        return None
    
    @classmethod
    def from_minimized_dfa(cls, dfa: MinimizedDFA) -> 'TransitionTable':
        """Build transition table from minimized DFA."""
        # Build symbol mapping
        index_to_symbol = sorted(dfa.alphabet)
        symbol_to_index = {s: i for i, s in enumerate(index_to_symbol)}
        num_symbols = len(index_to_symbol)
        
        # Build flat transition table
        table = [-1] * (dfa.num_states * num_symbols)
        
        for state, trans in dfa.transitions.items():
            for symbol, target in trans.items():
                idx = symbol_to_index.get(symbol, -1)
                if idx >= 0:
                    table[state * num_symbols + idx] = target
        
        # Build accept table
        accept_table: List[Optional[Tuple[str, int]]] = [None] * dfa.num_states
        for state, (token, priority) in dfa.accept_states.items():
            accept_table[state] = (token, priority)
        
        return cls(
            table=table,
            num_states=dfa.num_states,
            num_symbols=num_symbols,
            symbol_to_index=symbol_to_index,
            index_to_symbol=index_to_symbol,
            accept_table=accept_table,
            start_state=dfa.start_state
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "num_states": self.num_states,
            "num_symbols": self.num_symbols,
            "start_state": self.start_state,
            "error_state": self.error_state,
            "symbol_to_index": self.symbol_to_index,
            "index_to_symbol": self.index_to_symbol,
            "table": self.table,
            "accept_table": [
                {"token": a[0], "priority": a[1]} if a else None
                for a in self.accept_table
            ]
        }
    
    def __repr__(self):
        return f"TransitionTable(states={self.num_states}, symbols={self.num_symbols})"


class SubsetConstruction:
    """
    Convert NFA to DFA using subset construction (powerset construction).
    
    Each DFA state corresponds to a set of NFA states (the epsilon closure
    of reachable NFA states).
    """
    
    def __init__(self, nfa: NFA):
        self.nfa = nfa
        self.dfa_states: List[DFAState] = []
        self.state_map: Dict[FrozenSet[NFAState], DFAState] = {}
        self._dfa_counter = 0
    
    def convert(self) -> DFA:
        """
        Perform subset construction to convert NFA to DFA.
        
        Returns:
            Equivalent DFA
        """
        # Start with epsilon closure of NFA start state
        start_closure = self.nfa.epsilon_closure({self.nfa.start})
        start_dfa = self._get_or_create_state(start_closure)
        
        # Process states using worklist
        worklist = [start_dfa]
        visited: Set[FrozenSet[NFAState]] = {start_closure}
        
        while worklist:
            dfa_state = worklist.pop()
            
            # Process each symbol in alphabet
            for symbol in self.nfa.alphabet:
                # Compute move and epsilon closure
                move_result = self.nfa.move(set(dfa_state.nfa_states), symbol)
                if not move_result:
                    continue
                
                closure = self.nfa.epsilon_closure(move_result)
                target_dfa = self._get_or_create_state(closure)
                
                # Add transition
                dfa_state.add_transition(symbol, target_dfa)
                
                # Add to worklist if not visited
                if closure not in visited:
                    visited.add(closure)
                    worklist.append(target_dfa)
        
        # Collect accept states
        accept_states = [s for s in self.dfa_states if s.is_accept]
        
        return DFA(
            start=start_dfa,
            states=self.dfa_states,
            alphabet=self.nfa.alphabet,
            accept_states=accept_states
        )
    
    def _get_or_create_state(self, nfa_states: FrozenSet[NFAState]) -> DFAState:
        """Get existing or create new DFA state for NFA state set."""
        if nfa_states in self.state_map:
            return self.state_map[nfa_states]
        
        # Determine accept info (highest priority wins)
        is_accept = False
        accept_token = None
        accept_priority = -1
        
        for nfa_state in nfa_states:
            if nfa_state.is_accept:
                if nfa_state.accept_priority > accept_priority:
                    is_accept = True
                    accept_token = nfa_state.accept_token
                    accept_priority = nfa_state.accept_priority
        
        # Create new DFA state
        dfa_state = DFAState(
            id=self._dfa_counter,
            nfa_states=nfa_states,
            is_accept=is_accept,
            accept_token=accept_token,
            accept_priority=accept_priority
        )
        self._dfa_counter += 1
        self.dfa_states.append(dfa_state)
        self.state_map[nfa_states] = dfa_state
        
        return dfa_state


class HopcroftMinimizer:
    """
    Minimize DFA using Hopcroft's algorithm.
    
    Hopcroft's algorithm runs in O(n log n) time where n is the number
    of states. It works by iteratively refining partitions of states
    until no further refinement is possible.
    """
    
    def __init__(self, dfa: DFA):
        self.dfa = dfa
        self.partition: List[Set[DFAState]] = []
    
    def minimize(self) -> MinimizedDFA:
        """
        Minimize the DFA.
        
        Returns:
            Minimized DFA
        """
        if not self.dfa.states:
            return MinimizedDFA(
                start_state=0,
                num_states=0,
                transitions={},
                accept_states={},
                alphabet=[]
            )
        
        # Initial partition: group accept states by token type
        accept_groups: Dict[Optional[str], Set[DFAState]] = defaultdict(set)
        non_accept: Set[DFAState] = set()
        
        for state in self.dfa.states:
            if state.is_accept:
                accept_groups[state.accept_token].add(state)
            else:
                non_accept.add(state)
        
        # Build initial partition
        self.partition = [g for g in accept_groups.values() if g]
        if non_accept:
            self.partition.append(non_accept)
        
        # Handle edge case: single state
        if len(self.partition) <= 1:
            return self._build_minimized_dfa()
        
        # Worklist of sets to process
        worklist = list(self.partition)
        
        while worklist:
            splitter = worklist.pop()
            
            for symbol in self.dfa.alphabet:
                # Find states that transition to splitter on symbol
                inverse = self._inverse_transition(splitter, symbol)
                if not inverse:
                    continue
                
                # Try to split each partition
                new_partition = []
                for group in self.partition:
                    in_inverse = group & inverse
                    not_in_inverse = group - inverse
                    
                    if in_inverse and not_in_inverse:
                        # Split the group
                        new_partition.append(in_inverse)
                        new_partition.append(not_in_inverse)
                        
                        # Update worklist
                        if group in worklist:
                            worklist.remove(group)
                            worklist.append(in_inverse)
                            worklist.append(not_in_inverse)
                        else:
                            # Add smaller set to worklist (Hopcroft optimization)
                            if len(in_inverse) <= len(not_in_inverse):
                                worklist.append(in_inverse)
                            else:
                                worklist.append(not_in_inverse)
                    else:
                        new_partition.append(group)
                
                self.partition = new_partition
        
        return self._build_minimized_dfa()
    
    def _inverse_transition(self, targets: Set[DFAState], symbol: str) -> Set[DFAState]:
        """Find states that transition to any target state on symbol."""
        result = set()
        for state in self.dfa.states:
            target = state.get_transition(symbol)
            if target in targets:
                result.add(state)
        return result
    
    def _build_minimized_dfa(self) -> MinimizedDFA:
        """Build minimized DFA from partition."""
        if not self.partition:
            return MinimizedDFA(
                start_state=0,
                num_states=0,
                transitions={},
                accept_states={},
                alphabet=sorted(self.dfa.alphabet)
            )
        
        # Map each original state to its partition index
        state_to_partition: Dict[DFAState, int] = {}
        for i, group in enumerate(self.partition):
            for state in group:
                state_to_partition[state] = i
        
        # Find start state partition
        start_state = state_to_partition.get(self.dfa.start, 0)
        
        # Build transitions and accept states
        transitions: Dict[int, Dict[str, int]] = {i: {} for i in range(len(self.partition))}
        accept_states: Dict[int, Tuple[str, int]] = {}
        
        for i, group in enumerate(self.partition):
            # Use any representative state (they're equivalent)
            representative = next(iter(group))
            
            # Build transitions
            for symbol, target in representative.transitions.items():
                if target in state_to_partition:
                    transitions[i][symbol] = state_to_partition[target]
            
            # Record accept info
            if representative.is_accept:
                accept_states[i] = (representative.accept_token, representative.accept_priority)
        
        return MinimizedDFA(
            start_state=start_state,
            num_states=len(self.partition),
            transitions=transitions,
            accept_states=accept_states,
            alphabet=sorted(self.dfa.alphabet)
        )


def nfa_to_dfa(nfa: NFA) -> DFA:
    """
    Convert NFA to DFA using subset construction.
    
    Args:
        nfa: NFA to convert
        
    Returns:
        Equivalent DFA
    """
    return SubsetConstruction(nfa).convert()


def minimize_dfa(dfa: DFA) -> MinimizedDFA:
    """
    Minimize DFA using Hopcroft's algorithm.
    
    Args:
        dfa: DFA to minimize
        
    Returns:
        Minimized DFA
    """
    return HopcroftMinimizer(dfa).minimize()
