#!/usr/bin/env python3
"""Parse table data structures for LR and LL parsers.

This module provides:
- ParserType: Enumeration of supported parser types
- LRItem: LR(0)/LR(1) item representation
- LRItemSet: Set of LR items (parser state)
- ActionType: Types of parser actions
- Action: Parse table action
- Conflict: Parse table conflict
- ParseTable: LR parse table (ACTION/GOTO)
- LL1Table: LL(1) parse table
- LL1Conflict: LL(1) conflict
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Set, Optional, FrozenSet, Tuple, Any, Union

# Import from grammar module
try:
    from ir.grammar.symbol import Symbol, EPSILON, EOF
    from ir.grammar.production import ProductionRule
except ImportError:
    # Type stubs for development
    Symbol = Any
    ProductionRule = Any
    EPSILON = None
    EOF = None


class ParserType(Enum):
    """Supported parser types."""
    LR0 = auto()        # LR(0) - Simple LR
    SLR1 = auto()       # SLR(1) - Simple LR with lookahead
    LALR1 = auto()      # LALR(1) - Look-Ahead LR
    LR1 = auto()        # LR(1) - Canonical LR
    LL1 = auto()        # LL(1) - Predictive parsing
    RD = auto()         # Recursive Descent


class ActionType(Enum):
    """Types of parser actions."""
    SHIFT = auto()      # Shift input and push state
    REDUCE = auto()     # Reduce by production
    ACCEPT = auto()     # Accept input (successful parse)
    ERROR = auto()      # Parse error


@dataclass(frozen=True)
class LRItem:
    """An LR(0) or LR(1) item: A → α • β [, lookahead].
    
    Represents a production rule with a dot position indicating
    how much of the rule has been recognized.
    
    Attributes:
        production: The production rule
        dot_position: Position of the dot (0 = before first symbol)
        lookahead: Optional lookahead symbol (for LR(1)/LALR(1))
    
    Example:
        >>> E = nonterminal("E")
        >>> plus = terminal("+")
        >>> T = nonterminal("T")
        >>> prod = ProductionRule(E, (E, plus, T))
        >>> item = LRItem(prod, 1)  # E → E • + T
        >>> item.next_symbol()  # returns plus
    """
    production: ProductionRule
    dot_position: int
    lookahead: Optional[Symbol] = None
    
    def is_complete(self) -> bool:
        """Check if dot is at the end (reduce item).
        
        Returns:
            True if this is a complete/reduce item
        """
        body = self.production.body if self.production.body else ()
        return self.dot_position >= len(body)
    
    def next_symbol(self) -> Optional[Symbol]:
        """Get symbol after the dot.
        
        Returns:
            Symbol after dot, or None if item is complete
        """
        if self.is_complete():
            return None
        body = self.production.body if self.production.body else ()
        if not body:
            return None
        return body[self.dot_position]
    
    def advance(self) -> 'LRItem':
        """Return new item with dot advanced by one position.
        
        Returns:
            New LRItem with dot_position + 1
        """
        return LRItem(
            production=self.production,
            dot_position=self.dot_position + 1,
            lookahead=self.lookahead
        )
    
    def remaining_symbols(self) -> Tuple[Symbol, ...]:
        """Get symbols after the dot.
        
        Returns:
            Tuple of remaining symbols
        """
        body = self.production.body if self.production.body else ()
        return tuple(body[self.dot_position:])
    
    def symbols_before_dot(self) -> Tuple[Symbol, ...]:
        """Get symbols before the dot.
        
        Returns:
            Tuple of symbols before dot
        """
        body = self.production.body if self.production.body else ()
        return tuple(body[:self.dot_position])
    
    def core(self) -> 'LRItem':
        """Get the core (without lookahead) of this item.
        
        Returns:
            LRItem without lookahead
        """
        return LRItem(self.production, self.dot_position, None)
    
    def __str__(self) -> str:
        body = list(self.production.body) if self.production.body else []
        body_parts = [s.name for s in body[:self.dot_position]]
        body_parts.append('•')
        body_parts.extend(s.name for s in body[self.dot_position:])
        body_str = ' '.join(body_parts) if body_parts != ['•'] else '• ε'
        la_str = f", {self.lookahead.name}" if self.lookahead else ""
        return f"[{self.production.head.name} → {body_str}{la_str}]"
    
    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class LRItemSet:
    """A set of LR items forming a parser state.
    
    Attributes:
        items: Frozenset of LR items
        state_id: Unique state identifier
    """
    items: FrozenSet[LRItem]
    state_id: int = -1
    
    def kernel_items(self) -> FrozenSet[LRItem]:
        """Get kernel items (initial item or dot not at start).
        
        Kernel items are either:
        - The initial item S' → • S
        - Items with dot not at the beginning
        
        Returns:
            Frozenset of kernel items
        """
        return frozenset(
            item for item in self.items
            if item.dot_position > 0 or 
               (hasattr(item.production.head, 'name') and 
                item.production.head.name == "S'")
        )
    
    def closure_items(self) -> FrozenSet[LRItem]:
        """Get non-kernel (closure) items.
        
        Returns:
            Frozenset of closure items
        """
        kernel = self.kernel_items()
        return frozenset(item for item in self.items if item not in kernel)
    
    def get_complete_items(self) -> List[LRItem]:
        """Get all complete (reduce) items.
        
        Returns:
            List of complete items
        """
        return [item for item in self.items if item.is_complete()]
    
    def get_shift_symbols(self) -> Set[Symbol]:
        """Get all symbols that can be shifted from this state.
        
        Returns:
            Set of shiftable symbols
        """
        symbols = set()
        for item in self.items:
            sym = item.next_symbol()
            if sym:
                symbols.add(sym)
        return symbols
    
    def __hash__(self) -> int:
        return hash(self.items)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LRItemSet):
            return False
        return self.items == other.items
    
    def __str__(self) -> str:
        lines = [f"State {self.state_id}:"]
        for item in sorted(self.items, key=str):
            lines.append(f"  {item}")
        return "\n".join(lines)


@dataclass
class Action:
    """A parse table action.
    
    Attributes:
        action_type: Type of action (SHIFT, REDUCE, ACCEPT, ERROR)
        value: State number (SHIFT) or production index (REDUCE)
    """
    action_type: ActionType
    value: int = 0
    
    def is_shift(self) -> bool:
        """Check if this is a shift action."""
        return self.action_type == ActionType.SHIFT
    
    def is_reduce(self) -> bool:
        """Check if this is a reduce action."""
        return self.action_type == ActionType.REDUCE
    
    def is_accept(self) -> bool:
        """Check if this is an accept action."""
        return self.action_type == ActionType.ACCEPT
    
    def is_error(self) -> bool:
        """Check if this is an error action."""
        return self.action_type == ActionType.ERROR
    
    @staticmethod
    def shift(state: int) -> 'Action':
        """Create a shift action.
        
        Args:
            state: State to shift to
        
        Returns:
            Shift action
        """
        return Action(ActionType.SHIFT, state)
    
    @staticmethod
    def reduce(production_index: int) -> 'Action':
        """Create a reduce action.
        
        Args:
            production_index: Index of production to reduce by
        
        Returns:
            Reduce action
        """
        return Action(ActionType.REDUCE, production_index)
    
    @staticmethod
    def accept() -> 'Action':
        """Create an accept action.
        
        Returns:
            Accept action
        """
        return Action(ActionType.ACCEPT)
    
    @staticmethod
    def error() -> 'Action':
        """Create an error action.
        
        Returns:
            Error action
        """
        return Action(ActionType.ERROR)
    
    def __str__(self) -> str:
        if self.is_shift():
            return f"s{self.value}"
        elif self.is_reduce():
            return f"r{self.value}"
        elif self.is_accept():
            return "acc"
        return "err"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Action):
            return False
        return self.action_type == other.action_type and self.value == other.value
    
    def __hash__(self) -> int:
        return hash((self.action_type, self.value))


@dataclass
class Conflict:
    """A parse table conflict.
    
    Represents a conflict where two different actions are specified
    for the same (state, symbol) pair.
    
    Attributes:
        state: State where conflict occurs
        symbol: Symbol causing conflict
        action1: First conflicting action
        action2: Second conflicting action
    """
    state: int
    symbol: Symbol
    action1: Action
    action2: Action
    
    @property
    def conflict_type(self) -> str:
        """Get conflict type string.
        
        Returns:
            "shift-reduce" or "reduce-reduce"
        """
        if self.action1.is_shift() and self.action2.is_reduce():
            return "shift-reduce"
        elif self.action1.is_reduce() and self.action2.is_shift():
            return "shift-reduce"
        elif self.action1.is_reduce() and self.action2.is_reduce():
            return "reduce-reduce"
        return "unknown"
    
    def is_shift_reduce(self) -> bool:
        """Check if this is a shift-reduce conflict."""
        return self.conflict_type == "shift-reduce"
    
    def is_reduce_reduce(self) -> bool:
        """Check if this is a reduce-reduce conflict."""
        return self.conflict_type == "reduce-reduce"
    
    def __str__(self) -> str:
        sym_name = self.symbol.name if hasattr(self.symbol, 'name') else str(self.symbol)
        return f"{self.conflict_type} conflict in state {self.state} on '{sym_name}': {self.action1} vs {self.action2}"


@dataclass
class ParseTable:
    """LR parse table with ACTION and GOTO tables.
    
    Attributes:
        action: Dict mapping (state, terminal) to Action
        goto: Dict mapping (state, nonterminal) to state
        states: List of LR item sets (states)
        productions: List of productions (for reduce actions)
        conflicts: List of detected conflicts
        parser_type: Type of parser this table is for
    """
    action: Dict[Tuple[int, Symbol], Action] = field(default_factory=dict)
    goto: Dict[Tuple[int, Symbol], int] = field(default_factory=dict)
    states: List[LRItemSet] = field(default_factory=list)
    productions: List[ProductionRule] = field(default_factory=list)
    conflicts: List[Conflict] = field(default_factory=list)
    parser_type: ParserType = ParserType.LALR1
    
    def get_action(self, state: int, terminal: Symbol) -> Optional[Action]:
        """Get action for state and terminal.
        
        Args:
            state: Parser state
            terminal: Terminal symbol
        
        Returns:
            Action or None if not defined
        """
        return self.action.get((state, terminal))
    
    def set_action(self, state: int, terminal: Symbol, action: Action) -> Optional[Conflict]:
        """Set action, detecting conflicts.
        
        Args:
            state: Parser state
            terminal: Terminal symbol
            action: Action to set
        
        Returns:
            Conflict if one was detected, None otherwise
        """
        key = (state, terminal)
        if key in self.action:
            existing = self.action[key]
            if existing != action:
                conflict = Conflict(state, terminal, existing, action)
                self.conflicts.append(conflict)
                return conflict
        self.action[key] = action
        return None
    
    def get_goto(self, state: int, nonterminal: Symbol) -> Optional[int]:
        """Get goto state for state and nonterminal.
        
        Args:
            state: Current state
            nonterminal: Non-terminal symbol
        
        Returns:
            Target state or None if not defined
        """
        return self.goto.get((state, nonterminal))
    
    def set_goto(self, state: int, nonterminal: Symbol, target: int) -> None:
        """Set goto entry.
        
        Args:
            state: Current state
            nonterminal: Non-terminal symbol
            target: Target state
        """
        self.goto[(state, nonterminal)] = target
    
    def has_conflicts(self) -> bool:
        """Check if table has any conflicts.
        
        Returns:
            True if there are conflicts
        """
        return len(self.conflicts) > 0
    
    def state_count(self) -> int:
        """Get number of states.
        
        Returns:
            Number of parser states
        """
        return len(self.states)
    
    def get_terminals(self) -> Set[Symbol]:
        """Get all terminals used in ACTION table.
        
        Returns:
            Set of terminal symbols
        """
        return {key[1] for key in self.action.keys()}
    
    def get_nonterminals(self) -> Set[Symbol]:
        """Get all nonterminals used in GOTO table.
        
        Returns:
            Set of nonterminal symbols
        """
        return {key[1] for key in self.goto.keys()}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parse table to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        def symbol_key(sym: Symbol) -> str:
            return sym.name if hasattr(sym, 'name') else str(sym)
        
        action_dict = {}
        for (state, sym), action in self.action.items():
            key = f"{state},{symbol_key(sym)}"
            action_dict[key] = str(action)
        
        goto_dict = {}
        for (state, sym), target in self.goto.items():
            key = f"{state},{symbol_key(sym)}"
            goto_dict[key] = target
        
        return {
            "parser_type": self.parser_type.name,
            "state_count": self.state_count(),
            "production_count": len(self.productions),
            "conflict_count": len(self.conflicts),
            "action": action_dict,
            "goto": goto_dict,
        }
    
    def __str__(self) -> str:
        lines = [f"ParseTable ({self.parser_type.name})"]
        lines.append(f"States: {self.state_count()}")
        lines.append(f"Productions: {len(self.productions)}")
        lines.append(f"Conflicts: {len(self.conflicts)}")
        
        if self.conflicts:
            lines.append("\nConflicts:")
            for conflict in self.conflicts:
                lines.append(f"  {conflict}")
        
        return "\n".join(lines)


@dataclass
class LL1Conflict:
    """An LL(1) parse table conflict.
    
    Represents a conflict where two different productions are specified
    for the same (nonterminal, terminal) pair.
    
    Attributes:
        nonterminal: Non-terminal symbol
        terminal: Lookahead terminal
        production1: First conflicting production
        production2: Second conflicting production
    """
    nonterminal: Symbol
    terminal: Symbol
    production1: ProductionRule
    production2: ProductionRule
    
    def __str__(self) -> str:
        nt_name = self.nonterminal.name if hasattr(self.nonterminal, 'name') else str(self.nonterminal)
        t_name = self.terminal.name if hasattr(self.terminal, 'name') else str(self.terminal)
        return f"LL(1) conflict for {nt_name} on '{t_name}': {self.production1} vs {self.production2}"


@dataclass
class LL1Table:
    """LL(1) parse table.
    
    Attributes:
        table: Dict mapping (nonterminal, terminal) to production
        first_sets: FIRST sets for all symbols
        follow_sets: FOLLOW sets for all nonterminals
        conflicts: List of LL(1) conflicts
    """
    table: Dict[Tuple[Symbol, Symbol], ProductionRule] = field(default_factory=dict)
    first_sets: Dict[Symbol, Set[Symbol]] = field(default_factory=dict)
    follow_sets: Dict[Symbol, Set[Symbol]] = field(default_factory=dict)
    conflicts: List[LL1Conflict] = field(default_factory=list)
    
    def get_production(self, nonterminal: Symbol, terminal: Symbol) -> Optional[ProductionRule]:
        """Get production for nonterminal and lookahead terminal.
        
        Args:
            nonterminal: Non-terminal symbol
            terminal: Lookahead terminal
        
        Returns:
            Production rule or None if not defined
        """
        return self.table.get((nonterminal, terminal))
    
    def set_production(self, nonterminal: Symbol, terminal: Symbol, 
                       production: ProductionRule) -> Optional[LL1Conflict]:
        """Set production, detecting conflicts.
        
        Args:
            nonterminal: Non-terminal symbol
            terminal: Lookahead terminal
            production: Production to use
        
        Returns:
            LL1Conflict if one was detected, None otherwise
        """
        key = (nonterminal, terminal)
        if key in self.table:
            existing = self.table[key]
            if existing != production:
                conflict = LL1Conflict(nonterminal, terminal, existing, production)
                self.conflicts.append(conflict)
                return conflict
        self.table[key] = production
        return None
    
    def has_conflicts(self) -> bool:
        """Check if table has any conflicts.
        
        Returns:
            True if there are conflicts
        """
        return len(self.conflicts) > 0
    
    def get_nonterminals(self) -> Set[Symbol]:
        """Get all nonterminals in the table.
        
        Returns:
            Set of nonterminal symbols
        """
        return {key[0] for key in self.table.keys()}
    
    def get_terminals(self) -> Set[Symbol]:
        """Get all terminals in the table.
        
        Returns:
            Set of terminal symbols
        """
        return {key[1] for key in self.table.keys()}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert LL(1) table to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        def symbol_key(sym: Symbol) -> str:
            return sym.name if hasattr(sym, 'name') else str(sym)
        
        table_dict = {}
        for (nt, t), prod in self.table.items():
            key = f"{symbol_key(nt)},{symbol_key(t)}"
            table_dict[key] = str(prod)
        
        return {
            "table": table_dict,
            "conflict_count": len(self.conflicts),
        }
    
    def __str__(self) -> str:
        lines = ["LL(1) Parse Table"]
        lines.append(f"Entries: {len(self.table)}")
        lines.append(f"Conflicts: {len(self.conflicts)}")
        
        if self.conflicts:
            lines.append("\nConflicts:")
            for conflict in self.conflicts:
                lines.append(f"  {conflict}")
        
        return "\n".join(lines)
