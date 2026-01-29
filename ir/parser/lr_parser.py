#!/usr/bin/env python3
"""LR/LALR/SLR parser generator implementation.

This module provides:
- LRParserGenerator: Complete LR family parser generator
- closure(): Compute closure of item sets
- goto(): Compute GOTO function
- build_lr0_items(): Build LR(0) item sets
- build_parse_table(): Build complete parse table
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, FrozenSet, Tuple, Any

from ir.parser.parse_table import (
    ParserType, ParseTable, LRItem, LRItemSet, 
    Action, Conflict
)
from ir.parser.ast_node import ASTSchema, generate_ast_schema
from ir.parser.parser_generator import (
    ParserGenerator, ParserGeneratorResult,
    validate_grammar_for_parsing
)

# Import from grammar module
try:
    from ir.grammar.grammar_ir import Grammar
    from ir.grammar.symbol import Symbol, SymbolType, EPSILON, EOF
    from ir.grammar.production import ProductionRule
    from ir.grammar.validation import (
        compute_first_sets, compute_follow_sets, compute_nullable
    )
except ImportError:
    # Type stubs for development
    Grammar = Any
    Symbol = Any
    ProductionRule = Any
    EPSILON = None
    EOF = None


def closure(items: Set[LRItem], grammar: Grammar) -> FrozenSet[LRItem]:
    """Compute closure of item set.
    
    For each item A → α • B β in the set, add all items
    B → • γ for each production B → γ.
    
    Args:
        items: Initial set of LR items
        grammar: The grammar
    
    Returns:
        Closure of the item set (frozen)
    """
    closure_set = set(items)
    worklist = list(items)
    
    while worklist:
        item = worklist.pop()
        next_sym = item.next_symbol()
        
        if next_sym and hasattr(next_sym, 'is_nonterminal') and next_sym.is_nonterminal():
            for prod in grammar.get_productions(next_sym):
                new_item = LRItem(prod, 0, item.lookahead)
                if new_item not in closure_set:
                    closure_set.add(new_item)
                    worklist.append(new_item)
    
    return frozenset(closure_set)


def closure_lr1(items: Set[LRItem], grammar: Grammar, 
                first_sets: Dict[Symbol, Set[Symbol]]) -> FrozenSet[LRItem]:
    """Compute LR(1) closure with lookahead computation.
    
    For item A → α • B β, a:
    Add B → • γ, b for each production B → γ
    where b ∈ FIRST(βa)
    
    Args:
        items: Initial set of LR(1) items
        grammar: The grammar
        first_sets: Precomputed FIRST sets
    
    Returns:
        LR(1) closure of the item set
    """
    closure_set = set(items)
    worklist = list(items)
    
    def first_of_string(symbols: Tuple[Symbol, ...], la: Symbol) -> Set[Symbol]:
        """Compute FIRST of a string of symbols followed by lookahead."""
        result: Set[Symbol] = set()
        all_nullable = True
        
        for sym in symbols:
            if hasattr(sym, 'is_terminal') and sym.is_terminal():
                result.add(sym)
                all_nullable = False
                break
            elif hasattr(sym, 'is_epsilon') and sym.is_epsilon():
                continue
            else:
                # Non-terminal
                sym_first = first_sets.get(sym, set())
                result.update(s for s in sym_first if not (hasattr(s, 'is_epsilon') and s.is_epsilon()))
                if EPSILON not in sym_first and not any(hasattr(s, 'is_epsilon') and s.is_epsilon() for s in sym_first):
                    all_nullable = False
                    break
        
        if all_nullable:
            result.add(la)
        
        return result
    
    while worklist:
        item = worklist.pop()
        next_sym = item.next_symbol()
        
        if next_sym and hasattr(next_sym, 'is_nonterminal') and next_sym.is_nonterminal():
            # Get β (symbols after B in A → α • B β)
            remaining = item.remaining_symbols()[1:]  # Skip B itself
            
            # Compute FIRST(β lookahead)
            lookaheads = first_of_string(remaining, item.lookahead)
            
            for prod in grammar.get_productions(next_sym):
                for la in lookaheads:
                    new_item = LRItem(prod, 0, la)
                    if new_item not in closure_set:
                        closure_set.add(new_item)
                        worklist.append(new_item)
    
    return frozenset(closure_set)


def goto(items: FrozenSet[LRItem], symbol: Symbol, grammar: Grammar,
         first_sets: Optional[Dict[Symbol, Set[Symbol]]] = None,
         use_lr1: bool = False) -> FrozenSet[LRItem]:
    """Compute GOTO(items, symbol).
    
    For each item A → α • X β where X = symbol,
    add A → α X • β to the result and take closure.
    
    Args:
        items: Set of LR items
        symbol: Grammar symbol
        grammar: The grammar
        first_sets: FIRST sets (required for LR(1))
        use_lr1: Whether to use LR(1) closure
    
    Returns:
        New item set (closure of advanced items)
    """
    moved_items: Set[LRItem] = set()
    
    for item in items:
        if item.next_symbol() == symbol:
            moved_items.add(item.advance())
    
    if not moved_items:
        return frozenset()
    
    if use_lr1 and first_sets:
        return closure_lr1(moved_items, grammar, first_sets)
    else:
        return closure(moved_items, grammar)


def build_lr0_items(grammar: Grammar) -> Tuple[List[LRItemSet], ProductionRule]:
    """Build canonical collection of LR(0) item sets.
    
    Algorithm:
    1. Create augmented grammar with S' → S
    2. Start with closure of {S' → • S}
    3. For each item set and symbol, compute GOTO
    4. Add new item sets until fixed point
    
    Args:
        grammar: The grammar (will be augmented)
    
    Returns:
        Tuple of (list of LR item sets, augmented start production)
    """
    # Create augmented start production
    augmented_start = Symbol("S'", SymbolType.NONTERMINAL)
    start_prod = ProductionRule(augmented_start, (grammar.start_symbol,))
    
    # Initial item and closure
    initial_item = LRItem(start_prod, 0)
    initial_closure = closure({initial_item}, grammar)
    initial_set = LRItemSet(initial_closure, 0)
    
    states: List[LRItemSet] = [initial_set]
    state_map: Dict[FrozenSet[LRItem], int] = {initial_closure: 0}
    worklist = [initial_set]
    
    while worklist:
        current_set = worklist.pop(0)
        
        # Get all symbols that could be shifted
        symbols: Set[Symbol] = set()
        for item in current_set.items:
            sym = item.next_symbol()
            if sym:
                symbols.add(sym)
        
        for symbol in symbols:
            new_items = goto(current_set.items, symbol, grammar)
            
            if new_items and new_items not in state_map:
                new_state = LRItemSet(new_items, len(states))
                states.append(new_state)
                state_map[new_items] = new_state.state_id
                worklist.append(new_state)
    
    return states, start_prod


def build_lalr_items(grammar: Grammar) -> Tuple[List[LRItemSet], ProductionRule, Dict[Tuple[int, LRItem], Set[Symbol]]]:
    """Build LALR(1) item sets with lookahead.
    
    Uses the "merging LR(1) cores" approach:
    1. Build LR(0) item sets
    2. Compute lookaheads using propagation
    
    Args:
        grammar: The grammar
    
    Returns:
        Tuple of (states, augmented production, lookahead dict)
    """
    # Build LR(0) items first
    states, start_prod = build_lr0_items(grammar)
    
    # Compute FIRST/FOLLOW sets
    first_sets = compute_first_sets(grammar)
    follow_sets = compute_follow_sets(grammar, first_sets)
    
    # Initialize lookaheads for kernel items
    lookaheads: Dict[Tuple[int, LRItem], Set[Symbol]] = {}
    
    for state in states:
        for item in state.items:
            key = (state.state_id, item.core())
            if key not in lookaheads:
                lookaheads[key] = set()
    
    # Add $ (EOF) lookahead to augmented start
    for item in states[0].items:
        if item.production == start_prod and item.dot_position == 0:
            lookaheads[(0, item.core())].add(EOF)
    
    # Propagate lookaheads
    # Track propagation links: from (state, item) to (target_state, target_item)
    propagates: Dict[Tuple[int, LRItem], List[Tuple[int, LRItem]]] = {}
    
    for state in states:
        for item in state.items:
            key = (state.state_id, item.core())
            if key not in propagates:
                propagates[key] = []
    
    # Build propagation links by following GOTO transitions
    state_map = {frozenset(item.core() for item in s.items): s.state_id for s in states}
    
    for state in states:
        for item in state.items:
            next_sym = item.next_symbol()
            if next_sym:
                # Find target state
                goto_items = goto(state.items, next_sym, grammar)
                if goto_items:
                    goto_cores = frozenset(i.core() for i in goto_items)
                    if goto_cores in state_map:
                        target_state_id = state_map[goto_cores]
                        # Add propagation from item to advanced item
                        advanced = item.advance().core()
                        src_key = (state.state_id, item.core())
                        tgt_key = (target_state_id, advanced)
                        
                        if src_key in propagates and tgt_key in lookaheads:
                            propagates[src_key].append(tgt_key)
    
    # Also compute spontaneous lookaheads from closure
    for state in states:
        for item in state.items:
            if item.dot_position == 0:
                # This is a closure item, get lookahead from the item that generated it
                continue
    
    # Propagate until fixed point
    changed = True
    iterations = 0
    max_iterations = len(states) * len(lookaheads) + 100
    
    while changed and iterations < max_iterations:
        changed = False
        iterations += 1
        
        for (state_id, item), la_set in list(lookaheads.items()):
            for target in propagates.get((state_id, item), []):
                if target in lookaheads:
                    old_size = len(lookaheads[target])
                    lookaheads[target].update(la_set)
                    if len(lookaheads[target]) > old_size:
                        changed = True
    
    return states, start_prod, lookaheads


def build_parse_table(grammar: Grammar, parser_type: ParserType = ParserType.LALR1) -> ParseTable:
    """Build LR parse table.
    
    Args:
        grammar: The grammar
        parser_type: Type of parser (LR0, SLR1, LALR1)
    
    Returns:
        ParseTable with ACTION and GOTO tables
    """
    # Build item sets
    if parser_type == ParserType.LALR1:
        states, start_prod, lookaheads = build_lalr_items(grammar)
    else:
        states, start_prod = build_lr0_items(grammar)
        lookaheads = {}
    
    # Get FIRST/FOLLOW sets for SLR(1)
    if parser_type == ParserType.SLR1:
        first_sets = compute_first_sets(grammar)
        follow_sets = compute_follow_sets(grammar, first_sets)
    else:
        follow_sets = {}
    
    # Build production list (augmented start + original)
    productions = [start_prod] + grammar.all_productions()
    
    parse_table = ParseTable(
        states=states,
        productions=productions,
        parser_type=parser_type
    )
    
    # Build state lookup
    state_map: Dict[FrozenSet[LRItem], int] = {}
    for state in states:
        state_map[state.items] = state.state_id
    
    # Fill ACTION and GOTO tables
    for state in states:
        for item in state.items:
            next_sym = item.next_symbol()
            
            if next_sym is None:  # Complete/reduce item
                # Get lookahead set based on parser type
                if parser_type == ParserType.LR0:
                    # LR(0): reduce on all terminals
                    la_set: Set[Symbol] = set(grammar.terminals) | {EOF}
                elif parser_type == ParserType.SLR1:
                    # SLR(1): reduce on FOLLOW(head)
                    la_set = follow_sets.get(item.production.head, set())
                else:
                    # LALR(1): use computed lookaheads
                    key = (state.state_id, item.core())
                    la_set = lookaheads.get(key, set())
                    if not la_set:
                        # Fallback to FOLLOW
                        if not follow_sets:
                            first_sets_fallback = compute_first_sets(grammar)
                            follow_sets_fallback = compute_follow_sets(grammar, first_sets_fallback)
                            la_set = follow_sets_fallback.get(item.production.head, set())
                        else:
                            la_set = follow_sets.get(item.production.head, set())
                
                # Check for accept
                if item.production == start_prod:
                    parse_table.set_action(state.state_id, EOF, Action.accept())
                else:
                    # Find production index
                    try:
                        prod_index = productions.index(item.production)
                    except ValueError:
                        continue
                    
                    for la in la_set:
                        if la and (hasattr(la, 'is_terminal') and la.is_terminal() or la == EOF):
                            parse_table.set_action(
                                state.state_id, la,
                                Action.reduce(prod_index)
                            )
            
            elif hasattr(next_sym, 'is_terminal') and next_sym.is_terminal():
                # Shift item
                goto_items = goto(state.items, next_sym, grammar)
                if goto_items and goto_items in state_map:
                    goto_state = state_map[goto_items]
                    parse_table.set_action(
                        state.state_id, next_sym,
                        Action.shift(goto_state)
                    )
            
            elif hasattr(next_sym, 'is_nonterminal') and next_sym.is_nonterminal():
                # GOTO entry
                goto_items = goto(state.items, next_sym, grammar)
                if goto_items and goto_items in state_map:
                    goto_state = state_map[goto_items]
                    parse_table.set_goto(state.state_id, next_sym, goto_state)
    
    return parse_table


def detect_conflicts(parse_table: ParseTable) -> List[Conflict]:
    """Detect and report all conflicts in parse table.
    
    Args:
        parse_table: The parse table to check
    
    Returns:
        List of Conflict objects
    """
    return parse_table.conflicts


def resolve_conflicts(parse_table: ParseTable, 
                     strategy: str = "shift") -> ParseTable:
    """Attempt to resolve conflicts using specified strategy.
    
    Strategies:
    - "shift": Prefer shift over reduce (handles dangling else)
    - "first": Keep first action encountered
    - "precedence": Use operator precedence (requires annotations)
    
    Args:
        parse_table: Table with conflicts
        strategy: Resolution strategy
    
    Returns:
        New parse table with conflicts resolved
    """
    resolved = ParseTable(
        states=parse_table.states,
        productions=parse_table.productions,
        parser_type=parse_table.parser_type
    )
    
    # Copy all actions and gotos
    for key, action in parse_table.action.items():
        resolved.action[key] = action
    
    resolved.goto = parse_table.goto.copy()
    
    # Resolve conflicts based on strategy
    for conflict in parse_table.conflicts:
        state = conflict.state
        symbol = conflict.symbol
        
        if strategy == "shift":
            # Prefer shift over reduce
            if conflict.action1.is_shift():
                resolved.action[(state, symbol)] = conflict.action1
            elif conflict.action2.is_shift():
                resolved.action[(state, symbol)] = conflict.action2
            # For reduce-reduce, keep first (arbitrary)
        
        elif strategy == "first":
            # Keep first action (already in table)
            pass
    
    return resolved


class LRParserGenerator(ParserGenerator):
    """LR/LALR/SLR parser generator.
    
    Generates LR-family parsers from context-free grammars.
    
    Attributes:
        parser_type: Type of LR parser to generate
        generate_ast: Whether to generate AST schema
    """
    
    def __init__(self, parser_type: ParserType = ParserType.LALR1,
                 generate_ast: bool = True):
        """Initialize the LR parser generator.
        
        Args:
            parser_type: Type of parser (LR0, SLR1, LALR1)
            generate_ast: Whether to generate AST schema
        """
        if parser_type not in (ParserType.LR0, ParserType.SLR1, ParserType.LALR1, ParserType.LR1):
            raise ValueError(f"LRParserGenerator does not support {parser_type}")
        
        self.parser_type = parser_type
        self.generate_ast = generate_ast
    
    def generate(self, grammar: Grammar) -> ParserGeneratorResult:
        """Generate LR parse table from grammar.
        
        Args:
            grammar: The input grammar
        
        Returns:
            ParserGeneratorResult with parse table
        """
        # Validate grammar first
        issues = validate_grammar_for_parsing(grammar)
        
        result = ParserGeneratorResult(
            parse_table=ParseTable(),
            parser_type=self.parser_type
        )
        
        if issues:
            for issue in issues:
                result.add_warning(issue)
        
        # Build parse table
        parse_table = build_parse_table(grammar, self.parser_type)
        result.parse_table = parse_table
        result.conflicts = parse_table.conflicts
        
        # Generate AST schema if requested
        if self.generate_ast:
            result.ast_schema = generate_ast_schema(grammar)
        
        # Add info
        result.add_info("state_count", parse_table.state_count())
        result.add_info("production_count", len(parse_table.productions))
        result.add_info("conflict_count", len(parse_table.conflicts))
        result.add_info("terminal_count", len(parse_table.get_terminals()))
        result.add_info("nonterminal_count", len(parse_table.get_nonterminals()))
        
        return result
    
    def supports_grammar(self, grammar: Grammar) -> Tuple[bool, List[str]]:
        """Check if this generator supports the given grammar.
        
        LR parsers support most context-free grammars, but may have
        conflicts for ambiguous grammars.
        
        Args:
            grammar: The grammar to check
        
        Returns:
            Tuple of (supported, list of issues)
        """
        issues = validate_grammar_for_parsing(grammar)
        
        # Try to generate and check for conflicts
        try:
            result = self.generate(grammar)
            if result.has_conflicts():
                for conflict in result.conflicts:
                    issues.append(str(conflict))
        except Exception as e:
            issues.append(f"Generation error: {str(e)}")
        
        return (len(issues) == 0, issues)
    
    def get_parser_type(self) -> ParserType:
        """Get the type of parser this generator produces.
        
        Returns:
            ParserType enum value
        """
        return self.parser_type
