#!/usr/bin/env python3
"""Grammar validation algorithms.

This module provides comprehensive validation and analysis functions for grammars:

Validation:
- validate_grammar(): Comprehensive grammar validation

Analysis:
- compute_first_sets(): Compute FIRST sets for all symbols
- compute_follow_sets(): Compute FOLLOW sets for all non-terminals
- compute_nullable(): Find nullable non-terminals
- detect_left_recursion(): Find left-recursive cycles
- detect_ambiguity(): Find potential ambiguity
- find_unreachable_nonterminals(): Find unreachable symbols
"""

from typing import Dict, List, Set, Tuple, Optional

from ir.grammar.symbol import Symbol, SymbolType, EPSILON, EOF
from ir.grammar.production import ProductionRule
from ir.grammar.grammar_ir import Grammar, GrammarType, ValidationResult


def compute_nullable(grammar: Grammar) -> Set[Symbol]:
    """Compute the set of nullable non-terminals.
    
    A non-terminal is nullable if it can derive the empty string (ε).
    
    Algorithm:
    1. Add non-terminals with ε productions to nullable set
    2. Add non-terminal A if A → X1...Xn where all Xi are nullable
    3. Repeat until fixed point
    
    Args:
        grammar: The grammar to analyze
    
    Returns:
        Set of nullable non-terminal symbols
    """
    nullable: Set[Symbol] = set()
    
    # Add non-terminals with epsilon productions
    for prod in grammar.all_productions():
        if prod.is_epsilon_production():
            nullable.add(prod.head)
    
    # Iterate until fixed point
    changed = True
    while changed:
        changed = False
        for prod in grammar.all_productions():
            if prod.head in nullable:
                continue
            
            # Get actual symbols (not EBNF operators)
            body_syms = prod.body_symbols()
            
            # A is nullable if all symbols in body are nullable or epsilon
            if all(sym in nullable or sym.is_epsilon() for sym in body_syms):
                nullable.add(prod.head)
                changed = True
    
    return nullable


def compute_first_sets(grammar: Grammar) -> Dict[Symbol, Set[Symbol]]:
    """Compute FIRST sets for all symbols in the grammar.
    
    FIRST(X) = set of terminals that can begin strings derived from X
    
    Algorithm:
    1. For terminal a: FIRST(a) = {a}
    2. For non-terminal A:
       - If A → ε, add ε to FIRST(A)
       - If A → Y1 Y2 ... Yk:
         - Add FIRST(Y1) - {ε} to FIRST(A)
         - If ε ∈ FIRST(Y1), add FIRST(Y2) - {ε}
         - Continue until Yi doesn't derive ε
         - If all Yi derive ε, add ε to FIRST(A)
    3. Repeat until no changes
    
    Args:
        grammar: The grammar to analyze
    
    Returns:
        Dict mapping symbols to their FIRST sets
    """
    first: Dict[Symbol, Set[Symbol]] = {}
    
    # Initialize: FIRST(terminal) = {terminal}
    for t in grammar.terminals:
        first[t] = {t}
    
    # Add special symbols
    first[EPSILON] = {EPSILON}
    first[EOF] = {EOF}
    
    # Initialize: FIRST(nonterminal) = {}
    for nt in grammar.nonterminals:
        first[nt] = set()
    
    # Compute nullable set for reference
    nullable = compute_nullable(grammar)
    
    # Iterate until fixed point
    changed = True
    while changed:
        changed = False
        
        for prod in grammar.all_productions():
            head = prod.head
            body_syms = prod.body_symbols()
            
            if not body_syms or prod.is_epsilon_production():
                # A → ε: add ε to FIRST(A)
                if EPSILON not in first[head]:
                    first[head].add(EPSILON)
                    changed = True
            else:
                # Process each symbol in the body
                for sym in body_syms:
                    if sym.is_epsilon():
                        continue
                    
                    # Add FIRST(sym) - {ε} to FIRST(head)
                    before_size = len(first[head])
                    sym_first = first.get(sym, {sym} if sym.is_terminal() else set())
                    first[head] |= (sym_first - {EPSILON})
                    
                    if len(first[head]) > before_size:
                        changed = True
                    
                    # If sym is not nullable, stop
                    if sym not in nullable and EPSILON not in sym_first:
                        break
                else:
                    # All symbols can derive ε
                    if EPSILON not in first[head]:
                        first[head].add(EPSILON)
                        changed = True
    
    return first


def compute_first_of_string(symbols: List[Symbol], first: Dict[Symbol, Set[Symbol]]) -> Set[Symbol]:
    """Compute FIRST set of a sequence of symbols.
    
    Args:
        symbols: List of symbols (sequence)
        first: Precomputed FIRST sets
    
    Returns:
        FIRST set of the symbol sequence
    """
    result: Set[Symbol] = set()
    
    if not symbols:
        return {EPSILON}
    
    for sym in symbols:
        sym_first = first.get(sym, {sym} if sym.is_terminal() else set())
        result |= (sym_first - {EPSILON})
        
        if EPSILON not in sym_first:
            break
    else:
        result.add(EPSILON)
    
    return result


def compute_first_of_production(prod: ProductionRule, first: Dict[Symbol, Set[Symbol]]) -> Set[Symbol]:
    """Compute FIRST set of a production's body.
    
    Args:
        prod: The production rule
        first: Precomputed FIRST sets
    
    Returns:
        FIRST set of the production body
    """
    if prod.is_epsilon_production():
        return {EPSILON}
    
    return compute_first_of_string(prod.body_symbols(), first)


def compute_follow_sets(grammar: Grammar, first: Dict[Symbol, Set[Symbol]]) -> Dict[Symbol, Set[Symbol]]:
    """Compute FOLLOW sets for all non-terminals.
    
    FOLLOW(A) = set of terminals that can appear immediately after A
    
    Algorithm:
    1. Add $ to FOLLOW(start_symbol)
    2. For each production A → αBβ:
       - Add FIRST(β) - {ε} to FOLLOW(B)
       - If ε ∈ FIRST(β) or β is empty, add FOLLOW(A) to FOLLOW(B)
    3. Repeat until no changes
    
    Args:
        grammar: The grammar to analyze
        first: Precomputed FIRST sets
    
    Returns:
        Dict mapping non-terminals to their FOLLOW sets
    """
    follow: Dict[Symbol, Set[Symbol]] = {}
    
    # Initialize FOLLOW sets
    for nt in grammar.nonterminals:
        follow[nt] = set()
    
    # Add $ to FOLLOW(start)
    follow[grammar.start_symbol].add(EOF)
    
    # Iterate until fixed point
    changed = True
    while changed:
        changed = False
        
        for prod in grammar.all_productions():
            head = prod.head
            body_syms = prod.body_symbols()
            
            for i, sym in enumerate(body_syms):
                if not sym.is_nonterminal():
                    continue
                
                # β = everything after sym
                beta = body_syms[i + 1:]
                
                # Compute FIRST(β)
                first_beta = compute_first_of_string(beta, first)
                
                before_size = len(follow[sym])
                
                # Add FIRST(β) - {ε} to FOLLOW(sym)
                follow[sym] |= (first_beta - {EPSILON})
                
                # If ε ∈ FIRST(β) or β is empty, add FOLLOW(head) to FOLLOW(sym)
                if not beta or EPSILON in first_beta:
                    follow[sym] |= follow[head]
                
                if len(follow[sym]) > before_size:
                    changed = True
    
    return follow


def detect_left_recursion(grammar: Grammar) -> List[Tuple[Symbol, List[Symbol]]]:
    """Detect left recursion in the grammar.
    
    Left recursion occurs when a non-terminal A can derive Aα (directly or indirectly).
    
    Algorithm:
    - Build "left-derives" graph where A left-derives B if A → Bα
      or A → Cα where C is nullable and C left-derives B
    - Find cycles using DFS
    
    Args:
        grammar: The grammar to analyze
    
    Returns:
        List of (non-terminal, cycle) tuples where cycle shows the left-recursion path
    """
    results: List[Tuple[Symbol, List[Symbol]]] = []
    
    # Compute nullable non-terminals
    nullable = compute_nullable(grammar)
    
    # Build "left-derives" graph
    left_derives: Dict[Symbol, Set[Symbol]] = {nt: set() for nt in grammar.nonterminals}
    
    for prod in grammar.all_productions():
        head = prod.head
        body_syms = prod.body_symbols()
        
        for sym in body_syms:
            if sym.is_nonterminal():
                left_derives[head].add(sym)
            # Stop at first non-nullable symbol
            if sym not in nullable:
                break
    
    # DFS to find cycles starting from each non-terminal
    def find_cycle(start: Symbol) -> Optional[List[Symbol]]:
        visited: Set[Symbol] = set()
        path: List[Symbol] = []
        
        def dfs(current: Symbol) -> Optional[List[Symbol]]:
            if current == start and len(path) > 0:
                return path + [current]
            
            if current in visited:
                return None
            
            visited.add(current)
            path.append(current)
            
            for next_sym in left_derives.get(current, set()):
                result = dfs(next_sym)
                if result:
                    return result
            
            path.pop()
            return None
        
        # Start DFS from start symbol's successors
        for next_sym in left_derives.get(start, set()):
            result = dfs(next_sym)
            if result:
                return [start] + result
        
        return None
    
    # Find cycles for each non-terminal
    found_cycles: Set[frozenset] = set()
    for nt in grammar.nonterminals:
        cycle = find_cycle(nt)
        if cycle:
            # Normalize cycle to avoid duplicates
            cycle_set = frozenset(cycle[:-1])  # Exclude repeated start
            if cycle_set not in found_cycles:
                found_cycles.add(cycle_set)
                results.append((nt, cycle))
    
    return results


def find_unreachable_nonterminals(grammar: Grammar) -> Set[Symbol]:
    """Find non-terminals that cannot be reached from the start symbol.
    
    A non-terminal is unreachable if there's no derivation path from
    the start symbol to that non-terminal.
    
    Args:
        grammar: The grammar to analyze
    
    Returns:
        Set of unreachable non-terminal symbols
    """
    reachable: Set[Symbol] = set()
    worklist = [grammar.start_symbol]
    
    while worklist:
        current = worklist.pop()
        if current in reachable:
            continue
        
        reachable.add(current)
        
        for prod in grammar.get_productions(current):
            for sym in prod.body_symbols():
                if sym.is_nonterminal() and sym not in reachable:
                    worklist.append(sym)
    
    return grammar.nonterminals - reachable


def detect_ambiguity(grammar: Grammar) -> List[Tuple[Symbol, str]]:
    """Detect potential ambiguity in the grammar.
    
    Checks for:
    - LL(1) conflicts (FIRST/FIRST and FIRST/FOLLOW)
    - Multiple epsilon productions
    
    Note: This is a heuristic check; full ambiguity detection is undecidable.
    
    Args:
        grammar: The grammar to analyze
    
    Returns:
        List of (non-terminal, reason) tuples describing potential ambiguity
    """
    issues: List[Tuple[Symbol, str]] = []
    
    # Compute or use cached sets
    if grammar._first_sets is None:
        grammar._first_sets = compute_first_sets(grammar)
    if grammar._follow_sets is None:
        grammar._follow_sets = compute_follow_sets(grammar, grammar._first_sets)
    
    first = grammar._first_sets
    follow = grammar._follow_sets
    
    for nt in grammar.nonterminals:
        productions = grammar.get_productions(nt)
        if len(productions) < 2:
            continue
        
        # Check FIRST/FIRST conflicts
        for i, prod_i in enumerate(productions):
            first_i = compute_first_of_production(prod_i, first)
            
            for j, prod_j in enumerate(productions[i + 1:], i + 1):
                first_j = compute_first_of_production(prod_j, first)
                
                overlap = (first_i - {EPSILON}) & (first_j - {EPSILON})
                if overlap:
                    symbols = ", ".join(s.name for s in overlap)
                    issues.append((nt, f"FIRST/FIRST conflict on {symbols}"))
        
        # Check FIRST/FOLLOW conflicts for nullable productions
        for prod in productions:
            first_prod = compute_first_of_production(prod, first)
            if EPSILON in first_prod:
                conflict = (first_prod - {EPSILON}) & follow.get(nt, set())
                if conflict:
                    symbols = ", ".join(s.name for s in conflict)
                    issues.append((nt, f"FIRST/FOLLOW conflict on {symbols}"))
    
    return issues


def validate_grammar(grammar: Grammar) -> ValidationResult:
    """Perform comprehensive grammar validation.
    
    Checks:
    1. Start symbol is defined
    2. All non-terminals have at least one production
    3. No unreachable non-terminals
    4. No undefined non-terminal references
    5. Left recursion detection (warning for BNF/EBNF, error for PEG)
    6. Ambiguity detection (warning)
    
    Args:
        grammar: The grammar to validate
    
    Returns:
        ValidationResult with errors, warnings, and info
    """
    errors: List[str] = []
    warnings: List[str] = []
    info: Dict[str, any] = {}
    
    # Check start symbol is in non-terminals
    if grammar.start_symbol not in grammar.nonterminals:
        errors.append(f"Start symbol '{grammar.start_symbol.name}' not defined")
    
    # Check all non-terminals have productions
    for nt in grammar.nonterminals:
        if not grammar.get_productions(nt):
            errors.append(f"Non-terminal '{nt.name}' has no productions")
    
    # Check for undefined non-terminals in production bodies
    defined = grammar.nonterminals
    for prod in grammar.all_productions():
        for sym in prod.body_symbols():
            if sym.is_nonterminal() and sym not in defined:
                errors.append(f"Undefined non-terminal '{sym.name}' in production: {prod}")
    
    # Reachability analysis
    unreachable = find_unreachable_nonterminals(grammar)
    if unreachable:
        names = ", ".join(s.name for s in unreachable)
        warnings.append(f"Unreachable non-terminals: {names}")
    info["unreachable"] = [s.name for s in unreachable]
    
    # Left recursion detection
    left_recursive = detect_left_recursion(grammar)
    if left_recursive:
        for nt, cycle in left_recursive:
            cycle_str = " → ".join(s.name for s in cycle)
            if grammar.grammar_type == GrammarType.PEG:
                errors.append(f"Left recursion detected (invalid for PEG): {cycle_str}")
            else:
                warnings.append(f"Left recursion detected: {cycle_str}")
    info["left_recursive"] = [(nt.name, [s.name for s in cycle]) for nt, cycle in left_recursive]
    
    # Compute analysis sets
    first_sets = compute_first_sets(grammar)
    follow_sets = compute_follow_sets(grammar, first_sets)
    nullable = compute_nullable(grammar)
    
    info["first_sets"] = {k.name: [s.name for s in v] for k, v in first_sets.items() if k.is_nonterminal()}
    info["follow_sets"] = {k.name: [s.name for s in v] for k, v in follow_sets.items()}
    info["nullable"] = [s.name for s in nullable]
    
    # Cache computed sets in grammar
    grammar._first_sets = first_sets
    grammar._follow_sets = follow_sets
    grammar._nullable = nullable
    
    # Ambiguity detection (only if no errors so far)
    if not errors:
        ambiguities = detect_ambiguity(grammar)
        for nt, reason in ambiguities:
            warnings.append(f"Potential ambiguity in '{nt.name}': {reason}")
        info["ambiguities"] = [(nt.name, reason) for nt, reason in ambiguities]
    
    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        info=info
    )
