#!/usr/bin/env python3
"""Grammar transformation algorithms.

This module provides functions to transform grammars:
- eliminate_left_recursion(): Remove left recursion
- left_factor(): Apply left factoring
- convert_ebnf_to_bnf(): Convert EBNF to pure BNF
- normalize_grammar(): Normalize grammar to standard form
"""

from typing import Dict, List, Set, Tuple, Optional

from ir.grammar.symbol import Symbol, SymbolType, EPSILON
from ir.grammar.production import (
    ProductionRule, 
    BodyElement,
    EBNFOperator,
    OptionalOp,
    Repetition,
    OneOrMore,
    Group,
    Alternation,
)
from ir.grammar.grammar_ir import Grammar, GrammarType
from ir.grammar.validation import compute_nullable


def eliminate_left_recursion(grammar: Grammar) -> Grammar:
    """Eliminate left recursion from the grammar.
    
    For direct left recursion A → Aα | β:
    - Replace with: A → βA'
                    A' → αA' | ε
    
    For indirect left recursion, first substitutes to convert to direct.
    
    Args:
        grammar: The grammar to transform
    
    Returns:
        New grammar without left recursion
    """
    # Create new grammar
    new_grammar = Grammar(
        name=grammar.name + "_no_lr",
        grammar_type=grammar.grammar_type,
        start_symbol=grammar.start_symbol,
        metadata=grammar.metadata.copy()
    )
    
    # Get ordered list of non-terminals
    nonterminals = list(grammar.nonterminals)
    
    # Working copy of productions
    working_prods: Dict[Symbol, List[ProductionRule]] = {}
    for nt in nonterminals:
        working_prods[nt] = list(grammar.get_productions(nt))
    
    # Process each non-terminal
    for i, Ai in enumerate(nonterminals):
        # First, substitute Ai → Ajγ where j < i with Ai → δ1γ | δ2γ | ...
        for j in range(i):
            Aj = nonterminals[j]
            _substitute_productions(working_prods, Ai, Aj)
        
        # Then eliminate direct left recursion for Ai
        _eliminate_direct_left_recursion(working_prods, Ai, new_grammar)
    
    return new_grammar


def _substitute_productions(
    working_prods: Dict[Symbol, List[ProductionRule]], 
    Ai: Symbol, 
    Aj: Symbol
) -> None:
    """Substitute Aj productions into Ai productions.
    
    For Ai → Ajγ, replace with Ai → δ1γ | δ2γ | ... where Aj → δ1 | δ2 | ...
    """
    Ai_prods = working_prods.get(Ai, [])
    Aj_prods = working_prods.get(Aj, [])
    
    new_prods = []
    
    for prod in Ai_prods:
        body_syms = prod.body_symbols()
        
        if body_syms and body_syms[0] == Aj:
            # Ai → Aj γ: substitute all Aj productions
            gamma = tuple(prod.body[1:])  # Rest of the production
            
            for aj_prod in Aj_prods:
                delta = aj_prod.body
                new_body = delta + gamma
                new_prods.append(ProductionRule(Ai, new_body, prod.label))
        else:
            # Keep as-is
            new_prods.append(prod)
    
    working_prods[Ai] = new_prods


def _eliminate_direct_left_recursion(
    working_prods: Dict[Symbol, List[ProductionRule]],
    A: Symbol,
    new_grammar: Grammar
) -> None:
    """Eliminate direct left recursion for non-terminal A.
    
    For A → Aα1 | Aα2 | ... | β1 | β2 | ...
    Replace with:
        A → β1A' | β2A' | ...
        A' → α1A' | α2A' | ... | ε
    """
    productions = working_prods.get(A, [])
    
    # Separate recursive and non-recursive productions
    recursive = []      # A → Aα
    non_recursive = []  # A → β
    
    for prod in productions:
        body_syms = prod.body_symbols()
        if body_syms and body_syms[0] == A:
            recursive.append(prod)
        else:
            non_recursive.append(prod)
    
    if not recursive:
        # No left recursion, copy productions as-is
        for prod in productions:
            new_grammar.add_production(prod)
        return
    
    # Create A' non-terminal
    A_prime = Symbol(A.name + "'", SymbolType.NONTERMINAL)
    
    # A → βA' for each non-recursive production β
    for prod in non_recursive:
        if prod.is_epsilon_production():
            # A → ε becomes A → A'
            new_body = (A_prime,)
        else:
            new_body = tuple(list(prod.body) + [A_prime])
        new_grammar.add_production(ProductionRule(A, new_body, prod.label))
    
    # If no non-recursive productions, add A → A'
    if not non_recursive:
        new_grammar.add_production(ProductionRule(A, (A_prime,)))
    
    # A' → αA' for each recursive production (from A → Aα)
    for prod in recursive:
        alpha = prod.body[1:]  # Remove leading A
        if alpha:
            new_body = tuple(list(alpha) + [A_prime])
        else:
            new_body = (A_prime,)
        new_grammar.add_production(ProductionRule(A_prime, new_body))
    
    # A' → ε
    new_grammar.add_production(ProductionRule(A_prime, ()))


def left_factor(grammar: Grammar) -> Grammar:
    """Apply left factoring to the grammar.
    
    For productions A → αβ1 | αβ2 | ... | αβn:
    - Replace with: A → αA'
                    A' → β1 | β2 | ... | βn
    
    Args:
        grammar: The grammar to transform
    
    Returns:
        New left-factored grammar
    """
    new_grammar = Grammar(
        name=grammar.name + "_factored",
        grammar_type=grammar.grammar_type,
        start_symbol=grammar.start_symbol,
        metadata=grammar.metadata.copy()
    )
    
    counter = [0]  # Use list for closure mutation
    
    for nt in grammar.nonterminals:
        productions = grammar.get_productions(nt)
        _left_factor_productions(new_grammar, nt, productions, counter)
    
    return new_grammar


def _find_longest_common_prefix(productions: List[ProductionRule]) -> Tuple[BodyElement, ...]:
    """Find the longest common prefix among production bodies.
    
    Args:
        productions: List of production rules with the same head
    
    Returns:
        Tuple of common prefix symbols (may be empty)
    """
    if not productions or len(productions) < 2:
        return ()
    
    # Get minimum body length
    min_len = min(len(p.body) for p in productions)
    if min_len == 0:
        return ()
    
    prefix: List[BodyElement] = []
    
    for i in range(min_len):
        # Get all symbols at position i
        symbols_at_i = {p.body[i] for p in productions}
        
        if len(symbols_at_i) == 1:
            prefix.append(productions[0].body[i])
        else:
            break
    
    return tuple(prefix)


def _left_factor_productions(
    new_grammar: Grammar,
    nt: Symbol,
    productions: List[ProductionRule],
    counter: List[int]
) -> None:
    """Left factor productions for a single non-terminal.
    
    Args:
        new_grammar: The grammar to add productions to
        nt: The non-terminal being factored
        productions: Its current productions
        counter: Counter for generating unique names
    """
    if len(productions) <= 1:
        # No factoring needed
        for prod in productions:
            new_grammar.add_production(prod)
        return
    
    # Find common prefix
    prefix = _find_longest_common_prefix(productions)
    
    if not prefix:
        # No common prefix, add all productions as-is
        for prod in productions:
            new_grammar.add_production(prod)
        return
    
    # Group productions by whether they share the prefix
    with_prefix = []
    without_prefix = []
    
    for prod in productions:
        if len(prod.body) >= len(prefix) and prod.body[:len(prefix)] == prefix:
            with_prefix.append(prod)
        else:
            without_prefix.append(prod)
    
    if len(with_prefix) < 2:
        # Not enough to factor
        for prod in productions:
            new_grammar.add_production(prod)
        return
    
    # Create new non-terminal for suffixes
    counter[0] += 1
    new_nt = Symbol(f"{nt.name}_{counter[0]}", SymbolType.NONTERMINAL)
    
    # A → α A' (where α is the common prefix)
    new_body = prefix + (new_nt,)
    new_grammar.add_production(ProductionRule(nt, new_body))
    
    # A' → β1 | β2 | ... for each suffix
    suffixes = []
    for prod in with_prefix:
        suffix = prod.body[len(prefix):]
        if suffix:
            suffixes.append(ProductionRule(new_nt, suffix, prod.label))
        else:
            # Empty suffix → epsilon production
            suffixes.append(ProductionRule(new_nt, (), prod.label))
    
    # Recursively factor the suffixes
    _left_factor_productions(new_grammar, new_nt, suffixes, counter)
    
    # Add productions without the common prefix
    for prod in without_prefix:
        new_grammar.add_production(prod)


def convert_ebnf_to_bnf(grammar: Grammar) -> Grammar:
    """Convert EBNF grammar to pure BNF.
    
    Transformations:
    - [A] (optional) → A_opt → A | ε
    - {A} (repetition) → A_rep → A A_rep | ε
    - A+ (one or more) → A_plus → A A_rep
    - (A B C) (grouping) → A_grp → A B C
    - A | B (alternation) → handled via multiple productions
    
    Args:
        grammar: The EBNF grammar to convert
    
    Returns:
        New BNF grammar without EBNF operators
    """
    new_grammar = Grammar(
        name=grammar.name + "_bnf",
        grammar_type=GrammarType.BNF,
        start_symbol=grammar.start_symbol,
        metadata=grammar.metadata.copy()
    )
    
    counter = [0]
    generated_prods: List[ProductionRule] = []
    
    def generate_name(base: str) -> str:
        counter[0] += 1
        return f"_{base}_{counter[0]}"
    
    def transform_element(element: BodyElement) -> Symbol:
        """Transform an EBNF element to a symbol, generating helper productions."""
        if isinstance(element, Symbol):
            return element
        
        elif isinstance(element, OptionalOp):
            # [A] → A_opt → A | ε
            inner = transform_element(element.element)
            opt_name = generate_name("opt")
            opt_sym = Symbol(opt_name, SymbolType.NONTERMINAL)
            generated_prods.append(ProductionRule(opt_sym, (inner,)))
            generated_prods.append(ProductionRule(opt_sym, ()))
            return opt_sym
        
        elif isinstance(element, Repetition):
            # {A} → A_rep → A A_rep | ε
            inner = transform_element(element.element)
            rep_name = generate_name("rep")
            rep_sym = Symbol(rep_name, SymbolType.NONTERMINAL)
            generated_prods.append(ProductionRule(rep_sym, (inner, rep_sym)))
            generated_prods.append(ProductionRule(rep_sym, ()))
            return rep_sym
        
        elif isinstance(element, OneOrMore):
            # A+ → A_plus → A A_rep
            inner = transform_element(element.element)
            rep_name = generate_name("rep")
            plus_name = generate_name("plus")
            rep_sym = Symbol(rep_name, SymbolType.NONTERMINAL)
            plus_sym = Symbol(plus_name, SymbolType.NONTERMINAL)
            generated_prods.append(ProductionRule(rep_sym, (inner, rep_sym)))
            generated_prods.append(ProductionRule(rep_sym, ()))
            generated_prods.append(ProductionRule(plus_sym, (inner, rep_sym)))
            return plus_sym
        
        elif isinstance(element, Group):
            # (A B C) → A_grp → A B C
            transformed = [transform_element(e) for e in element.elements]
            if len(transformed) == 1:
                return transformed[0]
            group_name = generate_name("grp")
            group_sym = Symbol(group_name, SymbolType.NONTERMINAL)
            generated_prods.append(ProductionRule(group_sym, tuple(transformed)))
            return group_sym
        
        elif isinstance(element, Alternation):
            # a | b | c → alt → a | b | c (multiple productions)
            alt_name = generate_name("alt")
            alt_sym = Symbol(alt_name, SymbolType.NONTERMINAL)
            
            for alt in element.alternatives:
                if isinstance(alt, tuple):
                    # Sequence of elements
                    transformed = tuple(transform_element(e) for e in alt)
                    generated_prods.append(ProductionRule(alt_sym, transformed))
                else:
                    # Single element
                    transformed = transform_element(alt)
                    generated_prods.append(ProductionRule(alt_sym, (transformed,)))
            
            return alt_sym
        
        else:
            raise ValueError(f"Unknown EBNF element type: {type(element)}")
    
    # Transform all productions
    for prod in grammar.all_productions():
        new_body = []
        for element in prod.body:
            new_body.append(transform_element(element))
        
        # Add the main production
        new_grammar.add_production(ProductionRule(prod.head, tuple(new_body), prod.label))
    
    # Add generated helper productions
    for gen_prod in generated_prods:
        new_grammar.add_production(gen_prod)
    
    return new_grammar


def normalize_grammar(grammar: Grammar) -> Grammar:
    """Normalize grammar to a standard form.
    
    Steps:
    1. Remove duplicate productions
    2. Remove unreachable non-terminals
    3. Remove non-productive non-terminals
    4. Sort productions consistently
    
    Args:
        grammar: The grammar to normalize
    
    Returns:
        New normalized grammar
    """
    # Step 1: Remove duplicate productions
    seen: Set[Tuple[Symbol, Tuple[BodyElement, ...]]] = set()
    unique_prods: List[ProductionRule] = []
    
    for prod in grammar.all_productions():
        key = (prod.head, prod.body)
        if key not in seen:
            seen.add(key)
            unique_prods.append(prod)
    
    # Step 2: Find productive non-terminals
    # A non-terminal is productive if it can derive a string of terminals
    productive: Set[Symbol] = set()
    
    changed = True
    while changed:
        changed = False
        for prod in unique_prods:
            if prod.head in productive:
                continue
            
            body_syms = prod.body_symbols()
            
            # Production is productive if all symbols are productive
            # (terminals are always productive)
            if all(
                sym in productive or sym.is_terminal() or sym.is_epsilon()
                for sym in body_syms
            ):
                productive.add(prod.head)
                changed = True
    
    # Filter to keep only productive productions
    productive_prods = [p for p in unique_prods if p.head in productive]
    
    # Also filter body symbols to only reference productive non-terminals
    filtered_prods = []
    for prod in productive_prods:
        body_syms = prod.body_symbols()
        if all(
            sym.is_terminal() or sym.is_epsilon() or sym in productive
            for sym in body_syms
        ):
            filtered_prods.append(prod)
    
    # Step 3: Find reachable non-terminals
    reachable: Set[Symbol] = {grammar.start_symbol}
    
    changed = True
    while changed:
        changed = False
        for prod in filtered_prods:
            if prod.head not in reachable:
                continue
            
            for sym in prod.body_symbols():
                if sym.is_nonterminal() and sym not in reachable:
                    reachable.add(sym)
                    changed = True
    
    # Build new grammar with only reachable, productive productions
    new_grammar = Grammar(
        name=grammar.name + "_normalized",
        grammar_type=grammar.grammar_type,
        start_symbol=grammar.start_symbol,
        metadata=grammar.metadata.copy()
    )
    
    # Add productions in sorted order for consistency
    final_prods = [p for p in filtered_prods if p.head in reachable]
    final_prods.sort(key=lambda p: (p.head.name, str(p.body)))
    
    for prod in final_prods:
        new_grammar.add_production(prod)
    
    return new_grammar
