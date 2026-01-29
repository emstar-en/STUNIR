#!/usr/bin/env python3
"""LL(1) parser generator implementation.

This module provides:
- LLParserGenerator: LL(1) parser generator
- build_ll1_table(): Build LL(1) parse table
- check_ll1_conditions(): Check if grammar is LL(1)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any

from ir.parser.parse_table import (
    ParserType, LL1Table, LL1Conflict
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
        compute_first_sets, compute_follow_sets, compute_nullable,
        detect_left_recursion
    )
except ImportError:
    # Type stubs for development
    Grammar = Any
    Symbol = Any
    ProductionRule = Any
    EPSILON = None
    EOF = None


def compute_first_of_string(symbols: Tuple[Symbol, ...], 
                            first_sets: Dict[Symbol, Set[Symbol]],
                            nullable: Set[Symbol]) -> Set[Symbol]:
    """Compute FIRST set of a string of symbols.
    
    FIRST(X1 X2 ... Xn) = FIRST(X1) if X1 not nullable
                       = FIRST(X1) - {ε} ∪ FIRST(X2 ... Xn) if X1 nullable
    
    Args:
        symbols: Tuple of grammar symbols
        first_sets: Precomputed FIRST sets
        nullable: Set of nullable symbols
    
    Returns:
        FIRST set of the symbol string
    """
    result: Set[Symbol] = set()
    
    if not symbols:
        # Empty string: FIRST = {ε}
        if EPSILON:
            result.add(EPSILON)
        return result
    
    all_nullable = True
    
    for sym in symbols:
        if hasattr(sym, 'is_terminal') and sym.is_terminal():
            result.add(sym)
            all_nullable = False
            break
        elif hasattr(sym, 'is_epsilon') and sym.is_epsilon():
            continue
        elif hasattr(sym, 'is_nonterminal') and sym.is_nonterminal():
            # Add FIRST(sym) - {ε}
            sym_first = first_sets.get(sym, set())
            for s in sym_first:
                if not (hasattr(s, 'is_epsilon') and s.is_epsilon()):
                    result.add(s)
            
            if sym not in nullable:
                all_nullable = False
                break
        else:
            # Unknown symbol type, treat as terminal
            result.add(sym)
            all_nullable = False
            break
    
    if all_nullable and EPSILON:
        result.add(EPSILON)
    
    return result


def build_ll1_table(grammar: Grammar) -> LL1Table:
    """Build LL(1) parse table.
    
    Algorithm:
    For each production A → α:
    1. For each terminal a in FIRST(α), add A → α to M[A, a]
    2. If ε in FIRST(α), for each terminal b in FOLLOW(A), add A → α to M[A, b]
    
    Args:
        grammar: The grammar
    
    Returns:
        LL1Table with parse table and any conflicts
    """
    # Compute FIRST, FOLLOW, and nullable
    first_sets = compute_first_sets(grammar)
    follow_sets = compute_follow_sets(grammar, first_sets)
    nullable = compute_nullable(grammar)
    
    table = LL1Table(
        first_sets=first_sets,
        follow_sets=follow_sets
    )
    
    for production in grammar.all_productions():
        head = production.head
        body = production.body if production.body else ()
        
        # Compute FIRST(body)
        body_first = compute_first_of_string(body, first_sets, nullable)
        
        # Add entries for terminals in FIRST(body)
        for terminal in body_first:
            if hasattr(terminal, 'is_epsilon') and terminal.is_epsilon():
                continue
            if hasattr(terminal, 'is_terminal') and terminal.is_terminal():
                table.set_production(head, terminal, production)
        
        # If body can derive ε, add entries for FOLLOW(head)
        has_epsilon = any(hasattr(t, 'is_epsilon') and t.is_epsilon() for t in body_first)
        is_empty_body = len(body) == 0 or (len(body) == 1 and hasattr(body[0], 'is_epsilon') and body[0].is_epsilon())
        
        if has_epsilon or is_empty_body:
            for terminal in follow_sets.get(head, set()):
                if hasattr(terminal, 'is_terminal') and terminal.is_terminal() or terminal == EOF:
                    table.set_production(head, terminal, production)
    
    return table


def check_ll1_conditions(grammar: Grammar) -> Tuple[bool, List[str]]:
    """Check if a grammar satisfies LL(1) conditions.
    
    LL(1) conditions:
    1. No left recursion (direct or indirect)
    2. For each nonterminal A with productions A → α1 | α2 | ... | αn:
       - FIRST(αi) ∩ FIRST(αj) = ∅ for all i ≠ j
       - If αi ⇒* ε, then FIRST(αj) ∩ FOLLOW(A) = ∅ for all j ≠ i
    
    Args:
        grammar: The grammar to check
    
    Returns:
        Tuple of (is_ll1, list of issues)
    """
    issues: List[str] = []
    
    # Check for left recursion
    # detect_left_recursion returns List[Tuple[Symbol, List[Symbol]]]
    left_rec_cycles = detect_left_recursion(grammar)
    if left_rec_cycles:
        for nonterminal, cycle in left_rec_cycles:
            cycle_str = ' -> '.join(s.name if hasattr(s, 'name') else str(s) for s in cycle)
            issues.append(f"Left recursion detected for {nonterminal.name if hasattr(nonterminal, 'name') else nonterminal}: {cycle_str}")
    
    if issues:
        return (False, issues)
    
    # Compute sets
    first_sets = compute_first_sets(grammar)
    follow_sets = compute_follow_sets(grammar, first_sets)
    nullable = compute_nullable(grammar)
    
    # Check FIRST/FIRST conflicts and FIRST/FOLLOW conflicts
    for nonterminal in grammar.nonterminals:
        productions = grammar.get_productions(nonterminal)
        
        if len(productions) <= 1:
            continue
        
        # Compute FIRST sets for each production body
        prod_firsts: List[Tuple[ProductionRule, Set[Symbol]]] = []
        nullable_prods: List[ProductionRule] = []
        
        for prod in productions:
            body = prod.body if prod.body else ()
            body_first = compute_first_of_string(body, first_sets, nullable)
            
            # Remove epsilon from FIRST set
            clean_first = {s for s in body_first 
                          if not (hasattr(s, 'is_epsilon') and s.is_epsilon())}
            prod_firsts.append((prod, clean_first))
            
            if any(hasattr(s, 'is_epsilon') and s.is_epsilon() for s in body_first):
                nullable_prods.append(prod)
        
        # Check FIRST/FIRST conflicts
        for i in range(len(prod_firsts)):
            for j in range(i + 1, len(prod_firsts)):
                prod_i, first_i = prod_firsts[i]
                prod_j, first_j = prod_firsts[j]
                
                intersection = first_i & first_j
                if intersection:
                    syms = ', '.join(s.name if hasattr(s, 'name') else str(s) for s in intersection)
                    issues.append(
                        f"FIRST/FIRST conflict for {nonterminal.name}: "
                        f"'{prod_i}' and '{prod_j}' share FIRST symbols: {syms}"
                    )
        
        # Check FIRST/FOLLOW conflicts
        if nullable_prods:
            follow_A = follow_sets.get(nonterminal, set())
            
            for prod, first_set in prod_firsts:
                if prod not in nullable_prods:
                    intersection = first_set & follow_A
                    if intersection:
                        syms = ', '.join(s.name if hasattr(s, 'name') else str(s) for s in intersection)
                        issues.append(
                            f"FIRST/FOLLOW conflict for {nonterminal.name}: "
                            f"FIRST('{prod}') intersects FOLLOW({nonterminal.name}): {syms}"
                        )
    
    return (len(issues) == 0, issues)


class LLParserGenerator(ParserGenerator):
    """LL(1) parser generator.
    
    Generates LL(1) predictive parsers from suitable grammars.
    
    Attributes:
        generate_ast: Whether to generate AST schema
        check_conditions: Whether to check LL(1) conditions before generation
    """
    
    def __init__(self, generate_ast: bool = True, check_conditions: bool = True):
        """Initialize the LL parser generator.
        
        Args:
            generate_ast: Whether to generate AST schema
            check_conditions: Whether to check LL(1) conditions
        """
        self.generate_ast = generate_ast
        self.check_conditions = check_conditions
    
    def generate(self, grammar: Grammar) -> ParserGeneratorResult:
        """Generate LL(1) parse table from grammar.
        
        Args:
            grammar: The input grammar
        
        Returns:
            ParserGeneratorResult with parse table
        """
        result = ParserGeneratorResult(
            parse_table=LL1Table(),
            parser_type=ParserType.LL1
        )
        
        # Validate grammar
        issues = validate_grammar_for_parsing(grammar)
        for issue in issues:
            result.add_warning(issue)
        
        # Check LL(1) conditions
        if self.check_conditions:
            is_ll1, ll1_issues = check_ll1_conditions(grammar)
            for issue in ll1_issues:
                result.add_warning(issue)
        
        # Build parse table
        ll_table = build_ll1_table(grammar)
        result.parse_table = ll_table
        result.conflicts = ll_table.conflicts
        
        # Generate AST schema if requested
        if self.generate_ast:
            result.ast_schema = generate_ast_schema(grammar)
        
        # Add info
        result.add_info("table_entries", len(ll_table.table))
        result.add_info("conflict_count", len(ll_table.conflicts))
        result.add_info("nonterminal_count", len(ll_table.get_nonterminals()))
        result.add_info("terminal_count", len(ll_table.get_terminals()))
        
        return result
    
    def supports_grammar(self, grammar: Grammar) -> Tuple[bool, List[str]]:
        """Check if this generator supports the given grammar.
        
        LL(1) parsers require:
        - No left recursion
        - No FIRST/FIRST conflicts
        - No FIRST/FOLLOW conflicts
        
        Args:
            grammar: The grammar to check
        
        Returns:
            Tuple of (supported, list of issues)
        """
        issues = validate_grammar_for_parsing(grammar)
        
        # Check LL(1) conditions
        is_ll1, ll1_issues = check_ll1_conditions(grammar)
        issues.extend(ll1_issues)
        
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
            ParserType.LL1
        """
        return ParserType.LL1


def generate_recursive_descent(grammar: Grammar) -> str:
    """Generate recursive descent parser skeleton.
    
    Generates a simple recursive descent parser structure
    without full parsing logic (for demonstration).
    
    Args:
        grammar: The grammar
    
    Returns:
        Python code skeleton for recursive descent parser
    """
    lines = [
        '"""Recursive Descent Parser for ' + grammar.name + '"""',
        '',
        'class Token:',
        '    def __init__(self, type: str, value: str):',
        '        self.type = type',
        '        self.value = value',
        '',
        'class Parser:',
        '    def __init__(self, tokens: list):',
        '        self.tokens = tokens',
        '        self.pos = 0',
        '',
        '    def current(self) -> Token:',
        '        if self.pos < len(self.tokens):',
        '            return self.tokens[self.pos]',
        '        return Token("EOF", "")',
        '',
        '    def match(self, expected: str) -> Token:',
        '        token = self.current()',
        '        if token.type == expected:',
        '            self.pos += 1',
        '            return token',
        '        raise SyntaxError(f"Expected {expected}, got {token.type}")',
        '',
    ]
    
    # Generate a method for each nonterminal
    first_sets = compute_first_sets(grammar)
    
    for nonterminal in sorted(grammar.nonterminals, key=lambda s: s.name):
        method_name = f"parse_{nonterminal.name.lower().replace('-', '_')}"
        lines.append(f'    def {method_name}(self):')
        lines.append(f'        """Parse {nonterminal.name}."""')
        
        productions = grammar.get_productions(nonterminal)
        
        if len(productions) == 1:
            prod = productions[0]
            lines.append(f'        # {prod}')
            for sym in prod.body:
                if hasattr(sym, 'is_terminal') and sym.is_terminal():
                    lines.append(f'        self.match("{sym.name}")')
                elif hasattr(sym, 'is_nonterminal') and sym.is_nonterminal():
                    sub_method = f"parse_{sym.name.lower().replace('-', '_')}"
                    lines.append(f'        self.{sub_method}()')
            lines.append('        pass')
        else:
            lines.append('        token = self.current()')
            
            for i, prod in enumerate(productions):
                body_first = compute_first_of_string(
                    prod.body if prod.body else (), 
                    first_sets, 
                    compute_nullable(grammar)
                )
                
                conditions = []
                for sym in body_first:
                    if hasattr(sym, 'is_terminal') and sym.is_terminal():
                        conditions.append(f'token.type == "{sym.name}"')
                
                if conditions:
                    condition = ' or '.join(conditions)
                    prefix = 'if' if i == 0 else 'elif'
                    lines.append(f'        {prefix} {condition}:')
                    lines.append(f'            # {prod}')
                    lines.append('            pass')
            
            lines.append('        else:')
            lines.append(f'            raise SyntaxError(f"Unexpected {{token.type}} in {nonterminal.name}")')
        
        lines.append('')
    
    return '\n'.join(lines)
