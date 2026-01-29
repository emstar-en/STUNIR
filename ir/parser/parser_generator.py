#!/usr/bin/env python3
"""Parser generator interface and base classes.

This module provides:
- ParserGeneratorResult: Result of parser generation
- ParserGenerator: Abstract base class for parser generators
- ErrorRecoveryStrategy: Error recovery strategies
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Set, Optional, Any, Tuple, Union

from ir.parser.parse_table import (
    ParserType, ParseTable, LL1Table, 
    Conflict, LL1Conflict
)
from ir.parser.ast_node import ASTSchema

# Import from grammar module
try:
    from ir.grammar.grammar_ir import Grammar
except ImportError:
    Grammar = Any


class ErrorRecoveryStrategy(Enum):
    """Error recovery strategies for generated parsers."""
    PANIC_MODE = auto()       # Skip tokens until synchronization point
    PHRASE_LEVEL = auto()     # Insert/delete tokens locally
    ERROR_PRODUCTIONS = auto() # Use error productions in grammar


@dataclass
class ParserGeneratorResult:
    """Result of parser generation.
    
    Attributes:
        parse_table: The generated parse table (LR) or LL1Table (LL)
        ast_schema: Optional AST schema
        parser_type: Type of parser generated
        conflicts: List of conflicts detected
        warnings: List of warnings
        info: Additional information
    """
    parse_table: Union[ParseTable, LL1Table]
    ast_schema: Optional[ASTSchema] = None
    parser_type: ParserType = ParserType.LALR1
    conflicts: List[Union[Conflict, LL1Conflict]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)
    
    def has_conflicts(self) -> bool:
        """Check if generation had conflicts.
        
        Returns:
            True if there are conflicts
        """
        return len(self.conflicts) > 0
    
    def is_successful(self) -> bool:
        """Check if generation was successful (no conflicts).
        
        Returns:
            True if no conflicts
        """
        return not self.has_conflicts()
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message.
        
        Args:
            warning: Warning message
        """
        self.warnings.append(warning)
    
    def add_info(self, key: str, value: Any) -> None:
        """Add information.
        
        Args:
            key: Information key
            value: Information value
        """
        self.info[key] = value
    
    def __str__(self) -> str:
        lines = [f"ParserGeneratorResult ({self.parser_type.name})"]
        lines.append(f"Successful: {self.is_successful()}")
        lines.append(f"Conflicts: {len(self.conflicts)}")
        lines.append(f"Warnings: {len(self.warnings)}")
        
        if self.info:
            lines.append("Info:")
            for key, value in self.info.items():
                lines.append(f"  {key}: {value}")
        
        if self.conflicts:
            lines.append("Conflicts:")
            for conflict in self.conflicts[:5]:  # Show first 5
                lines.append(f"  {conflict}")
            if len(self.conflicts) > 5:
                lines.append(f"  ... and {len(self.conflicts) - 5} more")
        
        return "\n".join(lines)


class ParserGenerator(ABC):
    """Abstract interface for parser generators.
    
    Subclasses implement specific parser types (LR, LL, etc.).
    """
    
    @abstractmethod
    def generate(self, grammar: Grammar) -> ParserGeneratorResult:
        """Generate parser tables from grammar.
        
        Args:
            grammar: The input grammar
        
        Returns:
            ParserGeneratorResult with tables and metadata
        """
        pass
    
    @abstractmethod
    def supports_grammar(self, grammar: Grammar) -> Tuple[bool, List[str]]:
        """Check if this generator supports the given grammar.
        
        Args:
            grammar: The grammar to check
        
        Returns:
            Tuple of (supported, list of issues)
        """
        pass
    
    @abstractmethod
    def get_parser_type(self) -> ParserType:
        """Get the type of parser this generator produces.
        
        Returns:
            ParserType enum value
        """
        pass


def generate_error_recovery(grammar: Grammar, 
                           strategy: ErrorRecoveryStrategy) -> Dict[str, Any]:
    """Generate error recovery code/tables.
    
    Args:
        grammar: The grammar
        strategy: Recovery strategy to use
    
    Returns:
        Dict with recovery tables/code
    """
    recovery: Dict[str, Any] = {"strategy": strategy.name}
    
    if strategy == ErrorRecoveryStrategy.PANIC_MODE:
        # Compute synchronization tokens (typically ; } ) etc.)
        sync_tokens: Set[str] = set()
        for terminal in grammar.terminals:
            if hasattr(terminal, 'name') and terminal.name in {';', '}', ')', ']', 'end', 'EOF', '$'}:
                sync_tokens.add(terminal.name)
        recovery["sync_tokens"] = list(sync_tokens)
        
        # Add FOLLOW sets as potential sync points
        recovery["use_follow_sets"] = True
    
    elif strategy == ErrorRecoveryStrategy.PHRASE_LEVEL:
        # Suggest tokens that could be inserted/deleted
        recovery["insert_candidates"] = [';', ')']
        recovery["delete_on_error"] = True
    
    elif strategy == ErrorRecoveryStrategy.ERROR_PRODUCTIONS:
        # Error productions would be added to the grammar
        recovery["error_symbol"] = "error"
        recovery["needs_grammar_modification"] = True
    
    return recovery


def validate_grammar_for_parsing(grammar: Grammar) -> List[str]:
    """Validate grammar for parsing suitability.
    
    Checks common issues that might affect parser generation.
    
    Args:
        grammar: The grammar to validate
    
    Returns:
        List of issue descriptions
    """
    issues = []
    
    # Check for empty grammar
    if grammar.production_count() == 0:
        issues.append("Grammar has no productions")
        return issues
    
    # Check start symbol has productions
    start_prods = grammar.get_productions(grammar.start_symbol)
    if not start_prods:
        issues.append(f"Start symbol '{grammar.start_symbol.name}' has no productions")
    
    # Check for unreachable nonterminals
    reachable = {grammar.start_symbol}
    worklist = [grammar.start_symbol]
    
    while worklist:
        nt = worklist.pop()
        for prod in grammar.get_productions(nt):
            for sym in prod.body:
                if hasattr(sym, 'is_nonterminal') and sym.is_nonterminal():
                    if sym not in reachable:
                        reachable.add(sym)
                        worklist.append(sym)
    
    unreachable = grammar.nonterminals - reachable
    if unreachable:
        names = [nt.name for nt in unreachable]
        issues.append(f"Unreachable nonterminals: {', '.join(names)}")
    
    # Check for nonproductive nonterminals (no way to derive terminals)
    productive = set()
    changed = True
    
    while changed:
        changed = False
        for nt in grammar.nonterminals:
            if nt in productive:
                continue
            
            for prod in grammar.get_productions(nt):
                all_productive = True
                for sym in prod.body:
                    if hasattr(sym, 'is_nonterminal') and sym.is_nonterminal():
                        if sym not in productive:
                            all_productive = False
                            break
                
                if all_productive or len(prod.body) == 0:
                    productive.add(nt)
                    changed = True
                    break
    
    nonproductive = grammar.nonterminals - productive
    if nonproductive:
        names = [nt.name for nt in nonproductive]
        issues.append(f"Non-productive nonterminals: {', '.join(names)}")
    
    return issues
