#!/usr/bin/env python3
"""Core Grammar IR classes.

This module defines:
- GrammarType: Enumeration of supported grammar types (BNF, EBNF, PEG)
- Grammar: Complete grammar representation
- ValidationResult: Result of grammar validation
- EmitterResult: Result of grammar emission
- BaseGrammarEmitter: Abstract base class for grammar emitters
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Set, Optional, Any, Tuple

from ir.grammar.symbol import Symbol, SymbolType, EPSILON, EOF
from ir.grammar.production import ProductionRule, BodyElement


class GrammarType(Enum):
    """Types of supported grammars."""
    BNF = auto()             # Backus-Naur Form
    EBNF = auto()            # Extended BNF (with ?, *, +, grouping)
    PEG = auto()             # Parsing Expression Grammar (ordered choice)


@dataclass
class ValidationResult:
    """Result of grammar validation.
    
    Attributes:
        valid: True if the grammar is valid (no errors)
        errors: List of error messages
        warnings: List of warning messages
        info: Additional information (e.g., computed sets)
    """
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)
    
    def __bool__(self) -> bool:
        """Allow using result as boolean (True if valid)."""
        return self.valid


@dataclass
class EmitterResult:
    """Result of grammar emission.
    
    Attributes:
        code: Generated grammar text
        manifest: Build manifest dictionary
        format: Output format name (e.g., 'bnf', 'antlr')
        warnings: List of warning messages
    """
    code: str
    manifest: Dict[str, Any]
    format: str
    warnings: List[str] = field(default_factory=list)


@dataclass
class Grammar:
    """Complete grammar representation.
    
    A Grammar object represents a context-free grammar with:
    - A name for identification
    - A grammar type (BNF, EBNF, or PEG)
    - A start symbol
    - A set of production rules
    - Metadata for additional information
    
    The grammar maintains sets of terminals and non-terminals, which are
    automatically updated when productions are added.
    
    Attributes:
        name: Grammar name
        grammar_type: Type of grammar (BNF, EBNF, PEG)
        start_symbol: The start non-terminal
        productions: Dict mapping non-terminals to their productions
        terminals: Set of terminal symbols
        nonterminals: Set of non-terminal symbols
        metadata: Additional grammar metadata
    
    Example:
        >>> E = Symbol("E", SymbolType.NONTERMINAL)
        >>> T = Symbol("T", SymbolType.NONTERMINAL)
        >>> num = Symbol("num", SymbolType.TERMINAL)
        >>> plus = Symbol("+", SymbolType.TERMINAL)
        
        >>> grammar = Grammar("expr", GrammarType.BNF, E)
        >>> grammar.add_production(ProductionRule(E, (E, plus, T)))
        >>> grammar.add_production(ProductionRule(E, (T,)))
        >>> grammar.add_production(ProductionRule(T, (num,)))
    """
    name: str
    grammar_type: GrammarType
    start_symbol: Symbol
    productions: Dict[Symbol, List[ProductionRule]] = field(default_factory=dict)
    terminals: Set[Symbol] = field(default_factory=set)
    nonterminals: Set[Symbol] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Computed caches (populated during validation)
    _first_sets: Optional[Dict[Symbol, Set[Symbol]]] = field(default=None, repr=False)
    _follow_sets: Optional[Dict[Symbol, Set[Symbol]]] = field(default=None, repr=False)
    _nullable: Optional[Set[Symbol]] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Initialize grammar after dataclass construction."""
        # Ensure start symbol is a non-terminal
        if not self.start_symbol.is_nonterminal():
            raise ValueError(f"Start symbol must be a non-terminal: {self.start_symbol}")
        
        # Add start symbol to non-terminals
        self.nonterminals.add(self.start_symbol)
    
    def add_production(self, rule: ProductionRule) -> None:
        """Add a production rule to the grammar.
        
        Args:
            rule: The production rule to add
        
        Raises:
            ValueError: If rule head is not a non-terminal
        """
        # Validate rule
        if not rule.head.is_nonterminal():
            raise ValueError(f"Production head must be a non-terminal: {rule.head}")
        
        # Add to productions dict
        if rule.head not in self.productions:
            self.productions[rule.head] = []
        self.productions[rule.head].append(rule)
        
        # Update symbol sets
        self.nonterminals.add(rule.head)
        
        for sym in rule.body_symbols():
            if sym.is_terminal():
                self.terminals.add(sym)
            elif sym.is_nonterminal():
                self.nonterminals.add(sym)
        
        # Invalidate caches
        self._first_sets = None
        self._follow_sets = None
        self._nullable = None
    
    def add_productions(self, rules: List[ProductionRule]) -> None:
        """Add multiple production rules to the grammar.
        
        Args:
            rules: List of production rules to add
        """
        for rule in rules:
            self.add_production(rule)
    
    def get_productions(self, nonterminal: Symbol) -> List[ProductionRule]:
        """Get all productions for a non-terminal.
        
        Args:
            nonterminal: The non-terminal symbol
        
        Returns:
            List of production rules (empty if none)
        """
        return self.productions.get(nonterminal, [])
    
    def all_productions(self) -> List[ProductionRule]:
        """Get all production rules in the grammar.
        
        Returns:
            List of all production rules
        """
        result = []
        for prods in self.productions.values():
            result.extend(prods)
        return result
    
    def all_symbols(self) -> Set[Symbol]:
        """Get all symbols (terminals and non-terminals) in the grammar.
        
        Returns:
            Set of all symbols
        """
        return self.terminals | self.nonterminals
    
    def production_count(self) -> int:
        """Get the total number of production rules.
        
        Returns:
            Number of productions
        """
        return sum(len(prods) for prods in self.productions.values())
    
    def copy(self) -> 'Grammar':
        """Create a deep copy of the grammar.
        
        Returns:
            New Grammar with copied data
        """
        new_grammar = Grammar(
            name=self.name,
            grammar_type=self.grammar_type,
            start_symbol=self.start_symbol,
            metadata=self.metadata.copy()
        )
        
        for rule in self.all_productions():
            new_grammar.add_production(ProductionRule(
                head=rule.head,
                body=rule.body,
                label=rule.label,
                action=rule.action,
                precedence=rule.precedence
            ))
        
        return new_grammar
    
    def __str__(self) -> str:
        """Return string representation of the grammar."""
        lines = [f"Grammar: {self.name} ({self.grammar_type.name})"]
        lines.append(f"Start: {self.start_symbol.name}")
        lines.append("")
        
        for nt in sorted(self.nonterminals, key=lambda s: s.name):
            prods = self.get_productions(nt)
            for prod in prods:
                lines.append(str(prod))
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        return (f"Grammar(name={self.name!r}, type={self.grammar_type.name}, "
                f"productions={self.production_count()}, "
                f"terminals={len(self.terminals)}, "
                f"nonterminals={len(self.nonterminals)})")


class BaseGrammarEmitter(ABC):
    """Abstract base class for grammar emitters.
    
    Subclasses must implement:
    - emit(): Main emission method
    - emit_production(): Emit a single production rule
    
    Attributes:
        FORMAT: The output format name (override in subclass)
        FILE_EXTENSION: The typical file extension (override in subclass)
    """
    
    FORMAT: str = "base"
    FILE_EXTENSION: str = ".grammar"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the emitter.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._warnings: List[str] = []
    
    @abstractmethod
    def emit(self, grammar: Grammar) -> EmitterResult:
        """Emit grammar in target format.
        
        Args:
            grammar: The grammar to emit
        
        Returns:
            EmitterResult with generated code and manifest
        """
        pass
    
    @abstractmethod
    def emit_production(self, rule: ProductionRule) -> str:
        """Emit a single production rule.
        
        Args:
            rule: The production rule to emit
        
        Returns:
            String representation in target format
        """
        pass
    
    def _add_warning(self, message: str) -> None:
        """Add a warning message.
        
        Args:
            message: The warning message
        """
        self._warnings.append(message)
    
    def _generate_manifest(self, grammar: Grammar, code: str) -> Dict[str, Any]:
        """Generate deterministic manifest for the emitted grammar.
        
        Args:
            grammar: The source grammar
            code: The generated code
        
        Returns:
            Manifest dictionary
        """
        import hashlib
        import time
        import json
        
        def compute_sha256(data: bytes) -> str:
            return hashlib.sha256(data).hexdigest()
        
        def canonical_json(obj: Any) -> str:
            return json.dumps(obj, sort_keys=True, separators=(',', ':'))
        
        manifest = {
            "schema": f"stunir.grammar.{self.FORMAT}.v1",
            "generator": f"stunir.grammar.{self.FORMAT}_emitter",
            "epoch": int(time.time()),
            "grammar_name": grammar.name,
            "grammar_type": grammar.grammar_type.name,
            "output_format": self.FORMAT,
            "production_count": grammar.production_count(),
            "terminal_count": len(grammar.terminals),
            "nonterminal_count": len(grammar.nonterminals),
            "output_hash": compute_sha256(code.encode('utf-8')),
            "output_size": len(code),
        }
        
        # Compute manifest hash
        manifest_for_hash = {k: v for k, v in manifest.items() if k != "manifest_hash"}
        manifest["manifest_hash"] = compute_sha256(
            canonical_json(manifest_for_hash).encode('utf-8')
        )
        
        return manifest
    
    def _emit_header(self, grammar: Grammar) -> str:
        """Generate header comment for the output.
        
        Args:
            grammar: The source grammar
        
        Returns:
            Header comment string
        """
        import datetime
        
        lines = [
            f";;; Grammar: {grammar.name}",
            f";;; Type: {grammar.grammar_type.name}",
            f";;; Format: {self.FORMAT}",
            f";;; Generated: {datetime.datetime.now().isoformat()}",
            f";;; Generator: STUNIR Grammar Emitter",
            "",
        ]
        return "\n".join(lines)
