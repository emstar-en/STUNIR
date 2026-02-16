#!/usr/bin/env python3
"""BNF (Backus-Naur Form) grammar emitter.

Emits grammars in standard BNF notation:
- Non-terminals are wrapped in angle brackets: <expr>
- Productions use ::= as the definition operator
- Alternatives are separated by |
"""

from typing import Dict, List, Optional, Any

from ir.grammar.grammar_ir import Grammar, EmitterResult
from ir.grammar.production import ProductionRule
from ir.grammar.symbol import Symbol
from targets.grammar.base import GrammarEmitterBase


class BNFEmitter(GrammarEmitterBase):
    """Emitter for BNF (Backus-Naur Form) grammar format.
    
    BNF format:
        <expr> ::= <term> | <expr> "+" <term>
        <term> ::= <factor> | <term> "*" <factor>
        <factor> ::= "num" | "(" <expr> ")"
    
    Config options:
        wrap_terminals (bool): Wrap terminals in quotes (default: True)
        compact (bool): Use compact format with | on same line (default: True)
    """
    
    FORMAT = "bnf"
    FILE_EXTENSION = ".bnf"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.wrap_terminals = self.config.get('wrap_terminals', True)
        self.compact = self.config.get('compact', True)
    
    def emit(self, grammar: Grammar) -> EmitterResult:
        """Emit grammar in BNF format.
        
        Args:
            grammar: The grammar to emit
        
        Returns:
            EmitterResult with BNF code and manifest
        """
        self._warnings = []
        lines = []
        
        # Add header
        lines.append(self._emit_header(grammar))
        
        # Group productions by non-terminal
        if self.compact:
            # Emit compact form with | on same line
            for nt in sorted(grammar.nonterminals, key=lambda s: s.name):
                prods = grammar.get_productions(nt)
                if not prods:
                    continue
                
                nt_str = self._emit_nonterminal(nt)
                alternatives = [self._emit_body(p) for p in prods]
                
                if len(alternatives) == 1:
                    lines.append(f"{nt_str} ::= {alternatives[0]}")
                else:
                    # Multi-line alternatives
                    lines.append(f"{nt_str} ::= {alternatives[0]}")
                    for alt in alternatives[1:]:
                        lines.append(f"       | {alt}")
                lines.append("")
        else:
            # Emit one production per line
            for prod in grammar.all_productions():
                lines.append(self.emit_production(prod))
        
        code = "\n".join(lines)
        manifest = self._generate_manifest(grammar, code)
        
        return EmitterResult(
            code=code,
            manifest=manifest,
            format=self.FORMAT,
            warnings=self._warnings
        )
    
    def emit_production(self, rule: ProductionRule) -> str:
        """Emit a single production rule in BNF format.
        
        Args:
            rule: The production rule to emit
        
        Returns:
            BNF production string
        """
        head = self._emit_nonterminal(rule.head)
        body = self._emit_body(rule)
        return f"{head} ::= {body}"
    
    def _emit_nonterminal(self, sym: Symbol) -> str:
        """Emit a non-terminal with angle brackets."""
        return f"<{sym.name}>"
    
    def _emit_terminal(self, sym: Symbol) -> str:
        """Emit a terminal, optionally wrapped in quotes."""
        if self.wrap_terminals:
            return f'"{sym.name}"'
        return sym.name
    
    def _emit_body(self, rule: ProductionRule) -> str:
        """Emit the body of a production rule."""
        if rule.is_epsilon_production():
            return self._get_epsilon_repr()
        
        parts = []
        for elem in rule.body:
            if isinstance(elem, Symbol):
                if elem.is_nonterminal():
                    parts.append(self._emit_nonterminal(elem))
                elif elem.is_terminal():
                    parts.append(self._emit_terminal(elem))
                elif elem.is_epsilon():
                    parts.append(self._get_epsilon_repr())
                else:
                    parts.append(str(elem))
            else:
                # EBNF operator - warn and emit as string
                self._add_warning(f"EBNF operator in BNF output: {elem}")
                parts.append(self._emit_element(elem))
        
        return " ".join(parts)
    
    def _get_comment_char(self) -> str:
        return ";"
    
    def _get_epsilon_repr(self) -> str:
        return "ε"
