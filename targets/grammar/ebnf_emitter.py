#!/usr/bin/env python3
"""EBNF (Extended Backus-Naur Form) grammar emitter.

Emits grammars in EBNF notation with extensions:
- Optional: [x] or x?
- Repetition: {x} or x*
- One or more: x+
- Grouping: (x y z)
- Alternation: x | y | z
"""

from typing import Dict, List, Optional, Any

from ir.grammar.grammar_ir import Grammar, EmitterResult
from ir.grammar.production import ProductionRule, BodyElement
from ir.grammar.production import OptionalOp, Repetition, OneOrMore, Group, Alternation
from ir.grammar.symbol import Symbol
from targets.grammar.base import GrammarEmitterBase


class EBNFEmitter(GrammarEmitterBase):
    """Emitter for EBNF (Extended Backus-Naur Form) grammar format.
    
    EBNF format:
        expr = term { ("+" | "-") term } ;
        term = factor { ("*" | "/") factor } ;
        factor = "num" | "(" expr ")" ;
    
    Config options:
        iso_style (bool): Use ISO EBNF style with = and ; (default: True)
        wrap_terminals (bool): Wrap terminals in quotes (default: True)
    """
    
    FORMAT = "ebnf"
    FILE_EXTENSION = ".ebnf"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.iso_style = self.config.get('iso_style', True)
        self.wrap_terminals = self.config.get('wrap_terminals', True)
    
    def emit(self, grammar: Grammar) -> EmitterResult:
        """Emit grammar in EBNF format.
        
        Args:
            grammar: The grammar to emit
        
        Returns:
            EmitterResult with EBNF code and manifest
        """
        self._warnings = []
        lines = []
        
        # Add header
        lines.append(self._emit_header(grammar))
        
        # Emit productions grouped by non-terminal
        for nt in sorted(grammar.nonterminals, key=lambda s: s.name):
            prods = grammar.get_productions(nt)
            if not prods:
                continue
            
            # Combine alternatives
            alternatives = [self._emit_body(p) for p in prods]
            body = " | ".join(alternatives)
            
            if self.iso_style:
                lines.append(f"{nt.name} = {body} ;")
            else:
                lines.append(f"{nt.name} ::= {body}")
            
            lines.append("")
        
        code = "\n".join(lines)
        manifest = self._generate_manifest(grammar, code)
        
        return EmitterResult(
            code=code,
            manifest=manifest,
            format=self.FORMAT,
            warnings=self._warnings
        )
    
    def emit_production(self, rule: ProductionRule) -> str:
        """Emit a single production rule in EBNF format.
        
        Args:
            rule: The production rule to emit
        
        Returns:
            EBNF production string
        """
        head = rule.head.name
        body = self._emit_body(rule)
        
        if self.iso_style:
            return f"{head} = {body} ;"
        else:
            return f"{head} ::= {body}"
    
    def _emit_body(self, rule: ProductionRule) -> str:
        """Emit the body of a production rule."""
        if rule.is_epsilon_production():
            return self._get_epsilon_repr()
        
        parts = []
        for elem in rule.body:
            parts.append(self._emit_ebnf_element(elem))
        
        return " ".join(parts)
    
    def _emit_ebnf_element(self, element: BodyElement) -> str:
        """Emit a body element with EBNF notation."""
        if isinstance(element, Symbol):
            if element.is_nonterminal():
                return element.name
            elif element.is_terminal():
                if self.wrap_terminals:
                    return f'"{element.name}"'
                return element.name
            elif element.is_epsilon():
                return self._get_epsilon_repr()
            else:
                return str(element)
        elif isinstance(element, OptionalOp):
            inner = self._emit_ebnf_element(element.element)
            return f"[ {inner} ]"
        elif isinstance(element, Repetition):
            inner = self._emit_ebnf_element(element.element)
            return f"{{ {inner} }}"
        elif isinstance(element, OneOrMore):
            inner = self._emit_ebnf_element(element.element)
            return f"{inner} {{ {inner} }}"
        elif isinstance(element, Group):
            inner = " ".join(self._emit_ebnf_element(e) for e in element.elements)
            return f"( {inner} )"
        elif isinstance(element, Alternation):
            parts = []
            for alt in element.alternatives:
                if isinstance(alt, tuple):
                    parts.append(" ".join(self._emit_ebnf_element(e) for e in alt))
                else:
                    parts.append(self._emit_ebnf_element(alt))
            return f"( {' | '.join(parts)} )"
        else:
            return str(element)
    
    def _get_comment_char(self) -> str:
        return "(*"
    
    def _emit_header(self, grammar: Grammar) -> str:
        """Generate EBNF-style header comment."""
        import datetime
        
        lines = [
            f"(* Grammar: {grammar.name} *)",
            f"(* Type: {grammar.grammar_type.name} *)",
            f"(* Format: {self.FORMAT} *)",
            f"(* Generated: {datetime.datetime.now().isoformat()} *)",
            f"(* Generator: STUNIR Grammar Emitter *)",
            "",
        ]
        return "\n".join(lines)
    
    def _get_epsilon_repr(self) -> str:
        # ISO EBNF uses empty string or special symbol
        return "(* empty *)"
