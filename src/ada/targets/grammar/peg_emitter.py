#!/usr/bin/env python3
"""PEG (Parsing Expression Grammar) emitter.

Emits grammars in PEG notation:
- Ordered choice: e1 / e2 (not |)
- Optional: e?
- Zero or more: e*
- One or more: e+
- And-predicate: &e
- Not-predicate: !e
- Grouping: (e)
"""

from typing import Dict, List, Optional, Any

from ir.grammar.grammar_ir import Grammar, GrammarType, EmitterResult
from ir.grammar.production import ProductionRule, BodyElement
from ir.grammar.production import OptionalOp, Repetition, OneOrMore, Group, Alternation
from ir.grammar.symbol import Symbol
from targets.grammar.base import GrammarEmitterBase
from ir.grammar.validation import detect_left_recursion


class PEGEmitter(GrammarEmitterBase):
    """Emitter for PEG (Parsing Expression Grammar) format.
    
    PEG format:
        expr <- term (('+' / '-') term)*
        term <- factor (('*' / '/') factor)*
        factor <- 'num' / '(' expr ')'
    
    Config options:
        arrow_style (str): "<-" or "<-" (default: "<-")
        check_left_recursion (bool): Warn about left recursion (default: True)
    """
    
    FORMAT = "peg"
    FILE_EXTENSION = ".peg"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.arrow_style = self.config.get('arrow_style', '<-')
        self.check_left_recursion = self.config.get('check_left_recursion', True)
    
    def emit(self, grammar: Grammar) -> EmitterResult:
        """Emit grammar in PEG format.
        
        Args:
            grammar: The grammar to emit
        
        Returns:
            EmitterResult with PEG code and manifest
        """
        self._warnings = []
        lines = []
        
        # Check for left recursion (invalid in PEG)
        if self.check_left_recursion:
            lr = detect_left_recursion(grammar)
            if lr:
                for nt, cycle in lr:
                    cycle_str = " -> ".join(s.name for s in cycle)
                    self._add_warning(f"Left recursion detected (invalid for PEG): {cycle_str}")
        
        # Add header
        lines.append(self._emit_header(grammar))
        
        # Emit productions grouped by non-terminal
        for nt in sorted(grammar.nonterminals, key=lambda s: s.name):
            prods = grammar.get_productions(nt)
            if not prods:
                continue
            
            # Combine alternatives with ordered choice (/)
            alternatives = [self._emit_body(p) for p in prods]
            body = " / ".join(alternatives)
            
            lines.append(f"{nt.name} {self.arrow_style} {body}")
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
        """Emit a single production rule in PEG format.
        
        Args:
            rule: The production rule to emit
        
        Returns:
            PEG production string
        """
        head = rule.head.name
        body = self._emit_body(rule)
        return f"{head} {self.arrow_style} {body}"
    
    def _emit_body(self, rule: ProductionRule) -> str:
        """Emit the body of a production rule."""
        if rule.is_epsilon_production():
            return self._get_epsilon_repr()
        
        parts = []
        for elem in rule.body:
            parts.append(self._emit_peg_element(elem))
        
        return " ".join(parts)
    
    def _emit_peg_element(self, element: BodyElement) -> str:
        """Emit a body element with PEG notation."""
        if isinstance(element, Symbol):
            if element.is_nonterminal():
                return element.name
            elif element.is_terminal():
                # PEG uses single quotes for literals
                return f"'{element.name}'"
            elif element.is_epsilon():
                return self._get_epsilon_repr()
            else:
                return str(element)
        elif isinstance(element, OptionalOp):
            inner = self._emit_peg_element(element.element)
            return f"{inner}?"
        elif isinstance(element, Repetition):
            inner = self._emit_peg_element(element.element)
            return f"{inner}*"
        elif isinstance(element, OneOrMore):
            inner = self._emit_peg_element(element.element)
            return f"{inner}+"
        elif isinstance(element, Group):
            inner = " ".join(self._emit_peg_element(e) for e in element.elements)
            return f"({inner})"
        elif isinstance(element, Alternation):
            parts = []
            for alt in element.alternatives:
                if isinstance(alt, tuple):
                    parts.append(" ".join(self._emit_peg_element(e) for e in alt))
                else:
                    parts.append(self._emit_peg_element(alt))
            return f"({' / '.join(parts)})"
        else:
            return str(element)
    
    def _get_comment_char(self) -> str:
        return "#"
    
    def _emit_header(self, grammar: Grammar) -> str:
        """Generate PEG-style header comment."""
        import datetime
        
        lines = [
            f"# Grammar: {grammar.name}",
            f"# Type: {grammar.grammar_type.name}",
            f"# Format: {self.FORMAT}",
            f"# Generated: {datetime.datetime.now().isoformat()}",
            f"# Generator: STUNIR Grammar Emitter",
            "",
        ]
        return "\n".join(lines)
    
    def _get_epsilon_repr(self) -> str:
        # PEG uses empty string (epsilon)
        return "''"
