"""Clingo ASP Emitter.

This module implements the Clingo emitter for Answer Set Programming,
following the ASP-Core-2 standard and Clingo-specific extensions.

Part of Phase 7D: Answer Set Programming

Clingo syntax reference: https://potassco.org/clingo/
"""

import json
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

from ir.asp import (
    ASPProgram, Rule, Atom, Term, Literal,
    Aggregate, AggregateElement, BodyElement, HeadElement, ChoiceElement,
    ShowStatement, OptimizeStatement, ConstantDef, Comparison, Guard,
    RuleType, AggregateFunction, ComparisonOp, NegationType,
    DEFAULT_PRIORITY, DEFAULT_WEIGHT,
)


@dataclass
class EmitterResult:
    """Result of emission operation."""
    code: str                    # Generated code
    manifest: Dict[str, Any]     # Build manifest
    warnings: List[str] = field(default_factory=list)  # Non-fatal issues


@dataclass
class ClingoConfig:
    """Configuration for Clingo emitter."""
    include_comments: bool = True
    pretty_print: bool = True
    include_header: bool = True
    include_metadata: bool = True
    line_width: int = 80


class ClingoEmitter:
    """Emitter for Clingo ASP syntax.
    
    Generates Clingo-compatible ASP programs following the ASP-Core-2
    standard with Clingo-specific extensions like choice rules,
    aggregates, and optimization statements.
    """
    
    DIALECT = "clingo"
    VERSION = "1.0.0"
    
    def __init__(self, config: ClingoConfig = None):
        """Initialize emitter with configuration."""
        self.config = config or ClingoConfig()
        self._warnings: List[str] = []
    
    def emit(self, program: ASPProgram) -> EmitterResult:
        """Emit ASP program to Clingo format.
        
        Args:
            program: ASP program to emit
            
        Returns:
            EmitterResult with code, manifest, and warnings
        """
        self._warnings = []
        
        lines = []
        
        # Header
        if self.config.include_header:
            lines.extend(self._emit_header(program))
        
        # Constants
        if program.constants:
            lines.append("% Constants")
            for const in program.constants:
                lines.append(self._emit_constant(const))
            lines.append("")
        
        # User comments
        if self.config.include_comments and program.comments:
            lines.append("% Comments")
            for comment in program.comments:
                lines.append(f"% {comment}")
            lines.append("")
        
        # Facts
        facts = program.get_facts()
        if facts:
            lines.append("% Facts")
            for rule in facts:
                lines.append(self.emit_rule(rule))
            lines.append("")
        
        # Normal rules
        normal_rules = [r for r in program.rules if r.rule_type == RuleType.NORMAL and not r.is_fact]
        if normal_rules:
            lines.append("% Rules")
            for rule in normal_rules:
                lines.append(self.emit_rule(rule))
            lines.append("")
        
        # Choice rules
        choice_rules = program.get_choice_rules()
        if choice_rules:
            lines.append("% Choice Rules")
            for rule in choice_rules:
                lines.append(self.emit_rule(rule))
            lines.append("")
        
        # Disjunctive rules
        disjunctive_rules = [r for r in program.rules if r.is_disjunctive]
        if disjunctive_rules:
            lines.append("% Disjunctive Rules")
            for rule in disjunctive_rules:
                lines.append(self.emit_rule(rule))
            lines.append("")
        
        # Constraints
        constraints = program.get_constraints()
        if constraints:
            lines.append("% Constraints")
            for rule in constraints:
                lines.append(self.emit_rule(rule))
            lines.append("")
        
        # Weak constraints
        weak_constraints = [r for r in program.rules if r.is_weak]
        if weak_constraints:
            lines.append("% Weak Constraints")
            for rule in weak_constraints:
                lines.append(self.emit_rule(rule))
            lines.append("")
        
        # Optimization statements
        if program.optimize_statements:
            lines.append("% Optimization")
            for stmt in program.optimize_statements:
                lines.append(self._emit_optimize(stmt))
            lines.append("")
        
        # Show statements
        if program.show_statements:
            lines.append("% Output")
            for stmt in program.show_statements:
                lines.append(self._emit_show(stmt))
            lines.append("")
        
        code = "\n".join(lines)
        manifest = self._generate_manifest(program, code)
        
        return EmitterResult(code=code, manifest=manifest, warnings=self._warnings)
    
    def emit_rule(self, rule: Rule) -> str:
        """Emit a single rule.
        
        Args:
            rule: Rule to emit
            
        Returns:
            Clingo rule string
        """
        if rule.rule_type == RuleType.NORMAL:
            return self._emit_normal_rule(rule)
        elif rule.rule_type == RuleType.CHOICE:
            return self._emit_choice_rule(rule)
        elif rule.rule_type == RuleType.CONSTRAINT:
            return self._emit_constraint(rule)
        elif rule.rule_type == RuleType.DISJUNCTIVE:
            return self._emit_disjunctive_rule(rule)
        elif rule.rule_type == RuleType.WEAK:
            return self._emit_weak_constraint(rule)
        else:
            self._warnings.append(f"Unknown rule type: {rule.rule_type}")
            return f"% Unknown rule type: {rule.rule_type}"
    
    def emit_atom(self, atom: Atom) -> str:
        """Emit an atom.
        
        Args:
            atom: Atom to emit
            
        Returns:
            Clingo atom string
        """
        if not atom.terms:
            return atom.predicate
        terms_str = ", ".join(self.emit_term(t) for t in atom.terms)
        return f"{atom.predicate}({terms_str})"
    
    def emit_term(self, term: Term) -> str:
        """Emit a term.
        
        Args:
            term: Term to emit
            
        Returns:
            Clingo term string
        """
        if term.args is None:
            return term.name
        args_str = ", ".join(self.emit_term(a) for a in term.args)
        return f"{term.name}({args_str})"
    
    def emit_literal(self, literal: Literal) -> str:
        """Emit a literal.
        
        Args:
            literal: Literal to emit
            
        Returns:
            Clingo literal string
        """
        atom_str = self.emit_atom(literal.atom)
        if literal.negation == NegationType.NONE:
            return atom_str
        elif literal.negation == NegationType.DEFAULT:
            return f"not {atom_str}"
        else:  # CLASSICAL
            return f"-{atom_str}"
    
    def emit_aggregate(self, aggregate: Aggregate) -> str:
        """Emit an aggregate expression.
        
        Args:
            aggregate: Aggregate to emit
            
        Returns:
            Clingo aggregate string
        """
        func_str = aggregate.function.to_clingo()
        elements_str = "; ".join(self._emit_aggregate_element(e) for e in aggregate.elements)
        agg_str = f"{func_str} {{ {elements_str} }}"
        
        parts = []
        if aggregate.left_guard:
            parts.append(f"{self.emit_term(aggregate.left_guard.term)} {aggregate.left_guard.op.value}")
        parts.append(agg_str)
        if aggregate.right_guard:
            parts.append(f"{aggregate.right_guard.op.value} {self.emit_term(aggregate.right_guard.term)}")
        
        return " ".join(parts)
    
    def emit_body(self, body: List[BodyElement]) -> str:
        """Emit a rule body.
        
        Args:
            body: List of body elements
            
        Returns:
            Clingo body string
        """
        parts = []
        for elem in body:
            parts.append(self._emit_body_element(elem))
        return ", ".join(parts)
    
    # Private emission methods
    
    def _emit_header(self, program: ASPProgram) -> List[str]:
        """Emit program header."""
        lines = [
            f"% ASP Program: {program.name}",
            f"% Generated by STUNIR Clingo Emitter v{self.VERSION}",
            f"% Dialect: {self.DIALECT}",
        ]
        if self.config.include_metadata and program.metadata:
            lines.append(f"% Metadata: {json.dumps(program.metadata)}")
        lines.append("")
        return lines
    
    def _emit_constant(self, const: ConstantDef) -> str:
        """Emit a constant definition."""
        if isinstance(const.value, Term):
            value_str = self.emit_term(const.value)
        else:
            value_str = str(const.value)
        return f"#const {const.name} = {value_str}."
    
    def _emit_normal_rule(self, rule: Rule) -> str:
        """Emit a normal rule."""
        head_str = "; ".join(self.emit_atom(e.atom) for e in rule.head)
        if not rule.body:
            return f"{head_str}."
        body_str = self.emit_body(rule.body)
        return f"{head_str} :- {body_str}."
    
    def _emit_choice_rule(self, rule: Rule) -> str:
        """Emit a choice rule."""
        # Build choice elements
        if rule.choice_elements:
            elements = []
            for elem in rule.choice_elements:
                elements.append(self._emit_choice_element(elem))
            elements_str = "; ".join(elements)
        else:
            elements_str = "; ".join(self.emit_atom(e.atom) for e in rule.head)
        
        # Build choice head with bounds
        if rule.choice_lower is not None and rule.choice_upper is not None:
            head_str = f"{rule.choice_lower} {{ {elements_str} }} {rule.choice_upper}"
        elif rule.choice_lower is not None:
            head_str = f"{rule.choice_lower} {{ {elements_str} }}"
        elif rule.choice_upper is not None:
            head_str = f"{{ {elements_str} }} {rule.choice_upper}"
        else:
            head_str = f"{{ {elements_str} }}"
        
        if not rule.body:
            return f"{head_str}."
        body_str = self.emit_body(rule.body)
        return f"{head_str} :- {body_str}."
    
    def _emit_choice_element(self, elem: ChoiceElement) -> str:
        """Emit a choice element."""
        atom_str = self.emit_atom(elem.atom)
        if elem.condition:
            cond_str = ", ".join(self.emit_literal(lit) for lit in elem.condition)
            return f"{atom_str} : {cond_str}"
        return atom_str
    
    def _emit_constraint(self, rule: Rule) -> str:
        """Emit a constraint (integrity constraint)."""
        body_str = self.emit_body(rule.body)
        return f":- {body_str}."
    
    def _emit_disjunctive_rule(self, rule: Rule) -> str:
        """Emit a disjunctive rule."""
        head_str = " | ".join(self.emit_atom(e.atom) for e in rule.head)
        if not rule.body:
            return f"{head_str}."
        body_str = self.emit_body(rule.body)
        return f"{head_str} :- {body_str}."
    
    def _emit_weak_constraint(self, rule: Rule) -> str:
        """Emit a weak constraint."""
        body_str = self.emit_body(rule.body)
        weight = rule.weight if rule.weight is not None else DEFAULT_WEIGHT
        priority = rule.priority if rule.priority is not None else DEFAULT_PRIORITY
        
        if rule.terms:
            terms_str = ", ".join(self.emit_term(t) for t in rule.terms)
            return f":~ {body_str}. [{weight}@{priority}, {terms_str}]"
        return f":~ {body_str}. [{weight}@{priority}]"
    
    def _emit_body_element(self, elem: BodyElement) -> str:
        """Emit a body element."""
        if elem.literal:
            return self.emit_literal(elem.literal)
        elif elem.aggregate:
            return self.emit_aggregate(elem.aggregate)
        elif elem.comparison:
            return self._emit_comparison(elem.comparison)
        return ""
    
    def _emit_comparison(self, comp: Comparison) -> str:
        """Emit a comparison."""
        return f"{self.emit_term(comp.left)} {comp.op.value} {self.emit_term(comp.right)}"
    
    def _emit_aggregate_element(self, elem: AggregateElement) -> str:
        """Emit an aggregate element."""
        terms_str = ", ".join(self.emit_term(t) for t in elem.terms)
        if elem.condition:
            cond_str = ", ".join(self.emit_literal(lit) for lit in elem.condition)
            return f"{terms_str} : {cond_str}"
        return terms_str
    
    def _emit_optimize(self, stmt: OptimizeStatement) -> str:
        """Emit an optimization statement."""
        directive = "#minimize" if stmt.minimize else "#maximize"
        elements = []
        for elem in stmt.elements:
            terms_str = ", ".join(self.emit_term(t) for t in elem.terms)
            if elem.condition:
                cond_str = ", ".join(self.emit_literal(lit) for lit in elem.condition)
                elements.append(f"{terms_str}@{stmt.priority} : {cond_str}")
            else:
                elements.append(f"{terms_str}@{stmt.priority}")
        elements_str = "; ".join(elements)
        return f"{directive} {{ {elements_str} }}."
    
    def _emit_show(self, stmt: ShowStatement) -> str:
        """Emit a show statement."""
        sign = "" if stmt.positive else "-"
        return f"#show {sign}{stmt.predicate}/{stmt.arity}."
    
    def _generate_manifest(self, program: ASPProgram, code: str) -> Dict[str, Any]:
        """Generate build manifest."""
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        return {
            "schema": "stunir.asp.clingo.v1",
            "program_name": program.name,
            "dialect": self.DIALECT,
            "emitter_version": self.VERSION,
            "rules_count": len(program.rules),
            "facts_count": len(program.get_facts()),
            "constraints_count": len(program.get_constraints()),
            "predicates": program.get_predicates(),
            "code_hash": code_hash,
            "code_size": len(code),
        }


def emit_clingo(program: ASPProgram, config: ClingoConfig = None) -> EmitterResult:
    """Convenience function to emit Clingo code.
    
    Args:
        program: ASP program to emit
        config: Optional emitter configuration
        
    Returns:
        EmitterResult with code, manifest, and warnings
    """
    emitter = ClingoEmitter(config)
    return emitter.emit(program)


def emit_clingo_to_file(program: ASPProgram, path: str, config: ClingoConfig = None) -> EmitterResult:
    """Emit Clingo code to file.
    
    Args:
        program: ASP program to emit
        path: Output file path
        config: Optional emitter configuration
        
    Returns:
        EmitterResult with code, manifest, and warnings
    """
    result = emit_clingo(program, config)
    Path(path).write_text(result.code)
    return result
