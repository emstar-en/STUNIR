"""DLV ASP Emitter.

This module implements the DLV emitter for Answer Set Programming,
supporting disjunctive logic programming with DLV-specific syntax.

Part of Phase 7D: Answer Set Programming

DLV reference: https://www.dlvsystem.com/
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
class DLVConfig:
    """Configuration for DLV emitter."""
    include_comments: bool = True
    pretty_print: bool = True
    include_header: bool = True
    include_metadata: bool = True
    use_v_disjunction: bool = True  # Use 'v' instead of '|' for disjunction
    dlv2_compatible: bool = False   # Enable DLV2 specific features


class DLVEmitter:
    """Emitter for DLV ASP syntax.
    
    Generates DLV-compatible ASP programs supporting disjunctive
    logic programming with weak constraints.
    
    Key differences from Clingo:
    - Disjunction uses 'v' instead of '|'
    - Weak constraints use [weight:priority] instead of [weight@priority]
    - No native choice rule support (converted to disjunctive rules or error)
    - Different aggregate syntax in some cases
    """
    
    DIALECT = "dlv"
    VERSION = "1.0.0"
    
    def __init__(self, config: DLVConfig = None):
        """Initialize emitter with configuration."""
        self.config = config or DLVConfig()
        self._warnings: List[str] = []
    
    def emit(self, program: ASPProgram) -> EmitterResult:
        """Emit ASP program to DLV format.
        
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
        
        # Constants (DLV uses different syntax for some constants)
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
        
        # Choice rules (converted or warned)
        choice_rules = program.get_choice_rules()
        if choice_rules:
            lines.append("% Choice Rules (converted to disjunctive)")
            for rule in choice_rules:
                converted = self._convert_choice_rule(rule)
                if converted:
                    lines.append(converted)
            lines.append("")
        
        # Disjunctive rules (DLV specialty!)
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
        
        # Weak constraints (DLV syntax)
        weak_constraints = [r for r in program.rules if r.is_weak]
        if weak_constraints:
            lines.append("% Weak Constraints")
            for rule in weak_constraints:
                lines.append(self.emit_rule(rule))
            lines.append("")
        
        # Optimization statements (different syntax in DLV)
        if program.optimize_statements:
            lines.append("% Optimization")
            for stmt in program.optimize_statements:
                lines.append(self._emit_optimize(stmt))
            lines.append("")
        
        # Show statements (DLV uses different directive)
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
            DLV rule string
        """
        if rule.rule_type == RuleType.NORMAL:
            return self._emit_normal_rule(rule)
        elif rule.rule_type == RuleType.CHOICE:
            return self._convert_choice_rule(rule) or f"% Choice rule not fully supported: {rule}"
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
            DLV atom string
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
            DLV term string
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
            DLV literal string
        """
        atom_str = self.emit_atom(literal.atom)
        if literal.negation == NegationType.NONE:
            return atom_str
        elif literal.negation == NegationType.DEFAULT:
            return f"not {atom_str}"
        else:  # CLASSICAL (strong negation)
            return f"-{atom_str}"
    
    def emit_aggregate(self, aggregate: Aggregate) -> str:
        """Emit an aggregate expression.
        
        Args:
            aggregate: Aggregate to emit
            
        Returns:
            DLV aggregate string
        """
        func_str = aggregate.function.to_dlv()
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
            DLV body string
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
            f"% Generated by STUNIR DLV Emitter v{self.VERSION}",
            f"% Dialect: {self.DIALECT}",
        ]
        if self.config.include_metadata and program.metadata:
            lines.append(f"% Metadata: {json.dumps(program.metadata)}")
        lines.append("")
        return lines
    
    def _emit_constant(self, const: ConstantDef) -> str:
        """Emit a constant definition.
        
        Note: DLV uses different syntax for constants in some cases.
        """
        if isinstance(const.value, Term):
            value_str = self.emit_term(const.value)
        else:
            value_str = str(const.value)
        # DLV2 uses #const, classic DLV may use different approach
        return f"#const {const.name} = {value_str}."
    
    def _emit_normal_rule(self, rule: Rule) -> str:
        """Emit a normal rule."""
        head_str = "; ".join(self.emit_atom(e.atom) for e in rule.head)
        if not rule.body:
            return f"{head_str}."
        body_str = self.emit_body(rule.body)
        return f"{head_str} :- {body_str}."
    
    def _convert_choice_rule(self, rule: Rule) -> Optional[str]:
        """Convert a choice rule to DLV format.
        
        DLV doesn't natively support choice rules. This attempts to convert
        simple cases to disjunctive rules with constraints.
        """
        # For simple 1-choice rules, convert to disjunctive
        if rule.choice_elements:
            # Build disjunction: p(X) v not_p(X) :- body.
            # Then constraint: :- #count { X : p(X) } > upper.
            # This is a simplification; full conversion is complex
            
            if len(rule.choice_elements) == 1 and rule.choice_lower == 1 and rule.choice_upper == 1:
                # Exactly one: convert to disjunction with auxiliary
                elem = rule.choice_elements[0]
                atom_str = self.emit_atom(elem.atom)
                aux_atom = f"_not_{elem.atom.predicate}"
                
                if elem.condition:
                    cond_str = ", ".join(self.emit_literal(lit) for lit in elem.condition)
                    cond_part = f", {cond_str}"
                else:
                    cond_part = ""
                
                if rule.body:
                    body_str = self.emit_body(rule.body)
                    disjunction = "v" if self.config.use_v_disjunction else "|"
                    return f"{atom_str} {disjunction} {aux_atom}({', '.join(t.name for t in elem.atom.terms)}) :- {body_str}{cond_part}."
                else:
                    disjunction = "v" if self.config.use_v_disjunction else "|"
                    if cond_part:
                        return f"{atom_str} {disjunction} {aux_atom}({', '.join(t.name for t in elem.atom.terms)}) :- {cond_part[2:]}."
                    return f"{atom_str}."
            
            self._warnings.append(
                f"Complex choice rule cannot be directly converted to DLV: "
                f"{len(rule.choice_elements)} elements, bounds [{rule.choice_lower}, {rule.choice_upper}]"
            )
            
            # Emit as comment with approximation
            elements = [self._emit_dlv_choice_element(e) for e in rule.choice_elements]
            elements_str = "; ".join(elements)
            disjunction = "v" if self.config.use_v_disjunction else "|"
            head_str = f" {disjunction} ".join(elements)
            
            if rule.body:
                body_str = self.emit_body(rule.body)
                return f"% Choice rule approximation\n{head_str} :- {body_str}."
            return f"% Choice rule approximation\n{head_str}."
        
        self._warnings.append("Empty choice rule")
        return None
    
    def _emit_dlv_choice_element(self, elem: ChoiceElement) -> str:
        """Emit a choice element for DLV (without choice syntax)."""
        return self.emit_atom(elem.atom)
    
    def _emit_constraint(self, rule: Rule) -> str:
        """Emit a constraint (integrity constraint)."""
        body_str = self.emit_body(rule.body)
        return f":- {body_str}."
    
    def _emit_disjunctive_rule(self, rule: Rule) -> str:
        """Emit a disjunctive rule.
        
        This is where DLV shines - native support for disjunctive rules!
        """
        disjunction = "v" if self.config.use_v_disjunction else "|"
        head_str = f" {disjunction} ".join(self.emit_atom(e.atom) for e in rule.head)
        if not rule.body:
            return f"{head_str}."
        body_str = self.emit_body(rule.body)
        return f"{head_str} :- {body_str}."
    
    def _emit_weak_constraint(self, rule: Rule) -> str:
        """Emit a weak constraint (DLV syntax).
        
        DLV uses [weight:priority] instead of Clingo's [weight@priority].
        """
        body_str = self.emit_body(rule.body)
        weight = rule.weight if rule.weight is not None else DEFAULT_WEIGHT
        priority = rule.priority if rule.priority is not None else DEFAULT_PRIORITY
        
        # DLV syntax: :~ body. [weight:priority]
        # Note: DLV traditionally uses colon, but DLV2 may accept @
        if self.config.dlv2_compatible:
            # DLV2 can use @ like Clingo
            if rule.terms:
                terms_str = ", ".join(self.emit_term(t) for t in rule.terms)
                return f":~ {body_str}. [{weight}@{priority}, {terms_str}]"
            return f":~ {body_str}. [{weight}@{priority}]"
        else:
            # Classic DLV uses colon
            if rule.terms:
                terms_str = ", ".join(self.emit_term(t) for t in rule.terms)
                return f":~ {body_str}. [{weight}:{priority}, {terms_str}]"
            return f":~ {body_str}. [{weight}:{priority}]"
    
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
        """Emit an optimization statement (DLV syntax)."""
        # DLV uses #minimize/#maximize with slightly different syntax
        directive = "#minimize" if stmt.minimize else "#maximize"
        elements = []
        for elem in stmt.elements:
            terms_str = ", ".join(self.emit_term(t) for t in elem.terms)
            if elem.condition:
                cond_str = ", ".join(self.emit_literal(lit) for lit in elem.condition)
                elements.append(f"{terms_str} : {cond_str}")
            else:
                elements.append(terms_str)
        elements_str = "; ".join(elements)
        return f"{directive} {{ {elements_str} }}."
    
    def _emit_show(self, stmt: ShowStatement) -> str:
        """Emit a show statement (DLV syntax)."""
        # DLV uses different syntax for output control
        # Classic DLV uses -filter or specific command-line options
        # DLV2 may support #show
        sign = "" if stmt.positive else "-"
        return f"#show {sign}{stmt.predicate}/{stmt.arity}."
    
    def _generate_manifest(self, program: ASPProgram, code: str) -> Dict[str, Any]:
        """Generate build manifest."""
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        return {
            "schema": "stunir.asp.dlv.v1",
            "program_name": program.name,
            "dialect": self.DIALECT,
            "emitter_version": self.VERSION,
            "dlv2_compatible": self.config.dlv2_compatible,
            "rules_count": len(program.rules),
            "facts_count": len(program.get_facts()),
            "constraints_count": len(program.get_constraints()),
            "disjunctive_rules_count": len([r for r in program.rules if r.is_disjunctive]),
            "predicates": program.get_predicates(),
            "code_hash": code_hash,
            "code_size": len(code),
        }


def emit_dlv(program: ASPProgram, config: DLVConfig = None) -> EmitterResult:
    """Convenience function to emit DLV code.
    
    Args:
        program: ASP program to emit
        config: Optional emitter configuration
        
    Returns:
        EmitterResult with code, manifest, and warnings
    """
    emitter = DLVEmitter(config)
    return emitter.emit(program)


def emit_dlv_to_file(program: ASPProgram, path: str, config: DLVConfig = None) -> EmitterResult:
    """Emit DLV code to file.
    
    Args:
        program: ASP program to emit
        path: Output file path
        config: Optional emitter configuration
        
    Returns:
        EmitterResult with code, manifest, and warnings
    """
    result = emit_dlv(program, config)
    Path(path).write_text(result.code)
    return result
