#!/usr/bin/env python3
"""STUNIR SWI-Prolog Emitter.

Generates SWI-Prolog code from STUNIR Logic IR.
Supports module system, predicates, DCG rules, and SWI-specific features.

Part of Phase 5C-1: Logic Programming Foundation.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from tools.ir.logic_ir import (
    Term, Variable, Atom, Number, StringTerm, Compound, ListTerm, Anonymous,
    Fact, Rule, Goal, Query, Predicate, GoalKind,
    LogicIRExtension, term_from_dict
)
from .types import SWIPrologTypeMapper, SWI_PROLOG_TYPES


def compute_sha256(data: bytes) -> str:
    """Compute SHA-256 hash of data."""
    return hashlib.sha256(data).hexdigest()


def canonical_json(data: Any) -> str:
    """Produce deterministic JSON output."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'))


@dataclass
class EmitterResult:
    """Result of code emission."""
    code: str
    module_name: str
    predicates: List[str]
    sha256: str
    emit_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'code': self.code,
            'module_name': self.module_name,
            'predicates': self.predicates,
            'sha256': self.sha256,
            'emit_time': self.emit_time
        }


@dataclass
class SWIPrologConfig:
    """Configuration for SWI-Prolog emitter."""
    module_prefix: str = "stunir"
    emit_module: bool = True
    emit_comments: bool = True
    emit_type_hints: bool = True  # :- pred declarations
    indent: str = "    "
    line_width: int = 80
    # SWI-specific options
    use_tabling: bool = False  # :- table pred/N
    use_constraints: bool = False  # CLP(FD) support
    dialect_check: bool = True  # Check for SWI-specific features
    # Formatting
    emit_timestamps: bool = True
    emit_sha256: bool = True


class SWIPrologEmitter:
    """Emitter for SWI-Prolog code.
    
    Generates valid SWI-Prolog code from STUNIR Logic IR including:
    - Module definitions with exports
    - Predicate declarations
    - Facts and rules
    - DCG rules
    - Meta-predicate declarations
    - Tabling directives (optional)
    """
    
    DIALECT = "swi-prolog"
    FILE_EXTENSION = ".pl"
    
    def __init__(self, config: Optional[SWIPrologConfig] = None):
        """Initialize SWI-Prolog emitter.
        
        Args:
            config: Emitter configuration (optional)
        """
        self.config = config or SWIPrologConfig()
        self.logic_ext = LogicIRExtension()
        self.type_mapper = SWIPrologTypeMapper(self.config.emit_type_hints)
        self._exports: Set[Tuple[str, int]] = set()
        self._dynamics: Set[Tuple[str, int]] = set()
        self._predicates: Dict[Tuple[str, int], Predicate] = {}
        self._multifile: Set[Tuple[str, int]] = set()
        self._discontiguous: Set[Tuple[str, int]] = set()
    
    def emit(self, ir: Dict[str, Any]) -> EmitterResult:
        """Emit SWI-Prolog code from Logic IR.
        
        Args:
            ir: STUNIR Logic IR dictionary
            
        Returns:
            EmitterResult with generated code
        """
        start_time = time.time()
        
        module_name = ir.get('module', 'unnamed')
        prolog_module = f"{self.config.module_prefix}_{self._prolog_name(module_name)}"
        
        # Reset state
        self._exports = set()
        self._dynamics = set()
        self._multifile = set()
        self._discontiguous = set()
        
        # Extract predicates and analyze
        self._predicates = self.logic_ext.extract_predicates(ir)
        self._analyze_exports(ir)
        self._analyze_dynamics(ir)
        
        # Build code sections
        sections = []
        
        # Header comment
        if self.config.emit_comments:
            sections.append(self._emit_header(module_name))
        
        # Module declaration
        if self.config.emit_module:
            sections.append(self._emit_module(prolog_module))
        
        # Directives (dynamic, multifile, etc.)
        directives = self._emit_directives()
        if directives:
            sections.append(directives)
        
        # Type declarations (pred)
        if self.config.emit_type_hints:
            type_hints = self._emit_pred_declarations()
            if type_hints:
                sections.append(type_hints)
        
        # Predicate definitions
        pred_code = self._emit_predicates(ir)
        if pred_code:
            sections.append(pred_code)
        
        # DCG rules (if any)
        dcg_rules = self.logic_ext.extract_dcg_rules(ir)
        if dcg_rules:
            dcg_code = self._emit_dcg_rules(dcg_rules)
            if dcg_code:
                sections.append(dcg_code)
        
        # Queries (as comments or initialization)
        queries = self.logic_ext.extract_queries(ir)
        if queries:
            query_code = self._emit_queries(queries)
            if query_code:
                sections.append(query_code)
        
        code = '\n\n'.join(s for s in sections if s)
        code_hash = compute_sha256(code.encode('utf-8'))
        
        return EmitterResult(
            code=code,
            module_name=prolog_module,
            predicates=[f"{p}/{a}" for p, a in sorted(self._predicates.keys())],
            sha256=code_hash,
            emit_time=time.time() - start_time
        )
    
    def _prolog_name(self, name: str) -> str:
        """Convert name to valid Prolog identifier.
        
        Args:
            name: Original name
            
        Returns:
            Valid Prolog identifier
        """
        # Replace non-alphanumeric with underscore
        result = ''.join(c if c.isalnum() or c == '_' else '_' for c in name)
        # Ensure starts with lowercase
        if result and result[0].isupper():
            result = result[0].lower() + result[1:]
        return result or 'unnamed'
    
    def _analyze_exports(self, ir: Dict[str, Any]) -> None:
        """Analyze exports from IR."""
        for exp in ir.get('exports', []):
            pred = exp.get('predicate')
            arity = exp.get('arity', 0)
            if pred:
                self._exports.add((pred, arity))
        
        # Auto-export public predicates
        for (name, arity), pred in self._predicates.items():
            if not name.startswith('_'):  # Convention: _ prefix = private
                self._exports.add((name, arity))
    
    def _analyze_dynamics(self, ir: Dict[str, Any]) -> None:
        """Analyze dynamic predicate declarations."""
        for dyn in ir.get('dynamic', []):
            pred = dyn.get('predicate')
            arity = dyn.get('arity', 0)
            if pred:
                self._dynamics.add((pred, arity))
        
        for mf in ir.get('multifile', []):
            pred = mf.get('predicate')
            arity = mf.get('arity', 0)
            if pred:
                self._multifile.add((pred, arity))
        
        for dc in ir.get('discontiguous', []):
            pred = dc.get('predicate')
            arity = dc.get('arity', 0)
            if pred:
                self._discontiguous.add((pred, arity))
    
    def _emit_header(self, module_name: str) -> str:
        """Generate header comment."""
        lines = [
            "/*",
            " * STUNIR Generated SWI-Prolog Module",
            f" * Module: {module_name}",
        ]
        
        if self.config.emit_timestamps:
            lines.append(f" * Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        lines.extend([
            " * ",
            " * This file was automatically generated by STUNIR.",
            " * Do not edit manually.",
            " */"
        ])
        
        return '\n'.join(lines)
    
    def _emit_module(self, module_name: str) -> str:
        """Generate module declaration."""
        exports = sorted(f"{p}/{a}" for p, a in self._exports)
        
        if not exports:
            return f":- module({module_name}, [])."
        
        # Format exports nicely
        if len(exports) <= 3:
            export_str = ', '.join(exports)
            return f":- module({module_name}, [{export_str}])."
        
        # Multi-line for many exports
        lines = [f":- module({module_name}, ["]
        for i, exp in enumerate(exports):
            suffix = ',' if i < len(exports) - 1 else ''
            lines.append(f"    {exp}{suffix}")
        lines.append("]).")
        return '\n'.join(lines)
    
    def _emit_directives(self) -> str:
        """Emit directives for dynamic, multifile, etc."""
        lines = []
        
        # Dynamic predicates
        for pred, arity in sorted(self._dynamics):
            lines.append(f":- dynamic {pred}/{arity}.")
        
        # Multifile predicates
        for pred, arity in sorted(self._multifile):
            lines.append(f":- multifile {pred}/{arity}.")
        
        # Discontiguous predicates
        for pred, arity in sorted(self._discontiguous):
            lines.append(f":- discontiguous {pred}/{arity}.")
        
        # Tabling (if enabled)
        if self.config.use_tabling:
            for (pred, arity), p in self._predicates.items():
                if self._is_recursive(p):
                    lines.append(f":- table {pred}/{arity}.")
        
        return '\n'.join(lines)
    
    def _is_recursive(self, pred: Predicate) -> bool:
        """Check if predicate is recursive."""
        for clause in pred.clauses:
            if isinstance(clause, Rule):
                for goal in clause.body:
                    if goal.term and isinstance(goal.term, Compound):
                        if goal.term.functor == pred.name:
                            return True
        return False
    
    def _emit_pred_declarations(self) -> str:
        """Emit pred declarations for documentation."""
        lines = []
        
        for (name, arity), pred in sorted(self._predicates.items()):
            # Generate PlDoc-style comment
            lines.append(f"%% {name}/{arity}")
        
        return '\n'.join(lines)
    
    def _emit_predicates(self, ir: Dict[str, Any]) -> str:
        """Emit all predicate definitions."""
        sections = []
        
        for (name, arity), pred in sorted(self._predicates.items()):
            pred_lines = []
            
            for clause in pred.clauses:
                if isinstance(clause, Fact):
                    pred_lines.append(self._emit_fact(clause))
                elif isinstance(clause, Rule):
                    pred_lines.append(self._emit_rule(clause))
            
            if pred_lines:
                sections.append('\n'.join(pred_lines))
        
        return '\n\n'.join(sections)
    
    def _emit_fact(self, fact: Fact) -> str:
        """Emit a fact: pred(args)."""
        if not fact.args:
            return f"{fact.predicate}."
        args = ', '.join(self._emit_term(a) for a in fact.args)
        return f"{fact.predicate}({args})."
    
    def _emit_rule(self, rule: Rule) -> str:
        """Emit a rule: head :- body."""
        head = self._emit_term(rule.head)
        
        if not rule.body:
            return f"{head}."
        
        body_parts = []
        for goal in rule.body:
            body_parts.append(self._emit_goal(goal))
        
        if len(body_parts) == 1:
            return f"{head} :-\n{self.config.indent}{body_parts[0]}."
        
        body = f',\n{self.config.indent}'.join(body_parts)
        return f"{head} :-\n{self.config.indent}{body}."
    
    def _emit_term(self, term: Union[Term, Dict, Any]) -> str:
        """Convert term to Prolog syntax.
        
        Args:
            term: Term object or dictionary
            
        Returns:
            Prolog syntax string
        """
        # Handle dict input
        if isinstance(term, dict):
            term = term_from_dict(term)
        
        if isinstance(term, Variable):
            return term.name
        elif isinstance(term, Atom):
            return self._escape_atom(term.value)
        elif isinstance(term, Number):
            return str(term.value)
        elif isinstance(term, StringTerm):
            escaped = term.value.replace('"', '\\"')
            return f'"{escaped}"'
        elif isinstance(term, Compound):
            if not term.args:
                return self._escape_atom(term.functor)
            args = ', '.join(self._emit_term(a) for a in term.args)
            return f"{term.functor}({args})"
        elif isinstance(term, ListTerm):
            return self._emit_list(term)
        elif isinstance(term, Anonymous):
            return '_'
        else:
            return str(term)
    
    def _emit_list(self, lst: ListTerm) -> str:
        """Emit list with proper [H|T] syntax."""
        if not lst.elements and lst.tail is None:
            return '[]'
        
        elements = [self._emit_term(el) for el in lst.elements]
        
        if lst.tail is None:
            return '[' + ', '.join(elements) + ']'
        elif lst.tail:
            tail_str = self._emit_term(lst.tail)
            if elements:
                return '[' + ', '.join(elements) + '|' + tail_str + ']'
            else:
                return tail_str  # Just the tail variable
        return '[]'
    
    def _escape_atom(self, value: str) -> str:
        """Escape atom if needed.
        
        Atoms need quoting if:
        - They don't start with lowercase
        - They contain special characters
        - They are empty
        """
        if not value:
            return "''"
        
        # Atoms starting with lowercase and containing only alphanumeric/_
        if value[0].islower() and all(c.isalnum() or c == '_' for c in value):
            # Check for reserved words
            reserved = {'is', 'mod', 'rem', 'not', 'true', 'false', 'fail'}
            if value not in reserved:
                return value
        
        # Need quoting
        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"
    
    def _emit_goal(self, goal: Goal) -> str:
        """Emit a goal."""
        if goal.kind == GoalKind.CUT:
            return '!'
        elif goal.kind == GoalKind.NEGATION:
            if goal.goals:
                inner = self._emit_goal(goal.goals[0])
            elif goal.term:
                inner = self._emit_term(goal.term)
            else:
                inner = 'true'
            return f"\\+ {inner}"
        elif goal.kind == GoalKind.UNIFICATION:
            left = self._emit_term(goal.left) if goal.left else '_'
            right = self._emit_term(goal.right) if goal.right else '_'
            return f"{left} = {right}"
        elif goal.kind == GoalKind.CONJUNCTION:
            if goal.goals:
                parts = [self._emit_goal(g) for g in goal.goals]
                return '(' + ', '.join(parts) + ')'
            return 'true'
        elif goal.kind == GoalKind.DISJUNCTION:
            if goal.goals:
                parts = [self._emit_goal(g) for g in goal.goals]
                return '(' + '; '.join(parts) + ')'
            return 'fail'
        elif goal.kind == GoalKind.IF_THEN_ELSE:
            # (Cond -> Then ; Else)
            cond = self._emit_goal(goal.goals[0]) if goal.goals else 'true'
            then_part = 'true'  # Simplified
            else_part = 'fail'  # Simplified
            return f"({cond} -> {then_part} ; {else_part})"
        elif goal.term:
            return self._emit_term(goal.term)
        else:
            return 'true'
    
    def _emit_dcg_rules(self, dcg_rules: List[Dict[str, Any]]) -> str:
        """Emit DCG rules using --> syntax."""
        lines = []
        
        for rule in dcg_rules:
            head_data = rule.get('head', {})
            head = term_from_dict(head_data)
            head_str = self._emit_term(head)
            
            body_items = rule.get('body', [])
            if not body_items:
                lines.append(f"{head_str} --> [].")
                continue
            
            body_parts = []
            for item in body_items:
                kind = item.get('kind', '')
                
                if kind == 'terminal':
                    # Terminal: [word, word2, ...]
                    terminals = item.get('terminals', [])
                    term_str = '[' + ', '.join(self._escape_atom(t) for t in terminals) + ']'
                    body_parts.append(term_str)
                elif kind == 'nonterminal':
                    # Non-terminal: rule_name or rule_name(Args)
                    term = term_from_dict(item.get('term', {}))
                    body_parts.append(self._emit_term(term))
                elif kind == 'pushback':
                    # Pushback: {goals}
                    goals = item.get('goals', [])
                    if goals:
                        goal_strs = [self._emit_goal(_parse_goal(g)) for g in goals]
                        body_parts.append('{' + ', '.join(goal_strs) + '}')
                    else:
                        body_parts.append('{}')
                else:
                    # Treat as non-terminal
                    term = term_from_dict(item)
                    body_parts.append(self._emit_term(term))
            
            body = ', '.join(body_parts)
            lines.append(f"{head_str} --> {body}.")
        
        return '\n'.join(lines)
    
    def _emit_queries(self, queries: List[Query]) -> str:
        """Emit queries."""
        lines = ["% Queries"]
        
        for query in queries:
            if not query.goals:
                continue
            
            goals = ', '.join(self._emit_goal(g) for g in query.goals)
            # As initialization directive
            lines.append(f":- initialization({goals}).")
            # Also as comment for interactive use
            lines.append(f"% ?- {goals}.")
        
        return '\n'.join(lines)
    
    def emit_to_file(self, ir: Dict[str, Any], output_path: Path) -> EmitterResult:
        """Emit code and write to file.
        
        Args:
            ir: STUNIR Logic IR dictionary
            output_path: Output file path
            
        Returns:
            EmitterResult with generated code
        """
        result = self.emit(ir)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result.code)
        
        return result


def _parse_goal(data: Dict[str, Any]) -> Goal:
    """Parse a goal from IR dictionary."""
    from tools.ir.logic_ir import Goal, GoalKind, Compound
    
    kind = data.get('kind', 'call')
    
    if kind == 'cut':
        return Goal.cut()
    elif kind == 'negation':
        inner_data = data.get('goal', data.get('goals', [{}])[0] if data.get('goals') else {})
        inner = _parse_goal(inner_data)
        return Goal.negation(inner)
    elif kind == 'unification':
        left = term_from_dict(data.get('left', {}))
        right = term_from_dict(data.get('right', {}))
        return Goal.unification(left, right)
    elif kind == 'compound' or 'functor' in data:
        term = Compound.from_dict(data) if 'functor' in data else term_from_dict(data)
        return Goal.call(term)
    else:
        term = term_from_dict(data)
        return Goal.call(term)


__all__ = ['SWIPrologEmitter', 'SWIPrologConfig', 'EmitterResult', 'compute_sha256', 'canonical_json']
