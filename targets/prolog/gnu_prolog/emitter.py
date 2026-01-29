#!/usr/bin/env python3
"""STUNIR GNU Prolog Emitter.

Generates GNU Prolog code from STUNIR Logic IR.
Supports file-based organization (no modules), CLP(FD), CLP(B), CLP(R).

Key differences from SWI-Prolog:
- No module system (uses file-based organization)
- Different built-in predicates (fd_domain vs clpfd:in)
- Native CLP(FD), CLP(B), CLP(R) support
- Different syntax for some constructs

Part of Phase 5C-2: GNU Prolog with CLP support.
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
from .types import (
    GNUPrologTypeMapper, GNU_PROLOG_TYPES,
    CLPFD_OPERATORS, CLPB_OPERATORS, CLPFD_PREDICATES
)


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
    filename: str  # No module in GNU Prolog, use filename
    predicates: List[str]
    sha256: str
    emit_time: float
    clp_features: List[str]  # Track CLP features used
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'code': self.code,
            'filename': self.filename,
            'predicates': self.predicates,
            'sha256': self.sha256,
            'emit_time': self.emit_time,
            'clp_features': self.clp_features
        }


@dataclass
class GNUPrologConfig:
    """Configuration for GNU Prolog emitter."""
    file_prefix: str = "stunir"
    emit_comments: bool = True
    emit_public: bool = True  # :- public declarations
    emit_initialization: bool = True
    indent: str = "    "
    line_width: int = 80
    # CLP options
    enable_clpfd: bool = True   # Enable CLP(FD) constraints
    enable_clpb: bool = True    # Enable CLP(B) constraints
    enable_clpr: bool = False   # Enable CLP(R) constraints
    # GNU Prolog specific
    native_compile: bool = False  # Generate gplc commands
    emit_timestamps: bool = True
    emit_sha256: bool = True


class GNUPrologEmitter:
    """Emitter for GNU Prolog code.
    
    Generates valid GNU Prolog code from STUNIR Logic IR including:
    - File-based organization (no modules)
    - Public predicate declarations
    - Facts and rules
    - CLP(FD) constraints (finite domains)
    - CLP(B) constraints (booleans)
    - CLP(R) constraints (reals) - optional
    """
    
    DIALECT = "gnu-prolog"
    FILE_EXTENSION = ".pl"
    
    def __init__(self, config: Optional[GNUPrologConfig] = None):
        """Initialize GNU Prolog emitter.
        
        Args:
            config: Emitter configuration (optional)
        """
        self.config = config or GNUPrologConfig()
        self.logic_ext = LogicIRExtension()
        self.type_mapper = GNUPrologTypeMapper(self.config)
        self._publics: Set[Tuple[str, int]] = set()
        self._dynamics: Set[Tuple[str, int]] = set()
        self._predicates: Dict[Tuple[str, int], Predicate] = {}
        self._clp_features: Set[str] = set()  # Track CLP usage
    
    def emit(self, ir: Dict[str, Any]) -> EmitterResult:
        """Emit GNU Prolog code from Logic IR.
        
        Args:
            ir: STUNIR Logic IR dictionary
            
        Returns:
            EmitterResult with generated code
        """
        start_time = time.time()
        
        name = ir.get('module', 'unnamed')
        filename = f"{self.config.file_prefix}_{self._prolog_name(name)}"
        
        # Reset state
        self._publics = set()
        self._dynamics = set()
        self._clp_features = set()
        
        # Extract predicates and analyze
        self._predicates = self.logic_ext.extract_predicates(ir)
        self._analyze_publics(ir)
        self._analyze_dynamics(ir)
        self._detect_clp_usage(ir)
        
        # Build code sections
        sections = []
        
        # Header comment
        if self.config.emit_comments:
            sections.append(self._emit_header(name))
        
        # Public declarations (GNU Prolog's equivalent of exports)
        if self.config.emit_public and self._publics:
            sections.append(self._emit_public_declarations())
        
        # Dynamic declarations
        if self._dynamics:
            sections.append(self._emit_dynamic_declarations())
        
        # CLP directives/hints
        if self._clp_features:
            sections.append(self._emit_clp_header())
        
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
        
        # Queries
        queries = self.logic_ext.extract_queries(ir)
        if queries:
            sections.append(self._emit_queries(queries))
        
        # Initialization
        if self.config.emit_initialization and ir.get('initialization'):
            sections.append(self._emit_initialization(ir['initialization']))
        
        code = '\n\n'.join(s for s in sections if s)
        code_hash = compute_sha256(code.encode('utf-8'))
        
        return EmitterResult(
            code=code,
            filename=filename,
            predicates=[f"{p}/{a}" for p, a in sorted(self._predicates.keys())],
            sha256=code_hash,
            emit_time=time.time() - start_time,
            clp_features=list(sorted(self._clp_features))
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
    
    def _analyze_publics(self, ir: Dict[str, Any]) -> None:
        """Analyze public predicates from IR (exports in GNU Prolog)."""
        for exp in ir.get('exports', []):
            pred = exp.get('predicate')
            arity = exp.get('arity', 0)
            if pred:
                self._publics.add((pred, arity))
        
        # Auto-export non-private predicates
        for (name, arity), pred in self._predicates.items():
            if not name.startswith('_'):  # Convention: _ prefix = private
                self._publics.add((name, arity))
    
    def _analyze_dynamics(self, ir: Dict[str, Any]) -> None:
        """Analyze dynamic predicate declarations."""
        for dyn in ir.get('dynamic', []):
            pred = dyn.get('predicate')
            arity = dyn.get('arity', 0)
            if pred:
                self._dynamics.add((pred, arity))
    
    def _detect_clp_usage(self, ir: Dict[str, Any]) -> None:
        """Detect CLP features used in the IR."""
        ir_str = json.dumps(ir)
        
        # Check for CLP(FD)
        clpfd_indicators = ['fd_domain', 'fd_labeling', 'fd_all_different',
                           'domain', 'labeling', 'all_different',
                           '#=', '#\\=', '#<', '#>', '#=<', '#>=']
        for ind in clpfd_indicators:
            if ind in ir_str:
                self._clp_features.add('clpfd')
                break
        
        # Check for CLP(B)
        clpb_indicators = ['#/\\', '#\\/', '#\\', '#<=>', '#==>']
        for ind in clpb_indicators:
            if ind in ir_str:
                self._clp_features.add('clpb')
                break
        
        # Check for CLP(R)
        if self.config.enable_clpr and '{' in ir_str:
            self._clp_features.add('clpr')
    
    def _emit_header(self, name: str) -> str:
        """Generate header comment (no module declaration in GNU Prolog)."""
        lines = [
            "/*",
            " * STUNIR Generated GNU Prolog Source",
            f" * File: {name}",
        ]
        
        if self.config.emit_timestamps:
            lines.append(f" * Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self._clp_features:
            lines.append(f" * CLP Features: {', '.join(sorted(self._clp_features))}")
        
        lines.extend([
            " * ",
            " * This file was automatically generated by STUNIR.",
            f" * Compile with: gplc -o program {name}.pl",
            " * Do not edit manually.",
            " */"
        ])
        
        return '\n'.join(lines)
    
    def _emit_public_declarations(self) -> str:
        """Emit public declarations (GNU Prolog's export equivalent)."""
        lines = ["% Public predicates"]
        
        for pred, arity in sorted(self._publics):
            lines.append(f":- public({pred}/{arity}).")
        
        return '\n'.join(lines)
    
    def _emit_dynamic_declarations(self) -> str:
        """Emit dynamic declarations (GNU Prolog syntax)."""
        lines = ["% Dynamic predicates"]
        
        for pred, arity in sorted(self._dynamics):
            # GNU Prolog uses :- dynamic(pred/N). syntax
            lines.append(f":- dynamic({pred}/{arity}).")
        
        return '\n'.join(lines)
    
    def _emit_clp_header(self) -> str:
        """Emit CLP header comments.
        
        Note: GNU Prolog has built-in CLP(FD), no imports needed.
        """
        lines = ["% CLP Features used in this file:"]
        
        if 'clpfd' in self._clp_features:
            lines.extend([
                "% - CLP(FD): Finite Domain Constraints",
                "%   Operators: #=, #\\=, #<, #>, #=<, #>=",
                "%   Use fd_domain/3 or X in L..H for domains",
                "%   Use fd_labeling/1 for search"
            ])
        
        if 'clpb' in self._clp_features:
            lines.extend([
                "% - CLP(B): Boolean Constraints",
                "%   Operators: #/\\ (and), #\\/ (or), #\\ (not), #<=> (equiv)"
            ])
        
        if 'clpr' in self._clp_features:
            lines.extend([
                "% - CLP(R): Real Number Constraints",
                "%   Use {Constraint} syntax"
            ])
        
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
        """Convert term to GNU Prolog syntax.
        
        Args:
            term: Term object or dictionary
            
        Returns:
            GNU Prolog syntax string
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
            # Check for CLP constraints
            if self._is_clpfd_constraint(term):
                return self._emit_clpfd_constraint(term)
            elif self._is_clpb_constraint(term):
                return self._emit_clpb_constraint(term)
            
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
    
    def _is_clpfd_constraint(self, term: Compound) -> bool:
        """Check if compound is a CLP(FD) constraint."""
        functor = term.functor
        return (functor in CLPFD_OPERATORS or 
                functor in CLPFD_PREDICATES or
                functor.startswith('#') and not functor.startswith('#/'))
    
    def _is_clpb_constraint(self, term: Compound) -> bool:
        """Check if compound is a CLP(B) constraint."""
        functor = term.functor
        return functor in CLPB_OPERATORS
    
    def _emit_clpfd_constraint(self, term: Compound) -> str:
        """Emit CLP(FD) constraint.
        
        GNU Prolog uses operators: #=, #\\=, #<, #>, #=<, #>=
        """
        op = term.functor
        self._clp_features.add('clpfd')
        
        # Arithmetic constraint operators
        if op in CLPFD_OPERATORS:
            left = self._emit_term(term.args[0])
            right = self._emit_term(term.args[1])
            gnu_op = CLPFD_OPERATORS[op]
            return f"{left} {gnu_op} {right}"
        
        # CLP(FD) predicates
        if op in CLPFD_PREDICATES or op == 'domain':
            # fd_domain(Vars, Min, Max)
            gnu_pred = CLPFD_PREDICATES.get(op, op)
            args = ', '.join(self._emit_term(a) for a in term.args)
            return f"{gnu_pred}({args})"
        
        if op == 'in':
            # X in L..H syntax
            var = self._emit_term(term.args[0])
            range_term = term.args[1]
            if isinstance(range_term, Compound) and range_term.functor == '..':
                low = self._emit_term(range_term.args[0])
                high = self._emit_term(range_term.args[1])
                return f"{var} in {low}..{high}"
            return f"{var} in {self._emit_term(range_term)}"
        
        # Default compound handling
        args = ', '.join(self._emit_term(a) for a in term.args)
        return f"{term.functor}({args})"
    
    def _emit_clpb_constraint(self, term: Compound) -> str:
        """Emit CLP(B) constraint.
        
        GNU Prolog uses operators: #/\\ (and), #\\/ (or), #\\ (not), #<=> (equiv)
        """
        op = term.functor
        self._clp_features.add('clpb')
        
        if op in ('and', '#/\\'):
            left = self._emit_term(term.args[0])
            right = self._emit_term(term.args[1])
            return f"({left} #/\\ {right})"
        
        if op in ('or', '#\\/'):
            left = self._emit_term(term.args[0])
            right = self._emit_term(term.args[1])
            return f"({left} #\\/ {right})"
        
        if op in ('not', '#\\'):
            arg = self._emit_term(term.args[0])
            return f"#\\ {arg}"
        
        if op in ('equiv', '#<=>'):
            left = self._emit_term(term.args[0])
            right = self._emit_term(term.args[1])
            return f"({left} #<=> {right})"
        
        if op in ('implies', '#==>'):
            left = self._emit_term(term.args[0])
            right = self._emit_term(term.args[1])
            return f"({left} #==> {right})"
        
        # Default
        args = ', '.join(self._emit_term(a) for a in term.args)
        return f"{term.functor}({args})"
    
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
                        goal_strs = [self._emit_goal(self._parse_goal(g)) for g in goals]
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
    
    def _emit_initialization(self, init_data: Any) -> str:
        """Emit initialization directive."""
        if isinstance(init_data, list):
            goals = ', '.join(str(g) for g in init_data)
            return f":- initialization({goals})."
        return f":- initialization({init_data})."
    
    def _parse_goal(self, data: Dict[str, Any]) -> Goal:
        """Parse a goal from IR dictionary."""
        from tools.ir.logic_ir import Goal, GoalKind, Compound
        
        kind = data.get('kind', 'call')
        
        if kind == 'cut':
            return Goal.cut()
        elif kind == 'negation':
            inner_data = data.get('goal', data.get('goals', [{}])[0] if data.get('goals') else {})
            inner = self._parse_goal(inner_data)
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


__all__ = ['GNUPrologEmitter', 'GNUPrologConfig', 'EmitterResult', 'compute_sha256', 'canonical_json']
