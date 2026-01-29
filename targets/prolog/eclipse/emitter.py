#!/usr/bin/env python3
"""STUNIR ECLiPSe Emitter.

Generates ECLiPSe code from STUNIR Logic IR.
Supports constraint optimization, multiple CLP libraries (IC, FD, R, Q),
advanced search strategies, and branch-and-bound optimization.

Key differences from other Prolog systems:
- Focus on optimization (minimize/maximize), not just satisfaction
- Rich constraint libraries with clear separation (lib(ic), lib(fd), etc.)
- Different syntax for constraints ($= for IC, #= for FD)
- Advanced search with cost-based branch-and-bound
- Uses .ecl file extension

Part of Phase 5D-2: ECLiPSe with Constraint Optimization.
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
    ECLiPSeTypeMapper, ECLIPSE_TYPES,
    IC_OPERATORS, FD_OPERATORS, ECLIPSE_GLOBALS,
    ECLIPSE_OPTIMIZATION, ECLIPSE_SEARCH,
    ECLIPSE_SELECT_METHODS, ECLIPSE_CHOICE_METHODS
)


def compute_sha256(data: bytes) -> str:
    """Compute SHA-256 hash of data."""
    return hashlib.sha256(data).hexdigest()


def canonical_json(data: Any) -> str:
    """Produce deterministic JSON output."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'))


@dataclass
class EmitterResult:
    """Result of ECLiPSe code emission."""
    code: str
    module_name: str
    predicates: List[str]
    sha256: str
    emit_time: float
    libraries_used: List[str]
    has_optimization: bool
    search_strategies: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'code': self.code,
            'module_name': self.module_name,
            'predicates': self.predicates,
            'sha256': self.sha256,
            'emit_time': self.emit_time,
            'libraries_used': self.libraries_used,
            'has_optimization': self.has_optimization,
            'search_strategies': self.search_strategies
        }


@dataclass
class ECLiPSeConfig:
    """Configuration for ECLiPSe emitter."""
    module_prefix: str = "stunir"
    emit_comments: bool = True
    emit_timestamps: bool = True
    indent: str = "    "
    line_width: int = 80
    
    # Library selection (IC is recommended default)
    default_library: str = "ic"        # Primary constraint library: ic|fd
    use_ic_global: bool = True         # Use lib(ic_global) for global constraints
    use_ic_search: bool = True         # Use lib(ic_search) for advanced search
    use_branch_bound: bool = True      # Use lib(branch_and_bound)
    
    # Search defaults
    default_select: str = "first_fail"
    default_choice: str = "indomain_middle"
    
    # Optimization
    enable_optimization: bool = True
    optimization_timeout: Optional[int] = None


class ECLiPSeEmitter:
    """Emitter for ECLiPSe constraint logic programs.
    
    Generates valid ECLiPSe code from STUNIR Logic IR including:
    - Module declarations (:- module(name).)
    - Library imports (:- lib(ic).)
    - Facts and rules
    - Constraint operators (IC: $=, FD: #=)
    - Domain constraints (X :: 1..10)
    - Global constraints (alldifferent, element, etc.)
    - Optimization goals (minimize, maximize, bb_min)
    - Advanced search strategies (search/6)
    """
    
    DIALECT = "eclipse"
    FILE_EXTENSION = ".ecl"
    
    def __init__(self, config: Optional[ECLiPSeConfig] = None):
        """Initialize ECLiPSe emitter.
        
        Args:
            config: Emitter configuration (optional)
        """
        self.config = config or ECLiPSeConfig()
        self.logic_ext = LogicIRExtension()
        self.type_mapper = ECLiPSeTypeMapper(self.config)
        self._exports: Set[Tuple[str, int]] = set()
        self._dynamics: Set[Tuple[str, int]] = set()
        self._predicates: Dict[Tuple[str, int], Predicate] = {}
        self._libraries_used: Set[str] = set()
        self._has_optimization: bool = False
        self._search_strategies: Set[str] = set()
    
    def emit(self, ir: Dict[str, Any]) -> EmitterResult:
        """Emit ECLiPSe code from Logic IR.
        
        Args:
            ir: STUNIR Logic IR dictionary
            
        Returns:
            EmitterResult with generated code
        """
        start_time = time.time()
        
        name = ir.get('module', 'unnamed')
        module_name = self._eclipse_name(name)
        
        # Reset state
        self._exports = set()
        self._dynamics = set()
        self._libraries_used = set()
        self._has_optimization = False
        self._search_strategies = set()
        
        # Extract predicates and analyze
        self._predicates = self.logic_ext.extract_predicates(ir)
        self._analyze_exports(ir)
        self._analyze_dynamics(ir)
        self._detect_features(ir)
        
        # Build code sections
        sections = []
        
        # Header comment
        if self.config.emit_comments:
            sections.append(self._emit_header(name))
        
        # Module declaration
        sections.append(f":- module({module_name}).")
        
        # Library imports
        lib_section = self._emit_library_imports()
        if lib_section:
            sections.append(lib_section)
        
        # Export declarations
        if self._exports:
            sections.append(self._emit_export_declarations())
        
        # Dynamic declarations
        if self._dynamics:
            sections.append(self._emit_dynamic_declarations())
        
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
        
        # Queries / Goals
        queries = self.logic_ext.extract_queries(ir)
        if queries:
            sections.append(self._emit_queries(queries))
        
        # Optimization section
        if ir.get('optimization'):
            sections.append(self._emit_optimization(ir['optimization']))
        
        code = '\n\n'.join(s for s in sections if s)
        code_hash = compute_sha256(code.encode('utf-8'))
        
        return EmitterResult(
            code=code,
            module_name=module_name,
            predicates=[f"{p}/{a}" for p, a in sorted(self._predicates.keys())],
            sha256=code_hash,
            emit_time=time.time() - start_time,
            libraries_used=list(sorted(self._libraries_used)),
            has_optimization=self._has_optimization,
            search_strategies=list(sorted(self._search_strategies))
        )
    
    def _eclipse_name(self, name: str) -> str:
        """Convert name to valid ECLiPSe identifier.
        
        Args:
            name: Original name
            
        Returns:
            Valid ECLiPSe identifier (lowercase start)
        """
        # Replace non-alphanumeric with underscore
        result = ''.join(c if c.isalnum() or c == '_' else '_' for c in name)
        # Ensure starts with lowercase
        if result and result[0].isupper():
            result = result[0].lower() + result[1:]
        return result or 'unnamed'
    
    def _analyze_exports(self, ir: Dict[str, Any]) -> None:
        """Analyze exported predicates from IR."""
        for exp in ir.get('exports', []):
            pred = exp.get('predicate')
            arity = exp.get('arity', 0)
            if pred:
                self._exports.add((pred, arity))
        
        # Auto-export non-private predicates
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
    
    def _detect_features(self, ir: Dict[str, Any]) -> None:
        """Detect features used in the IR to determine required libraries."""
        ir_str = json.dumps(ir)
        
        # Check for constraint operators
        constraint_indicators = ['$=', '$<', '$>', '$=<', '$>=', '$\\=',
                                '#=', '#<', '#>', '#=<', '#>=', '#\\=',
                                '::', 'domain', 'fd_domain']
        for ind in constraint_indicators:
            if ind in ir_str:
                self._libraries_used.add(self.config.default_library)
                break
        
        # Check for global constraints
        global_indicators = ['alldifferent', 'all_different', 'element', 
                            'cumulative', 'circuit', 'gcc']
        for ind in global_indicators:
            if ind in ir_str.lower():
                self._libraries_used.add(self.config.default_library)
                if self.config.use_ic_global and self.config.default_library == 'ic':
                    self._libraries_used.add('ic_global')
                break
        
        # Check for optimization
        opt_indicators = ['minimize', 'maximize', 'bb_min', 'bb_max']
        for ind in opt_indicators:
            if ind in ir_str.lower():
                self._has_optimization = True
                if self.config.use_branch_bound:
                    self._libraries_used.add('branch_and_bound')
                break
        
        # Check for search strategies
        search_indicators = ['search', 'labeling', 'first_fail', 'indomain']
        for ind in search_indicators:
            if ind in ir_str.lower():
                if self.config.use_ic_search and self.config.default_library == 'ic':
                    self._libraries_used.add('ic_search')
                break
    
    def _emit_header(self, name: str) -> str:
        """Generate header comment."""
        lines = [
            "/*",
            " * STUNIR Generated ECLiPSe Source",
            f" * Module: {name}",
        ]
        
        if self.config.emit_timestamps:
            lines.append(f" * Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self._libraries_used:
            lines.append(f" * Libraries: {', '.join(sorted(self._libraries_used))}")
        
        if self._has_optimization:
            lines.append(" * Features: Constraint Optimization")
        
        lines.extend([
            " * ",
            " * This file was automatically generated by STUNIR.",
            " * Run with: eclipse -b file.ecl -e 'main'",
            " * Do not edit manually.",
            " */"
        ])
        
        return '\n'.join(lines)
    
    def _emit_library_imports(self) -> str:
        """Emit library import directives."""
        if not self._libraries_used:
            return ""
        
        lines = ["% Library imports"]
        for lib in sorted(self._libraries_used):
            lines.append(f":- lib({lib}).")
        
        return '\n'.join(lines)
    
    def _emit_export_declarations(self) -> str:
        """Emit export declarations (ECLiPSe uses :- export)."""
        lines = ["% Exported predicates"]
        
        for pred, arity in sorted(self._exports):
            lines.append(f":- export({pred}/{arity}).")
        
        return '\n'.join(lines)
    
    def _emit_dynamic_declarations(self) -> str:
        """Emit dynamic declarations."""
        lines = ["% Dynamic predicates"]
        
        for pred, arity in sorted(self._dynamics):
            lines.append(f":- dynamic({pred}/{arity}).")
        
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
        """Convert term to ECLiPSe syntax.
        
        Args:
            term: Term object or dictionary
            
        Returns:
            ECLiPSe syntax string
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
            # Check for domain constraint (X :: L..H)
            if self._is_domain_constraint(term):
                return self._emit_domain_constraint(term)
            # Check for constraint operators
            if self._is_constraint_operator(term):
                return self._emit_constraint_operator(term)
            # Check for global constraints
            if self._is_global_constraint(term):
                return self._emit_global_constraint(term)
            # Check for optimization
            if self._is_optimization_goal(term):
                return self._emit_optimization_goal(term)
            # Check for search
            if self._is_search_predicate(term):
                return self._emit_search(term)
            
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
                return tail_str
        return '[]'
    
    def _escape_atom(self, value: str) -> str:
        """Escape atom if needed."""
        if not value:
            return "''"
        
        # Atoms starting with lowercase and containing only alphanumeric/_
        if value[0].islower() and all(c.isalnum() or c == '_' for c in value):
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
            cond = self._emit_goal(goal.goals[0]) if goal.goals else 'true'
            then_part = 'true'
            else_part = 'fail'
            return f"({cond} -> {then_part} ; {else_part})"
        elif goal.term:
            return self._emit_term(goal.term)
        else:
            return 'true'
    
    def _is_domain_constraint(self, term: Compound) -> bool:
        """Check if compound is a domain constraint (X :: L..H)."""
        return term.functor in ('::', 'domain', 'fd_domain', 'in')
    
    def _emit_domain_constraint(self, term: Compound) -> str:
        """Emit domain constraint using ECLiPSe :: syntax."""
        self._libraries_used.add(self.config.default_library)
        
        if term.functor == '::':
            # X :: L..H
            var = self._emit_term(term.args[0])
            range_term = term.args[1]
            if isinstance(range_term, Compound) and range_term.functor == '..':
                low = self._emit_term(range_term.args[0])
                high = self._emit_term(range_term.args[1])
                return f"{var} :: {low}..{high}"
            return f"{var} :: {self._emit_term(range_term)}"
        
        if term.functor in ('domain', 'fd_domain'):
            # domain(X, L, H) -> X :: L..H
            if len(term.args) >= 3:
                var = self._emit_term(term.args[0])
                low = self._emit_term(term.args[1])
                high = self._emit_term(term.args[2])
                return f"{var} :: {low}..{high}"
        
        if term.functor == 'in':
            # X in L..H -> X :: L..H
            var = self._emit_term(term.args[0])
            range_term = term.args[1]
            if isinstance(range_term, Compound) and range_term.functor == '..':
                low = self._emit_term(range_term.args[0])
                high = self._emit_term(range_term.args[1])
                return f"{var} :: {low}..{high}"
            return f"{var} :: {self._emit_term(range_term)}"
        
        # Fallback
        args = ', '.join(self._emit_term(a) for a in term.args)
        return f"{term.functor}({args})"
    
    def _is_constraint_operator(self, term: Compound) -> bool:
        """Check if compound is a constraint operator."""
        return self.type_mapper.is_constraint_operator(term.functor)
    
    def _emit_constraint_operator(self, term: Compound) -> str:
        """Emit constraint operator using IC or FD syntax."""
        self._libraries_used.add(self.config.default_library)
        
        op = self.type_mapper.map_constraint_operator(term.functor)
        left = self._emit_term(term.args[0])
        right = self._emit_term(term.args[1])
        return f"{left} {op} {right}"
    
    def _is_global_constraint(self, term: Compound) -> bool:
        """Check if compound is a global constraint."""
        return self.type_mapper.is_global_constraint(term.functor)
    
    def _emit_global_constraint(self, term: Compound) -> str:
        """Emit global constraint."""
        self._libraries_used.add(self.config.default_library)
        if self.config.use_ic_global and self.config.default_library == 'ic':
            self._libraries_used.add('ic_global')
        
        pred = self.type_mapper.map_global_constraint(term.functor)
        args = ', '.join(self._emit_term(a) for a in term.args)
        return f"{pred}({args})"
    
    def _is_optimization_goal(self, term: Compound) -> bool:
        """Check if compound is an optimization goal."""
        return self.type_mapper.is_optimization_predicate(term.functor)
    
    def _emit_optimization_goal(self, term: Compound) -> str:
        """Emit optimization goal."""
        self._has_optimization = True
        if self.config.use_branch_bound:
            self._libraries_used.add('branch_and_bound')
        
        pred = self.type_mapper.map_optimization_predicate(term.functor)
        
        if pred in ('minimize', 'maximize'):
            # minimize(Goal, Cost) / maximize(Goal, Cost)
            goal = self._emit_term(term.args[0])
            cost = self._emit_term(term.args[1]) if len(term.args) > 1 else goal
            return f"{pred}({goal}, {cost})"
        
        if pred in ('bb_min', 'bb_max'):
            # bb_min(Goal, Cost, Options)
            goal = self._emit_term(term.args[0])
            cost = self._emit_term(term.args[1]) if len(term.args) > 1 else '_'
            opts = "bb_options{}" if len(term.args) < 3 else self._emit_term(term.args[2])
            return f"{pred}({goal}, {cost}, {opts})"
        
        args = ', '.join(self._emit_term(a) for a in term.args)
        return f"{pred}({args})"
    
    def _is_search_predicate(self, term: Compound) -> bool:
        """Check if compound is a search predicate."""
        return term.functor.lower() in ('search', 'labeling', 'label', 'fd_labeling')
    
    def _emit_search(self, term: Compound) -> str:
        """Emit search predicate."""
        if self.config.use_ic_search and self.config.default_library == 'ic':
            self._libraries_used.add('ic_search')
        
        functor = term.functor.lower()
        
        if functor in ('labeling', 'label', 'fd_labeling'):
            # Simple labeling
            if len(term.args) == 1:
                vars_term = self._emit_term(term.args[0])
                self._search_strategies.add('labeling')
                return f"labeling({vars_term})"
            # Labeling with options
            if len(term.args) >= 2:
                vars_term = self._emit_term(term.args[1])
                opts = self._emit_term(term.args[0])
                self._search_strategies.add('labeling')
                return f"labeling({vars_term})"
        
        if functor == 'search':
            # search(Vars, ArgNo, Select, Choice, Method, Options)
            if len(term.args) >= 4:
                vars_term = self._emit_term(term.args[0])
                arg_no = self._emit_term(term.args[1]) if len(term.args) > 1 else '0'
                select = self.type_mapper.map_select_method(
                    term.args[2].value if hasattr(term.args[2], 'value') else str(term.args[2])
                ) if len(term.args) > 2 else self.config.default_select
                choice = self.type_mapper.map_choice_method(
                    term.args[3].value if hasattr(term.args[3], 'value') else str(term.args[3])
                ) if len(term.args) > 3 else self.config.default_choice
                method = 'complete'
                opts = '[]'
                
                self._search_strategies.add(f"search({select},{choice})")
                return f"search({vars_term}, {arg_no}, {select}, {choice}, {method}, {opts})"
        
        # Default: pass through
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
                    terminals = item.get('terminals', [])
                    term_str = '[' + ', '.join(self._escape_atom(t) for t in terminals) + ']'
                    body_parts.append(term_str)
                elif kind == 'nonterminal':
                    term = term_from_dict(item.get('term', {}))
                    body_parts.append(self._emit_term(term))
                elif kind == 'pushback':
                    goals = item.get('goals', [])
                    if goals:
                        goal_strs = [self._emit_goal(self._parse_goal(g)) for g in goals]
                        body_parts.append('{' + ', '.join(goal_strs) + '}')
                    else:
                        body_parts.append('{}')
                else:
                    term = term_from_dict(item)
                    body_parts.append(self._emit_term(term))
            
            body = ', '.join(body_parts)
            lines.append(f"{head_str} --> {body}.")
        
        return '\n'.join(lines)
    
    def _emit_queries(self, queries: List[Query]) -> str:
        """Emit queries."""
        lines = ["% Queries / Goals"]
        
        for query in queries:
            if not query.goals:
                continue
            
            goals = ', '.join(self._emit_goal(g) for g in query.goals)
            lines.append(f":- {goals}.")
            lines.append(f"% ?- {goals}.")
        
        return '\n'.join(lines)
    
    def _emit_optimization(self, opt_data: Dict[str, Any]) -> str:
        """Emit optimization section."""
        self._has_optimization = True
        if self.config.use_branch_bound:
            self._libraries_used.add('branch_and_bound')
        
        lines = ["% Optimization"]
        
        goal = opt_data.get('goal', 'minimize')
        var = opt_data.get('var', 'Cost')
        predicate = opt_data.get('predicate', 'solve')
        
        if goal in ('minimize', 'min'):
            lines.append(f"% minimize({predicate}({var}), {var})")
        elif goal in ('maximize', 'max'):
            lines.append(f"% maximize({predicate}({var}), {var})")
        
        return '\n'.join(lines)
    
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


__all__ = ['ECLiPSeEmitter', 'ECLiPSeConfig', 'EmitterResult', 'compute_sha256', 'canonical_json']
