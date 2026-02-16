#!/usr/bin/env python3
"""STUNIR Datalog Emitter.

Generates Datalog code from STUNIR Logic IR with:
- Strict Datalog restriction enforcement
- Stratified negation support
- Range restriction checking
- Multiple dialect support (standard, Souffle)

Datalog is a syntactic subset of Prolog that guarantees termination
through restrictions on recursion and function symbols.

Key differences from Prolog:
- Bottom-up evaluation (vs top-down)
- No function symbols in rule heads
- Set semantics (no duplicates)
- Stratified negation only
- No cut or side effects

Part of Phase 5C-4: Datalog Emitter.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from tools.ir.logic_ir import (
    Term, Variable, Atom, Number, StringTerm, Compound, ListTerm, Anonymous,
    Fact, Rule, Goal, Query, Predicate, GoalKind,
    LogicIRExtension, term_from_dict
)
from .types import (
    DatalogTypeMapper, DATALOG_TYPES,
    escape_atom, escape_string, format_variable
)


def compute_sha256(data: bytes) -> str:
    """Compute SHA-256 hash of data."""
    return hashlib.sha256(data).hexdigest()


def canonical_json(data: Any) -> str:
    """Produce deterministic JSON output."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'))


class StratificationError(Exception):
    """Raised when negation cannot be stratified."""
    pass


class DatalogRestrictionError(Exception):
    """Raised when Datalog restrictions are violated."""
    pass


class ValidationLevel(Enum):
    """Validation strictness level."""
    STRICT = "strict"      # Reject any violation
    WARN = "warn"          # Warn but continue
    LENIENT = "lenient"    # Best-effort conversion


@dataclass
class DatalogConfig:
    """Configuration for Datalog emitter."""
    # Output options
    emit_comments: bool = True
    emit_timestamps: bool = True
    emit_sha256: bool = True
    
    # Validation
    validation_level: ValidationLevel = ValidationLevel.STRICT
    check_range_restriction: bool = True
    check_stratification: bool = True
    
    # Output format
    file_extension: str = ".dl"
    indent: str = "    "
    line_width: int = 80
    
    # Dialect compatibility
    dialect: str = "standard"  # standard, souffle


@dataclass
class StratificationResult:
    """Result of stratification analysis."""
    strata: Dict[str, int]  # predicate -> stratum number
    is_stratifiable: bool
    dependency_graph: Dict[str, Set[Tuple[str, bool]]]  # pred -> {(dep, is_negated)}
    errors: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'strata': self.strata,
            'is_stratifiable': self.is_stratifiable,
            'errors': self.errors,
            'dependency_graph': {
                k: [(d, n) for d, n in v] 
                for k, v in self.dependency_graph.items()
            }
        }


@dataclass
class ValidationResult:
    """Result of Datalog validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    restricted_terms: List[str]  # Terms that violated restrictions


@dataclass
class EmitterResult:
    """Result of code emission."""
    code: str
    predicates: List[str]
    facts_count: int
    rules_count: int
    queries_count: int
    sha256: str
    emit_time: float
    stratification: Optional[StratificationResult]
    validation: ValidationResult
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'code': self.code,
            'predicates': self.predicates,
            'facts_count': self.facts_count,
            'rules_count': self.rules_count,
            'queries_count': self.queries_count,
            'sha256': self.sha256,
            'emit_time': self.emit_time,
            'stratification': self.stratification.to_dict() if self.stratification else None,
            'validation': {
                'is_valid': self.validation.is_valid,
                'errors': self.validation.errors,
                'warnings': self.validation.warnings,
            }
        }


class DatalogEmitter:
    """Emitter for Datalog code.
    
    Generates valid Datalog code with:
    - Strict Datalog restriction enforcement
    - Stratified negation support
    - Range restriction checking
    - Multiple dialect support
    """
    
    DIALECT = "datalog"
    FILE_EXTENSION = ".dl"
    
    def __init__(self, config: Optional[DatalogConfig] = None):
        """Initialize Datalog emitter.
        
        Args:
            config: Emitter configuration (optional)
        """
        self.config = config or DatalogConfig()
        self.logic_ext = LogicIRExtension()
        self.type_mapper = DatalogTypeMapper()
        self._predicates: Dict[Tuple[str, int], Predicate] = {}
        self._errors: List[str] = []
        self._warnings: List[str] = []
        self._restricted_terms: List[str] = []
    
    def emit(self, ir: Dict[str, Any]) -> EmitterResult:
        """Emit Datalog code from Logic IR.
        
        Args:
            ir: STUNIR Logic IR dictionary
            
        Returns:
            EmitterResult with generated code
            
        Raises:
            DatalogRestrictionError: If validation fails in STRICT mode
        """
        start_time = time.time()
        
        # Reset state
        self._errors = []
        self._warnings = []
        self._restricted_terms = []
        
        # Extract predicates
        self._predicates = self.logic_ext.extract_predicates(ir)
        
        # Validate
        validation = self.validate(ir)
        if not validation.is_valid and self.config.validation_level == ValidationLevel.STRICT:
            raise DatalogRestrictionError('\n'.join(validation.errors))
        
        # Stratify if needed
        stratification = None
        if self.config.check_stratification:
            stratification = self.stratify(ir)
            if not stratification.is_stratifiable and self.config.validation_level == ValidationLevel.STRICT:
                raise StratificationError('\n'.join(stratification.errors))
        
        # Build code sections
        sections = []
        
        # Header comment
        if self.config.emit_comments:
            sections.append(self._emit_header(ir.get('module', 'unnamed')))
        
        # Facts
        facts, facts_count = self._emit_facts(ir)
        if facts:
            sections.append(facts)
        
        # Rules (ordered by stratum if stratified)
        rules, rules_count = self._emit_rules(ir, stratification)
        if rules:
            sections.append(rules)
        
        # Queries
        queries = self.logic_ext.extract_queries(ir)
        queries_count = len(queries)
        if queries:
            query_code = self._emit_queries(queries)
            if query_code:
                sections.append(query_code)
        
        code = '\n\n'.join(s for s in sections if s)
        code_hash = compute_sha256(code.encode('utf-8'))
        
        return EmitterResult(
            code=code,
            predicates=[f"{p}/{a}" for p, a in sorted(self._predicates.keys())],
            facts_count=facts_count,
            rules_count=rules_count,
            queries_count=queries_count,
            sha256=code_hash,
            emit_time=time.time() - start_time,
            stratification=stratification,
            validation=validation
        )
    
    def validate(self, ir: Dict[str, Any]) -> ValidationResult:
        """Validate IR against Datalog restrictions.
        
        Checks:
        1. No function symbols in heads
        2. Range restriction (head vars in positive body)
        3. No cut operator
        4. No side effects
        
        Args:
            ir: STUNIR Logic IR dictionary
            
        Returns:
            ValidationResult with errors/warnings
        """
        errors = []
        warnings = []
        restricted = []
        
        for pred_data in ir.get('predicates', []):
            for clause in pred_data.get('clauses', []):
                if clause.get('kind') == 'rule':
                    # Check head restriction
                    head_errors = self._check_head_restriction(clause.get('head', {}))
                    errors.extend(head_errors)
                    
                    # Check range restriction
                    if self.config.check_range_restriction:
                        range_errors = self._check_range_restriction_dict(clause)
                        errors.extend(range_errors)
                    
                    # Check for cut
                    for goal in clause.get('body', []):
                        if goal.get('kind') == 'cut':
                            errors.append("Cut operator not allowed in Datalog")
                            restricted.append("cut")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            restricted_terms=restricted
        )
    
    def stratify(self, ir: Dict[str, Any]) -> StratificationResult:
        """Compute stratification for negation.
        
        Algorithm:
        1. Build dependency graph: pred -> {(dep_pred, is_negated)}
        2. Find strongly connected components (SCCs)
        3. Check for negative cycles within SCCs (unstratifiable)
        4. Topologically sort SCCs to assign strata
        
        Args:
            ir: STUNIR Logic IR dictionary
            
        Returns:
            StratificationResult with stratum assignments
        """
        # Step 1: Build dependency graph
        deps = self._build_dependency_graph(ir)
        
        # Step 2: Find SCCs
        sccs = self._find_sccs(deps)
        
        # Step 3: Check for negative cycles
        errors = []
        for scc in sccs:
            if self._has_negative_cycle(scc, deps):
                errors.append(f"Unstratifiable: negative cycle in {{{', '.join(sorted(scc))}}}")
        
        if errors:
            return StratificationResult(
                strata={},
                is_stratifiable=False,
                dependency_graph=deps,
                errors=errors
            )
        
        # Step 4: Compute strata
        strata = self._compute_strata(deps, sccs)
        
        return StratificationResult(
            strata=strata,
            is_stratifiable=True,
            dependency_graph=deps,
            errors=[]
        )
    
    def _build_dependency_graph(self, ir: Dict[str, Any]) -> Dict[str, Set[Tuple[str, bool]]]:
        """Build predicate dependency graph.
        
        For each rule: head :- body
        Add edges: head depends on each predicate in body
        Mark edge as negative if predicate appears under negation.
        """
        deps: Dict[str, Set[Tuple[str, bool]]] = {}
        
        for pred_data in ir.get('predicates', []):
            pred_name = pred_data.get('name')
            if pred_name and pred_name not in deps:
                deps[pred_name] = set()
            
            for clause in pred_data.get('clauses', []):
                if clause.get('kind') == 'rule':
                    for goal in clause.get('body', []):
                        goal_pred = self._extract_predicate_name(goal)
                        is_negated = goal.get('kind') == 'negation'
                        if goal_pred and pred_name:
                            deps[pred_name].add((goal_pred, is_negated))
                            # Ensure the dependency is also in the graph
                            if goal_pred not in deps:
                                deps[goal_pred] = set()
        
        return deps
    
    def _extract_predicate_name(self, goal: Dict) -> Optional[str]:
        """Extract predicate name from a goal."""
        if goal.get('kind') == 'negation':
            inner = goal.get('inner', {})
            return self._extract_predicate_name(inner)
        
        term = goal.get('term', goal)
        if term.get('kind') == 'compound':
            return term.get('functor')
        if term.get('kind') == 'atom':
            return term.get('value')
        
        return None
    
    def _find_sccs(self, graph: Dict[str, Set[Tuple[str, bool]]]) -> List[Set[str]]:
        """Find strongly connected components using Tarjan's algorithm."""
        index_counter = [0]
        stack: List[str] = []
        lowlinks: Dict[str, int] = {}
        index: Dict[str, int] = {}
        on_stack: Dict[str, bool] = {}
        sccs: List[Set[str]] = []
        
        def strongconnect(v: str) -> None:
            index[v] = index_counter[0]
            lowlinks[v] = index_counter[0]
            index_counter[0] += 1
            on_stack[v] = True
            stack.append(v)
            
            for (w, _) in graph.get(v, set()):
                if w not in index:
                    strongconnect(w)
                    lowlinks[v] = min(lowlinks[v], lowlinks[w])
                elif on_stack.get(w, False):
                    lowlinks[v] = min(lowlinks[v], index[w])
            
            if lowlinks[v] == index[v]:
                scc: Set[str] = set()
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc.add(w)
                    if w == v:
                        break
                sccs.append(scc)
        
        for v in graph:
            if v not in index:
                strongconnect(v)
        
        return sccs
    
    def _has_negative_cycle(self, scc: Set[str], 
                           deps: Dict[str, Set[Tuple[str, bool]]]) -> bool:
        """Check if SCC contains a negative edge (unstratifiable)."""
        for pred in scc:
            for (dep, is_negated) in deps.get(pred, set()):
                if dep in scc and is_negated:
                    return True
        return False
    
    def _compute_strata(self, deps: Dict[str, Set[Tuple[str, bool]]],
                       sccs: List[Set[str]]) -> Dict[str, int]:
        """Compute stratum for each predicate."""
        strata: Dict[str, int] = {}
        
        # Map predicates to their SCC
        pred_to_scc: Dict[str, int] = {}
        for i, scc in enumerate(sccs):
            for pred in scc:
                pred_to_scc[pred] = i
        
        # Build SCC dependency graph
        scc_deps: Dict[int, Set[Tuple[int, bool]]] = {i: set() for i in range(len(sccs))}
        for pred, dependencies in deps.items():
            pred_scc = pred_to_scc.get(pred, -1)
            if pred_scc < 0:
                continue
            for (dep, is_negated) in dependencies:
                dep_scc = pred_to_scc.get(dep, -1)
                if dep_scc >= 0 and dep_scc != pred_scc:
                    scc_deps[pred_scc].add((dep_scc, is_negated))
        
        # Compute SCC strata (topological order + negation bumps)
        scc_strata: Dict[int, int] = {}
        visited: Set[int] = set()
        
        def compute_scc_stratum(scc_idx: int) -> int:
            if scc_idx in scc_strata:
                return scc_strata[scc_idx]
            
            if scc_idx in visited:
                return 0  # Cycle detected (shouldn't happen after SCC)
            
            visited.add(scc_idx)
            
            max_stratum = 0
            for (dep_scc, is_negated) in scc_deps.get(scc_idx, set()):
                dep_stratum = compute_scc_stratum(dep_scc)
                # Negation requires higher stratum
                if is_negated:
                    max_stratum = max(max_stratum, dep_stratum + 1)
                else:
                    max_stratum = max(max_stratum, dep_stratum)
            
            scc_strata[scc_idx] = max_stratum
            return max_stratum
        
        # Compute strata for all SCCs
        for i in range(len(sccs)):
            compute_scc_stratum(i)
        
        # Assign predicate strata from SCC strata
        for pred, scc_idx in pred_to_scc.items():
            strata[pred] = scc_strata.get(scc_idx, 0)
        
        return strata
    
    def _check_head_restriction(self, head: Dict) -> List[str]:
        """Check that head contains no function symbols.
        
        Valid heads: atoms, variables, simple predicates
        Invalid: compound terms in arguments
        """
        errors = []
        
        if head.get('kind') == 'compound':
            for arg in head.get('args', []):
                if arg.get('kind') == 'compound':
                    functor = arg.get('functor', 'unknown')
                    errors.append(
                        f"Function symbol '{functor}' not allowed in Datalog head"
                    )
                elif arg.get('kind') == 'list_term':
                    errors.append("List terms not allowed in Datalog head")
        
        return errors
    
    def _check_range_restriction_dict(self, clause: Dict) -> List[str]:
        """Check range restriction from IR dictionary.
        
        Every head variable must appear in a positive body literal.
        """
        errors = []
        
        # Get head variables
        head_vars = self._get_variables_from_dict(clause.get('head', {}))
        
        # Get variables from positive body literals only
        positive_vars: Set[str] = set()
        for goal in clause.get('body', []):
            if goal.get('kind') != 'negation':
                positive_vars.update(self._get_variables_from_dict(goal))
        
        # Check restriction
        unbound = head_vars - positive_vars
        # Filter out anonymous variables
        unbound = {v for v in unbound if not v.startswith('_')}
        
        if unbound:
            errors.append(
                f"Variables {{{', '.join(sorted(unbound))}}} in head but not in "
                f"positive body (range restriction violated)"
            )
        
        return errors
    
    def _get_variables_from_dict(self, term: Dict) -> Set[str]:
        """Extract variable names from IR dictionary term."""
        variables: Set[str] = set()
        
        kind = term.get('kind')
        
        if kind == 'variable':
            name = term.get('name', '')
            if name:
                variables.add(name)
        elif kind == 'compound':
            for arg in term.get('args', []):
                variables.update(self._get_variables_from_dict(arg))
        elif kind == 'list_term':
            for elem in term.get('elements', []):
                variables.update(self._get_variables_from_dict(elem))
            if term.get('tail'):
                variables.update(self._get_variables_from_dict(term['tail']))
        elif kind in ('call', 'negation'):
            inner = term.get('term') or term.get('inner', {})
            if inner:
                variables.update(self._get_variables_from_dict(inner))
        
        return variables
    
    def _emit_header(self, module_name: str) -> str:
        """Generate header comment."""
        lines = [
            "%",
            "% STUNIR Generated Datalog Program",
            f"% Module: {module_name}",
        ]
        
        if self.config.emit_timestamps:
            lines.append(f"% Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        lines.extend([
            "% ",
            "% This file was automatically generated by STUNIR.",
            "% Datalog: Bottom-up evaluation, stratified negation",
            "%"
        ])
        
        return '\n'.join(lines)
    
    def _emit_facts(self, ir: Dict[str, Any]) -> Tuple[str, int]:
        """Emit all facts."""
        lines = []
        count = 0
        
        if self.config.emit_comments:
            lines.append("% Facts")
        
        for pred_data in ir.get('predicates', []):
            for clause in pred_data.get('clauses', []):
                if clause.get('kind') == 'fact':
                    fact_line = self._emit_fact_dict(clause)
                    if fact_line:
                        lines.append(fact_line)
                        count += 1
        
        if count == 0:
            return '', 0
        
        return '\n'.join(lines), count
    
    def _emit_fact_dict(self, fact: Dict) -> str:
        """Emit a single fact from dictionary."""
        pred = fact.get('predicate', 'unknown')
        args = fact.get('args', [])
        
        if not args:
            return f"{escape_atom(pred)}."
        
        arg_strs = [self._emit_term_dict(a) for a in args]
        return f"{escape_atom(pred)}({', '.join(arg_strs)})."
    
    def _emit_rules(self, ir: Dict[str, Any], 
                   stratification: Optional[StratificationResult]) -> Tuple[str, int]:
        """Emit all rules, ordered by stratum if available."""
        sections = []
        count = 0
        
        # Collect rules by predicate
        rules_by_pred: Dict[str, List[Dict]] = {}
        for pred_data in ir.get('predicates', []):
            pred_name = pred_data.get('name', 'unknown')
            for clause in pred_data.get('clauses', []):
                if clause.get('kind') == 'rule':
                    if pred_name not in rules_by_pred:
                        rules_by_pred[pred_name] = []
                    rules_by_pred[pred_name].append(clause)
        
        # Order by stratum if stratification available
        if stratification and stratification.is_stratifiable:
            sorted_preds = sorted(
                rules_by_pred.keys(),
                key=lambda p: stratification.strata.get(p, 0)
            )
        else:
            sorted_preds = sorted(rules_by_pred.keys())
        
        for pred in sorted_preds:
            rules = rules_by_pred[pred]
            pred_lines = []
            
            if self.config.emit_comments and stratification and stratification.is_stratifiable:
                stratum = stratification.strata.get(pred, 0)
                pred_lines.append(f"% Stratum {stratum}: {pred}")
            
            for rule in rules:
                rule_line = self._emit_rule_dict(rule)
                if rule_line:
                    pred_lines.append(rule_line)
                    count += 1
            
            if pred_lines:
                sections.append('\n'.join(pred_lines))
        
        if count == 0:
            return '', 0
        
        return '\n\n'.join(sections), count
    
    def _emit_rule_dict(self, rule: Dict) -> str:
        """Emit a rule from dictionary."""
        head = rule.get('head', {})
        body = rule.get('body', [])
        
        head_str = self._emit_term_dict(head)
        
        if not body:
            return f"{head_str}."
        
        body_parts = [self._emit_goal_dict(g) for g in body]
        body_str = ', '.join(body_parts)
        
        return f"{head_str} :- {body_str}."
    
    def _emit_goal_dict(self, goal: Dict) -> str:
        """Emit a goal from dictionary."""
        kind = goal.get('kind')
        
        if kind == 'negation':
            inner = goal.get('inner', {})
            inner_str = self._emit_goal_dict(inner)
            return f"not {inner_str}"
        
        if kind == 'call':
            term = goal.get('term', {})
            return self._emit_term_dict(term)
        
        if kind == 'conjunction':
            goals = goal.get('goals', [])
            parts = [self._emit_goal_dict(g) for g in goals]
            return ', '.join(parts)
        
        if kind == 'disjunction':
            goals = goal.get('goals', [])
            parts = [self._emit_goal_dict(g) for g in goals]
            return f"({'; '.join(parts)})"
        
        # Treat as term
        return self._emit_term_dict(goal)
    
    def _emit_term_dict(self, term: Dict) -> str:
        """Emit a term from dictionary."""
        kind = term.get('kind')
        
        if kind == 'variable':
            name = term.get('name', '_')
            return format_variable(name)
        
        if kind == 'atom':
            value = term.get('value', '')
            return escape_atom(value)
        
        if kind == 'number':
            value = term.get('value', 0)
            return str(value)
        
        if kind == 'string_term':
            value = term.get('value', '')
            return f'"{escape_string(value)}"'
        
        if kind == 'compound':
            functor = term.get('functor', 'unknown')
            args = term.get('args', [])
            if not args:
                return escape_atom(functor)
            arg_strs = [self._emit_term_dict(a) for a in args]
            return f"{escape_atom(functor)}({', '.join(arg_strs)})"
        
        if kind == 'anonymous':
            return '_'
        
        # Fallback for direct values
        if isinstance(term, (int, float)):
            return str(term)
        if isinstance(term, str):
            return escape_atom(term)
        
        return str(term)
    
    def _emit_queries(self, queries: List[Query]) -> str:
        """Emit queries."""
        if not queries:
            return ''
        
        lines = []
        if self.config.emit_comments:
            lines.append("% Queries")
        
        for query in queries:
            goals_str = ', '.join(self._emit_goal(g) for g in query.goals)
            lines.append(f"?- {goals_str}.")
        
        return '\n'.join(lines)
    
    def _emit_goal(self, goal: Goal) -> str:
        """Emit a Goal object."""
        if goal.kind == GoalKind.NEGATION:
            inner = self._emit_term(goal.term) if goal.term else ''
            return f"not {inner}"
        
        if goal.term:
            return self._emit_term(goal.term)
        
        return ''
    
    def _emit_term(self, term: Term) -> str:
        """Emit a Term object."""
        if isinstance(term, Variable):
            return format_variable(term.name)
        
        if isinstance(term, Atom):
            return escape_atom(term.value)
        
        if isinstance(term, Number):
            return str(term.value)
        
        if isinstance(term, StringTerm):
            return f'"{escape_string(term.value)}"'
        
        if isinstance(term, Compound):
            if not term.args:
                return escape_atom(term.functor)
            args = ', '.join(self._emit_term(a) for a in term.args)
            return f"{escape_atom(term.functor)}({args})"
        
        if isinstance(term, Anonymous):
            return '_'
        
        if isinstance(term, ListTerm):
            # Lists are restricted in Datalog but allowed in bodies
            elements = ', '.join(self._emit_term(e) for e in term.elements)
            if term.tail:
                tail = self._emit_term(term.tail)
                return f"[{elements}|{tail}]"
            return f"[{elements}]"
        
        return str(term)
    
    def emit_to_file(self, ir: Dict[str, Any], output_path: Path) -> EmitterResult:
        """Emit Datalog code and write to file.
        
        Args:
            ir: STUNIR Logic IR dictionary
            output_path: Path to output file
            
        Returns:
            EmitterResult with generated code
        """
        result = self.emit(ir)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result.code, encoding='utf-8')
        
        return result


__all__ = [
    'DatalogEmitter',
    'DatalogConfig',
    'EmitterResult',
    'StratificationResult',
    'ValidationResult',
    'ValidationLevel',
    'DatalogRestrictionError',
    'StratificationError',
]
