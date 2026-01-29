#!/usr/bin/env python3
"""STUNIR XSB Prolog Emitter.

Generates XSB Prolog code from STUNIR Logic IR.
Supports advanced tabling features including incremental tabling,
answer subsumption, well-founded semantics, and lattice operations.

Key XSB features:
- Advanced tabling (more sophisticated than YAP)
- Incremental evaluation (`:- table pred/N as incremental`)
- Answer subsumption (`:- table pred/N as subsumptive`)
- Well-founded semantics (WFS) for negation
- Lattice tabling for aggregation
- Different module syntax (`:- export(...)`, `:- import(...)`)
- Trie-based indexing

Part of Phase 5D-1: XSB with Advanced Tabling.
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
    XSBPrologTypeMapper, XSB_PROLOG_TYPES, XSB_TABLING_MODES,
    XSB_LATTICE_OPS, TablingMode
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
    module_name: str
    predicates: List[str]
    sha256: str
    emit_time: float
    tabled_predicates: List[str] = field(default_factory=list)
    incremental_predicates: List[str] = field(default_factory=list)
    subsumptive_predicates: List[str] = field(default_factory=list)
    wfs_predicates: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'code': self.code,
            'module_name': self.module_name,
            'predicates': self.predicates,
            'sha256': self.sha256,
            'emit_time': self.emit_time,
            'tabled_predicates': self.tabled_predicates,
            'incremental_predicates': self.incremental_predicates,
            'subsumptive_predicates': self.subsumptive_predicates,
            'wfs_predicates': self.wfs_predicates,
        }


@dataclass
class XSBPrologConfig:
    """Configuration for XSB Prolog emitter."""
    module_prefix: str = "stunir"
    emit_exports: bool = True
    emit_comments: bool = True
    emit_type_hints: bool = True  # Comment-style type hints
    indent: str = "    "
    line_width: int = 80
    # XSB-specific options
    enable_tabling: bool = True            # Auto-table recursive predicates
    default_tabling_mode: str = 'variant'  # variant, incremental, subsumptive
    auto_incremental: bool = True          # Auto-detect incremental candidates
    auto_subsumptive: bool = False         # Auto-detect subsumptive candidates
    enable_wfs: bool = True                # Enable well-founded semantics hints
    max_answers: Optional[int] = None      # Default max answers (None = unlimited)
    # Formatting
    emit_timestamps: bool = True
    emit_sha256: bool = True
    file_extension: str = ".P"             # XSB convention: .P (uppercase)


@dataclass
class TablingSpec:
    """Specification for a tabled predicate."""
    name: str
    arity: int
    mode: TablingMode = TablingMode.VARIANT
    additional_modes: List[TablingMode] = field(default_factory=list)
    lattice_op: Optional[str] = None
    max_answers: Optional[int] = None
    
    def to_directive(self, mapper: XSBPrologTypeMapper) -> str:
        """Generate XSB tabling directive."""
        return mapper.format_tabling_directive(
            self.name, self.arity, self.mode,
            self.lattice_op, self.max_answers, self.additional_modes
        )


class XSBPrologEmitter:
    """Emitter for XSB Prolog code.
    
    Generates valid XSB Prolog code from STUNIR Logic IR including:
    - Export/import directives (XSB module syntax)
    - Predicate declarations
    - Facts and rules
    - Advanced tabling directives:
      - Standard tabling
      - Incremental tabling
      - Subsumptive tabling
      - Lattice tabling
    - Well-founded semantics support
    - DCG rules
    - Queries
    """
    
    DIALECT = "xsb-prolog"
    FILE_EXTENSION = ".P"
    
    def __init__(self, config: Optional[XSBPrologConfig] = None):
        """Initialize XSB Prolog emitter.
        
        Args:
            config: Emitter configuration (optional)
        """
        self.config = config or XSBPrologConfig()
        self.logic_ext = LogicIRExtension()
        self.type_mapper = XSBPrologTypeMapper(self.config.emit_type_hints)
        self._exports: Set[Tuple[str, int]] = set()
        self._imports: Dict[Tuple[str, int], str] = {}  # (pred, arity) -> module
        self._dynamics: Set[Tuple[str, int]] = set()
        self._predicates: Dict[Tuple[str, int], Predicate] = {}
        self._multifile: Set[Tuple[str, int]] = set()
        self._discontiguous: Set[Tuple[str, int]] = set()
        self._tabling_specs: Dict[Tuple[str, int], TablingSpec] = {}
        self._wfs_predicates: Set[Tuple[str, int]] = set()
    
    def emit(self, ir: Dict[str, Any]) -> EmitterResult:
        """Emit XSB Prolog code from Logic IR.
        
        Args:
            ir: STUNIR Logic IR dictionary
            
        Returns:
            EmitterResult with generated code
        """
        start_time = time.time()
        
        module_name = ir.get('module', 'unnamed')
        xsb_module = f"{self.config.module_prefix}_{self._xsb_name(module_name)}"
        
        # Reset state
        self._exports = set()
        self._imports = {}
        self._dynamics = set()
        self._multifile = set()
        self._discontiguous = set()
        self._tabling_specs = {}
        self._wfs_predicates = set()
        
        # Extract predicates and analyze
        self._predicates = self.logic_ext.extract_predicates(ir)
        self._analyze_exports(ir)
        self._analyze_imports(ir)
        self._analyze_dynamics(ir)
        self._analyze_tabling(ir)
        self._analyze_wfs(ir)
        
        # Build code sections
        sections = []
        
        # Header comment
        if self.config.emit_comments:
            sections.append(self._emit_header(module_name))
        
        # Export directives (XSB-specific)
        if self.config.emit_exports:
            exports = self._emit_exports()
            if exports:
                sections.append(exports)
        
        # Import directives (XSB-specific)
        imports = self._emit_imports()
        if imports:
            sections.append(imports)
        
        # Directives (dynamic, multifile, etc.)
        directives = self._emit_directives()
        if directives:
            sections.append(directives)
        
        # Tabling directives
        tabling = self._emit_tabling_directives()
        if tabling:
            sections.append(tabling)
        
        # Type declarations (comments)
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
        
        # Collect tabling statistics
        tabled = [f"{p}/{a}" for (p, a) in sorted(self._tabling_specs.keys())]
        incremental = [
            f"{p}/{a}" for (p, a), spec in self._tabling_specs.items()
            if spec.mode == TablingMode.INCREMENTAL or 
               TablingMode.INCREMENTAL in spec.additional_modes
        ]
        subsumptive = [
            f"{p}/{a}" for (p, a), spec in self._tabling_specs.items()
            if spec.mode == TablingMode.SUBSUMPTIVE or 
               TablingMode.SUBSUMPTIVE in spec.additional_modes
        ]
        wfs = [f"{p}/{a}" for p, a in sorted(self._wfs_predicates)]
        
        return EmitterResult(
            code=code,
            module_name=xsb_module,
            predicates=[f"{p}/{a}" for p, a in sorted(self._predicates.keys())],
            sha256=code_hash,
            emit_time=time.time() - start_time,
            tabled_predicates=tabled,
            incremental_predicates=incremental,
            subsumptive_predicates=subsumptive,
            wfs_predicates=wfs
        )
    
    def _xsb_name(self, name: str) -> str:
        """Convert name to valid XSB Prolog identifier.
        
        Args:
            name: Original name
            
        Returns:
            Valid XSB Prolog identifier
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
        
        # Auto-export public predicates (not starting with _)
        for (name, arity), pred in self._predicates.items():
            if not name.startswith('_'):
                self._exports.add((name, arity))
    
    def _analyze_imports(self, ir: Dict[str, Any]) -> None:
        """Analyze imports from IR."""
        for imp in ir.get('imports', []):
            pred = imp.get('predicate')
            arity = imp.get('arity', 0)
            module = imp.get('from', imp.get('module', 'basics'))
            if pred:
                self._imports[(pred, arity)] = module
    
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
    
    def _analyze_tabling(self, ir: Dict[str, Any]) -> None:
        """Analyze predicates for tabling.
        
        Detects recursive predicates, explicit tabling annotations,
        and determines optimal tabling mode (incremental, subsumptive, etc.)
        """
        # Process explicit tabling annotations
        for tab in ir.get('tabled', []):
            pred = tab.get('predicate')
            arity = tab.get('arity', 0)
            if pred:
                mode_str = tab.get('mode', self.config.default_tabling_mode)
                mode = self.type_mapper.get_tabling_mode(mode_str)
                
                # Check for additional modes
                add_modes = []
                for m in tab.get('additional_modes', []):
                    add_modes.append(self.type_mapper.get_tabling_mode(m))
                
                spec = TablingSpec(
                    name=pred,
                    arity=arity,
                    mode=mode,
                    additional_modes=add_modes,
                    lattice_op=tab.get('lattice_op'),
                    max_answers=tab.get('max_answers', self.config.max_answers)
                )
                self._tabling_specs[(pred, arity)] = spec
        
        # Auto-detect tabling candidates if enabled
        if self.config.enable_tabling:
            for (name, arity), pred in self._predicates.items():
                if (name, arity) not in self._tabling_specs:
                    if self._is_recursive(pred):
                        mode = self._detect_tabling_mode(pred, ir)
                        spec = TablingSpec(name=name, arity=arity, mode=mode)
                        
                        # Add incremental for dynamic predicates
                        if self.config.auto_incremental and (name, arity) in self._dynamics:
                            if mode != TablingMode.INCREMENTAL:
                                spec.additional_modes.append(TablingMode.INCREMENTAL)
                        
                        self._tabling_specs[(name, arity)] = spec
    
    def _analyze_wfs(self, ir: Dict[str, Any]) -> None:
        """Analyze predicates for well-founded semantics.
        
        Detects predicates with negation that may benefit from WFS.
        """
        if not self.config.enable_wfs:
            return
        
        for (name, arity), pred in self._predicates.items():
            if self._has_negation(pred):
                self._wfs_predicates.add((name, arity))
    
    def _is_recursive(self, pred: Predicate) -> bool:
        """Check if predicate is recursive (directly or indirectly).
        
        Args:
            pred: Predicate to check
            
        Returns:
            True if predicate calls itself
        """
        for clause in pred.clauses:
            if isinstance(clause, Rule):
                for goal in clause.body:
                    if goal.term and isinstance(goal.term, Compound):
                        if goal.term.functor == pred.name:
                            return True
                    # Check nested goals
                    if goal.goals:
                        for nested in goal.goals:
                            if nested.term and isinstance(nested.term, Compound):
                                if nested.term.functor == pred.name:
                                    return True
        return False
    
    def _has_negation(self, pred: Predicate) -> bool:
        """Check if predicate uses negation.
        
        Args:
            pred: Predicate to check
            
        Returns:
            True if predicate contains negation
        """
        for clause in pred.clauses:
            if isinstance(clause, Rule):
                for goal in clause.body:
                    if goal.kind == GoalKind.NEGATION:
                        return True
                    # Check nested goals
                    if goal.goals:
                        for nested in goal.goals:
                            if nested.kind == GoalKind.NEGATION:
                                return True
        return False
    
    def _detect_tabling_mode(self, pred: Predicate, ir: Dict[str, Any]) -> TablingMode:
        """Detect optimal tabling mode for a predicate.
        
        Args:
            pred: Predicate to analyze
            ir: Full IR for context
            
        Returns:
            Optimal TablingMode
        """
        name, arity = pred.name, pred.arity
        
        # Dynamic predicates benefit from incremental tabling
        if self.config.auto_incremental and (name, arity) in self._dynamics:
            return TablingMode.INCREMENTAL
        
        # Predicates with negation may need careful handling (WFS)
        if self._has_negation(pred):
            return TablingMode.VARIANT
        
        # Subsumptive for certain patterns (if enabled)
        if self.config.auto_subsumptive:
            # Could analyze for subsumption patterns here
            pass
        
        return TablingMode.VARIANT
    
    def _emit_header(self, module_name: str) -> str:
        """Generate header comment."""
        lines = [
            "/*",
            " * STUNIR Generated XSB Prolog Module",
            f" * Module: {module_name}",
        ]
        
        if self.config.emit_timestamps:
            lines.append(f" * Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self._tabling_specs:
            lines.append(f" * Tabled predicates: {len(self._tabling_specs)}")
            
            # Count by mode
            incremental = sum(1 for s in self._tabling_specs.values() 
                            if s.mode == TablingMode.INCREMENTAL or 
                               TablingMode.INCREMENTAL in s.additional_modes)
            subsumptive = sum(1 for s in self._tabling_specs.values()
                            if s.mode == TablingMode.SUBSUMPTIVE or
                               TablingMode.SUBSUMPTIVE in s.additional_modes)
            if incremental:
                lines.append(f" *   - Incremental: {incremental}")
            if subsumptive:
                lines.append(f" *   - Subsumptive: {subsumptive}")
        
        if self._wfs_predicates:
            lines.append(f" * WFS predicates: {len(self._wfs_predicates)}")
        
        lines.extend([
            " * ",
            " * This file was automatically generated by STUNIR.",
            " * XSB Prolog target with advanced tabling support.",
            " * Do not edit manually.",
            " */"
        ])
        
        return '\n'.join(lines)
    
    def _emit_exports(self) -> str:
        """Generate export directives (XSB-specific syntax)."""
        if not self._exports:
            return ''
        
        lines = ['% Exports']
        for pred, arity in sorted(self._exports):
            lines.append(f":- export({pred}/{arity}).")
        
        return '\n'.join(lines)
    
    def _emit_imports(self) -> str:
        """Generate import directives (XSB-specific syntax)."""
        if not self._imports:
            return ''
        
        lines = ['% Imports']
        for (pred, arity), module in sorted(self._imports.items()):
            lines.append(f":- import({pred}/{arity} from {module}).")
        
        return '\n'.join(lines)
    
    def _emit_directives(self) -> str:
        """Emit directives for dynamic, multifile, etc."""
        lines = []
        
        # Dynamic predicates
        if self._dynamics:
            # Check if any dynamic predicates are also incremental
            incr_dynamics = []
            regular_dynamics = []
            for pred, arity in sorted(self._dynamics):
                if (pred, arity) in self._tabling_specs:
                    spec = self._tabling_specs[(pred, arity)]
                    if (spec.mode == TablingMode.INCREMENTAL or 
                        TablingMode.INCREMENTAL in spec.additional_modes):
                        incr_dynamics.append((pred, arity))
                        continue
                regular_dynamics.append((pred, arity))
            
            for pred, arity in regular_dynamics:
                lines.append(f":- dynamic {pred}/{arity}.")
            
            # Incremental dynamics need special handling
            for pred, arity in incr_dynamics:
                lines.append(f":- dynamic {pred}/{arity} as incremental.")
        
        # Multifile predicates
        for pred, arity in sorted(self._multifile):
            lines.append(f":- multifile {pred}/{arity}.")
        
        # Discontiguous predicates
        for pred, arity in sorted(self._discontiguous):
            lines.append(f":- discontiguous {pred}/{arity}.")
        
        return '\n'.join(lines)
    
    def _emit_tabling_directives(self) -> str:
        """Emit tabling directives with XSB-specific modes.
        
        Generates:
        - Standard tabling: :- table pred/N.
        - Incremental: :- table pred/N as incremental.
        - Subsumptive: :- table pred/N as subsumptive.
        - Combined: :- table pred/N as (incremental, subsumptive).
        - Lattice: :- table pred(_,_,lattice(min/3)).
        """
        if not self._tabling_specs:
            return ''
        
        lines = ['% Tabling directives (XSB advanced tabling)']
        
        for (pred, arity), spec in sorted(self._tabling_specs.items()):
            directive = spec.to_directive(self.type_mapper)
            
            # Add comment for WFS predicates
            if (pred, arity) in self._wfs_predicates:
                lines.append(f"% WFS: {pred}/{arity} uses negation - well-founded semantics applies")
            
            lines.append(directive)
        
        return '\n'.join(lines)
    
    def _emit_pred_declarations(self) -> str:
        """Emit pred declarations for documentation."""
        lines = []
        
        for (name, arity), pred in sorted(self._predicates.items()):
            marks = []
            if (name, arity) in self._tabling_specs:
                spec = self._tabling_specs[(name, arity)]
                marks.append(spec.mode.name.lower())
                for m in spec.additional_modes:
                    marks.append(m.name.lower())
            if (name, arity) in self._wfs_predicates:
                marks.append('wfs')
            
            mark_str = f" [{', '.join(marks)}]" if marks else ''
            lines.append(f"%% {name}/{arity}{mark_str}")
        
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
        """Convert term to XSB Prolog syntax.
        
        Args:
            term: Term object or dictionary
            
        Returns:
            XSB Prolog syntax string
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
                return tail_str
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
            reserved = {'is', 'mod', 'rem', 'not', 'true', 'false', 'fail',
                       'tnot', 'table', 'import', 'export'}
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
            # XSB uses tnot for tabled negation, \+ for standard NAF
            if goal.goals:
                inner = self._emit_goal(goal.goals[0])
            elif goal.term:
                inner = self._emit_term(goal.term)
            else:
                inner = 'true'
            # Use tnot for tabled predicates
            if goal.term and isinstance(goal.term, Compound):
                pred_key = (goal.term.functor, goal.term.arity)
                if pred_key in self._tabling_specs:
                    return f"tnot({inner})"
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
            then_part = 'true'
            else_part = 'fail'
            return f"({cond} -> {then_part} ; {else_part})"
        elif goal.term:
            return self._emit_term(goal.term)
        else:
            return 'true'
    
    def _emit_dcg_rules(self, dcg_rules: List[Dict[str, Any]]) -> str:
        """Emit DCG rules using --> syntax."""
        lines = ['% DCG Rules']
        
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
                        goal_strs = [self._emit_goal(_parse_goal(g)) for g in goals]
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
        lines = ["% Queries"]
        
        for query in queries:
            if not query.goals:
                continue
            
            goals = ', '.join(self._emit_goal(g) for g in query.goals)
            # XSB uses :- for initialization
            lines.append(f":- {goals}.")
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


__all__ = ['XSBPrologEmitter', 'XSBPrologConfig', 'EmitterResult', 
           'TablingSpec', 'compute_sha256', 'canonical_json']
