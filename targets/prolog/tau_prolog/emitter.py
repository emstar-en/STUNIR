#!/usr/bin/env python3
"""STUNIR Tau Prolog Emitter.

Generates Tau Prolog code from STUNIR Logic IR.
Tau Prolog is a JavaScript-based Prolog implementation that runs
in web browsers and Node.js with DOM manipulation and JS interop.

Part of Phase 5D-4: Extended Prolog Targets (Tau Prolog).
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
    TauPrologTypeMapper,
    TAU_PROLOG_TYPES,
    DOM_PREDICATES,
    JS_PREDICATES,
    TAU_LIBRARIES,
)


def compute_sha256(data: bytes) -> str:
    """Compute SHA-256 hash of data."""
    return hashlib.sha256(data).hexdigest()


def canonical_json(data: Any) -> str:
    """Produce deterministic JSON output."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'))


@dataclass
class TauPrologEmitterResult:
    """Result of Tau Prolog code emission."""
    code: str
    module_name: str
    predicates: List[str]
    sha256: str
    emit_time: float
    
    # Tau Prolog specific
    libraries_used: List[str]
    has_dom: bool
    has_js_interop: bool
    target_runtime: str
    loader_js: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'code': self.code,
            'module_name': self.module_name,
            'predicates': self.predicates,
            'sha256': self.sha256,
            'emit_time': self.emit_time,
            'libraries_used': self.libraries_used,
            'has_dom': self.has_dom,
            'has_js_interop': self.has_js_interop,
            'target_runtime': self.target_runtime,
            'loader_js': self.loader_js,
        }


@dataclass
class TauPrologConfig:
    """Configuration for Tau Prolog emitter."""
    module_prefix: str = "stunir"
    emit_module: bool = True
    emit_comments: bool = True
    emit_type_hints: bool = True
    indent: str = "    "
    line_width: int = 80
    
    # Tau Prolog specific options
    enable_dom: bool = False          # Include DOM predicates
    enable_js_interop: bool = True    # Enable JavaScript interop
    target_runtime: str = "browser"   # "browser" or "node"
    emit_loader_js: bool = False      # Emit JavaScript loader code
    
    # Library options
    auto_include_libraries: bool = True
    default_libraries: List[str] = field(default_factory=lambda: ['lists'])
    
    # Output options
    emit_timestamps: bool = True
    emit_sha256: bool = True


class TauPrologEmitter:
    """Emitter for Tau Prolog code.
    
    Generates valid Tau Prolog code from STUNIR Logic IR including:
    - Module definitions with exports
    - Library imports (use_module)
    - Predicate declarations
    - Facts and rules
    - DCG rules
    - JavaScript interop predicates
    - DOM manipulation predicates (browser mode)
    
    Tau Prolog runs in JavaScript environments (browsers, Node.js)
    and provides unique features for web integration.
    """
    
    DIALECT = "tau-prolog"
    FILE_EXTENSION = ".pl"
    
    def __init__(self, config: Optional[TauPrologConfig] = None):
        """Initialize Tau Prolog emitter.
        
        Args:
            config: Emitter configuration (optional)
        """
        self.config = config or TauPrologConfig()
        self.logic_ext = LogicIRExtension()
        self.type_mapper = TauPrologTypeMapper(self.config.enable_js_interop)
        self._exports: Set[Tuple[str, int]] = set()
        self._dynamics: Set[Tuple[str, int]] = set()
        self._predicates: Dict[Tuple[str, int], Predicate] = {}
        self._used_predicates: Set[str] = set()
        self._libraries_needed: Set[str] = set()
    
    def emit(self, ir: Dict[str, Any]) -> TauPrologEmitterResult:
        """Emit Tau Prolog code from Logic IR.
        
        Args:
            ir: STUNIR Logic IR dictionary
            
        Returns:
            TauPrologEmitterResult with generated code
        """
        start_time = time.time()
        
        module_name = ir.get('module', 'unnamed')
        prolog_module = f"{self.config.module_prefix}_{self._prolog_name(module_name)}"
        
        # Reset state
        self._exports = set()
        self._dynamics = set()
        self._used_predicates = set()
        self._libraries_needed = set(self.config.default_libraries)
        
        # Extract predicates and analyze
        self._predicates = self.logic_ext.extract_predicates(ir)
        self._analyze_exports(ir)
        self._analyze_dynamics(ir)
        self._analyze_used_predicates(ir)
        
        # Build code sections
        sections = []
        
        # Header comment
        if self.config.emit_comments:
            sections.append(self._emit_header(module_name))
        
        # Module declaration
        if self.config.emit_module:
            sections.append(self._emit_module(prolog_module))
        
        # Library imports
        lib_imports = self._emit_library_imports()
        if lib_imports:
            sections.append(lib_imports)
        
        # Directives (dynamic, etc.)
        directives = self._emit_directives()
        if directives:
            sections.append(directives)
        
        # Type declarations
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
        
        # Queries (as initialization)
        queries = self.logic_ext.extract_queries(ir)
        if queries:
            query_code = self._emit_queries(queries)
            if query_code:
                sections.append(query_code)
        
        code = '\n\n'.join(s for s in sections if s)
        code_hash = compute_sha256(code.encode('utf-8'))
        
        # Check for DOM and JS interop usage
        has_dom = any(p in DOM_PREDICATES for p in self._used_predicates)
        has_js_interop = any(p in JS_PREDICATES for p in self._used_predicates)
        
        # Generate JavaScript loader if requested
        loader_js = None
        if self.config.emit_loader_js:
            loader_js = self._generate_js_loader(prolog_module, code)
        
        return TauPrologEmitterResult(
            code=code,
            module_name=prolog_module,
            predicates=[f"{p}/{a}" for p, a in sorted(self._predicates.keys())],
            sha256=code_hash,
            emit_time=time.time() - start_time,
            libraries_used=sorted(self._libraries_needed),
            has_dom=has_dom,
            has_js_interop=has_js_interop,
            target_runtime=self.config.target_runtime,
            loader_js=loader_js,
        )
    
    def _prolog_name(self, name: str) -> str:
        """Convert name to valid Prolog identifier."""
        result = ''.join(c if c.isalnum() or c == '_' else '_' for c in name)
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
    
    def _analyze_dynamics(self, ir: Dict[str, Any]) -> None:
        """Analyze dynamic predicate declarations."""
        for dyn in ir.get('dynamic', []):
            pred = dyn.get('predicate')
            arity = dyn.get('arity', 0)
            if pred:
                self._dynamics.add((pred, arity))
    
    def _analyze_used_predicates(self, ir: Dict[str, Any]) -> None:
        """Analyze predicates used to determine library needs."""
        self._used_predicates = set()
        
        def extract_functor(term_data: Any) -> None:
            if isinstance(term_data, dict):
                functor = term_data.get('functor')
                if functor:
                    self._used_predicates.add(functor)
                # Recurse into args
                for arg in term_data.get('args', []):
                    extract_functor(arg)
                # Recurse into body
                for body_item in term_data.get('body', []):
                    extract_functor(body_item)
                # Recurse into term
                if 'term' in term_data:
                    extract_functor(term_data['term'])
        
        # Analyze all clauses
        for clause in ir.get('clauses', []):
            extract_functor(clause)
        
        # Analyze predicates
        for pred_data in ir.get('predicates', []):
            for clause in pred_data.get('clauses', []):
                extract_functor(clause)
        
        # Determine libraries needed
        if self.config.auto_include_libraries:
            detected_libs = self.type_mapper.get_required_libraries(self._used_predicates)
            self._libraries_needed.update(detected_libs)
            
            # Add dom library if DOM predicates detected and enabled
            if self.config.enable_dom and any(p in DOM_PREDICATES for p in self._used_predicates):
                self._libraries_needed.add('dom')
            
            # Add js library if JS interop predicates detected
            if self.config.enable_js_interop and any(p in JS_PREDICATES for p in self._used_predicates):
                self._libraries_needed.add('js')
    
    def _emit_header(self, module_name: str) -> str:
        """Generate header comment."""
        lines = [
            "/*",
            " * STUNIR Generated Tau Prolog Module",
            f" * Module: {module_name}",
        ]
        
        if self.config.emit_timestamps:
            lines.append(f" * Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        lines.append(f" * Target Runtime: {self.config.target_runtime}")
        
        if self._libraries_needed:
            lines.append(f" * Libraries: {', '.join(sorted(self._libraries_needed))}")
        
        lines.extend([
            " * ",
            " * This file was automatically generated by STUNIR.",
            " * Tau Prolog: JavaScript-based Prolog for browsers and Node.js",
            " * https://tau-prolog.org/",
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
    
    def _emit_library_imports(self) -> str:
        """Emit use_module directives for required libraries."""
        lines = []
        
        for lib in sorted(self._libraries_needed):
            if lib in TAU_LIBRARIES:
                lines.append(f":- use_module(library({lib})).")
        
        return '\n'.join(lines)
    
    def _emit_directives(self) -> str:
        """Emit directives for dynamic predicates."""
        lines = []
        
        # Dynamic predicates
        for pred, arity in sorted(self._dynamics):
            lines.append(f":- dynamic({pred}/{arity}).")
        
        return '\n'.join(lines)
    
    def _emit_pred_declarations(self) -> str:
        """Emit pred declarations for documentation."""
        lines = []
        
        for (name, arity), pred in sorted(self._predicates.items()):
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
        """Convert term to Prolog syntax."""
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
        """Escape atom if needed."""
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
            # As comment for interactive use (Tau Prolog uses JS API for queries)
            lines.append(f"% ?- {goals}.")
        
        return '\n'.join(lines)
    
    def _generate_js_loader(self, module_name: str, prolog_code: str) -> str:
        """Generate JavaScript loader code for the Prolog program."""
        # Escape the Prolog code for JS string
        escaped_code = prolog_code.replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$')
        
        libs_to_load = []
        for lib in sorted(self._libraries_needed):
            if lib in TAU_LIBRARIES:
                if self.config.target_runtime == 'browser':
                    libs_to_load.append(f'    // Load {lib} module from CDN')
                    libs_to_load.append(f'    // <script src="https://cdn.tau-prolog.org/modules/{lib}.js"></script>')
                else:
                    libs_to_load.append(f'    require("tau-prolog/modules/{lib}")(pl);')
        
        if self.config.target_runtime == 'browser':
            return f'''// STUNIR Generated Tau Prolog Loader (Browser)
// Module: {module_name}

(function() {{
    // Initialize Tau Prolog session
    const session = pl.create();
    
{chr(10).join(libs_to_load)}
    
    // STUNIR Generated Prolog Program
    const program = `{escaped_code}`;
    
    // Load the program
    session.consult(program, {{
        success: function() {{
            console.log("{module_name} loaded successfully");
        }},
        error: function(err) {{
            console.error("Error loading program:", err);
        }}
    }});
    
    // Export session for external use
    window.stunirSession = session;
    window.stunirQuery = function(queryStr, callback) {{
        session.query(queryStr);
        session.answer(callback);
    }};
}})();
'''
        else:  # Node.js
            return f'''// STUNIR Generated Tau Prolog Loader (Node.js)
// Module: {module_name}

const pl = require("tau-prolog");
{chr(10).join(libs_to_load)}

// STUNIR Generated Prolog Program
const program = `{escaped_code}`;

// Create and initialize session
const session = pl.create();
session.consult(program);

// Export for module use
module.exports = {{
    session: session,
    query: function(queryStr, callback) {{
        session.query(queryStr);
        session.answer(callback);
    }},
    formatAnswer: pl.format_answer
}};
'''
    
    def emit_to_file(self, ir: Dict[str, Any], output_path: Path) -> TauPrologEmitterResult:
        """Emit code and write to file.
        
        Args:
            ir: STUNIR Logic IR dictionary
            output_path: Output file path
            
        Returns:
            TauPrologEmitterResult with generated code
        """
        result = self.emit(ir)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result.code)
        
        # Write JavaScript loader if generated
        if result.loader_js:
            js_path = output_path.with_suffix('.loader.js')
            with open(js_path, 'w', encoding='utf-8') as f:
                f.write(result.loader_js)
        
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


__all__ = [
    'TauPrologEmitter',
    'TauPrologConfig',
    'TauPrologEmitterResult',
    'compute_sha256',
    'canonical_json',
]
