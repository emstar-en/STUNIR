#!/usr/bin/env python3
"""STUNIR Mercury Emitter.

Generates Mercury code from STUNIR Logic IR with full type, mode,
and determinism declarations as required by the Mercury compiler.

Mercury is a pure declarative logic/functional programming language
that requires mandatory declarations unlike traditional Prolog.

Key features:
- Strong static typing with type inference
- Mode declarations (input/output annotations)
- Determinism declarations (det, semidet, multi, nondet, etc.)
- Module system with interface/implementation sections
- Functions in addition to predicates
- Purity tracking (pure, semipure, impure)

Part of Phase 5D-3: Mercury Emitter.
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
    MercuryTypeMapper, MercuryMode, Determinism, Purity,
    MERCURY_TYPES, MERCURY_RESERVED, MERCURY_IMPORTS
)


def compute_sha256(data: bytes) -> str:
    """Compute SHA-256 hash of data."""
    return hashlib.sha256(data).hexdigest()


def canonical_json(data: Any) -> str:
    """Produce deterministic JSON output."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'))


@dataclass
class EmitterResult:
    """Result of Mercury code emission."""
    code: str                   # Generated Mercury code
    module_name: str            # Mercury module name
    predicates: List[str]       # List of pred/arity strings
    functions: List[str]        # List of func/arity strings
    types: List[str]            # Declared types
    sha256: str                 # Hash of generated code
    emit_time: float            # Time taken to emit
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'code': self.code,
            'module_name': self.module_name,
            'predicates': self.predicates,
            'functions': self.functions,
            'types': self.types,
            'sha256': self.sha256,
            'emit_time': self.emit_time
        }


@dataclass
class MercuryConfig:
    """Configuration for Mercury emitter."""
    module_prefix: str = "stunir"
    emit_interface: bool = True           # Emit :- interface section
    emit_implementation: bool = True       # Emit :- implementation section
    emit_comments: bool = True             # Emit documentation comments
    emit_type_declarations: bool = True    # Emit :- type declarations
    emit_pred_declarations: bool = True    # Emit :- pred declarations
    emit_func_declarations: bool = True    # Emit :- func declarations
    emit_mode_declarations: bool = True    # Emit :- mode declarations
    emit_determinism: bool = True          # Emit determinism annotations
    infer_types: bool = True               # Auto-infer types from IR
    infer_modes: bool = True               # Auto-infer modes
    infer_determinism: bool = True         # Auto-infer determinism
    default_determinism: str = "det"       # Default if cannot infer
    default_mode: str = "in_out"           # Default if cannot infer
    emit_timestamps: bool = True           # Emit generation timestamps
    emit_end_module: bool = True           # Emit :- end_module directive
    auto_imports: bool = True              # Auto-import standard modules
    indent: str = "    "
    line_width: int = 80


class MercuryEmitter:
    """Emitter for Mercury code from STUNIR Logic IR.
    
    Generates valid Mercury source files with mandatory type, mode,
    and determinism declarations. Unlike Prolog, Mercury requires
    explicit declarations for compile-time type checking.
    
    Module structure:
        :- module name.
        :- interface.
            % Public declarations
        :- implementation.
            % Imports and definitions
        :- end_module name.
    """
    
    DIALECT = "mercury"
    FILE_EXTENSION = ".m"
    
    def __init__(self, config: Optional[MercuryConfig] = None):
        """Initialize Mercury emitter.
        
        Args:
            config: Emitter configuration (optional)
        """
        self.config = config or MercuryConfig()
        self.logic_ext = LogicIRExtension()
        self.type_mapper = MercuryTypeMapper(
            infer_types=self.config.infer_types,
            infer_modes=self.config.infer_modes,
            infer_determinism=self.config.infer_determinism
        )
        
        # State for current emission
        self._exports: Set[Tuple[str, int]] = set()
        self._predicates: Dict[Tuple[str, int], Predicate] = {}
        self._functions: Dict[str, Dict[str, Any]] = {}
        self._types: List[Dict[str, Any]] = []
        self._imports: Set[str] = set()
        self._warnings: List[str] = []
    
    def emit(self, ir: Dict[str, Any]) -> EmitterResult:
        """Emit Mercury code from Logic IR.
        
        Args:
            ir: STUNIR Logic IR dictionary
            
        Returns:
            EmitterResult with generated code
        """
        start_time = time.time()
        
        module_name = ir.get('module', 'unnamed')
        mercury_module = f"{self.config.module_prefix}_{self._mercury_name(module_name)}"
        
        # Reset state
        self._reset_state()
        
        # Extract and analyze IR
        self._predicates = self.logic_ext.extract_predicates(ir)
        self._analyze_exports(ir)
        self._analyze_types(ir)
        self._analyze_functions(ir)
        self._determine_imports()
        
        # Build code sections
        sections = []
        
        # Header comment
        if self.config.emit_comments:
            sections.append(self._emit_header(module_name))
        
        # Module declaration
        sections.append(self._emit_module_declaration(mercury_module))
        
        # Interface section
        if self.config.emit_interface:
            interface = self._emit_interface_section()
            if interface:
                sections.append(interface)
        
        # Implementation section
        if self.config.emit_implementation:
            implementation = self._emit_implementation_section(ir)
            if implementation:
                sections.append(implementation)
        
        # End module
        if self.config.emit_end_module:
            sections.append(f":- end_module {mercury_module}.")
        
        code = '\n\n'.join(s for s in sections if s)
        code_hash = compute_sha256(code.encode('utf-8'))
        
        return EmitterResult(
            code=code,
            module_name=mercury_module,
            predicates=[f"{p}/{a}" for p, a in sorted(self._predicates.keys())],
            functions=list(self._functions.keys()),
            types=[t.get('name', '') for t in self._types],
            sha256=code_hash,
            emit_time=time.time() - start_time
        )
    
    def _reset_state(self) -> None:
        """Reset emission state."""
        self._exports = set()
        self._predicates = {}
        self._functions = {}
        self._types = []
        self._imports = set()
        self._warnings = []
    
    def _mercury_name(self, name: str) -> str:
        """Convert name to valid Mercury identifier.
        
        Mercury identifiers must start with lowercase letter
        and contain only alphanumeric and underscore.
        """
        result = ''.join(c if c.isalnum() or c == '_' else '_' for c in name)
        if result and result[0].isupper():
            result = result[0].lower() + result[1:]
        if not result or not result[0].isalpha():
            result = 'mod_' + result
        if result in MERCURY_RESERVED:
            result = result + '_'
        return result or 'unnamed'
    
    def _analyze_exports(self, ir: Dict[str, Any]) -> None:
        """Analyze exports from IR."""
        for exp in ir.get('exports', []):
            pred = exp.get('predicate')
            arity = exp.get('arity', 0)
            if pred:
                self._exports.add((pred, arity))
        
        # Auto-export non-private predicates
        for (name, arity), pred in self._predicates.items():
            if not name.startswith('_'):
                self._exports.add((name, arity))
    
    def _analyze_types(self, ir: Dict[str, Any]) -> None:
        """Analyze type declarations from IR."""
        self._types = ir.get('types', [])
    
    def _analyze_functions(self, ir: Dict[str, Any]) -> None:
        """Analyze function definitions from IR."""
        for func in ir.get('functions', []):
            name = func.get('name', '')
            if name:
                self._functions[name] = func
    
    def _determine_imports(self) -> None:
        """Determine required imports based on used types/predicates."""
        if self.config.auto_imports:
            # Standard imports
            self._imports = set(MERCURY_IMPORTS)
    
    def _emit_header(self, module_name: str) -> str:
        """Generate header comment."""
        lines = [
            "%--------------------------------------------------------------------------%",
            "% STUNIR Generated Mercury Module",
            f"% Module: {module_name}",
        ]
        
        if self.config.emit_timestamps:
            lines.append(f"% Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        lines.extend([
            "% ",
            "% This file was automatically generated by STUNIR.",
            "% Do not edit manually.",
            "%--------------------------------------------------------------------------%"
        ])
        
        return '\n'.join(lines)
    
    def _emit_module_declaration(self, module_name: str) -> str:
        """Generate module declaration."""
        return f":- module {module_name}."
    
    def _emit_interface_section(self) -> str:
        """Generate interface section with public declarations."""
        lines = [":- interface."]
        
        # Type declarations (public)
        for type_def in self._types:
            type_decl = self._emit_type_declaration(type_def)
            if type_decl:
                lines.append("")
                lines.append(type_decl)
        
        # Predicate declarations (public)
        if self.config.emit_pred_declarations:
            for (name, arity), pred in sorted(self._predicates.items()):
                if (name, arity) in self._exports:
                    decl = self._emit_pred_declaration(pred)
                    lines.append("")
                    lines.append(decl)
        
        # Function declarations (public)
        if self.config.emit_func_declarations:
            for name, func in sorted(self._functions.items()):
                decl = self._emit_func_declaration(func)
                lines.append("")
                lines.append(decl)
        
        return '\n'.join(lines)
    
    def _emit_implementation_section(self, ir: Dict[str, Any]) -> str:
        """Generate implementation section."""
        lines = [":- implementation."]
        
        # Imports
        if self._imports:
            lines.append("")
            for imp in sorted(self._imports):
                lines.append(f":- import_module {imp}.")
        
        # Predicate implementations
        for (name, arity), pred in sorted(self._predicates.items()):
            pred_code = self._emit_predicate(pred)
            if pred_code:
                lines.append("")
                lines.append(pred_code)
        
        # Function implementations
        for name, func in sorted(self._functions.items()):
            func_code = self._emit_function(func)
            if func_code:
                lines.append("")
                lines.append(func_code)
        
        # Queries (as initialization)
        queries = self.logic_ext.extract_queries(ir)
        if queries:
            query_code = self._emit_queries(queries)
            if query_code:
                lines.append("")
                lines.append(query_code)
        
        return '\n'.join(lines)
    
    def _emit_type_declaration(self, type_def: Dict[str, Any]) -> str:
        """Emit Mercury type declaration.
        
        Supports:
        - Enumeration types: :- type color ---> red ; green ; blue.
        - Algebraic types: :- type maybe(T) ---> yes(T) ; no.
        - Record types: :- type point ---> point(x :: int, y :: int).
        """
        name = type_def.get('name', '')
        if not name:
            return ''
        
        kind = type_def.get('kind', 'enum')
        params = type_def.get('params', [])
        constructors = type_def.get('constructors', [])
        
        # Build type name with parameters
        if params:
            type_name = f"{name}({', '.join(params)})"
        else:
            type_name = name
        
        # Build constructors
        if kind == 'enum' or (kind == 'algebraic' and all(isinstance(c, str) for c in constructors)):
            # Simple enumeration
            cons_str = ' ; '.join(str(c) for c in constructors)
            return f":- type {type_name} ---> {cons_str}."
        
        elif kind == 'algebraic':
            # Algebraic data type with arguments
            cons_strs = []
            for cons in constructors:
                if isinstance(cons, dict):
                    cons_name = cons.get('name', '')
                    cons_args = cons.get('args', [])
                    if cons_args:
                        args_str = ', '.join(
                            f"{a.get('name', '_')} :: {self.type_mapper.map_type(a.get('type', 'any'))}"
                            for a in cons_args
                        )
                        cons_strs.append(f"{cons_name}({args_str})")
                    else:
                        cons_strs.append(cons_name)
                else:
                    cons_strs.append(str(cons))
            
            cons_str = ' ; '.join(cons_strs)
            return f":- type {type_name} ---> {cons_str}."
        
        elif kind == 'record':
            # Record type
            fields = type_def.get('fields', [])
            field_strs = []
            for fld in fields:
                fname = fld.get('name', '_')
                ftype = self.type_mapper.map_type(fld.get('type', 'any'))
                field_strs.append(f"{fname} :: {ftype}")
            
            return f":- type {type_name} ---> {name}({', '.join(field_strs)})."
        
        # Fallback
        return f":- type {type_name}."
    
    def _emit_pred_declaration(self, pred: Predicate) -> str:
        """Emit predicate declaration with types, modes, determinism.
        
        Format: :- pred name(type1::mode1, type2::mode2) is det.
        """
        name = pred.name
        arity = pred.arity
        
        # Get parameter info
        params = self._infer_pred_params(pred)
        
        # Infer determinism
        has_cut = self._predicate_has_cut(pred)
        det = self.type_mapper.infer_determinism_from_clauses(pred.clauses, has_cut)
        
        return self.type_mapper.format_pred_declaration(name, params, det)
    
    def _infer_pred_params(self, pred: Predicate) -> List[Dict[str, Any]]:
        """Infer parameter types and modes for a predicate."""
        params = []
        
        # Try to get from first clause head
        if pred.clauses:
            first_clause = pred.clauses[0]
            if isinstance(first_clause, Fact):
                for i, arg in enumerate(first_clause.args):
                    params.append(self._infer_param_from_term(arg, i, pred.arity))
            elif isinstance(first_clause, Rule) and first_clause.head:
                head = first_clause.head
                if isinstance(head, Compound):
                    for i, arg in enumerate(head.args):
                        params.append(self._infer_param_from_term(arg, i, pred.arity))
        
        # Fill in missing params
        while len(params) < pred.arity:
            idx = len(params)
            params.append({
                'name': f'Arg{idx}',
                'type': 'any',
                'mode': 'output' if idx == pred.arity - 1 else 'input'
            })
        
        return params
    
    def _infer_param_from_term(self, term: Term, index: int, arity: int) -> Dict[str, Any]:
        """Infer parameter info from a term in clause head."""
        param = {'name': f'Arg{index}', 'type': 'any', 'mode': 'input'}
        
        if isinstance(term, Variable):
            param['name'] = term.name
            # Variables as last arg often outputs
            if index == arity - 1:
                param['mode'] = 'output'
        elif isinstance(term, Number):
            param['type'] = 'float' if isinstance(term.value, float) else 'i32'
        elif isinstance(term, Atom):
            param['type'] = 'string'
        elif isinstance(term, StringTerm):
            param['type'] = 'string'
        elif isinstance(term, ListTerm):
            param['type'] = 'list'
        elif isinstance(term, Compound):
            # Could be a structured type
            param['type'] = term.functor
        
        return param
    
    def _predicate_has_cut(self, pred: Predicate) -> bool:
        """Check if predicate contains cut."""
        for clause in pred.clauses:
            if isinstance(clause, Rule):
                for goal in clause.body:
                    if goal.kind == GoalKind.CUT:
                        return True
        return False
    
    def _emit_func_declaration(self, func: Dict[str, Any]) -> str:
        """Emit function declaration.
        
        Format: :- func name(type1, type2) = return_type is det.
        """
        name = func.get('name', 'unknown')
        params = func.get('params', [])
        return_type = func.get('return_type', 'any')
        
        arg_types = [p.get('type', 'any') for p in params]
        
        return self.type_mapper.format_func_declaration(
            name, arg_types, return_type, Determinism.DET
        )
    
    def _emit_predicate(self, pred: Predicate) -> str:
        """Emit predicate implementation (clauses)."""
        lines = []
        
        # Comment with signature
        if self.config.emit_comments:
            lines.append(f"% {pred.name}/{pred.arity}")
        
        # Emit each clause
        for clause in pred.clauses:
            clause_str = self._emit_clause(clause)
            if clause_str:
                lines.append(clause_str)
        
        return '\n'.join(lines)
    
    def _emit_function(self, func: Dict[str, Any]) -> str:
        """Emit function implementation.
        
        Mercury functions use = instead of :- and must be deterministic.
        """
        name = func.get('name', 'unknown')
        params = func.get('params', [])
        body = func.get('body')
        
        # Build parameter list
        param_names = [p.get('name', f'X{i}') for i, p in enumerate(params)]
        
        if body:
            body_str = self._emit_expression(body)
            if param_names:
                return f"{name}({', '.join(param_names)}) = {body_str}."
            return f"{name} = {body_str}."
        
        # No body - emit stub
        if param_names:
            return f"{name}({', '.join(param_names)}) = _ :- fail."
        return f"{name} = _ :- fail."
    
    def _emit_expression(self, expr: Dict[str, Any]) -> str:
        """Emit an expression for function body."""
        kind = expr.get('kind', '')
        
        if kind == 'literal':
            return self._emit_literal(expr.get('value'))
        elif kind == 'var':
            return expr.get('name', '_')
        elif kind == 'binary_op':
            op = expr.get('op', '+')
            left = self._emit_expression(expr.get('left', {}))
            right = self._emit_expression(expr.get('right', {}))
            return f"({left} {op} {right})"
        elif kind == 'unary_op':
            op = expr.get('op', '-')
            operand = self._emit_expression(expr.get('operand', {}))
            return f"({op} {operand})"
        elif kind == 'call':
            func = expr.get('function', '')
            args = expr.get('args', [])
            args_str = ', '.join(self._emit_expression(a) for a in args)
            return f"{func}({args_str})" if args_str else func
        else:
            return '_'
    
    def _emit_literal(self, value: Any) -> str:
        """Emit a literal value."""
        if value is None:
            return '_'
        if isinstance(value, bool):
            return 'yes' if value else 'no'
        if isinstance(value, str):
            escaped = value.replace('"', '\\"')
            return f'"{escaped}"'
        return str(value)
    
    def _emit_clause(self, clause: Union[Fact, Rule]) -> str:
        """Emit a single clause (fact or rule)."""
        if isinstance(clause, Fact):
            return self._emit_fact(clause)
        elif isinstance(clause, Rule):
            return self._emit_rule(clause)
        return ''
    
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
        
        body_parts = [self._emit_goal(g) for g in rule.body]
        
        if len(body_parts) == 1:
            return f"{head} :-\n{self.config.indent}{body_parts[0]}."
        
        body = f',\n{self.config.indent}'.join(body_parts)
        return f"{head} :-\n{self.config.indent}{body}."
    
    def _emit_term(self, term: Union[Term, Dict, Any]) -> str:
        """Convert term to Mercury syntax."""
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
        """Emit list with Mercury syntax."""
        if not lst.elements and lst.tail is None:
            return '[]'
        
        elements = [self._emit_term(el) for el in lst.elements]
        
        if lst.tail is None:
            return '[' + ', '.join(elements) + ']'
        elif lst.tail:
            tail_str = self._emit_term(lst.tail)
            if elements:
                return '[' + ', '.join(elements) + ' | ' + tail_str + ']'
            return tail_str
        return '[]'
    
    def _escape_atom(self, value: str) -> str:
        """Escape atom if needed for Mercury."""
        if not value:
            return "''"
        
        # Mercury atoms (similar to Prolog)
        if value[0].islower() and all(c.isalnum() or c == '_' for c in value):
            if value not in MERCURY_RESERVED:
                return value
        
        # Need quoting
        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"
    
    def _emit_goal(self, goal: Goal) -> str:
        """Emit a goal in Mercury syntax."""
        if goal.kind == GoalKind.CUT:
            # Mercury doesn't have traditional cut, use committed choice
            return "true"  # Simplified - Mercury handles this differently
        elif goal.kind == GoalKind.NEGATION:
            if goal.goals:
                inner = self._emit_goal(goal.goals[0])
            elif goal.term:
                inner = self._emit_term(goal.term)
            else:
                inner = 'true'
            return f"not ({inner})"
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
                return '(' + ' ; '.join(parts) + ')'
            return 'fail'
        elif goal.kind == GoalKind.IF_THEN_ELSE:
            if goal.goals and len(goal.goals) >= 2:
                cond = self._emit_goal(goal.goals[0])
                then_part = self._emit_goal(goal.goals[1])
                else_part = self._emit_goal(goal.goals[2]) if len(goal.goals) > 2 else 'fail'
                return f"( if {cond} then {then_part} else {else_part} )"
            return 'true'
        elif goal.term:
            return self._emit_term(goal.term)
        else:
            return 'true'
    
    def _emit_queries(self, queries: List[Query]) -> str:
        """Emit queries as initialization predicates."""
        lines = ["% Queries"]
        
        for query in queries:
            if not query.goals:
                continue
            
            goals = ', '.join(self._emit_goal(g) for g in query.goals)
            # Mercury uses :- promise_pure for initialization
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


__all__ = [
    'MercuryEmitter',
    'MercuryConfig',
    'EmitterResult',
    'compute_sha256',
    'canonical_json',
]
