"""PDDL planning domain/problem emitter.

This module emits planning models to PDDL (Planning Domain Definition Language)
format, the standard input format for automated planners.
"""

import time
import hashlib
import json
from typing import Optional, List, Dict, Any

from ir.planning import (
    Domain, Problem, Action, Predicate, Function, Formula, Effect,
    TypeDef, Parameter, ObjectDef, Atom, InitialState, Metric,
    DerivedPredicate, FunctionApplication,
    PDDLRequirement, FormulaType, EffectType,
    PlanningEmitterResult
)


def canonical_json(obj: Any) -> str:
    """Generate canonical JSON (sorted keys, no extra whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(',', ':'))


def compute_sha256(data: bytes) -> str:
    """Compute SHA256 hash of data."""
    return hashlib.sha256(data).hexdigest()


class PDDLEmitter:
    """PDDL planning domain/problem emitter.
    
    Emits planning models to PDDL format, the standard
    input format for automated planners.
    """
    
    DIALECT = "pddl"
    VERSION = "2.1"
    FILE_EXTENSION = ".pddl"
    
    # Requirement to PDDL string mapping
    REQUIREMENT_MAP = {
        PDDLRequirement.STRIPS: ":strips",
        PDDLRequirement.TYPING: ":typing",
        PDDLRequirement.NEGATIVE_PRECONDITIONS: ":negative-preconditions",
        PDDLRequirement.DISJUNCTIVE_PRECONDITIONS: ":disjunctive-preconditions",
        PDDLRequirement.EQUALITY: ":equality",
        PDDLRequirement.EXISTENTIAL_PRECONDITIONS: ":existential-preconditions",
        PDDLRequirement.UNIVERSAL_PRECONDITIONS: ":universal-preconditions",
        PDDLRequirement.QUANTIFIED_PRECONDITIONS: ":quantified-preconditions",
        PDDLRequirement.CONDITIONAL_EFFECTS: ":conditional-effects",
        PDDLRequirement.FLUENTS: ":fluents",
        PDDLRequirement.NUMERIC_FLUENTS: ":numeric-fluents",
        PDDLRequirement.OBJECT_FLUENTS: ":object-fluents",
        PDDLRequirement.ADL: ":adl",
        PDDLRequirement.DURATIVE_ACTIONS: ":durative-actions",
        PDDLRequirement.DURATION_INEQUALITIES: ":duration-inequalities",
        PDDLRequirement.CONTINUOUS_EFFECTS: ":continuous-effects",
        PDDLRequirement.DERIVED_PREDICATES: ":derived-predicates",
        PDDLRequirement.TIMED_INITIAL_LITERALS: ":timed-initial-literals",
        PDDLRequirement.PREFERENCES: ":preferences",
        PDDLRequirement.CONSTRAINTS: ":constraints",
        PDDLRequirement.ACTION_COSTS: ":action-costs",
    }
    
    def __init__(self, indent: int = 2):
        """Initialize emitter.
        
        Args:
            indent: Number of spaces for indentation
        """
        self._indent = indent
        self._output: List[str] = []
        self._warnings: List[str] = []
        self._level = 0
    
    def emit_domain(self, domain: Domain) -> PlanningEmitterResult:
        """Emit a PDDL domain file.
        
        Args:
            domain: Domain to emit
            
        Returns:
            PlanningEmitterResult with domain PDDL code
        """
        self._output = []
        self._warnings = []
        self._level = 0
        
        # Validate domain
        errors = domain.validate()
        for error in errors:
            self._warn(f"Validation warning: {error}")
        
        # Emit domain structure
        self._emit_domain_header(domain)
        self._emit_requirements(domain.requirements)
        self._emit_types(domain.types)
        self._emit_constants(domain.constants)
        self._emit_predicates(domain.predicates)
        self._emit_functions(domain.functions)
        self._emit_derived_predicates(domain.derived_predicates)
        self._emit_actions(domain.actions)
        self._emit_domain_footer()
        
        code = self._get_code()
        manifest = self._generate_domain_manifest(domain, code)
        
        return PlanningEmitterResult(
            domain_code=code,
            manifest=manifest,
            warnings=self._warnings
        )
    
    def emit_problem(self, problem: Problem) -> PlanningEmitterResult:
        """Emit a PDDL problem file.
        
        Args:
            problem: Problem to emit
            
        Returns:
            PlanningEmitterResult with problem PDDL code
        """
        self._output = []
        self._warnings = []
        self._level = 0
        
        # Validate problem
        errors = problem.validate()
        for error in errors:
            self._warn(f"Validation warning: {error}")
        
        # Emit problem structure
        self._emit_problem_header(problem)
        self._emit_problem_domain(problem)
        self._emit_problem_requirements(problem.requirements)
        self._emit_objects(problem.objects)
        self._emit_init(problem.init)
        self._emit_goal(problem.goal)
        self._emit_metric(problem.metric)
        self._emit_problem_footer()
        
        code = self._get_code()
        manifest = self._generate_problem_manifest(problem, code)
        
        return PlanningEmitterResult(
            domain_code="",  # Empty when emitting only problem
            problem_code=code,
            manifest=manifest,
            warnings=self._warnings
        )
    
    def emit(self, domain: Domain, problem: Optional[Problem] = None) -> PlanningEmitterResult:
        """Emit domain and optionally problem.
        
        Args:
            domain: Domain to emit
            problem: Optional problem to emit
            
        Returns:
            PlanningEmitterResult with both domain and problem code
        """
        domain_result = self.emit_domain(domain)
        
        if problem is None:
            return domain_result
        
        problem_result = self.emit_problem(problem)
        
        # Merge manifests
        combined_manifest = {
            **domain_result.manifest,
            'problem': problem_result.manifest.get('problem', {}),
            'problem_output': problem_result.manifest.get('output', {}),
        }
        combined_manifest['manifest_hash'] = compute_sha256(
            canonical_json({k: v for k, v in combined_manifest.items() 
                          if k != 'manifest_hash'}).encode('utf-8')
        )
        
        return PlanningEmitterResult(
            domain_code=domain_result.domain_code,
            problem_code=problem_result.problem_code,
            manifest=combined_manifest,
            warnings=domain_result.warnings + problem_result.warnings
        )
    
    # --- Domain emission methods ---
    
    def _emit_domain_header(self, domain: Domain) -> None:
        """Emit domain header."""
        self._line(f";; PDDL Domain: {domain.name}")
        self._line(f";; Generated by STUNIR PDDL Emitter v{self.VERSION}")
        self._line()
        self._line(f"(define (domain {domain.name})")
        self._level += 1
    
    def _emit_requirements(self, requirements: List[PDDLRequirement]) -> None:
        """Emit requirements section."""
        if not requirements:
            return
        
        req_strs = [self.REQUIREMENT_MAP.get(r, f":{r.name.lower().replace('_', '-')}") 
                   for r in requirements]
        self._line(f"(:requirements {' '.join(req_strs)})")
        self._line()
    
    def _emit_types(self, types: List[TypeDef]) -> None:
        """Emit types section."""
        if not types:
            return
        
        self._line("(:types")
        self._level += 1
        
        # Group types by parent
        by_parent: Dict[str, List[str]] = {}
        for t in types:
            by_parent.setdefault(t.parent, []).append(t.name)
        
        for parent, children in sorted(by_parent.items()):
            if parent == "object":
                self._line(f"{' '.join(children)}")
            else:
                self._line(f"{' '.join(children)} - {parent}")
        
        self._level -= 1
        self._line(")")
        self._line()
    
    def _emit_constants(self, constants: List[ObjectDef]) -> None:
        """Emit constants section."""
        if not constants:
            return
        
        self._line("(:constants")
        self._level += 1
        self._emit_typed_list(constants)
        self._level -= 1
        self._line(")")
        self._line()
    
    def _emit_predicates(self, predicates: List[Predicate]) -> None:
        """Emit predicates section."""
        if not predicates:
            return
        
        self._line("(:predicates")
        self._level += 1
        
        for pred in predicates:
            params = self._format_parameters(pred.parameters)
            if params:
                self._line(f"({pred.name} {params})")
            else:
                self._line(f"({pred.name})")
        
        self._level -= 1
        self._line(")")
        self._line()
    
    def _emit_functions(self, functions: List[Function]) -> None:
        """Emit functions section."""
        if not functions:
            return
        
        self._line("(:functions")
        self._level += 1
        
        for func in functions:
            params = self._format_parameters(func.parameters)
            if params:
                self._line(f"({func.name} {params})")
            else:
                self._line(f"({func.name})")
        
        self._level -= 1
        self._line(")")
        self._line()
    
    def _emit_derived_predicates(self, derived: List[DerivedPredicate]) -> None:
        """Emit derived predicates (axioms)."""
        for dp in derived:
            params = self._format_parameters(dp.predicate.parameters)
            self._line(f"(:derived ({dp.predicate.name} {params})")
            self._level += 1
            self._emit_formula(dp.condition)
            self._level -= 1
            self._line(")")
            self._line()
    
    def _emit_actions(self, actions: List[Action]) -> None:
        """Emit action definitions."""
        for action in actions:
            self._emit_action(action)
    
    def _emit_action(self, action: Action) -> None:
        """Emit a single action."""
        self._line(f"(:action {action.name}")
        self._level += 1
        
        # Parameters
        params = self._format_parameters(action.parameters)
        self._line(f":parameters ({params})")
        
        # Precondition
        if action.precondition:
            self._line(":precondition")
            self._level += 1
            self._emit_formula(action.precondition)
            self._level -= 1
        
        # Effect
        if action.effect:
            self._line(":effect")
            self._level += 1
            self._emit_effect(action.effect)
            self._level -= 1
        
        self._level -= 1
        self._line(")")
        self._line()
    
    def _emit_domain_footer(self) -> None:
        """Emit domain footer."""
        self._level -= 1
        self._line(")")
    
    # --- Problem emission methods ---
    
    def _emit_problem_header(self, problem: Problem) -> None:
        """Emit problem header."""
        self._line(f";; PDDL Problem: {problem.name}")
        self._line(f";; Domain: {problem.domain_name}")
        self._line(f";; Generated by STUNIR PDDL Emitter v{self.VERSION}")
        self._line()
        self._line(f"(define (problem {problem.name})")
        self._level += 1
    
    def _emit_problem_domain(self, problem: Problem) -> None:
        """Emit domain reference."""
        self._line(f"(:domain {problem.domain_name})")
        self._line()
    
    def _emit_problem_requirements(self, requirements: List[PDDLRequirement]) -> None:
        """Emit problem-specific requirements."""
        if not requirements:
            return
        
        req_strs = [self.REQUIREMENT_MAP.get(r, f":{r.name.lower().replace('_', '-')}") 
                   for r in requirements]
        self._line(f"(:requirements {' '.join(req_strs)})")
        self._line()
    
    def _emit_objects(self, objects: List[ObjectDef]) -> None:
        """Emit objects section."""
        if not objects:
            return
        
        self._line("(:objects")
        self._level += 1
        self._emit_typed_list(objects)
        self._level -= 1
        self._line(")")
        self._line()
    
    def _emit_init(self, init: InitialState) -> None:
        """Emit initial state section."""
        self._line("(:init")
        self._level += 1
        
        # Ground atoms
        for fact in init.facts:
            self._emit_atom(fact)
        
        # Numeric values
        for func_str, value in sorted(init.numeric_values.items()):
            self._line(f"(= ({func_str}) {value})")
        
        # Timed literals
        for time_point, atom in init.timed_literals:
            args = ' '.join(atom.arguments)
            self._line(f"(at {time_point} ({atom.predicate} {args}))")
        
        self._level -= 1
        self._line(")")
        self._line()
    
    def _emit_goal(self, goal: Optional[Formula]) -> None:
        """Emit goal section."""
        if goal is None:
            return
        
        self._line("(:goal")
        self._level += 1
        self._emit_formula(goal)
        self._level -= 1
        self._line(")")
        self._line()
    
    def _emit_metric(self, metric: Optional[Metric]) -> None:
        """Emit metric section."""
        if metric is None:
            return
        
        self._line(f"(:metric {metric.direction} ({metric.expression}))")
        self._line()
    
    def _emit_problem_footer(self) -> None:
        """Emit problem footer."""
        self._level -= 1
        self._line(")")
    
    # --- Formula and effect emission ---
    
    def _emit_formula(self, formula: Formula) -> None:
        """Emit a logical formula."""
        if formula.formula_type == FormulaType.ATOM:
            self._emit_atom(formula.atom)
        
        elif formula.formula_type == FormulaType.AND:
            if len(formula.children) == 0:
                self._line("(and)")
            elif len(formula.children) == 1:
                self._emit_formula(formula.children[0])
            else:
                self._line("(and")
                self._level += 1
                for child in formula.children:
                    self._emit_formula(child)
                self._level -= 1
                self._line(")")
        
        elif formula.formula_type == FormulaType.OR:
            self._line("(or")
            self._level += 1
            for child in formula.children:
                self._emit_formula(child)
            self._level -= 1
            self._line(")")
        
        elif formula.formula_type == FormulaType.NOT:
            child = formula.children[0] if formula.children else None
            if child and child.formula_type == FormulaType.ATOM:
                # Simple negation
                args = ' '.join(child.atom.arguments) if child.atom else ""
                pred = child.atom.predicate if child.atom else "?"
                if args:
                    self._line(f"(not ({pred} {args}))")
                else:
                    self._line(f"(not ({pred}))")
            else:
                self._line("(not")
                self._level += 1
                if child:
                    self._emit_formula(child)
                self._level -= 1
                self._line(")")
        
        elif formula.formula_type == FormulaType.IMPLY:
            self._line("(imply")
            self._level += 1
            for child in formula.children[:2]:
                self._emit_formula(child)
            self._level -= 1
            self._line(")")
        
        elif formula.formula_type == FormulaType.EXISTS:
            vars_str = self._format_parameters(formula.variables)
            self._line(f"(exists ({vars_str})")
            self._level += 1
            if formula.children:
                self._emit_formula(formula.children[0])
            self._level -= 1
            self._line(")")
        
        elif formula.formula_type == FormulaType.FORALL:
            vars_str = self._format_parameters(formula.variables)
            self._line(f"(forall ({vars_str})")
            self._level += 1
            if formula.children:
                self._emit_formula(formula.children[0])
            self._level -= 1
            self._line(")")
        
        elif formula.formula_type == FormulaType.EQUALS:
            self._line(f"(= {formula.left_term} {formula.right_term})")
        
        elif formula.formula_type == FormulaType.WHEN:
            self._line("(when")
            self._level += 1
            for child in formula.children[:2]:
                self._emit_formula(child)
            self._level -= 1
            self._line(")")
    
    def _emit_effect(self, effect: Effect) -> None:
        """Emit an action effect."""
        if effect.effect_type == EffectType.POSITIVE:
            if effect.formula and effect.formula.atom:
                self._emit_atom(effect.formula.atom)
            elif effect.formula:
                self._emit_formula(effect.formula)
        
        elif effect.effect_type == EffectType.NEGATIVE:
            if effect.formula and effect.formula.atom:
                args = ' '.join(effect.formula.atom.arguments)
                if args:
                    self._line(f"(not ({effect.formula.atom.predicate} {args}))")
                else:
                    self._line(f"(not ({effect.formula.atom.predicate}))")
            elif effect.formula:
                self._line("(not")
                self._level += 1
                self._emit_formula(effect.formula)
                self._level -= 1
                self._line(")")
        
        elif effect.effect_type == EffectType.COMPOUND:
            if len(effect.children) == 1:
                self._emit_effect(effect.children[0])
            else:
                self._line("(and")
                self._level += 1
                for child in effect.children:
                    self._emit_effect(child)
                self._level -= 1
                self._line(")")
        
        elif effect.effect_type == EffectType.CONDITIONAL:
            self._line("(when")
            self._level += 1
            if effect.condition:
                self._emit_formula(effect.condition)
            if effect.children:
                self._emit_effect(effect.children[0])
            self._level -= 1
            self._line(")")
        
        elif effect.effect_type == EffectType.FORALL:
            vars_str = self._format_parameters(effect.variables)
            self._line(f"(forall ({vars_str})")
            self._level += 1
            if effect.children:
                self._emit_effect(effect.children[0])
            self._level -= 1
            self._line(")")
        
        elif effect.effect_type in (EffectType.INCREASE, EffectType.DECREASE, 
                                    EffectType.ASSIGN, EffectType.SCALE_UP, 
                                    EffectType.SCALE_DOWN):
            op_map = {
                EffectType.INCREASE: "increase",
                EffectType.DECREASE: "decrease",
                EffectType.ASSIGN: "assign",
                EffectType.SCALE_UP: "scale-up",
                EffectType.SCALE_DOWN: "scale-down",
            }
            op = op_map.get(effect.effect_type, "assign")
            if effect.function_app:
                func_args = ' '.join(effect.function_app.arguments)
                if func_args:
                    self._line(f"({op} ({effect.function_app.function} {func_args}) {effect.value})")
                else:
                    self._line(f"({op} ({effect.function_app.function}) {effect.value})")
    
    def _emit_atom(self, atom: Optional[Atom]) -> None:
        """Emit an atom."""
        if atom is None:
            return
        
        args = ' '.join(atom.arguments)
        if args:
            self._line(f"({atom.predicate} {args})")
        else:
            self._line(f"({atom.predicate})")
    
    # --- Helper methods ---
    
    def _format_parameters(self, params: List[Parameter]) -> str:
        """Format parameter list."""
        if not params:
            return ""
        
        # Group by type for more compact output
        by_type: Dict[str, List[str]] = {}
        for p in params:
            by_type.setdefault(p.param_type, []).append(p.name)
        
        parts = []
        for param_type, names in by_type.items():
            if param_type == "object":
                parts.append(' '.join(names))
            else:
                parts.append(f"{' '.join(names)} - {param_type}")
        
        return ' '.join(parts)
    
    def _emit_typed_list(self, objects: List[ObjectDef]) -> None:
        """Emit a typed list of objects/constants."""
        # Group by type
        by_type: Dict[str, List[str]] = {}
        for obj in objects:
            by_type.setdefault(obj.obj_type, []).append(obj.name)
        
        for obj_type, names in sorted(by_type.items()):
            if obj_type == "object":
                self._line(' '.join(names))
            else:
                self._line(f"{' '.join(names)} - {obj_type}")
    
    def _line(self, text: str = "") -> None:
        """Add a line to output."""
        if text:
            indent = ' ' * (self._level * self._indent)
            self._output.append(f"{indent}{text}")
        else:
            self._output.append("")
    
    def _warn(self, message: str) -> None:
        """Add a warning."""
        self._warnings.append(message)
    
    def _get_code(self) -> str:
        """Get generated code."""
        return '\n'.join(self._output)
    
    # --- Manifest generation ---
    
    def _generate_domain_manifest(self, domain: Domain, code: str) -> Dict[str, Any]:
        """Generate manifest for domain."""
        code_hash = compute_sha256(code.encode('utf-8'))
        
        manifest = {
            'schema': 'stunir.manifest.planning.v1',
            'generator': f'stunir.pddl.emitter.v{self.VERSION}',
            'epoch': int(time.time()),
            'domain': {
                'name': domain.name,
                'actions': len(domain.actions),
                'predicates': len(domain.predicates),
                'types': len(domain.types),
                'functions': len(domain.functions),
                'constants': len(domain.constants),
                'requirements': [r.name for r in domain.requirements],
            },
            'output': {
                'hash': code_hash,
                'size': len(code),
                'format': 'pddl',
            },
        }
        
        manifest['manifest_hash'] = compute_sha256(
            canonical_json({k: v for k, v in manifest.items() 
                          if k != 'manifest_hash'}).encode('utf-8')
        )
        
        return manifest
    
    def _generate_problem_manifest(self, problem: Problem, code: str) -> Dict[str, Any]:
        """Generate manifest for problem."""
        code_hash = compute_sha256(code.encode('utf-8'))
        
        manifest = {
            'schema': 'stunir.manifest.planning.v1',
            'generator': f'stunir.pddl.emitter.v{self.VERSION}',
            'epoch': int(time.time()),
            'problem': {
                'name': problem.name,
                'domain': problem.domain_name,
                'objects': len(problem.objects),
                'init_facts': len(problem.init.facts),
                'has_metric': problem.metric is not None,
            },
            'output': {
                'hash': code_hash,
                'size': len(code),
                'format': 'pddl',
            },
        }
        
        manifest['manifest_hash'] = compute_sha256(
            canonical_json({k: v for k, v in manifest.items() 
                          if k != 'manifest_hash'}).encode('utf-8')
        )
        
        return manifest
