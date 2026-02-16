#!/usr/bin/env python3
"""STUNIR Elixir Code Emitter.

This module provides the Elixir code emitter for the actor model IR,
generating idiomatic Elixir code with OTP behaviors.

Features:
    - Module definitions (defmodule)
    - Function definitions (def, defp)
    - Process spawning (spawn, Task.async)
    - Message passing (send, receive)
    - OTP behaviors (GenServer, Supervisor)
    - Pipe operator (|>)
    - Pattern matching

Usage:
    from targets.beam import ElixirEmitter
    from ir.actor import ActorModule
    
    emitter = ElixirEmitter()
    result = emitter.emit_module(module)
    print(result.code)
"""

import re
from typing import Dict, Any, List, Optional
from targets.beam.base import BEAMEmitterBase, EmitterResult, compute_sha256

# Import IR types for type hints
from ir.actor import (
    ActorModule, FunctionDef, FunctionClause,
    Expr, LiteralExpr, VarExpr, AtomExpr, TupleExpr, ListExpr, MapExpr,
    CallExpr, CaseExpr, CaseClause, FunExpr, FunClause,
    ListCompExpr, Generator, Filter, TryExpr, CatchClause,
    Guard, BinaryExpr, BinarySegment,
    SpawnExpr, SpawnStrategy, SendExpr, ReceiveExpr, ReceiveClause,
    SelfExpr, LinkExpr, MonitorExpr,
    Pattern, WildcardPattern, VarPattern, LiteralPattern, AtomPattern,
    TuplePattern, ListPattern, MapPattern, BinaryPattern,
    BehaviorImpl, BehaviorType,
    GenServerImpl, GenServerState, Callback,
    SupervisorImpl, SupervisorFlags, SupervisorStrategy,
    ChildSpec, ChildRestart, ChildType,
    ApplicationImpl,
)


class ElixirEmitter(BEAMEmitterBase):
    """Elixir code emitter.
    
    Generates Elixir source code from Actor Model IR,
    including GenServer and Supervisor implementations.
    """
    
    LANGUAGE = 'elixir'
    VERSION = '1.0.0'
    
    def _emit_module_impl(self, module: ActorModule) -> str:
        """Emit Elixir module."""
        lines = []
        
        # Module definition
        mod_name = self._to_elixir_module_name(module.name)
        lines.append(f'defmodule {mod_name} do')
        self.indent_level += 1
        
        # Use behavior
        for behavior in module.behaviors:
            behavior_name = self._behavior_type_to_elixir(behavior.behavior_type)
            lines.append(f'{self.indent()}use {behavior_name}')
        if module.behaviors:
            lines.append('')
        
        # Module attributes
        for key, value in module.attributes.items():
            lines.append(f'{self.indent()}@{key} {self._emit_term(value)}')
        if module.attributes:
            lines.append('')
        
        # Functions
        for func in module.functions:
            func_code = self._emit_function(func)
            lines.append(func_code)
            lines.append('')
        
        # Behavior implementations
        for behavior in module.behaviors:
            impl_lines = self._emit_behavior(behavior)
            lines.extend(impl_lines)
        
        self.indent_level -= 1
        lines.append('end')
        
        return '\n'.join(lines)
    
    def _to_elixir_module_name(self, name: str) -> str:
        """Convert to Elixir module name (CamelCase)."""
        if '.' in name:
            return '.'.join(self._capitalize_word(p) for p in name.split('.'))
        return self._capitalize_word(name)
    
    def _capitalize_word(self, word: str) -> str:
        """Capitalize a word for module name."""
        if not word:
            return word
        # Handle snake_case
        if '_' in word:
            return ''.join(p.capitalize() for p in word.split('_'))
        return word[0].upper() + word[1:]
    
    def _to_snake_case(self, name: str) -> str:
        """Convert name to snake_case."""
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    # =========================================================================
    # Function Emission
    # =========================================================================
    
    def _emit_function(self, func: FunctionDef) -> str:
        """Emit function definition."""
        lines = []
        fn_type = 'def' if func.exported else 'defp'
        
        for clause in func.clauses:
            clause_code = self._emit_function_clause(fn_type, func.name, clause)
            lines.append(clause_code)
        
        return '\n'.join(lines)
    
    def _emit_function_clause(self, fn_type: str, name: str, clause: FunctionClause) -> str:
        """Emit a single function clause."""
        fn_name = self._to_snake_case(name)
        args = ', '.join(self.emit_pattern(arg) for arg in clause.args)
        
        # Guards
        guard_str = ''
        if clause.guards:
            guard_str = f' when {self._emit_guards(clause.guards)}'
        
        # Body
        if not clause.body:
            return f'{self.indent()}{fn_type} {fn_name}({args}){guard_str}, do: nil'
        
        if len(clause.body) == 1:
            body = self._emit_expr(clause.body[0])
            return f'{self.indent()}{fn_type} {fn_name}({args}){guard_str}, do: {body}'
        else:
            lines = [f'{self.indent()}{fn_type} {fn_name}({args}){guard_str} do']
            self.indent_level += 1
            for expr in clause.body:
                lines.append(f'{self.indent()}{self._emit_expr(expr)}')
            self.indent_level -= 1
            lines.append(f'{self.indent()}end')
            return '\n'.join(lines)
    
    # =========================================================================
    # Variable/Literal Transformation
    # =========================================================================
    
    def _transform_var(self, name: str) -> str:
        """Transform variable name to Elixir syntax (lowercase/snake_case)."""
        if name == '_':
            return '_'
        if not name:
            return '_'
        return self._to_snake_case(name)
    
    def emit_literal(self, value: Any, literal_type: str = 'auto') -> str:
        """Emit literal value in Elixir syntax."""
        if literal_type == 'atom':
            atom_val = value[1:] if isinstance(value, str) and value.startswith(':') else value
            if self._needs_quoting(str(atom_val)):
                return f':"{atom_val}"'
            return f':{atom_val}'
        
        if isinstance(value, bool):
            return 'true' if value else 'false'
        elif isinstance(value, int):
            return str(value)
        elif isinstance(value, float):
            return str(value)
        elif isinstance(value, str):
            if value.startswith(':'):
                # Atom
                atom_val = value[1:]
                if self._needs_quoting(atom_val):
                    return f':"{atom_val}"'
                return f':{atom_val}'
            # String
            escaped = value.replace('\\', '\\\\').replace('"', '\\"')
            return f'"{escaped}"'
        elif value is None:
            return 'nil'
        return str(value)
    
    def _needs_quoting(self, atom: str) -> bool:
        """Check if atom needs quoting."""
        if not atom:
            return True
        if not (atom[0].islower() or atom[0] == '_'):
            return True
        return not all(c.isalnum() or c in '_!?' for c in atom)
    
    # =========================================================================
    # Guard Emission
    # =========================================================================
    
    def _join_guard_conditions(self, conditions: List[str]) -> str:
        """Join conditions within a guard group (and)."""
        return ' and '.join(conditions)
    
    def _join_guard_groups(self, groups: List[str]) -> str:
        """Join guard groups (or)."""
        return ' or '.join(groups)
    
    # =========================================================================
    # Expression Emission
    # =========================================================================
    
    def _emit_literal_expr(self, expr: LiteralExpr) -> str:
        """Emit literal expression."""
        return self.emit_literal(expr.value, expr.literal_type)
    
    def _emit_atom_expr(self, expr: AtomExpr) -> str:
        """Emit atom expression."""
        return self.emit_literal(expr.value, 'atom')
    
    def _emit_tuple_expr(self, expr: TupleExpr) -> str:
        """Emit tuple expression."""
        elements = ', '.join(self._emit_expr(e) for e in expr.elements)
        return f'{{{elements}}}'
    
    def _emit_list_expr(self, expr: ListExpr) -> str:
        """Emit list expression."""
        elements = ', '.join(self._emit_expr(e) for e in expr.elements)
        if expr.tail:
            tail = self._emit_expr(expr.tail)
            return f'[{elements} | {tail}]'
        return f'[{elements}]'
    
    def _emit_map_expr(self, expr: MapExpr) -> str:
        """Emit map expression."""
        pairs = ', '.join(
            f'{self._emit_expr(k)} => {self._emit_expr(v)}'
            for k, v in expr.pairs
        )
        return f'%{{{pairs}}}'
    
    def _emit_binary_expr(self, expr: BinaryExpr) -> str:
        """Emit binary expression."""
        segments = ', '.join(self._emit_binary_segment_elixir(s) for s in expr.segments)
        return f'<<{segments}>>'
    
    def _emit_binary_segment_elixir(self, s: BinarySegment) -> str:
        """Emit binary segment for Elixir."""
        if hasattr(s.value, 'kind'):
            value_str = self.emit_pattern(s.value)
        else:
            value_str = str(s.value)
        
        specs = []
        if s.size:
            specs.append(f'size({s.size})')
        if s.type_spec != 'integer':
            specs.append(s.type_spec)
        if specs:
            return f'{value_str}::{"-".join(specs)}'
        return value_str
    
    def _emit_call_expr(self, expr: CallExpr) -> str:
        """Emit function call."""
        args = ', '.join(self._emit_expr(a) for a in expr.args)
        fn_name = self._to_snake_case(expr.function)
        if expr.module:
            mod_name = self._to_elixir_module_name(expr.module)
            return f'{mod_name}.{fn_name}({args})'
        return f'{fn_name}({args})'
    
    def _emit_send_expr(self, expr: SendExpr) -> str:
        """Emit send expression."""
        target = self._emit_expr(expr.target)
        message = self._emit_expr(expr.message)
        return f'send({target}, {message})'
    
    def _emit_receive_expr(self, expr: ReceiveExpr) -> str:
        """Emit receive block."""
        lines = ['receive do']
        self.indent_level += 1
        
        for clause in expr.clauses:
            clause_str = self._emit_receive_clause(clause)
            lines.append(f'{self.indent()}{clause_str}')
        
        self.indent_level -= 1
        
        if expr.timeout is not None:
            lines.append('after')
            self.indent_level += 1
            lines.append(f'{self.indent()}{self._emit_expr(expr.timeout)} ->')
            self.indent_level += 1
            for e in expr.timeout_body:
                lines.append(f'{self.indent()}{self._emit_expr(e)}')
            self.indent_level -= 2
        
        lines.append('end')
        return '\n'.join(lines)
    
    def _emit_receive_clause(self, clause: ReceiveClause) -> str:
        """Emit receive clause."""
        pattern = self.emit_pattern(clause.pattern)
        
        if clause.guards:
            guard_str = self._emit_guards(clause.guards)
            pattern = f'{pattern} when {guard_str}'
        
        if len(clause.body) == 1:
            body = self._emit_expr(clause.body[0])
            return f'{pattern} -> {body}'
        
        lines = [f'{pattern} ->']
        self.indent_level += 1
        for e in clause.body:
            lines.append(f'{self.indent()}{self._emit_expr(e)}')
        self.indent_level -= 1
        return '\n'.join(lines)
    
    def _emit_spawn_expr(self, expr: SpawnExpr) -> str:
        """Emit spawn expression."""
        args = ', '.join(self._emit_expr(a) for a in expr.args)
        
        spawn_fn = {
            SpawnStrategy.SPAWN: 'spawn',
            SpawnStrategy.SPAWN_LINK: 'spawn_link',
            SpawnStrategy.SPAWN_MONITOR: 'spawn_monitor',
        }.get(expr.strategy, 'spawn')
        
        if expr.module:
            mod_name = self._to_elixir_module_name(expr.module)
            fn_name = self._to_snake_case(expr.function)
            return f'{spawn_fn}({mod_name}, :{fn_name}, [{args}])'
        fn_name = self._to_snake_case(expr.function)
        return f'{spawn_fn}(fn -> {fn_name}({args}) end)'
    
    def _emit_link_expr(self, expr: LinkExpr) -> str:
        """Emit link expression."""
        return f'Process.link({self._emit_expr(expr.target)})'
    
    def _emit_monitor_expr(self, expr: MonitorExpr) -> str:
        """Emit monitor expression."""
        return f'Process.monitor({self._emit_expr(expr.target)})'
    
    def _emit_case_expr(self, expr: CaseExpr) -> str:
        """Emit case expression."""
        lines = [f'case {self._emit_expr(expr.expr)} do']
        self.indent_level += 1
        
        for clause in expr.clauses:
            clause_str = self._emit_case_clause(clause)
            lines.append(f'{self.indent()}{clause_str}')
        
        self.indent_level -= 1
        lines.append('end')
        return '\n'.join(lines)
    
    def _emit_case_clause(self, clause: CaseClause) -> str:
        """Emit case clause."""
        pattern = self.emit_pattern(clause.pattern)
        
        if clause.guards:
            guard_str = self._emit_guards(clause.guards)
            pattern = f'{pattern} when {guard_str}'
        
        if len(clause.body) == 1:
            body = self._emit_expr(clause.body[0])
            return f'{pattern} -> {body}'
        
        lines = [f'{pattern} ->']
        self.indent_level += 1
        for e in clause.body:
            lines.append(f'{self.indent()}{self._emit_expr(e)}')
        self.indent_level -= 1
        return '\n'.join(lines)
    
    def _emit_fun_expr(self, expr: FunExpr) -> str:
        """Emit anonymous function."""
        if len(expr.clauses) == 1:
            clause = expr.clauses[0]
            args = ', '.join(self.emit_pattern(a) for a in clause.args)
            body = ', '.join(self._emit_expr(e) for e in clause.body)
            return f'fn {args} -> {body} end'
        
        # Multi-clause anonymous function
        lines = ['fn']
        self.indent_level += 1
        for clause in expr.clauses:
            args = ', '.join(self.emit_pattern(a) for a in clause.args)
            body = ', '.join(self._emit_expr(e) for e in clause.body)
            lines.append(f'{self.indent()}{args} -> {body}')
        self.indent_level -= 1
        lines.append('end')
        return '\n'.join(lines)
    
    def _emit_list_comp_expr(self, expr: ListCompExpr) -> str:
        """Emit list comprehension (for)."""
        result = self._emit_expr(expr.expr)
        qualifiers = ', '.join(self._emit_qualifier(q) for q in expr.qualifiers)
        return f'for {qualifiers}, do: {result}'
    
    def _emit_qualifier(self, q) -> str:
        """Emit qualifier."""
        if q.kind == 'generator':
            pattern = self.emit_pattern(q.pattern)
            list_expr = self._emit_expr(q.list_expr)
            return f'{pattern} <- {list_expr}'
        elif q.kind == 'filter':
            return self._emit_expr(q.condition)
        return ''
    
    def _emit_try_expr(self, expr: TryExpr) -> str:
        """Emit try expression."""
        lines = ['try do']
        self.indent_level += 1
        for e in expr.body:
            lines.append(f'{self.indent()}{self._emit_expr(e)}')
        self.indent_level -= 1
        
        if expr.catch_clauses:
            lines.append('rescue')
            self.indent_level += 1
            for clause in expr.catch_clauses:
                clause_str = self._emit_rescue_clause(clause)
                lines.append(f'{self.indent()}{clause_str}')
            self.indent_level -= 1
        
        if expr.after_body:
            lines.append('after')
            self.indent_level += 1
            for e in expr.after_body:
                lines.append(f'{self.indent()}{self._emit_expr(e)}')
            self.indent_level -= 1
        
        lines.append('end')
        return '\n'.join(lines)
    
    def _emit_rescue_clause(self, clause: CatchClause) -> str:
        """Emit rescue clause."""
        pattern = self.emit_pattern(clause.pattern)
        body = ', '.join(self._emit_expr(e) for e in clause.body)
        return f'{pattern} -> {body}'
    
    def _emit_pipe_expr(self, expr) -> str:
        """Emit pipe expression."""
        left = self._emit_expr(expr.left)
        right = self._emit_expr(expr.right)
        return f'{left} |> {right}'
    
    # =========================================================================
    # Pattern Emission
    # =========================================================================
    
    def _emit_tuple_pattern(self, p: TuplePattern) -> str:
        """Emit tuple pattern."""
        elements = ', '.join(self.emit_pattern(e) for e in p.elements)
        return f'{{{elements}}}'
    
    def _emit_list_pattern(self, p: ListPattern) -> str:
        """Emit list pattern."""
        elements = ', '.join(self.emit_pattern(e) for e in p.elements)
        if p.tail:
            return f'[{elements} | {self.emit_pattern(p.tail)}]'
        return f'[{elements}]'
    
    def _emit_map_pattern(self, p: MapPattern) -> str:
        """Emit map pattern."""
        pairs = ', '.join(
            f'{self._emit_term(k)} => {self.emit_pattern(v)}'
            for k, v in p.pairs
        )
        return f'%{{{pairs}}}'
    
    def _emit_binary_pattern(self, p: BinaryPattern) -> str:
        """Emit binary pattern."""
        segments = ', '.join(self._emit_binary_segment_elixir(s) for s in p.segments)
        return f'<<{segments}>>'
    
    # =========================================================================
    # OTP Behavior Emission
    # =========================================================================
    
    def _behavior_type_to_elixir(self, bt: BehaviorType) -> str:
        """Convert behavior type to Elixir module name."""
        mapping = {
            BehaviorType.GEN_SERVER: 'GenServer',
            BehaviorType.SUPERVISOR: 'Supervisor',
            BehaviorType.APPLICATION: 'Application',
            BehaviorType.GEN_STATEM: 'GenStateMachine',
        }
        return mapping.get(bt, bt.value)
    
    def _emit_behavior(self, behavior: BehaviorImpl) -> List[str]:
        """Emit behavior implementation."""
        bt = behavior.behavior_type
        impl = behavior.implementation
        
        if bt == BehaviorType.GEN_SERVER:
            return self._emit_gen_server(impl)
        elif bt == BehaviorType.SUPERVISOR:
            return self._emit_supervisor(impl)
        elif bt == BehaviorType.APPLICATION:
            return self._emit_application(impl)
        return []
    
    def _emit_gen_server(self, impl: GenServerImpl) -> List[str]:
        """Emit GenServer callbacks."""
        lines = []
        
        # Client API
        lines.append(f'{self.indent()}# Client API')
        lines.append(f'{self.indent()}def start_link(args \\\\ []) do')
        self.indent_level += 1
        lines.append(f'{self.indent()}GenServer.start_link(__MODULE__, args, name: __MODULE__)')
        self.indent_level -= 1
        lines.append(f'{self.indent()}end')
        lines.append('')
        
        # Server callbacks
        lines.append(f'{self.indent()}# Server Callbacks')
        lines.append(f'{self.indent()}@impl true')
        
        # init/1
        if impl.init_callback:
            lines.append(self._emit_elixir_callback(impl.init_callback))
        else:
            initial = self._emit_expr(impl.state.initial_state) if impl.state and impl.state.initial_state else '%{}'
            lines.append(f'{self.indent()}def init(_args), do: {{:ok, {initial}}}')
        lines.append('')
        
        # handle_call/3
        lines.append(f'{self.indent()}@impl true')
        for cb in impl.handle_call_callbacks:
            lines.append(self._emit_elixir_callback(cb))
        if not impl.handle_call_callbacks:
            lines.append(f'{self.indent()}def handle_call(_request, _from, state), do: {{:reply, :ok, state}}')
        lines.append('')
        
        # handle_cast/2
        lines.append(f'{self.indent()}@impl true')
        for cb in impl.handle_cast_callbacks:
            lines.append(self._emit_elixir_callback(cb))
        if not impl.handle_cast_callbacks:
            lines.append(f'{self.indent()}def handle_cast(_msg, state), do: {{:noreply, state}}')
        lines.append('')
        
        # handle_info/2
        lines.append(f'{self.indent()}@impl true')
        for cb in impl.handle_info_callbacks:
            lines.append(self._emit_elixir_callback(cb))
        if not impl.handle_info_callbacks:
            lines.append(f'{self.indent()}def handle_info(_info, state), do: {{:noreply, state}}')
        
        return lines
    
    def _emit_elixir_callback(self, cb: Callback) -> str:
        """Emit callback as Elixir function."""
        args = ', '.join(cb.args)
        fn_name = self._to_snake_case(cb.name)
        
        if len(cb.body) == 1:
            body = self._emit_expr(cb.body[0])
            return f'{self.indent()}def {fn_name}({args}), do: {body}'
        
        lines = [f'{self.indent()}def {fn_name}({args}) do']
        self.indent_level += 1
        for expr in cb.body:
            lines.append(f'{self.indent()}{self._emit_expr(expr)}')
        self.indent_level -= 1
        lines.append(f'{self.indent()}end')
        return '\n'.join(lines)
    
    def _emit_supervisor(self, impl: SupervisorImpl) -> List[str]:
        """Emit Supervisor callbacks."""
        lines = []
        
        # start_link
        lines.append(f'{self.indent()}def start_link(init_arg) do')
        self.indent_level += 1
        lines.append(f'{self.indent()}Supervisor.start_link(__MODULE__, init_arg, name: __MODULE__)')
        self.indent_level -= 1
        lines.append(f'{self.indent()}end')
        lines.append('')
        
        # init/1
        lines.append(f'{self.indent()}@impl true')
        lines.append(f'{self.indent()}def init(_init_arg) do')
        self.indent_level += 1
        
        # Children
        lines.append(f'{self.indent()}children = [')
        self.indent_level += 1
        for child in impl.children:
            child_spec = self._emit_elixir_child_spec(child)
            lines.append(f'{self.indent()}{child_spec},')
        self.indent_level -= 1
        lines.append(f'{self.indent()}]')
        lines.append('')
        
        # Strategy
        flags = impl.flags or SupervisorFlags()
        strategy = f':{flags.strategy.value}'
        lines.append(f'{self.indent()}Supervisor.init(children, strategy: {strategy}, max_restarts: {flags.intensity}, max_seconds: {flags.period})')
        
        self.indent_level -= 1
        lines.append(f'{self.indent()}end')
        
        return lines
    
    def _emit_elixir_child_spec(self, child: ChildSpec) -> str:
        """Emit child spec for Elixir Supervisor."""
        mod, func, args = child.start
        mod_name = self._to_elixir_module_name(mod)
        
        # Simple form: {Module, args}
        if func == 'start_link' and len(args) <= 1:
            arg = self._emit_term(args[0]) if args else '[]'
            return f'{{{mod_name}, {arg}}}'
        
        # Full spec
        return (
            f'%{{'
            f'id: :{child.id}, '
            f'start: {{{mod_name}, :{func}, {self._emit_term(list(args))}}}, '
            f'restart: :{child.restart.value}'
            f'}}'
        )
    
    def _emit_application(self, impl: ApplicationImpl) -> List[str]:
        """Emit Application callbacks."""
        mod, args = impl.mod
        mod_name = self._to_elixir_module_name(mod)
        
        return [
            f'{self.indent()}@impl true',
            f'{self.indent()}def start(_type, _args) do',
            f'{self.indent()}  {mod_name}.start_link({self._emit_term(args)})',
            f'{self.indent()}end',
        ]
    
    def _emit_term(self, value: Any) -> str:
        """Emit Elixir term."""
        if isinstance(value, str):
            if value.startswith(':'):
                return value  # Already an atom
            return f'"{value}"'
        elif isinstance(value, bool):
            return 'true' if value else 'false'
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, list):
            return f'[{", ".join(self._emit_term(v) for v in value)}]'
        elif isinstance(value, tuple):
            return f'{{", ".join(self._emit_term(v) for v in value)}}'
        elif isinstance(value, dict):
            pairs = ', '.join(f'{k}: {self._emit_term(v)}' for k, v in value.items())
            return f'%{{{pairs}}}'
        elif value is None:
            return 'nil'
        return str(value)


# Additional Elixir-specific expression types
from dataclasses import dataclass, field

@dataclass
class PipeExpr(Expr):
    """Pipe operator expression (|>)."""
    left: Expr = None
    right: Expr = None
    kind: str = 'pipe'


@dataclass
class WithExpr(Expr):
    """With expression for chained pattern matches."""
    clauses: List[tuple] = field(default_factory=list)  # [(pattern, expr), ...]
    body: List[Expr] = field(default_factory=list)
    else_clauses: List[tuple] = field(default_factory=list)
    kind: str = 'with'


@dataclass
class StructExpr(Expr):
    """Struct expression %Module{}."""
    module: str = ''
    fields: Dict[str, Expr] = field(default_factory=dict)
    kind: str = 'struct'


@dataclass
class SigilExpr(Expr):
    """Sigil expression ~s(), ~r(), etc."""
    sigil: str = ''  # s, r, w, etc.
    content: str = ''
    modifiers: str = ''
    kind: str = 'sigil'
