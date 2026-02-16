#!/usr/bin/env python3
"""STUNIR Erlang Code Emitter.

This module provides the Erlang code emitter for the actor model IR,
generating idiomatic Erlang/OTP code.

Features:
    - Module definitions (-module, -export)
    - Function definitions with pattern matching
    - Process spawning (spawn, spawn_link, spawn_monitor)
    - Message passing (!, receive)
    - OTP behaviors (gen_server, supervisor)
    - Binary pattern matching

Usage:
    from targets.beam import ErlangEmitter
    from ir.actor import ActorModule
    
    emitter = ErlangEmitter()
    result = emitter.emit_module(module)
    print(result.code)
"""

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


class ErlangEmitter(BEAMEmitterBase):
    """Erlang code emitter.
    
    Generates Erlang source code from Actor Model IR,
    including OTP behavior implementations.
    """
    
    LANGUAGE = 'erlang'
    VERSION = '1.0.0'
    
    def _emit_module_impl(self, module: ActorModule) -> str:
        """Emit Erlang module."""
        lines = []
        
        # Module declaration
        lines.append(f"-module({module.name}).")
        lines.append("")
        
        # Behavior declarations
        for behavior in module.behaviors:
            behavior_name = self._behavior_type_to_erlang(behavior.behavior_type)
            lines.append(f"-behaviour({behavior_name}).")
        if module.behaviors:
            lines.append("")
        
        # Exports
        if module.exports:
            exports = ', '.join(module.exports)
            lines.append(f"-export([{exports}]).")
        
        # OTP callback exports
        for behavior in module.behaviors:
            callbacks = self._get_behavior_callbacks(behavior)
            if callbacks:
                lines.append(f"-export([{', '.join(callbacks)}]).")
        lines.append("")
        
        # Custom attributes
        for key, value in module.attributes.items():
            lines.append(f"-{key}({self._emit_term(value)}).")
        if module.attributes:
            lines.append("")
        
        # Functions
        for func in module.functions:
            lines.append(self._emit_function(func))
            lines.append("")
        
        # Behavior implementations
        for behavior in module.behaviors:
            behavior_lines = self._emit_behavior(behavior)
            lines.extend(behavior_lines)
        
        return '\n'.join(lines)
    
    # =========================================================================
    # Function Emission
    # =========================================================================
    
    def _emit_function(self, func: FunctionDef) -> str:
        """Emit function definition."""
        if not func.clauses:
            return f"{func.name}() -> ok."
        
        clauses = []
        for i, clause in enumerate(func.clauses):
            clause_str = self._emit_function_clause(func.name, clause)
            # Add period or semicolon
            if i < len(func.clauses) - 1:
                clause_str += ';'
            else:
                clause_str += '.'
            clauses.append(clause_str)
        return '\n'.join(clauses)
    
    def _emit_function_clause(self, name: str, clause: FunctionClause) -> str:
        """Emit a single function clause."""
        args = ', '.join(self.emit_pattern(arg) for arg in clause.args)
        head = f"{name}({args})"
        
        # Guards
        if clause.guards:
            guard_str = self._emit_guards(clause.guards)
            head = f"{head} when {guard_str}"
        
        # Body
        if not clause.body:
            return f"{head} ->\n    ok"
        
        body_lines = [self._emit_expr(e) for e in clause.body]
        body = ',\n    '.join(body_lines)
        
        return f"{head} ->\n    {body}"
    
    # =========================================================================
    # Variable/Literal Transformation
    # =========================================================================
    
    def _transform_var(self, name: str) -> str:
        """Transform variable name to Erlang syntax (capitalized)."""
        if name == '_':
            return '_'
        if not name:
            return '_'
        # Capitalize first letter for Erlang
        return name[0].upper() + name[1:]
    
    def emit_literal(self, value: Any, literal_type: str = 'auto') -> str:
        """Emit literal value in Erlang syntax."""
        if literal_type == 'atom':
            atom_val = value[1:] if isinstance(value, str) and value.startswith(':') else value
            if self._needs_quoting(str(atom_val)):
                return f"'{atom_val}'"
            return str(atom_val)
        
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
                    return f"'{atom_val}'"
                return atom_val
            # String
            escaped = value.replace('\\', '\\\\').replace('"', '\\"')
            return f'"{escaped}"'
        elif value is None:
            return 'undefined'
        return str(value)
    
    def _needs_quoting(self, atom: str) -> bool:
        """Check if atom needs quoting."""
        if not atom:
            return True
        if not atom[0].islower():
            return True
        return not all(c.isalnum() or c == '_' for c in atom)
    
    # =========================================================================
    # Guard Emission
    # =========================================================================
    
    def _join_guard_conditions(self, conditions: List[str]) -> str:
        """Join conditions within a guard group (comma = AND)."""
        return ', '.join(conditions)
    
    def _join_guard_groups(self, groups: List[str]) -> str:
        """Join guard groups (semicolon = OR)."""
        return '; '.join(groups)
    
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
            return f'[{elements}|{tail}]'
        return f'[{elements}]'
    
    def _emit_map_expr(self, expr: MapExpr) -> str:
        """Emit map expression."""
        pairs = ', '.join(
            f'{self._emit_expr(k)} => {self._emit_expr(v)}'
            for k, v in expr.pairs
        )
        return f'#{{{pairs}}}'
    
    def _emit_binary_expr(self, expr: BinaryExpr) -> str:
        """Emit binary expression."""
        segments = ', '.join(self._emit_binary_segment(s) for s in expr.segments)
        return f'<<{segments}>>'
    
    def _emit_binary_segment(self, s: BinarySegment) -> str:
        """Emit binary segment."""
        if hasattr(s.value, 'kind'):
            value_str = self.emit_pattern(s.value)
        else:
            value_str = str(s.value)
        
        specs = []
        if s.size:
            specs.append(str(s.size))
        if s.type_spec != 'integer':
            specs.append(s.type_spec)
        if specs:
            return f'{value_str}:{"/".join(specs)}'
        return value_str
    
    def _emit_call_expr(self, expr: CallExpr) -> str:
        """Emit function call."""
        args = ', '.join(self._emit_expr(a) for a in expr.args)
        if expr.module:
            return f'{expr.module}:{expr.function}({args})'
        return f'{expr.function}({args})'
    
    def _emit_send_expr(self, expr: SendExpr) -> str:
        """Emit send expression (!)."""
        target = self._emit_expr(expr.target)
        message = self._emit_expr(expr.message)
        return f'{target} ! {message}'
    
    def _emit_receive_expr(self, expr: ReceiveExpr) -> str:
        """Emit receive block."""
        lines = ['receive']
        for i, clause in enumerate(expr.clauses):
            clause_str = self._emit_receive_clause(clause)
            if i < len(expr.clauses) - 1:
                clause_str += ';'
            lines.append(f'    {clause_str}')
        
        if expr.timeout is not None:
            lines.append(f'after {self._emit_expr(expr.timeout)} ->')
            for e in expr.timeout_body:
                lines.append(f'    {self._emit_expr(e)}')
        
        lines.append('end')
        return '\n'.join(lines)
    
    def _emit_receive_clause(self, clause: ReceiveClause) -> str:
        """Emit receive clause."""
        pattern = self.emit_pattern(clause.pattern)
        
        if clause.guards:
            guard_str = self._emit_guards(clause.guards)
            pattern = f'{pattern} when {guard_str}'
        
        body_lines = [self._emit_expr(e) for e in clause.body]
        body = ',\n        '.join(body_lines)
        
        return f'{pattern} ->\n        {body}'
    
    def _emit_spawn_expr(self, expr: SpawnExpr) -> str:
        """Emit spawn expression."""
        args = ', '.join(self._emit_expr(a) for a in expr.args)
        
        spawn_fn = {
            SpawnStrategy.SPAWN: 'spawn',
            SpawnStrategy.SPAWN_LINK: 'spawn_link',
            SpawnStrategy.SPAWN_MONITOR: 'spawn_monitor',
        }.get(expr.strategy, 'spawn')
        
        if expr.module:
            return f'{spawn_fn}({expr.module}, {expr.function}, [{args}])'
        return f'{spawn_fn}(fun() -> {expr.function}({args}) end)'
    
    def _emit_link_expr(self, expr: LinkExpr) -> str:
        """Emit link expression."""
        return f'link({self._emit_expr(expr.target)})'
    
    def _emit_monitor_expr(self, expr: MonitorExpr) -> str:
        """Emit monitor expression."""
        return f'monitor(process, {self._emit_expr(expr.target)})'
    
    def _emit_case_expr(self, expr: CaseExpr) -> str:
        """Emit case expression."""
        lines = [f'case {self._emit_expr(expr.expr)} of']
        for i, clause in enumerate(expr.clauses):
            clause_str = self._emit_case_clause(clause)
            if i < len(expr.clauses) - 1:
                clause_str += ';'
            lines.append(f'    {clause_str}')
        lines.append('end')
        return '\n'.join(lines)
    
    def _emit_case_clause(self, clause: CaseClause) -> str:
        """Emit case clause."""
        pattern = self.emit_pattern(clause.pattern)
        
        if clause.guards:
            guard_str = self._emit_guards(clause.guards)
            pattern = f'{pattern} when {guard_str}'
        
        body_lines = [self._emit_expr(e) for e in clause.body]
        body = ',\n        '.join(body_lines)
        
        return f'{pattern} ->\n        {body}'
    
    def _emit_fun_expr(self, expr: FunExpr) -> str:
        """Emit anonymous function."""
        clauses = []
        for i, clause in enumerate(expr.clauses):
            args = ', '.join(self.emit_pattern(a) for a in clause.args)
            body = ', '.join(self._emit_expr(e) for e in clause.body)
            
            clause_str = f'({args}) -> {body}'
            if clause.guards:
                guard_str = self._emit_guards(clause.guards)
                clause_str = f'({args}) when {guard_str} -> {body}'
            
            if i < len(expr.clauses) - 1:
                clause_str += ';'
            clauses.append(clause_str)
        
        return f'fun {"; ".join(clauses)} end'
    
    def _emit_list_comp_expr(self, expr: ListCompExpr) -> str:
        """Emit list comprehension."""
        result = self._emit_expr(expr.expr)
        qualifiers = ', '.join(self._emit_qualifier(q) for q in expr.qualifiers)
        return f'[{result} || {qualifiers}]'
    
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
        """Emit try-catch expression."""
        lines = ['try']
        for e in expr.body:
            lines.append(f'    {self._emit_expr(e)}')
        
        if expr.catch_clauses:
            lines.append('catch')
            for i, clause in enumerate(expr.catch_clauses):
                clause_str = self._emit_catch_clause(clause)
                if i < len(expr.catch_clauses) - 1:
                    clause_str += ';'
                lines.append(f'    {clause_str}')
        
        if expr.after_body:
            lines.append('after')
            for e in expr.after_body:
                lines.append(f'    {self._emit_expr(e)}')
        
        lines.append('end')
        return '\n'.join(lines)
    
    def _emit_catch_clause(self, clause: CatchClause) -> str:
        """Emit catch clause."""
        pattern = self.emit_pattern(clause.pattern)
        exc_type = clause.exception_type
        
        if clause.stacktrace:
            pattern = f'{exc_type}:{pattern}:{clause.stacktrace}'
        else:
            pattern = f'{exc_type}:{pattern}'
        
        body = ', '.join(self._emit_expr(e) for e in clause.body)
        return f'{pattern} -> {body}'
    
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
            return f'[{elements}|{self.emit_pattern(p.tail)}]'
        return f'[{elements}]'
    
    def _emit_map_pattern(self, p: MapPattern) -> str:
        """Emit map pattern."""
        pairs = ', '.join(
            f'{self._emit_term(k)} := {self.emit_pattern(v)}'
            for k, v in p.pairs
        )
        return f'#{{{pairs}}}'
    
    def _emit_binary_pattern(self, p: BinaryPattern) -> str:
        """Emit binary pattern."""
        segments = ', '.join(self._emit_binary_segment(s) for s in p.segments)
        return f'<<{segments}>>'
    
    # =========================================================================
    # OTP Behavior Emission
    # =========================================================================
    
    def _behavior_type_to_erlang(self, bt: BehaviorType) -> str:
        """Convert behavior type to Erlang name."""
        return bt.value
    
    def _get_behavior_callbacks(self, behavior: BehaviorImpl) -> List[str]:
        """Get callback exports for behavior."""
        bt = behavior.behavior_type
        if bt == BehaviorType.GEN_SERVER:
            return ['init/1', 'handle_call/3', 'handle_cast/2', 'handle_info/2',
                    'terminate/2', 'code_change/3']
        elif bt == BehaviorType.SUPERVISOR:
            return ['init/1']
        elif bt == BehaviorType.APPLICATION:
            return ['start/2', 'stop/1']
        return []
    
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
        """Emit gen_server callbacks."""
        lines = []
        
        # init/1
        if impl.init_callback:
            lines.append(self._emit_callback_as_function(impl.init_callback))
        else:
            initial = self._emit_expr(impl.state.initial_state) if impl.state and impl.state.initial_state else '#{}'
            lines.append(f'init(_Args) ->\n    {{ok, {initial}}}.')
        lines.append('')
        
        # handle_call/3
        for cb in impl.handle_call_callbacks:
            lines.append(self._emit_callback_as_function(cb))
        if not impl.handle_call_callbacks:
            lines.append('handle_call(_Request, _From, State) ->\n    {reply, ok, State}.')
        lines.append('')
        
        # handle_cast/2
        for cb in impl.handle_cast_callbacks:
            lines.append(self._emit_callback_as_function(cb))
        if not impl.handle_cast_callbacks:
            lines.append('handle_cast(_Msg, State) ->\n    {noreply, State}.')
        lines.append('')
        
        # handle_info/2
        for cb in impl.handle_info_callbacks:
            lines.append(self._emit_callback_as_function(cb))
        if not impl.handle_info_callbacks:
            lines.append('handle_info(_Info, State) ->\n    {noreply, State}.')
        lines.append('')
        
        # terminate/2
        if impl.terminate_callback:
            lines.append(self._emit_callback_as_function(impl.terminate_callback))
        else:
            lines.append('terminate(_Reason, _State) ->\n    ok.')
        lines.append('')
        
        # code_change/3
        if impl.code_change_callback:
            lines.append(self._emit_callback_as_function(impl.code_change_callback))
        else:
            lines.append('code_change(_OldVsn, State, _Extra) ->\n    {ok, State}.')
        
        return lines
    
    def _emit_callback_as_function(self, cb: Callback) -> str:
        """Emit callback as function."""
        args = ', '.join(cb.args)
        body_lines = [self._emit_expr(e) for e in cb.body]
        body = ',\n    '.join(body_lines) if body_lines else 'ok'
        return f'{cb.name}({args}) ->\n    {body}.'
    
    def _emit_supervisor(self, impl: SupervisorImpl) -> List[str]:
        """Emit supervisor init/1."""
        lines = []
        
        # Flags
        flags = impl.flags or SupervisorFlags()
        flags_str = (
            f'#{{strategy => {flags.strategy.value}, '
            f'intensity => {flags.intensity}, '
            f'period => {flags.period}}}'
        )
        
        # Child specs
        children = [self._emit_child_spec(child) for child in impl.children]
        children_str = ',\n        '.join(children)
        
        lines.append(f'init(_Args) ->')
        lines.append(f'    SupFlags = {flags_str},')
        lines.append(f'    ChildSpecs = [\n        {children_str}\n    ],')
        lines.append(f'    {{ok, {{SupFlags, ChildSpecs}}}}.')
        
        return lines
    
    def _emit_child_spec(self, child: ChildSpec) -> str:
        """Emit child specification."""
        mod, func, args = child.start
        args_str = ', '.join(self._emit_term(a) for a in args)
        
        return (
            f'#{{id => {child.id}, '
            f'start => {{{mod}, {func}, [{args_str}]}}, '
            f'restart => {child.restart.value}, '
            f'shutdown => {child.shutdown}, '
            f'type => {child.type.value}}}'
        )
    
    def _emit_application(self, impl: ApplicationImpl) -> List[str]:
        """Emit application callbacks."""
        mod, args = impl.mod
        args_str = self._emit_term(args)
        
        return [
            f'start(_StartType, _StartArgs) ->',
            f'    {mod}:start_link({args_str}).',
            '',
            f'stop(_State) ->',
            f'    ok.'
        ]
    
    def _emit_term(self, value: Any) -> str:
        """Emit Erlang term."""
        if isinstance(value, str):
            if value.startswith(':'):
                return value[1:]  # Atom
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
            pairs = ', '.join(f'{k} => {self._emit_term(v)}' for k, v in value.items())
            return f'#{{{pairs}}}'
        elif value is None:
            return 'undefined'
        return str(value)
