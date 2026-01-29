#!/usr/bin/env python3
"""Tests for Actor Model IR.

Tests cover:
    - Core IR classes (ActorModule, expressions)
    - Process definitions and spawning
    - Message passing and patterns
    - OTP behavior implementations
    - Serialization
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ir.actor import (
    # Core
    ActorModule, ActorNode, ActorNodeKind,
    # Expressions
    Expr, LiteralExpr, VarExpr, AtomExpr, TupleExpr, ListExpr, MapExpr,
    CallExpr, CaseExpr, CaseClause, FunExpr, FunClause,
    ListCompExpr, Generator, Filter, TryExpr, CatchClause,
    Guard, BinaryExpr, BinarySegment,
    # Functions
    FunctionDef, FunctionClause,
    # Process
    ProcessDef, SpawnExpr, SpawnStrategy, LinkExpr, MonitorExpr, SelfExpr,
    ProcessFlag,
    # Messages
    SendExpr, ReceiveExpr, ReceiveClause,
    Pattern, WildcardPattern, VarPattern, LiteralPattern, AtomPattern,
    TuplePattern, ListPattern, MapPattern, BinaryPattern,
    # OTP
    BehaviorType, BehaviorImpl,
    GenServerImpl, GenServerState, Callback,
    SupervisorImpl, SupervisorFlags, SupervisorStrategy,
    ChildSpec, ChildRestart, ChildType,
    ApplicationImpl,
)


class TestActorModule:
    """Tests for ActorModule."""
    
    def test_create_empty_module(self):
        """Test creating an empty actor module."""
        module = ActorModule(name='test')
        assert module.name == 'test'
        assert module.kind == 'actor_module'
        assert module.exports == []
        assert module.functions == []
    
    def test_create_module_with_exports(self):
        """Test creating module with exports."""
        module = ActorModule(
            name='my_server',
            exports=['start/0', 'handle/1']
        )
        assert 'start/0' in module.exports
        assert 'handle/1' in module.exports
    
    def test_module_with_functions(self):
        """Test creating module with functions."""
        func = FunctionDef(
            name='hello',
            arity=0,
            exported=True,
            clauses=[FunctionClause(args=[], body=[AtomExpr(value='world')])]
        )
        module = ActorModule(name='test', functions=[func])
        assert len(module.functions) == 1
        assert module.functions[0].name == 'hello'


class TestExpressions:
    """Tests for expression types."""
    
    def test_literal_expr(self):
        """Test literal expression."""
        expr = LiteralExpr(value=42, literal_type='integer')
        assert expr.value == 42
        assert expr.kind == 'literal'
    
    def test_atom_expr(self):
        """Test atom expression."""
        expr = AtomExpr(value='ok')
        assert expr.value == 'ok'
        assert expr.kind == 'atom'
    
    def test_var_expr(self):
        """Test variable expression."""
        expr = VarExpr(name='state')
        assert expr.name == 'state'
        assert expr.kind == 'var'
    
    def test_tuple_expr(self):
        """Test tuple expression."""
        expr = TupleExpr(elements=[
            AtomExpr(value='ok'),
            LiteralExpr(value=42)
        ])
        assert len(expr.elements) == 2
        assert expr.kind == 'tuple'
    
    def test_list_expr(self):
        """Test list expression."""
        expr = ListExpr(elements=[
            LiteralExpr(value=1),
            LiteralExpr(value=2)
        ])
        assert len(expr.elements) == 2
        assert expr.tail is None
    
    def test_list_with_tail(self):
        """Test list with tail (cons)."""
        expr = ListExpr(
            elements=[VarExpr(name='head')],
            tail=VarExpr(name='tail')
        )
        assert expr.tail is not None
    
    def test_map_expr(self):
        """Test map expression."""
        expr = MapExpr(pairs=[
            (AtomExpr(value='key'), LiteralExpr(value='value'))
        ])
        assert len(expr.pairs) == 1
    
    def test_call_expr(self):
        """Test function call expression."""
        expr = CallExpr(
            module='io',
            function='format',
            args=[LiteralExpr(value='Hello~n', literal_type='string')]
        )
        assert expr.module == 'io'
        assert expr.function == 'format'
    
    def test_case_expr(self):
        """Test case expression."""
        expr = CaseExpr(
            expr=VarExpr(name='x'),
            clauses=[
                CaseClause(
                    pattern=LiteralPattern(value=0),
                    body=[AtomExpr(value='zero')]
                ),
                CaseClause(
                    pattern=WildcardPattern(),
                    body=[AtomExpr(value='other')]
                )
            ]
        )
        assert len(expr.clauses) == 2
    
    def test_fun_expr(self):
        """Test anonymous function expression."""
        expr = FunExpr(clauses=[
            FunClause(
                args=[VarPattern(name='x')],
                body=[VarExpr(name='x')]
            )
        ])
        assert len(expr.clauses) == 1


class TestPatterns:
    """Tests for pattern matching."""
    
    def test_wildcard_pattern(self):
        """Test wildcard pattern."""
        pattern = WildcardPattern()
        assert pattern.kind == 'wildcard_pattern'
    
    def test_var_pattern(self):
        """Test variable pattern."""
        pattern = VarPattern(name='x')
        assert pattern.name == 'x'
    
    def test_tuple_pattern(self):
        """Test tuple pattern."""
        pattern = TuplePattern(elements=[
            AtomPattern(value='ok'),
            VarPattern(name='result')
        ])
        assert len(pattern.elements) == 2
    
    def test_list_pattern_with_tail(self):
        """Test list pattern with tail."""
        pattern = ListPattern(
            elements=[VarPattern(name='head')],
            tail=VarPattern(name='tail')
        )
        assert pattern.tail is not None
    
    def test_map_pattern(self):
        """Test map pattern."""
        pattern = MapPattern(pairs=[
            ('key', VarPattern(name='value'))
        ])
        assert len(pattern.pairs) == 1
    
    def test_binary_pattern(self):
        """Test binary pattern."""
        pattern = BinaryPattern(segments=[
            BinarySegment(value=VarPattern(name='a'), size=8),
            BinarySegment(value=VarPattern(name='b'), size=16)
        ])
        assert len(pattern.segments) == 2


class TestProcessDefinitions:
    """Tests for process-related IR."""
    
    def test_process_def(self):
        """Test process definition."""
        proc = ProcessDef(
            name='worker',
            entry_function='loop',
            args=[LiteralExpr(value=0)]
        )
        assert proc.name == 'worker'
        assert proc.entry_function == 'loop'
    
    def test_spawn_expr(self):
        """Test spawn expression."""
        spawn = SpawnExpr(
            strategy=SpawnStrategy.SPAWN_LINK,
            module='my_module',
            function='loop',
            args=[LiteralExpr(value=0)]
        )
        assert spawn.strategy == SpawnStrategy.SPAWN_LINK
        assert spawn.module == 'my_module'
    
    def test_spawn_strategies(self):
        """Test different spawn strategies."""
        assert SpawnStrategy.SPAWN.value == 'spawn'
        assert SpawnStrategy.SPAWN_LINK.value == 'spawn_link'
        assert SpawnStrategy.SPAWN_MONITOR.value == 'spawn_monitor'
    
    def test_link_expr(self):
        """Test link expression."""
        link = LinkExpr(target=VarExpr(name='pid'))
        assert link.kind == 'link'
    
    def test_monitor_expr(self):
        """Test monitor expression."""
        monitor = MonitorExpr(target=VarExpr(name='pid'))
        assert monitor.kind == 'monitor'
    
    def test_self_expr(self):
        """Test self() expression."""
        self_expr = SelfExpr()
        assert self_expr.kind == 'self'


class TestMessagePassing:
    """Tests for message passing IR."""
    
    def test_send_expr(self):
        """Test send expression."""
        send = SendExpr(
            target=VarExpr(name='pid'),
            message=TupleExpr(elements=[
                AtomExpr(value='hello'),
                LiteralExpr(value='world')
            ])
        )
        assert send.kind == 'send'
    
    def test_receive_expr(self):
        """Test receive expression."""
        receive = ReceiveExpr(
            clauses=[
                ReceiveClause(
                    pattern=TuplePattern(elements=[
                        AtomPattern(value='msg'),
                        VarPattern(name='data')
                    ]),
                    body=[VarExpr(name='data')]
                )
            ]
        )
        assert len(receive.clauses) == 1
    
    def test_receive_with_timeout(self):
        """Test receive with timeout."""
        receive = ReceiveExpr(
            clauses=[
                ReceiveClause(
                    pattern=WildcardPattern(),
                    body=[AtomExpr(value='ok')]
                )
            ],
            timeout=LiteralExpr(value=5000),
            timeout_body=[AtomExpr(value='timeout')]
        )
        assert receive.timeout is not None
        assert len(receive.timeout_body) == 1
    
    def test_receive_clause_with_guards(self):
        """Test receive clause with guards."""
        clause = ReceiveClause(
            pattern=VarPattern(name='x'),
            guards=[Guard(conditions=[
                CallExpr(function='is_integer', args=[VarExpr(name='x')])
            ])],
            body=[VarExpr(name='x')]
        )
        assert len(clause.guards) == 1


class TestOTPBehaviors:
    """Tests for OTP behavior IR."""
    
    def test_gen_server_impl(self):
        """Test GenServer implementation."""
        gen_server = GenServerImpl(
            name='counter',
            state=GenServerState(
                initial_state=LiteralExpr(value=0)
            )
        )
        assert gen_server.name == 'counter'
        assert gen_server.state is not None
    
    def test_gen_server_with_callbacks(self):
        """Test GenServer with custom callbacks."""
        gen_server = GenServerImpl(
            name='kv_store',
            state=GenServerState(
                initial_state=MapExpr(pairs=[])
            ),
            handle_call_callbacks=[
                Callback(
                    name='handle_call',
                    args=['{get, Key}', '_From', 'State'],
                    body=[TupleExpr(elements=[
                        AtomExpr(value='reply'),
                        CallExpr(function='maps:get', args=[
                            VarExpr(name='Key'),
                            VarExpr(name='State')
                        ]),
                        VarExpr(name='State')
                    ])]
                )
            ]
        )
        assert len(gen_server.handle_call_callbacks) == 1
    
    def test_supervisor_impl(self):
        """Test Supervisor implementation."""
        supervisor = SupervisorImpl(
            name='my_sup',
            flags=SupervisorFlags(
                strategy=SupervisorStrategy.ONE_FOR_ONE,
                intensity=3,
                period=5
            ),
            children=[
                ChildSpec(
                    id='worker1',
                    start=('worker_module', 'start_link', []),
                    restart=ChildRestart.PERMANENT,
                    type=ChildType.WORKER
                )
            ]
        )
        assert supervisor.flags.strategy == SupervisorStrategy.ONE_FOR_ONE
        assert len(supervisor.children) == 1
    
    def test_supervisor_strategies(self):
        """Test supervisor strategies."""
        assert SupervisorStrategy.ONE_FOR_ONE.value == 'one_for_one'
        assert SupervisorStrategy.ONE_FOR_ALL.value == 'one_for_all'
        assert SupervisorStrategy.REST_FOR_ONE.value == 'rest_for_one'
    
    def test_child_restart_types(self):
        """Test child restart types."""
        assert ChildRestart.PERMANENT.value == 'permanent'
        assert ChildRestart.TEMPORARY.value == 'temporary'
        assert ChildRestart.TRANSIENT.value == 'transient'
    
    def test_behavior_impl_wrapper(self):
        """Test BehaviorImpl wrapper."""
        behavior = BehaviorImpl(
            behavior_type=BehaviorType.GEN_SERVER,
            implementation=GenServerImpl(name='test')
        )
        assert behavior.behavior_type == BehaviorType.GEN_SERVER
    
    def test_application_impl(self):
        """Test Application implementation."""
        app = ApplicationImpl(
            name='my_app',
            mod=('MyApp.Supervisor', []),
            applications=['kernel', 'stdlib']
        )
        assert app.name == 'my_app'
        assert 'kernel' in app.applications


class TestSerialization:
    """Tests for IR serialization."""
    
    def test_to_dict_simple(self):
        """Test simple node serialization."""
        expr = AtomExpr(value='ok')
        d = expr.to_dict()
        assert d['kind'] == 'atom'
        assert d['value'] == 'ok'
    
    def test_to_dict_nested(self):
        """Test nested node serialization."""
        expr = TupleExpr(elements=[
            AtomExpr(value='ok'),
            LiteralExpr(value=42)
        ])
        d = expr.to_dict()
        assert d['kind'] == 'tuple'
        assert len(d['elements']) == 2
        assert d['elements'][0]['kind'] == 'atom'
    
    def test_to_dict_module(self):
        """Test module serialization."""
        module = ActorModule(
            name='test',
            functions=[
                FunctionDef(
                    name='hello',
                    arity=0,
                    clauses=[
                        FunctionClause(
                            args=[],
                            body=[AtomExpr(value='world')]
                        )
                    ]
                )
            ]
        )
        d = module.to_dict()
        assert d['kind'] == 'actor_module'
        assert d['name'] == 'test'
        assert len(d['functions']) == 1
    
    def test_to_dict_enum(self):
        """Test enum serialization."""
        spawn = SpawnExpr(strategy=SpawnStrategy.SPAWN_LINK)
        d = spawn.to_dict()
        assert d['strategy'] == 'spawn_link'  # Enum value


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
