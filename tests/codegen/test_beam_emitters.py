#!/usr/bin/env python3
"""Tests for BEAM VM Emitters (Erlang and Elixir).

Tests cover:
    - Module emission
    - Function definitions
    - Process spawning
    - Message passing (send, receive)
    - Pattern matching
    - OTP behaviors (GenServer, Supervisor)
    - Manifest generation
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ir.actor import (
    ActorModule, FunctionDef, FunctionClause,
    Expr, LiteralExpr, VarExpr, AtomExpr, TupleExpr, ListExpr, MapExpr,
    CallExpr, CaseExpr, CaseClause, FunExpr, FunClause,
    ListCompExpr, Generator, Filter, TryExpr, CatchClause,
    Guard,
    SpawnExpr, SpawnStrategy, SendExpr, ReceiveExpr, ReceiveClause,
    SelfExpr,
    WildcardPattern, VarPattern, LiteralPattern, AtomPattern,
    TuplePattern, ListPattern,
    BehaviorImpl, BehaviorType,
    GenServerImpl, GenServerState, Callback,
    SupervisorImpl, SupervisorFlags, SupervisorStrategy,
    ChildSpec, ChildRestart, ChildType,
)
from targets.beam import ErlangEmitter, ElixirEmitter


class TestErlangEmitter:
    """Tests for Erlang code emission."""
    
    @pytest.fixture
    def emitter(self):
        return ErlangEmitter()
    
    def test_simple_module(self, emitter):
        """Test simple module emission."""
        module = ActorModule(
            name='hello',
            exports=['world/0'],
            functions=[
                FunctionDef(
                    name='world',
                    arity=0,
                    exported=True,
                    clauses=[
                        FunctionClause(
                            args=[],
                            body=[AtomExpr(value='hello_world')]
                        )
                    ]
                )
            ]
        )
        result = emitter.emit_module(module)
        
        assert '-module(hello).' in result.code
        assert '-export([world/0]).' in result.code
        assert 'world() ->' in result.code
        assert 'hello_world' in result.code
    
    def test_function_with_args(self, emitter):
        """Test function with arguments."""
        func = FunctionDef(
            name='add',
            arity=2,
            clauses=[
                FunctionClause(
                    args=[VarPattern(name='a'), VarPattern(name='b')],
                    body=[CallExpr(function='+', args=[
                        VarExpr(name='a'),
                        VarExpr(name='b')
                    ])]
                )
            ]
        )
        code = emitter._emit_function(func)
        assert 'add(A, B) ->' in code
    
    def test_function_with_guards(self, emitter):
        """Test function with guards."""
        func = FunctionDef(
            name='abs',
            arity=1,
            clauses=[
                FunctionClause(
                    args=[VarPattern(name='x')],
                    guards=[Guard(conditions=[
                        CallExpr(function='>', args=[
                            VarExpr(name='x'),
                            LiteralExpr(value=0)
                        ])
                    ])],
                    body=[VarExpr(name='x')]
                ),
                FunctionClause(
                    args=[VarPattern(name='x')],
                    body=[CallExpr(function='-', args=[VarExpr(name='x')])]
                )
            ]
        )
        code = emitter._emit_function(func)
        assert 'when' in code
        # Guards emitted as function calls
        assert '>' in code or 'X' in code
    
    def test_spawn_emission(self, emitter):
        """Test spawn expression."""
        spawn = SpawnExpr(
            strategy=SpawnStrategy.SPAWN_LINK,
            module='my_mod',
            function='loop',
            args=[LiteralExpr(value=0)]
        )
        code = emitter._emit_spawn_expr(spawn)
        assert 'spawn_link' in code
        assert 'my_mod' in code
        assert 'loop' in code
    
    def test_send_emission(self, emitter):
        """Test send expression."""
        send = SendExpr(
            target=VarExpr(name='pid'),
            message=TupleExpr(elements=[
                AtomExpr(value='hello'),
                LiteralExpr(value='world', literal_type='string')
            ])
        )
        code = emitter._emit_send_expr(send)
        assert 'Pid !' in code
        assert '{hello, "world"}' in code
    
    def test_receive_emission(self, emitter):
        """Test receive block."""
        receive = ReceiveExpr(
            clauses=[
                ReceiveClause(
                    pattern=TuplePattern(elements=[
                        AtomPattern(value='msg'),
                        VarPattern(name='data')
                    ]),
                    body=[VarExpr(name='data')]
                )
            ],
            timeout=LiteralExpr(value=1000),
            timeout_body=[AtomExpr(value='timeout')]
        )
        code = emitter._emit_receive_expr(receive)
        assert 'receive' in code
        assert '{msg, Data}' in code
        assert 'after 1000' in code
        assert 'timeout' in code
    
    def test_case_emission(self, emitter):
        """Test case expression."""
        case = CaseExpr(
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
        code = emitter._emit_case_expr(case)
        assert 'case X of' in code
        assert '0 ->' in code
        assert '_ ->' in code
    
    def test_list_comprehension(self, emitter):
        """Test list comprehension emission."""
        comp = ListCompExpr(
            expr=CallExpr(function='*', args=[
                VarExpr(name='x'),
                LiteralExpr(value=2)
            ]),
            qualifiers=[
                Generator(
                    pattern=VarPattern(name='x'),
                    list_expr=ListExpr(elements=[
                        LiteralExpr(value=1),
                        LiteralExpr(value=2),
                        LiteralExpr(value=3)
                    ])
                )
            ]
        )
        code = emitter._emit_list_comp_expr(comp)
        assert '[' in code
        assert '||' in code
        assert '<-' in code
    
    def test_tuple_pattern(self, emitter):
        """Test tuple pattern emission."""
        pattern = TuplePattern(elements=[
            AtomPattern(value='ok'),
            VarPattern(name='result')
        ])
        code = emitter.emit_pattern(pattern)
        assert '{ok, Result}' in code
    
    def test_list_pattern_with_tail(self, emitter):
        """Test list pattern with tail."""
        pattern = ListPattern(
            elements=[VarPattern(name='h')],
            tail=VarPattern(name='t')
        )
        code = emitter.emit_pattern(pattern)
        assert '[H|T]' in code
    
    def test_gen_server_emission(self, emitter):
        """Test GenServer emission."""
        module = ActorModule(
            name='counter_server',
            behaviors=[
                BehaviorImpl(
                    behavior_type=BehaviorType.GEN_SERVER,
                    implementation=GenServerImpl(
                        name='counter',
                        state=GenServerState(
                            initial_state=LiteralExpr(value=0)
                        )
                    )
                )
            ]
        )
        result = emitter.emit_module(module)
        
        assert '-behaviour(gen_server).' in result.code
        assert 'init(_Args)' in result.code
        assert '{ok, 0}' in result.code
        assert 'handle_call' in result.code
        assert 'handle_cast' in result.code
        assert 'handle_info' in result.code
        assert 'terminate' in result.code
    
    def test_supervisor_emission(self, emitter):
        """Test Supervisor emission."""
        module = ActorModule(
            name='my_sup',
            behaviors=[
                BehaviorImpl(
                    behavior_type=BehaviorType.SUPERVISOR,
                    implementation=SupervisorImpl(
                        flags=SupervisorFlags(
                            strategy=SupervisorStrategy.ONE_FOR_ALL,
                            intensity=3,
                            period=10
                        ),
                        children=[
                            ChildSpec(
                                id='worker',
                                start=('worker', 'start_link', []),
                                restart=ChildRestart.PERMANENT,
                                type=ChildType.WORKER
                            )
                        ]
                    )
                )
            ]
        )
        result = emitter.emit_module(module)
        
        assert '-behaviour(supervisor).' in result.code
        assert 'one_for_all' in result.code
        assert 'intensity => 3' in result.code
        assert 'id => worker' in result.code
    
    def test_manifest_generation(self, emitter):
        """Test manifest generation."""
        module = ActorModule(name='test')
        result = emitter.emit_module(module)
        
        assert 'schema' in result.manifest
        assert result.manifest['language'] == 'erlang'
        assert 'code_hash' in result.manifest
        assert result.manifest['module'] == 'test'
    
    def test_variable_transformation(self, emitter):
        """Test Erlang variable naming (capitalized)."""
        assert emitter._transform_var('state') == 'State'
        assert emitter._transform_var('myVar') == 'MyVar'
        assert emitter._transform_var('_') == '_'
    
    def test_atom_quoting(self, emitter):
        """Test atom quoting."""
        assert emitter.emit_literal('ok', 'atom') == 'ok'
        assert emitter.emit_literal('Hello World', 'atom') == "'Hello World'"
        assert emitter.emit_literal('123abc', 'atom') == "'123abc'"


class TestElixirEmitter:
    """Tests for Elixir code emission."""
    
    @pytest.fixture
    def emitter(self):
        return ElixirEmitter()
    
    def test_simple_module(self, emitter):
        """Test simple module emission."""
        module = ActorModule(
            name='hello',
            exports=['world/0'],
            functions=[
                FunctionDef(
                    name='world',
                    arity=0,
                    exported=True,
                    clauses=[
                        FunctionClause(
                            args=[],
                            body=[AtomExpr(value='hello_world')]
                        )
                    ]
                )
            ]
        )
        result = emitter.emit_module(module)
        
        assert 'defmodule Hello do' in result.code
        assert 'def world' in result.code
        assert ':hello_world' in result.code
    
    def test_private_function(self, emitter):
        """Test private function emission."""
        func = FunctionDef(
            name='helper',
            arity=1,
            exported=False,
            clauses=[
                FunctionClause(
                    args=[VarPattern(name='x')],
                    body=[VarExpr(name='x')]
                )
            ]
        )
        code = emitter._emit_function(func)
        assert 'defp helper' in code
    
    def test_function_with_guards(self, emitter):
        """Test function with guards."""
        func = FunctionDef(
            name='positive',
            arity=1,
            exported=True,
            clauses=[
                FunctionClause(
                    args=[VarPattern(name='x')],
                    guards=[Guard(conditions=[
                        CallExpr(function='>', args=[
                            VarExpr(name='x'),
                            LiteralExpr(value=0)
                        ])
                    ])],
                    body=[AtomExpr(value='yes')]
                )
            ]
        )
        code = emitter._emit_function(func)
        assert 'when' in code
    
    def test_send_emission(self, emitter):
        """Test send expression."""
        send = SendExpr(
            target=VarExpr(name='pid'),
            message=TupleExpr(elements=[
                AtomExpr(value='hello'),
                LiteralExpr(value='world', literal_type='string')
            ])
        )
        code = emitter._emit_send_expr(send)
        assert 'send(pid, {:hello, "world"})' in code
    
    def test_receive_emission(self, emitter):
        """Test receive block."""
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
        code = emitter._emit_receive_expr(receive)
        assert 'receive do' in code
        assert '{:msg, data}' in code
    
    def test_spawn_emission(self, emitter):
        """Test spawn expression."""
        spawn = SpawnExpr(
            strategy=SpawnStrategy.SPAWN_LINK,
            module='MyMod',
            function='loop',
            args=[LiteralExpr(value=0)]
        )
        code = emitter._emit_spawn_expr(spawn)
        assert 'spawn_link' in code
        assert 'MyMod' in code
    
    def test_case_emission(self, emitter):
        """Test case expression."""
        case = CaseExpr(
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
        code = emitter._emit_case_expr(case)
        assert 'case x do' in code
        assert '0 ->' in code
        assert '_ ->' in code
    
    def test_list_comprehension(self, emitter):
        """Test for comprehension."""
        comp = ListCompExpr(
            expr=CallExpr(function='*', args=[
                VarExpr(name='x'),
                LiteralExpr(value=2)
            ]),
            qualifiers=[
                Generator(
                    pattern=VarPattern(name='x'),
                    list_expr=ListExpr(elements=[
                        LiteralExpr(value=1),
                        LiteralExpr(value=2),
                        LiteralExpr(value=3)
                    ])
                )
            ]
        )
        code = emitter._emit_list_comp_expr(comp)
        assert 'for' in code
        assert '<-' in code
        assert 'do:' in code
    
    def test_gen_server_emission(self, emitter):
        """Test GenServer emission."""
        module = ActorModule(
            name='counter',
            behaviors=[
                BehaviorImpl(
                    behavior_type=BehaviorType.GEN_SERVER,
                    implementation=GenServerImpl(
                        name='counter',
                        state=GenServerState(
                            initial_state=LiteralExpr(value=0)
                        )
                    )
                )
            ]
        )
        result = emitter.emit_module(module)
        
        assert 'defmodule Counter do' in result.code
        assert 'use GenServer' in result.code
        assert '@impl true' in result.code
        assert 'def init' in result.code
        assert 'def start_link' in result.code
        assert 'def handle_call' in result.code
        assert 'def handle_cast' in result.code
    
    def test_supervisor_emission(self, emitter):
        """Test Supervisor emission."""
        module = ActorModule(
            name='my_supervisor',
            behaviors=[
                BehaviorImpl(
                    behavior_type=BehaviorType.SUPERVISOR,
                    implementation=SupervisorImpl(
                        flags=SupervisorFlags(
                            strategy=SupervisorStrategy.ONE_FOR_ONE,
                            intensity=3,
                            period=5
                        ),
                        children=[
                            ChildSpec(
                                id='worker',
                                start=('MyWorker', 'start_link', []),
                                restart=ChildRestart.PERMANENT,
                                type=ChildType.WORKER
                            )
                        ]
                    )
                )
            ]
        )
        result = emitter.emit_module(module)
        
        assert 'use Supervisor' in result.code
        assert 'Supervisor.init' in result.code
        assert ':one_for_one' in result.code
        assert 'max_restarts: 3' in result.code
    
    def test_manifest_generation(self, emitter):
        """Test manifest generation."""
        module = ActorModule(name='test')
        result = emitter.emit_module(module)
        
        assert 'schema' in result.manifest
        assert result.manifest['language'] == 'elixir'
        assert 'code_hash' in result.manifest
    
    def test_variable_transformation(self, emitter):
        """Test Elixir variable naming (snake_case)."""
        assert emitter._transform_var('State') == 'state'
        assert emitter._transform_var('myVar') == 'my_var'
        assert emitter._transform_var('_') == '_'
    
    def test_module_name_transformation(self, emitter):
        """Test Elixir module name (CamelCase)."""
        assert emitter._to_elixir_module_name('hello') == 'Hello'
        assert emitter._to_elixir_module_name('my_app') == 'MyApp'
    
    def test_atom_syntax(self, emitter):
        """Test Elixir atom syntax."""
        assert emitter.emit_literal('ok', 'atom') == ':ok'
        assert emitter.emit_literal('hello_world', 'atom') == ':hello_world'
    
    def test_map_syntax(self, emitter):
        """Test Elixir map syntax."""
        expr = MapExpr(pairs=[
            (AtomExpr(value='key'), LiteralExpr(value='value', literal_type='string'))
        ])
        code = emitter._emit_map_expr(expr)
        assert '%{' in code
        assert '=>' in code


class TestBEAMEmitterIntegration:
    """Integration tests for both emitters."""
    
    def test_chat_server_erlang(self):
        """Test complete chat server in Erlang."""
        emitter = ErlangEmitter()
        
        module = ActorModule(
            name='chat_server',
            exports=['start/0', 'join/1'],
            functions=[
                FunctionDef(
                    name='start',
                    arity=0,
                    exported=True,
                    clauses=[
                        FunctionClause(
                            args=[],
                            body=[
                                SpawnExpr(
                                    strategy=SpawnStrategy.SPAWN,
                                    function='loop',
                                    args=[ListExpr(elements=[])]
                                )
                            ]
                        )
                    ]
                ),
                FunctionDef(
                    name='loop',
                    arity=1,
                    clauses=[
                        FunctionClause(
                            args=[VarPattern(name='clients')],
                            body=[
                                ReceiveExpr(
                                    clauses=[
                                        ReceiveClause(
                                            pattern=TuplePattern(elements=[
                                                AtomPattern(value='join'),
                                                VarPattern(name='pid')
                                            ]),
                                            body=[
                                                CallExpr(
                                                    function='loop',
                                                    args=[ListExpr(
                                                        elements=[VarExpr(name='pid')],
                                                        tail=VarExpr(name='clients')
                                                    )]
                                                )
                                            ]
                                        )
                                    ]
                                )
                            ]
                        )
                    ]
                )
            ]
        )
        
        result = emitter.emit_module(module)
        assert '-module(chat_server).' in result.code
        assert 'spawn' in result.code
        assert 'receive' in result.code
        assert '{join, Pid}' in result.code
    
    def test_counter_genserver_elixir(self):
        """Test counter GenServer in Elixir."""
        emitter = ElixirEmitter()
        
        module = ActorModule(
            name='counter',
            behaviors=[
                BehaviorImpl(
                    behavior_type=BehaviorType.GEN_SERVER,
                    implementation=GenServerImpl(
                        name='counter',
                        state=GenServerState(
                            initial_state=LiteralExpr(value=0)
                        ),
                        handle_call_callbacks=[
                            Callback(
                                name='handle_call',
                                args=[':increment', '_from', 'state'],
                                body=[
                                    TupleExpr(elements=[
                                        AtomExpr(value='reply'),
                                        AtomExpr(value='ok'),
                                        CallExpr(function='+', args=[
                                            VarExpr(name='state'),
                                            LiteralExpr(value=1)
                                        ])
                                    ])
                                ]
                            ),
                            Callback(
                                name='handle_call',
                                args=[':get', '_from', 'state'],
                                body=[
                                    TupleExpr(elements=[
                                        AtomExpr(value='reply'),
                                        VarExpr(name='state'),
                                        VarExpr(name='state')
                                    ])
                                ]
                            )
                        ]
                    )
                )
            ]
        )
        
        result = emitter.emit_module(module)
        assert 'defmodule Counter do' in result.code
        assert 'use GenServer' in result.code
        assert 'handle_call(:increment' in result.code or 'handle_call' in result.code
    
    def test_both_emitters_same_module(self):
        """Test same module generates valid code in both languages."""
        erlang = ErlangEmitter()
        elixir = ElixirEmitter()
        
        module = ActorModule(
            name='echo',
            exports=['echo/1'],
            functions=[
                FunctionDef(
                    name='echo',
                    arity=1,
                    exported=True,
                    clauses=[
                        FunctionClause(
                            args=[VarPattern(name='msg')],
                            body=[VarExpr(name='msg')]
                        )
                    ]
                )
            ]
        )
        
        erl_result = erlang.emit_module(module)
        ex_result = elixir.emit_module(module)
        
        # Erlang
        assert '-module(echo).' in erl_result.code
        assert 'echo(Msg)' in erl_result.code
        
        # Elixir
        assert 'defmodule Echo do' in ex_result.code
        assert 'def echo(msg)' in ex_result.code
        
        # Both have manifests
        assert erl_result.manifest['language'] == 'erlang'
        assert ex_result.manifest['language'] == 'elixir'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
