#!/usr/bin/env python3
"""STUNIR Actor Model IR Package.

This package provides IR constructs for actor-based concurrency,
as used in BEAM VM languages (Erlang, Elixir).

Core Components:
    - actor_ir: Base classes and expressions
    - process: Process spawning and management
    - message: Message passing and pattern matching
    - otp: OTP behavior implementations

Usage:
    from ir.actor import (
        # Module
        ActorModule, ActorNode,
        
        # Expressions
        Expr, LiteralExpr, VarExpr, AtomExpr,
        TupleExpr, ListExpr, MapExpr, BinaryExpr,
        CallExpr, CaseExpr, IfExpr, FunExpr, TryExpr,
        ListCompExpr, Generator, Filter,
        
        # Functions
        FunctionDef, FunctionClause, Guard,
        
        # Processes
        ProcessDef, SpawnExpr, SpawnStrategy,
        LinkExpr, MonitorExpr, SelfExpr,
        
        # Messages
        SendExpr, ReceiveExpr, ReceiveClause,
        
        # Patterns
        Pattern, WildcardPattern, VarPattern,
        LiteralPattern, TuplePattern, ListPattern,
        MapPattern, BinaryPattern,
        
        # OTP Behaviors
        BehaviorType, BehaviorImpl,
        GenServerImpl, GenServerState, Callback,
        SupervisorImpl, SupervisorFlags, SupervisorStrategy,
        ChildSpec, ChildRestart, ChildType,
        ApplicationImpl
    )
"""

# Core classes and expressions
from ir.actor.actor_ir import (
    ActorNodeKind,
    ActorNode,
    ActorModule,
    # Expressions
    Expr,
    LiteralExpr,
    VarExpr,
    AtomExpr,
    TupleExpr,
    ListExpr,
    MapExpr,
    BinaryExpr,
    BinarySegment,
    CallExpr,
    CaseExpr,
    CaseClause,
    IfExpr,
    FunExpr,
    FunClause,
    ListCompExpr,
    Qualifier,
    Generator,
    Filter,
    TryExpr,
    CatchClause,
    Guard,
    # Functions
    FunctionClause,
    FunctionDef,
)

# Process management
from ir.actor.process import (
    SpawnStrategy,
    ProcessFlag,
    ProcessDef,
    SpawnExpr,
    LinkExpr,
    UnlinkExpr,
    MonitorExpr,
    DemonitorExpr,
    SelfExpr,
    ProcessInfoExpr,
    RegisterExpr,
    WhereisExpr,
    ExitExpr,
    ProcessFlagExpr,
)

# Message passing
from ir.actor.message import (
    # Patterns
    Pattern,
    WildcardPattern,
    VarPattern,
    LiteralPattern,
    AtomPattern,
    TuplePattern,
    ListPattern,
    MapPattern,
    BinaryPattern,
    RecordPattern,
    PinPattern,
    # Message passing
    SendExpr,
    ReceiveClause,
    ReceiveExpr,
    FlushExpr,
    SelectiveReceiveExpr,
)

# OTP behaviors
from ir.actor.otp import (
    BehaviorType,
    Callback,
    # GenServer
    GenServerState,
    GenServerImpl,
    # Supervisor
    SupervisorStrategy,
    ChildRestart,
    ChildType,
    ChildShutdown,
    ChildSpec,
    SupervisorFlags,
    SupervisorImpl,
    # Application
    ApplicationImpl,
    # GenStatem
    CallbackMode,
    StateFunction,
    GenStatemImpl,
    # Wrapper
    BehaviorImpl,
)

__all__ = [
    # Core
    'ActorNodeKind',
    'ActorNode',
    'ActorModule',
    # Expressions
    'Expr',
    'LiteralExpr',
    'VarExpr',
    'AtomExpr',
    'TupleExpr',
    'ListExpr',
    'MapExpr',
    'BinaryExpr',
    'BinarySegment',
    'CallExpr',
    'CaseExpr',
    'CaseClause',
    'IfExpr',
    'FunExpr',
    'FunClause',
    'ListCompExpr',
    'Qualifier',
    'Generator',
    'Filter',
    'TryExpr',
    'CatchClause',
    'Guard',
    # Functions
    'FunctionClause',
    'FunctionDef',
    # Processes
    'SpawnStrategy',
    'ProcessFlag',
    'ProcessDef',
    'SpawnExpr',
    'LinkExpr',
    'UnlinkExpr',
    'MonitorExpr',
    'DemonitorExpr',
    'SelfExpr',
    'ProcessInfoExpr',
    'RegisterExpr',
    'WhereisExpr',
    'ExitExpr',
    'ProcessFlagExpr',
    # Patterns
    'Pattern',
    'WildcardPattern',
    'VarPattern',
    'LiteralPattern',
    'AtomPattern',
    'TuplePattern',
    'ListPattern',
    'MapPattern',
    'BinaryPattern',
    'RecordPattern',
    'PinPattern',
    # Messages
    'SendExpr',
    'ReceiveClause',
    'ReceiveExpr',
    'FlushExpr',
    'SelectiveReceiveExpr',
    # OTP
    'BehaviorType',
    'Callback',
    'GenServerState',
    'GenServerImpl',
    'SupervisorStrategy',
    'ChildRestart',
    'ChildType',
    'ChildShutdown',
    'ChildSpec',
    'SupervisorFlags',
    'SupervisorImpl',
    'ApplicationImpl',
    'CallbackMode',
    'StateFunction',
    'GenStatemImpl',
    'BehaviorImpl',
]
