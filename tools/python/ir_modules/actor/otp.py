#!/usr/bin/env python3
"""STUNIR Actor Model IR - OTP Behaviors.

This module provides IR constructs for OTP (Open Telecom Platform) behaviors,
including GenServer, Supervisor, Application, and gen_statem.

OTP behaviors provide battle-tested patterns for building fault-tolerant,
concurrent systems.

Usage:
    from ir.actor import (
        BehaviorType, GenServerImpl, GenServerState, Callback,
        SupervisorImpl, SupervisorFlags, SupervisorStrategy,
        ChildSpec, ChildRestart, ChildType,
        ApplicationImpl, BehaviorImpl
    )
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from ir.actor.actor_ir import ActorNode, Expr


# =============================================================================
# Behavior Types
# =============================================================================

class BehaviorType(Enum):
    """OTP behavior types.
    
    - GEN_SERVER: Generic server for request-response
    - GEN_STATEM: Generic state machine
    - GEN_EVENT: Generic event handling
    - SUPERVISOR: Process supervision
    - APPLICATION: OTP application
    """
    GEN_SERVER = 'gen_server'
    GEN_STATEM = 'gen_statem'
    GEN_EVENT = 'gen_event'
    SUPERVISOR = 'supervisor'
    APPLICATION = 'application'


# =============================================================================
# Callback Definition
# =============================================================================

@dataclass
class Callback(ActorNode):
    """OTP callback function.
    
    Represents a callback that implements part of a behavior.
    
    Attributes:
        name: Callback name (e.g., 'init', 'handle_call')
        args: Argument names/patterns
        body: Callback body expressions
        return_type: Documentation of expected return
    """
    name: str = ''
    args: List[str] = field(default_factory=list)
    body: List[Expr] = field(default_factory=list)
    return_type: str = ''
    kind: str = 'callback'


# =============================================================================
# GenServer
# =============================================================================

@dataclass
class GenServerState(ActorNode):
    """GenServer state definition.
    
    Attributes:
        initial_state: Expression for initial state
        state_type: Optional type annotation
    """
    initial_state: Expr = None
    state_type: Optional[str] = None
    kind: str = 'gen_server_state'


@dataclass
class GenServerImpl(ActorNode):
    """GenServer behavior implementation.
    
    GenServer is the most common OTP behavior, implementing
    a server that handles synchronous calls and async casts.
    
    Required callbacks:
        - init/1: Initialize state
    
    Optional callbacks:
        - handle_call/3: Handle synchronous requests
        - handle_cast/2: Handle async messages
        - handle_info/2: Handle other messages
        - terminate/2: Cleanup on shutdown
        - code_change/3: Hot code upgrade
    
    Attributes:
        name: Server name
        state: State definition
        init_callback: init/1 implementation
        handle_call_callbacks: handle_call/3 implementations
        handle_cast_callbacks: handle_cast/2 implementations
        handle_info_callbacks: handle_info/2 implementations
        terminate_callback: terminate/2 implementation
        code_change_callback: code_change/3 implementation
    """
    name: str = ''
    state: GenServerState = None
    init_callback: Callback = None
    handle_call_callbacks: List[Callback] = field(default_factory=list)
    handle_cast_callbacks: List[Callback] = field(default_factory=list)
    handle_info_callbacks: List[Callback] = field(default_factory=list)
    terminate_callback: Optional[Callback] = None
    code_change_callback: Optional[Callback] = None
    kind: str = 'gen_server_impl'


# =============================================================================
# Supervisor
# =============================================================================

class SupervisorStrategy(Enum):
    """Supervisor restart strategies.
    
    - ONE_FOR_ONE: Only restart failed child
    - ONE_FOR_ALL: Restart all children if one fails
    - REST_FOR_ONE: Restart failed child and those started after it
    - SIMPLE_ONE_FOR_ONE: Dynamic children, same spec
    """
    ONE_FOR_ONE = 'one_for_one'
    ONE_FOR_ALL = 'one_for_all'
    REST_FOR_ONE = 'rest_for_one'
    SIMPLE_ONE_FOR_ONE = 'simple_one_for_one'


class ChildRestart(Enum):
    """Child restart type.
    
    - PERMANENT: Always restart
    - TEMPORARY: Never restart
    - TRANSIENT: Restart only on abnormal exit
    """
    PERMANENT = 'permanent'
    TEMPORARY = 'temporary'
    TRANSIENT = 'transient'


class ChildType(Enum):
    """Child process type."""
    WORKER = 'worker'
    SUPERVISOR = 'supervisor'


class ChildShutdown(Enum):
    """Child shutdown behavior."""
    BRUTAL_KILL = 'brutal_kill'
    INFINITY = 'infinity'
    # Numeric timeout is also allowed


@dataclass
class ChildSpec(ActorNode):
    """Supervisor child specification.
    
    Defines how a child process should be started and supervised.
    
    Attributes:
        id: Unique identifier for child
        start: MFA tuple (Module, Function, Args)
        restart: Restart strategy for this child
        shutdown: Shutdown timeout or strategy
        type: Worker or supervisor
        modules: List of modules (for code upgrade)
    """
    id: str = ''
    start: tuple = ()  # (Module, Function, Args)
    restart: ChildRestart = ChildRestart.PERMANENT
    shutdown: Union[int, str] = 5000  # timeout, 'infinity', or 'brutal_kill'
    type: ChildType = ChildType.WORKER
    modules: List[str] = field(default_factory=list)
    kind: str = 'child_spec'


@dataclass
class SupervisorFlags(ActorNode):
    """Supervisor flags configuration.
    
    Attributes:
        strategy: How to handle child failures
        intensity: Max restarts allowed in period
        period: Time period for intensity (seconds)
    """
    strategy: SupervisorStrategy = SupervisorStrategy.ONE_FOR_ONE
    intensity: int = 1
    period: int = 5
    kind: str = 'supervisor_flags'


@dataclass
class SupervisorImpl(ActorNode):
    """Supervisor behavior implementation.
    
    Supervisors monitor child processes and restart them
    according to a defined strategy.
    
    Attributes:
        name: Supervisor name
        flags: Supervision flags (strategy, intensity, period)
        children: Child specifications
    """
    name: str = ''
    flags: SupervisorFlags = None
    children: List[ChildSpec] = field(default_factory=list)
    kind: str = 'supervisor_impl'


# =============================================================================
# Application
# =============================================================================

@dataclass
class ApplicationImpl(ActorNode):
    """Application behavior implementation.
    
    OTP applications are the top-level organization unit,
    typically starting a supervision tree.
    
    Attributes:
        name: Application name
        mod: Start module and args
        env: Application environment
        applications: Dependency applications
    """
    name: str = ''
    mod: tuple = ()  # (Module, Args)
    env: Dict[str, Any] = field(default_factory=dict)
    applications: List[str] = field(default_factory=list)
    kind: str = 'application_impl'


# =============================================================================
# GenStatem (State Machine)
# =============================================================================

class CallbackMode(Enum):
    """gen_statem callback mode."""
    STATE_FUNCTIONS = 'state_functions'
    HANDLE_EVENT_FUNCTION = 'handle_event_function'


@dataclass
class StateFunction(ActorNode):
    """State function for gen_statem.
    
    In state_functions mode, each state has its own function.
    """
    state_name: str = ''
    event_type: str = ''  # 'call', 'cast', 'info', etc.
    pattern: 'Pattern' = None
    guards: List['Guard'] = field(default_factory=list)
    body: List[Expr] = field(default_factory=list)
    kind: str = 'state_function'


@dataclass
class GenStatemImpl(ActorNode):
    """gen_statem behavior implementation.
    
    Attributes:
        name: State machine name
        callback_mode: How callbacks are organized
        initial_state: Starting state name
        initial_data: Starting state data
        state_functions: State handling functions
    """
    name: str = ''
    callback_mode: CallbackMode = CallbackMode.STATE_FUNCTIONS
    initial_state: str = ''
    initial_data: Expr = None
    state_functions: List[StateFunction] = field(default_factory=list)
    kind: str = 'gen_statem_impl'


# =============================================================================
# Behavior Wrapper
# =============================================================================

@dataclass
class BehaviorImpl(ActorNode):
    """Generic behavior implementation wrapper.
    
    Wraps specific behavior implementations (GenServer, Supervisor, etc.)
    for uniform handling.
    """
    behavior_type: BehaviorType = BehaviorType.GEN_SERVER
    implementation: Union[GenServerImpl, SupervisorImpl, ApplicationImpl, GenStatemImpl] = None
    kind: str = 'behavior_impl'
