#!/usr/bin/env python3
"""STUNIR Actor Model IR - Process Definitions.

This module provides IR constructs for process management in the actor model,
including process spawning, linking, and monitoring.

Usage:
    from ir.actor import (
        ProcessDef, SpawnExpr, SpawnStrategy,
        LinkExpr, MonitorExpr, SelfExpr
    )
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from ir.actor.actor_ir import ActorNode, Expr


class SpawnStrategy(Enum):
    """Process spawn strategies.
    
    - SPAWN: Basic process spawn
    - SPAWN_LINK: Spawn with bidirectional link (crashes propagate)
    - SPAWN_MONITOR: Spawn with monitoring (receive DOWN message)
    """
    SPAWN = 'spawn'
    SPAWN_LINK = 'spawn_link'
    SPAWN_MONITOR = 'spawn_monitor'


class ProcessFlag(Enum):
    """Process flags that can be set."""
    TRAP_EXIT = 'trap_exit'
    PRIORITY = 'priority'
    MESSAGE_QUEUE_DATA = 'message_queue_data'
    MIN_HEAP_SIZE = 'min_heap_size'
    MIN_BIN_VHEAP_SIZE = 'min_bin_vheap_size'
    MAX_HEAP_SIZE = 'max_heap_size'
    SENSITIVE = 'sensitive'


@dataclass
class ProcessDef(ActorNode):
    """Definition of a process.
    
    Represents a process that can be spawned in the actor system.
    
    Attributes:
        name: Process name (for registration)
        entry_function: Entry point function name
        args: Arguments to pass to entry function
        flags: Process flags to set
        initial_state: Initial state (for stateful processes)
    """
    name: str = ''
    entry_function: str = ''
    args: List[Expr] = field(default_factory=list)
    flags: List[ProcessFlag] = field(default_factory=list)
    initial_state: Optional[Expr] = None
    kind: str = 'process_def'


@dataclass
class SpawnExpr(ActorNode):
    """Expression to spawn a new process.
    
    Attributes:
        strategy: How to spawn (spawn, spawn_link, spawn_monitor)
        module: Optional module name (for MFA spawn)
        function: Function to call in new process
        args: Arguments to pass to function
    """
    strategy: SpawnStrategy = SpawnStrategy.SPAWN
    module: Optional[str] = None
    function: str = ''
    args: List[Expr] = field(default_factory=list)
    kind: str = 'spawn'


@dataclass
class LinkExpr(ActorNode):
    """Link current process to another process.
    
    Creates a bidirectional link - if one process dies,
    the other receives an exit signal.
    """
    target: Expr = None  # PID expression
    kind: str = 'link'


@dataclass
class UnlinkExpr(ActorNode):
    """Unlink current process from another process."""
    target: Expr = None  # PID expression
    kind: str = 'unlink'


@dataclass
class MonitorExpr(ActorNode):
    """Monitor a process.
    
    Creates a unidirectional monitor - if target dies,
    monitoring process receives a 'DOWN' message.
    """
    target: Expr = None  # PID or registered name
    kind: str = 'monitor'


@dataclass
class DemonitorExpr(ActorNode):
    """Remove a monitor."""
    reference: Expr = None  # Monitor reference
    kind: str = 'demonitor'


@dataclass
class SelfExpr(ActorNode):
    """Get current process PID (self())."""
    kind: str = 'self'


@dataclass
class ProcessInfoExpr(ActorNode):
    """Get process information.
    
    Attributes:
        pid: Process to query (None for self)
        item: Information item to get
    """
    pid: Optional[Expr] = None
    item: str = ''  # 'registered_name', 'memory', 'message_queue_len', etc.
    kind: str = 'process_info'


@dataclass
class RegisterExpr(ActorNode):
    """Register a process with a name."""
    name: str = ''
    pid: Expr = None
    kind: str = 'register'


@dataclass
class WhereisExpr(ActorNode):
    """Look up a registered process name."""
    name: str = ''
    kind: str = 'whereis'


@dataclass
class ExitExpr(ActorNode):
    """Send exit signal to a process."""
    pid: Expr = None
    reason: Expr = None
    kind: str = 'exit'


@dataclass
class ProcessFlagExpr(ActorNode):
    """Set a process flag."""
    flag: ProcessFlag = ProcessFlag.TRAP_EXIT
    value: Expr = None
    kind: str = 'process_flag'
