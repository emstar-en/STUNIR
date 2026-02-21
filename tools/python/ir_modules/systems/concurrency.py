#!/usr/bin/env python3
"""STUNIR Systems IR - Concurrency constructs.

This module defines IR nodes for concurrent programming features
including Ada tasks, protected objects, entries, and select statements.

Usage:
    from ir.systems.concurrency import TaskType, ProtectedType, Entry
    
    # Create a task type with entries
    task = TaskType(
        name='Worker',
        entries=[Entry(name='Start'), Entry(name='Stop')]
    )
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from ir.systems.systems_ir import (
    SystemsNode, TypeRef, Expr, Statement, 
    Parameter, Subprogram, Declaration, Visibility,
    ComponentDecl, Discriminant
)


# =============================================================================
# Task Types (Ada)
# =============================================================================

@dataclass
class Entry(SystemsNode):
    """Task or protected entry declaration.
    
    Ada: entry Name [(Family_Index)] [(params)];
    
    Examples:
        Ada: entry Start;
             entry Put(Item : in Integer);
             entry Request(Priority)(Data : Integer);  -- entry family
    """
    name: str = ''
    parameters: List[Parameter] = field(default_factory=list)
    family_index: Optional[TypeRef] = None  # For entry families
    kind: str = 'entry'


@dataclass
class TaskType(SystemsNode):
    """Task type definition (Ada).
    
    Ada tasks provide concurrent execution with rendezvous-style
    communication through entries.
    
    D equivalent: Thread or std.concurrency
    
    Examples:
        Ada:
            task type Worker_Task is
               entry Start;
               entry Stop;
               entry Put(Item : in Integer);
            end Worker_Task;
    """
    name: str = ''
    discriminants: List[Discriminant] = field(default_factory=list)
    entries: List[Entry] = field(default_factory=list)
    body: List[Statement] = field(default_factory=list)
    local_declarations: List[Declaration] = field(default_factory=list)
    priority: Optional[Expr] = None  # Task priority
    interfaces: List[TypeRef] = field(default_factory=list)  # Task interface
    visibility: Visibility = Visibility.PUBLIC
    spark_mode: bool = False
    kind: str = 'task_type'


@dataclass
class SingleTask(SystemsNode):
    """Single task declaration (Ada).
    
    A task object rather than a task type.
    
    Ada: task Name is ... end Name;
    """
    name: str = ''
    entries: List[Entry] = field(default_factory=list)
    body: List[Statement] = field(default_factory=list)
    local_declarations: List[Declaration] = field(default_factory=list)
    priority: Optional[Expr] = None
    kind: str = 'single_task'


# =============================================================================
# Protected Types (Ada)
# =============================================================================

@dataclass
class ProtectedType(SystemsNode):
    """Protected type definition (Ada).
    
    Protected types provide mutually exclusive access to shared data.
    They can have entries (with barriers), procedures, and functions.
    
    D equivalent: synchronized class or shared
    
    Examples:
        Ada:
            protected type Counter is
               procedure Increment;
               function Get return Integer;
               entry Wait_For_Zero;
            private
               Value : Integer := 0;
            end Counter;
    """
    name: str = ''
    discriminants: List[Discriminant] = field(default_factory=list)
    entries: List[Entry] = field(default_factory=list)
    procedures: List[Subprogram] = field(default_factory=list)
    functions: List[Subprogram] = field(default_factory=list)
    private_components: List[ComponentDecl] = field(default_factory=list)
    interfaces: List[TypeRef] = field(default_factory=list)
    visibility: Visibility = Visibility.PUBLIC
    spark_mode: bool = False
    kind: str = 'protected_type'


@dataclass
class ProtectedBody(SystemsNode):
    """Protected body implementation.
    
    Contains the implementations of protected procedures, functions,
    and entries with their barriers.
    """
    name: str = ''
    entry_bodies: List['EntryBody'] = field(default_factory=list)
    procedure_bodies: List[Subprogram] = field(default_factory=list)
    function_bodies: List[Subprogram] = field(default_factory=list)
    local_declarations: List[Declaration] = field(default_factory=list)
    kind: str = 'protected_body'


@dataclass
class EntryBody(SystemsNode):
    """Protected entry body with barrier.
    
    Ada: entry Name when Barrier is
            begin
               ...
            end Name;
    """
    name: str = ''
    parameters: List[Parameter] = field(default_factory=list)
    family_index_spec: Optional[str] = None  # For entry family parameter name
    barrier: Expr = None  # Boolean condition (when clause)
    body: List[Statement] = field(default_factory=list)
    local_declarations: List[Declaration] = field(default_factory=list)
    kind: str = 'entry_body'


@dataclass
class SingleProtected(SystemsNode):
    """Single protected object declaration (Ada).
    
    A protected object rather than a protected type.
    """
    name: str = ''
    entries: List[Entry] = field(default_factory=list)
    procedures: List[Subprogram] = field(default_factory=list)
    functions: List[Subprogram] = field(default_factory=list)
    private_components: List[ComponentDecl] = field(default_factory=list)
    kind: str = 'single_protected'


# =============================================================================
# Accept and Entry Call Statements
# =============================================================================

@dataclass
class AcceptStatement(Statement):
    """Accept statement (Ada task rendezvous).
    
    Ada: accept Entry_Name [(Index)] [(params)] [do
            ...
         end Entry_Name];
    
    Examples:
        Ada: accept Start;
             accept Put(Item : in Integer) do
                Buffer := Item;
             end Put;
    """
    entry_name: str = ''
    index: Optional[Expr] = None  # For entry families
    parameters: List[Parameter] = field(default_factory=list)
    body: List[Statement] = field(default_factory=list)  # Optional body
    kind: str = 'accept_statement'


@dataclass
class EntryCallStatement(Statement):
    """Entry call statement.
    
    Ada: Task_Name.Entry_Name[(args)];
    """
    target: Expr = None  # Task or protected object
    entry_name: str = ''
    arguments: List[Expr] = field(default_factory=list)
    kind: str = 'entry_call_statement'


@dataclass 
class RequeueStatement(Statement):
    """Requeue statement (Ada).
    
    Ada: requeue Entry_Name [with abort];
    
    Transfers a caller from one entry to another.
    """
    target_entry: str = ''  # Entry name to requeue to
    target_object: Optional[Expr] = None  # Different task/protected object
    with_abort: bool = False
    kind: str = 'requeue_statement'


# =============================================================================
# Select Statements (Ada)
# =============================================================================

@dataclass
class SelectAlternative(SystemsNode):
    """Base class for select statement alternatives."""
    guard: Optional[Expr] = None  # when condition
    kind: str = 'select_alternative'


@dataclass
class AcceptAlternative(SelectAlternative):
    """Accept alternative in selective accept.
    
    Ada: when Condition => accept Entry_Name do ... end;
    """
    accept_statement: AcceptStatement = None
    statements: List[Statement] = field(default_factory=list)  # After accept
    kind: str = 'accept_alternative'


@dataclass
class DelayAlternative(SelectAlternative):
    """Delay alternative in select.
    
    Ada: delay Duration;  -- or delay until Time;
    """
    delay_expr: Expr = None  # Duration or time
    is_until: bool = False  # delay until vs delay
    statements: List[Statement] = field(default_factory=list)
    kind: str = 'delay_alternative'


@dataclass
class TerminateAlternative(SelectAlternative):
    """Terminate alternative.
    
    Ada: terminate;
    
    Allows task to terminate when no clients are waiting.
    """
    kind: str = 'terminate_alternative'


@dataclass
class SelectStatement(Statement):
    """Selective accept statement (Ada).
    
    Ada: select
            [when Cond =>] accept Entry_Name do ... end;
         or
            [when Cond =>] accept Other_Entry;
         or
            delay Duration;
         or
            terminate;
         else
            ...
         end select;
    """
    alternatives: List[SelectAlternative] = field(default_factory=list)
    else_part: Optional[List[Statement]] = None
    kind: str = 'select_statement'


@dataclass
class ConditionalEntryCall(Statement):
    """Conditional entry call statement.
    
    Ada: select
            Entry_Call;
         else
            ...
         end select;
    
    Calls entry if immediately available, otherwise executes else part.
    """
    entry_call: EntryCallStatement = None
    then_statements: List[Statement] = field(default_factory=list)
    else_statements: List[Statement] = field(default_factory=list)
    kind: str = 'conditional_entry_call'


@dataclass
class TimedEntryCall(Statement):
    """Timed entry call statement.
    
    Ada: select
            Entry_Call;
         or
            delay Duration;
         end select;
    
    Waits for entry for specified duration.
    """
    entry_call: EntryCallStatement = None
    then_statements: List[Statement] = field(default_factory=list)
    delay_alternative: DelayAlternative = None
    kind: str = 'timed_entry_call'


@dataclass 
class AsynchronousSelect(Statement):
    """Asynchronous select statement (Ada).
    
    Ada: select
            delay Duration;
            -- abortable statements
         then abort
            -- triggering statements
         end select;
    """
    triggering_alternative: SelectAlternative = None  # Entry call or delay
    abortable_part: List[Statement] = field(default_factory=list)
    kind: str = 'asynchronous_select'


# =============================================================================
# Abort Statement (Ada)
# =============================================================================

@dataclass
class AbortStatement(Statement):
    """Abort statement (Ada).
    
    Ada: abort Task1, Task2, ...;
    
    Terminates one or more tasks.
    """
    tasks: List[Expr] = field(default_factory=list)  # Task objects to abort
    kind: str = 'abort_statement'


# =============================================================================
# Delay Statement
# =============================================================================

@dataclass
class DelayStatement(Statement):
    """Delay statement.
    
    Ada: delay Duration;  -- relative delay
         delay until Time;  -- absolute delay
    D: Thread.sleep(Duration)
    """
    duration: Expr = None
    is_until: bool = False  # delay until vs delay
    kind: str = 'delay_statement'


# =============================================================================
# D Concurrency
# =============================================================================

@dataclass
class SharedVariable(SystemsNode):
    """D shared variable annotation.
    
    D: shared int counter;
    
    Variables marked as shared can be accessed from multiple threads.
    """
    name: str = ''
    type_ref: TypeRef = None
    initializer: Optional[Expr] = None
    kind: str = 'shared_variable'


@dataclass
class SynchronizedBlock(Statement):
    """D synchronized block.
    
    D: synchronized { ... }
       synchronized(mutex) { ... }
    """
    mutex: Optional[Expr] = None  # Optional mutex object
    body: List[Statement] = field(default_factory=list)
    kind: str = 'synchronized_block'


@dataclass
class AtomicOp(SystemsNode):
    """D atomic operation.
    
    D: atomicOp!"operation"(shared_var, value)
    """
    operation: str = ''  # 'load', 'store', 'cas', 'fetchAdd', etc.
    target: Expr = None
    value: Optional[Expr] = None
    kind: str = 'atomic_op'
