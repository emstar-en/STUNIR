#!/usr/bin/env python3
"""STUNIR Memory Management Module.

Provides memory management patterns for different target languages including
manual allocation, reference counting, ownership/borrowing, and garbage collection.

This module is part of the STUNIR code generation enhancement suite.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple
from enum import Enum, auto
from abc import ABC, abstractmethod


class MemoryStrategy(Enum):
    """Memory management strategies."""
    MANUAL = auto()         # C: malloc/free
    REFERENCE_COUNTED = auto()  # Python, Swift
    OWNERSHIP = auto()      # Rust: ownership and borrowing
    GARBAGE_COLLECTED = auto()  # Python, Haskell, Java
    RAII = auto()           # C++, Rust: Resource Acquisition Is Initialization
    ARENA = auto()          # Custom arena allocator
    POOL = auto()           # Object pool
    STACK = auto()          # Stack allocation only


class AllocationKind(Enum):
    """Kinds of memory allocations."""
    STACK = auto()
    HEAP = auto()
    STATIC = auto()
    ARENA = auto()
    POOL = auto()


@dataclass
class Allocation:
    """Represents a memory allocation."""
    id: int
    kind: AllocationKind
    size: Optional[int] = None
    type_name: str = 'void'
    location: Optional[str] = None
    line: Optional[int] = None
    is_freed: bool = False
    freed_at: Optional[str] = None
    ref_count: int = 1
    
    def increment_ref(self) -> None:
        """Increment reference count."""
        self.ref_count += 1
    
    def decrement_ref(self) -> bool:
        """Decrement reference count. Returns True if should be freed."""
        self.ref_count -= 1
        return self.ref_count <= 0


@dataclass
class MemoryOperation:
    """Represents a memory operation."""
    op_type: str  # 'alloc', 'free', 'ref', 'deref', 'copy', 'move'
    allocation_id: Optional[int] = None
    source_var: Optional[str] = None
    dest_var: Optional[str] = None
    location: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'op': self.op_type,
            'alloc_id': self.allocation_id,
            'source': self.source_var,
            'dest': self.dest_var,
            'location': self.location
        }


class MemoryManager(ABC):
    """Abstract base class for memory managers."""
    
    def __init__(self):
        self.allocations: Dict[int, Allocation] = {}
        self.operations: List[MemoryOperation] = []
        self._next_alloc_id = 1
    
    @property
    @abstractmethod
    def strategy(self) -> MemoryStrategy:
        """Return the memory management strategy."""
        pass
    
    @abstractmethod
    def emit_allocation(self, var_name: str, type_name: str, 
                       size: Optional[int] = None) -> str:
        """Emit code for allocation."""
        pass
    
    @abstractmethod
    def emit_deallocation(self, var_name: str) -> str:
        """Emit code for deallocation."""
        pass
    
    def track_allocation(self, kind: AllocationKind, type_name: str,
                        size: Optional[int] = None,
                        location: Optional[str] = None) -> int:
        """Track a new allocation."""
        alloc_id = self._next_alloc_id
        self._next_alloc_id += 1
        
        alloc = Allocation(
            id=alloc_id,
            kind=kind,
            size=size,
            type_name=type_name,
            location=location
        )
        self.allocations[alloc_id] = alloc
        
        self.operations.append(MemoryOperation(
            op_type='alloc',
            allocation_id=alloc_id,
            location=location
        ))
        
        return alloc_id
    
    def track_free(self, alloc_id: int, location: Optional[str] = None) -> None:
        """Track a deallocation."""
        if alloc_id in self.allocations:
            self.allocations[alloc_id].is_freed = True
            self.allocations[alloc_id].freed_at = location
        
        self.operations.append(MemoryOperation(
            op_type='free',
            allocation_id=alloc_id,
            location=location
        ))
    
    def get_leaks(self) -> List[Allocation]:
        """Get list of unfreed heap allocations."""
        return [
            a for a in self.allocations.values()
            if a.kind == AllocationKind.HEAP and not a.is_freed
        ]


class ManualMemoryManager(MemoryManager):
    """Memory manager for manual allocation (C style)."""
    
    @property
    def strategy(self) -> MemoryStrategy:
        return MemoryStrategy.MANUAL
    
    def emit_allocation(self, var_name: str, type_name: str,
                       size: Optional[int] = None) -> str:
        """Emit malloc call."""
        if size is None:
            return f'{type_name}* {var_name} = ({type_name}*)malloc(sizeof({type_name}));'
        else:
            return f'{type_name}* {var_name} = ({type_name}*)malloc({size});'
    
    def emit_deallocation(self, var_name: str) -> str:
        """Emit free call."""
        return f'free({var_name});\n{var_name} = NULL;'
    
    def emit_array_allocation(self, var_name: str, type_name: str, 
                             count: int) -> str:
        """Emit array allocation."""
        return f'{type_name}* {var_name} = ({type_name}*)malloc({count} * sizeof({type_name}));'
    
    def emit_reallocation(self, var_name: str, type_name: str,
                         new_size: int) -> str:
        """Emit realloc call."""
        return f'{var_name} = ({type_name}*)realloc({var_name}, {new_size});'
    
    def emit_calloc(self, var_name: str, type_name: str, count: int) -> str:
        """Emit calloc call (zero-initialized)."""
        return f'{type_name}* {var_name} = ({type_name}*)calloc({count}, sizeof({type_name}));'
    
    def emit_null_check(self, var_name: str) -> str:
        """Emit null pointer check."""
        return f'''if ({var_name} == NULL) {{
    fprintf(stderr, "Memory allocation failed\\n");
    exit(1);
}}'''


class RefCountedMemoryManager(MemoryManager):
    """Memory manager for reference counting."""
    
    @property
    def strategy(self) -> MemoryStrategy:
        return MemoryStrategy.REFERENCE_COUNTED
    
    def emit_allocation(self, var_name: str, type_name: str,
                       size: Optional[int] = None) -> str:
        """Emit reference-counted allocation."""
        return f'{var_name} = {type_name}()  # ref_count = 1'
    
    def emit_deallocation(self, var_name: str) -> str:
        """Emit reference decrement."""
        return f'del {var_name}  # Decrement ref count, free if 0'
    
    def emit_reference(self, source: str, dest: str) -> str:
        """Emit reference (increment ref count)."""
        return f'{dest} = {source}  # Increment ref count'
    
    def emit_weak_reference(self, source: str, dest: str) -> str:
        """Emit weak reference (no ref count increment)."""
        return f'{dest} = weakref.ref({source})  # Weak reference'
    
    def emit_ref_count_check(self, var_name: str) -> str:
        """Emit reference count check."""
        return f'sys.getrefcount({var_name})'


class OwnershipMemoryManager(MemoryManager):
    """Memory manager for ownership model (Rust style)."""
    
    def __init__(self):
        super().__init__()
        self.owners: Dict[str, int] = {}  # var_name -> alloc_id
        self.borrows: Dict[str, Set[str]] = {}  # owner -> borrowed refs
    
    @property
    def strategy(self) -> MemoryStrategy:
        return MemoryStrategy.OWNERSHIP
    
    def emit_allocation(self, var_name: str, type_name: str,
                       size: Optional[int] = None) -> str:
        """Emit owned allocation."""
        return f'let mut {var_name}: {type_name} = {type_name}::new();'
    
    def emit_deallocation(self, var_name: str) -> str:
        """Emit drop (usually implicit in Rust)."""
        return f'drop({var_name});  // Explicit drop (usually implicit at end of scope)'
    
    def emit_box_allocation(self, var_name: str, type_name: str) -> str:
        """Emit Box allocation (heap)."""
        return f'let {var_name}: Box<{type_name}> = Box::new({type_name}::default());'
    
    def emit_move(self, source: str, dest: str) -> str:
        """Emit move (transfer ownership)."""
        return f'let {dest} = {source};  // {source} is moved, no longer valid'
    
    def emit_borrow(self, source: str, dest: str, mutable: bool = False) -> str:
        """Emit borrow (create reference)."""
        if mutable:
            return f'let {dest} = &mut {source};  // Mutable borrow'
        return f'let {dest} = &{source};  // Immutable borrow'
    
    def emit_clone(self, source: str, dest: str) -> str:
        """Emit clone (deep copy)."""
        return f'let {dest} = {source}.clone();  // Clone data'
    
    def emit_rc_allocation(self, var_name: str, type_name: str) -> str:
        """Emit Rc allocation (reference counted)."""
        return f'let {var_name} = Rc::new({type_name}::new());'
    
    def emit_arc_allocation(self, var_name: str, type_name: str) -> str:
        """Emit Arc allocation (atomic reference counted)."""
        return f'let {var_name} = Arc::new({type_name}::new());'
    
    def track_move(self, source: str, dest: str) -> None:
        """Track ownership transfer."""
        if source in self.owners:
            alloc_id = self.owners[source]
            del self.owners[source]
            self.owners[dest] = alloc_id
            
            self.operations.append(MemoryOperation(
                op_type='move',
                allocation_id=alloc_id,
                source_var=source,
                dest_var=dest
            ))
    
    def track_borrow(self, owner: str, borrower: str) -> None:
        """Track a borrow."""
        if owner not in self.borrows:
            self.borrows[owner] = set()
        self.borrows[owner].add(borrower)
        
        self.operations.append(MemoryOperation(
            op_type='ref',
            source_var=owner,
            dest_var=borrower
        ))


class GCMemoryManager(MemoryManager):
    """Memory manager for garbage-collected languages."""
    
    @property
    def strategy(self) -> MemoryStrategy:
        return MemoryStrategy.GARBAGE_COLLECTED
    
    def emit_allocation(self, var_name: str, type_name: str,
                       size: Optional[int] = None) -> str:
        """Emit allocation (GC handles cleanup)."""
        return f'{var_name} = {type_name}()'
    
    def emit_deallocation(self, var_name: str) -> str:
        """Emit reference removal (GC will collect)."""
        return f'{var_name} = None  # Allow GC to collect'
    
    def emit_gc_collect(self) -> str:
        """Emit explicit GC collection request."""
        return 'gc.collect()  # Request GC collection'
    
    def emit_gc_disable(self) -> str:
        """Emit GC disable (for performance)."""
        return 'gc.disable()  # Disable GC temporarily'
    
    def emit_gc_enable(self) -> str:
        """Emit GC enable."""
        return 'gc.enable()  # Re-enable GC'


class RAIIMemoryManager(MemoryManager):
    """Memory manager for RAII pattern (C++/Rust)."""
    
    @property
    def strategy(self) -> MemoryStrategy:
        return MemoryStrategy.RAII
    
    def emit_allocation(self, var_name: str, type_name: str,
                       size: Optional[int] = None) -> str:
        """Emit RAII object creation."""
        return f'{type_name} {var_name};  // RAII: destructor called at scope exit'
    
    def emit_deallocation(self, var_name: str) -> str:
        """Emit nothing (RAII handles cleanup)."""
        return f'// {var_name} cleaned up by RAII at scope exit'
    
    def emit_unique_ptr(self, var_name: str, type_name: str) -> str:
        """Emit unique_ptr (C++)."""
        return f'std::unique_ptr<{type_name}> {var_name} = std::make_unique<{type_name}>();'
    
    def emit_shared_ptr(self, var_name: str, type_name: str) -> str:
        """Emit shared_ptr (C++)."""
        return f'std::shared_ptr<{type_name}> {var_name} = std::make_shared<{type_name}>();'
    
    def emit_weak_ptr(self, source: str, dest: str) -> str:
        """Emit weak_ptr from shared_ptr."""
        return f'std::weak_ptr<auto> {dest}({source});'
    
    def emit_scope_guard(self, cleanup_code: str) -> str:
        """Emit scope guard for cleanup."""
        return f'''auto cleanup = []() {{ {cleanup_code} }};
scope_guard guard(cleanup);'''


class ArenaMemoryManager(MemoryManager):
    """Memory manager for arena allocation."""
    
    @property
    def strategy(self) -> MemoryStrategy:
        return MemoryStrategy.ARENA
    
    def emit_allocation(self, var_name: str, type_name: str,
                       size: Optional[int] = None) -> str:
        """Emit arena allocation."""
        return f'{type_name}* {var_name} = arena_alloc(&arena, sizeof({type_name}));'
    
    def emit_deallocation(self, var_name: str) -> str:
        """Emit nothing (arena frees all at once)."""
        return f'// {var_name}: Arena allocation, freed when arena is destroyed'
    
    def emit_arena_create(self, arena_name: str, size: int) -> str:
        """Emit arena creation."""
        return f'Arena {arena_name} = arena_create({size});'
    
    def emit_arena_destroy(self, arena_name: str) -> str:
        """Emit arena destruction (frees all allocations)."""
        return f'arena_destroy(&{arena_name});  // Frees all arena allocations'
    
    def emit_arena_reset(self, arena_name: str) -> str:
        """Emit arena reset (reuse without freeing)."""
        return f'arena_reset(&{arena_name});  // Reset arena for reuse'


def create_memory_manager(strategy: str) -> MemoryManager:
    """Factory function to create a memory manager."""
    strategy_map = {
        'manual': ManualMemoryManager,
        'c': ManualMemoryManager,
        'refcount': RefCountedMemoryManager,
        'python': RefCountedMemoryManager,
        'ownership': OwnershipMemoryManager,
        'rust': OwnershipMemoryManager,
        'gc': GCMemoryManager,
        'haskell': GCMemoryManager,
        'java': GCMemoryManager,
        'raii': RAIIMemoryManager,
        'cpp': RAIIMemoryManager,
        'arena': ArenaMemoryManager,
    }
    
    manager_class = strategy_map.get(strategy.lower(), ManualMemoryManager)
    return manager_class()


__all__ = [
    'MemoryStrategy', 'AllocationKind', 'Allocation', 'MemoryOperation',
    'MemoryManager', 'ManualMemoryManager', 'RefCountedMemoryManager',
    'OwnershipMemoryManager', 'GCMemoryManager', 'RAIIMemoryManager',
    'ArenaMemoryManager', 'create_memory_manager'
]
