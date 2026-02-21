"""STUNIR Memory Management Module.

Provides memory management patterns and safety analysis for
different target languages.
"""

from .manager import (
    MemoryStrategy, AllocationKind, Allocation, MemoryOperation,
    MemoryManager, ManualMemoryManager, RefCountedMemoryManager,
    OwnershipMemoryManager, GCMemoryManager, RAIIMemoryManager,
    ArenaMemoryManager, create_memory_manager
)

from .safety import (
    SafetyViolationKind, ViolationSeverity, SafetyViolation,
    PointerState, BufferBounds, MemorySafetyAnalyzer
)

__all__ = [
    # Manager
    'MemoryStrategy', 'AllocationKind', 'Allocation', 'MemoryOperation',
    'MemoryManager', 'ManualMemoryManager', 'RefCountedMemoryManager',
    'OwnershipMemoryManager', 'GCMemoryManager', 'RAIIMemoryManager',
    'ArenaMemoryManager', 'create_memory_manager',
    # Safety
    'SafetyViolationKind', 'ViolationSeverity', 'SafetyViolation',
    'PointerState', 'BufferBounds', 'MemorySafetyAnalyzer'
]
