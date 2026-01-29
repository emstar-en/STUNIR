#!/usr/bin/env python3
"""STUNIR Memory Safety Module.

Provides memory safety analysis including use-after-free detection,
double-free detection, memory leak detection, null pointer detection,
and buffer overflow detection.

This module is part of the STUNIR code generation enhancement suite.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple
from enum import Enum, auto

from .manager import Allocation, AllocationKind, MemoryOperation


class SafetyViolationKind(Enum):
    """Kinds of memory safety violations."""
    USE_AFTER_FREE = auto()
    DOUBLE_FREE = auto()
    MEMORY_LEAK = auto()
    NULL_DEREFERENCE = auto()
    BUFFER_OVERFLOW = auto()
    BUFFER_UNDERFLOW = auto()
    UNINITIALIZED_READ = auto()
    DANGLING_POINTER = auto()
    INVALID_FREE = auto()
    STACK_BUFFER_OVERFLOW = auto()
    HEAP_BUFFER_OVERFLOW = auto()
    USE_OF_UNINITIALIZED_MEMORY = auto()
    ALIASING_VIOLATION = auto()


class ViolationSeverity(Enum):
    """Severity of memory safety violations."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


@dataclass
class SafetyViolation:
    """Represents a memory safety violation."""
    kind: SafetyViolationKind
    message: str
    severity: ViolationSeverity = ViolationSeverity.ERROR
    location: Optional[str] = None
    line: Optional[int] = None
    allocation_id: Optional[int] = None
    variable: Optional[str] = None
    suggestion: Optional[str] = None
    
    def __str__(self) -> str:
        sev = self.severity.name
        loc = f" at {self.location}" if self.location else ""
        line_str = f":{self.line}" if self.line else ""
        var_str = f" (variable: {self.variable})" if self.variable else ""
        return f"[{sev}]{loc}{line_str}: {self.kind.name}: {self.message}{var_str}"


@dataclass
class PointerState:
    """State of a pointer variable."""
    name: str
    allocation_id: Optional[int] = None
    is_null: bool = False
    is_valid: bool = True
    is_freed: bool = False
    is_initialized: bool = False
    offset: int = 0  # Offset from base allocation
    
    def invalidate(self) -> None:
        """Mark pointer as invalid."""
        self.is_valid = False
        self.is_freed = True


@dataclass
class BufferBounds:
    """Bounds information for a buffer."""
    allocation_id: int
    base_address: int = 0
    size: int = 0
    element_size: int = 1
    
    def is_valid_offset(self, offset: int) -> bool:
        """Check if offset is within bounds."""
        return 0 <= offset < self.size
    
    def is_valid_range(self, start: int, end: int) -> bool:
        """Check if range is within bounds."""
        return 0 <= start <= end <= self.size


class MemorySafetyAnalyzer:
    """Analyzes code for memory safety violations."""
    
    def __init__(self):
        self.violations: List[SafetyViolation] = []
        self.allocations: Dict[int, Allocation] = {}
        self.pointers: Dict[str, PointerState] = {}
        self.buffers: Dict[int, BufferBounds] = {}
        self.freed_allocations: Set[int] = set()
        self._next_alloc_id = 1
    
    def track_allocation(self, var_name: str, size: int,
                        kind: AllocationKind = AllocationKind.HEAP,
                        location: Optional[str] = None) -> int:
        """Track a new allocation."""
        alloc_id = self._next_alloc_id
        self._next_alloc_id += 1
        
        alloc = Allocation(
            id=alloc_id,
            kind=kind,
            size=size,
            location=location
        )
        self.allocations[alloc_id] = alloc
        
        self.pointers[var_name] = PointerState(
            name=var_name,
            allocation_id=alloc_id,
            is_initialized=True,
            is_valid=True
        )
        
        self.buffers[alloc_id] = BufferBounds(
            allocation_id=alloc_id,
            size=size
        )
        
        return alloc_id
    
    def track_free(self, var_name: str, location: Optional[str] = None) -> None:
        """Track a deallocation."""
        ptr = self.pointers.get(var_name)
        
        if ptr is None:
            self.violations.append(SafetyViolation(
                kind=SafetyViolationKind.INVALID_FREE,
                message=f"Free of unknown pointer '{var_name}'",
                severity=ViolationSeverity.ERROR,
                location=location,
                variable=var_name
            ))
            return
        
        if ptr.allocation_id is None:
            self.violations.append(SafetyViolation(
                kind=SafetyViolationKind.NULL_DEREFERENCE,
                message=f"Free of NULL pointer '{var_name}'",
                severity=ViolationSeverity.ERROR,
                location=location,
                variable=var_name
            ))
            return
        
        if ptr.allocation_id in self.freed_allocations:
            self.violations.append(SafetyViolation(
                kind=SafetyViolationKind.DOUBLE_FREE,
                message=f"Double free of pointer '{var_name}'",
                severity=ViolationSeverity.CRITICAL,
                location=location,
                variable=var_name,
                allocation_id=ptr.allocation_id,
                suggestion="Check control flow to ensure single free"
            ))
            return
        
        # Mark as freed
        self.freed_allocations.add(ptr.allocation_id)
        ptr.invalidate()
        
        if ptr.allocation_id in self.allocations:
            self.allocations[ptr.allocation_id].is_freed = True
            self.allocations[ptr.allocation_id].freed_at = location
    
    def track_use(self, var_name: str, location: Optional[str] = None) -> None:
        """Track use of a pointer."""
        ptr = self.pointers.get(var_name)
        
        if ptr is None:
            self.violations.append(SafetyViolation(
                kind=SafetyViolationKind.USE_OF_UNINITIALIZED_MEMORY,
                message=f"Use of uninitialized pointer '{var_name}'",
                severity=ViolationSeverity.ERROR,
                location=location,
                variable=var_name
            ))
            return
        
        if not ptr.is_initialized:
            self.violations.append(SafetyViolation(
                kind=SafetyViolationKind.UNINITIALIZED_READ,
                message=f"Read of uninitialized pointer '{var_name}'",
                severity=ViolationSeverity.ERROR,
                location=location,
                variable=var_name
            ))
            return
        
        if ptr.is_null:
            self.violations.append(SafetyViolation(
                kind=SafetyViolationKind.NULL_DEREFERENCE,
                message=f"Dereference of NULL pointer '{var_name}'",
                severity=ViolationSeverity.CRITICAL,
                location=location,
                variable=var_name
            ))
            return
        
        if ptr.is_freed or ptr.allocation_id in self.freed_allocations:
            self.violations.append(SafetyViolation(
                kind=SafetyViolationKind.USE_AFTER_FREE,
                message=f"Use after free of pointer '{var_name}'",
                severity=ViolationSeverity.CRITICAL,
                location=location,
                variable=var_name,
                allocation_id=ptr.allocation_id,
                suggestion="Ensure pointer is not used after free"
            ))
    
    def track_index(self, var_name: str, index: int,
                   location: Optional[str] = None) -> None:
        """Track array/buffer indexing."""
        ptr = self.pointers.get(var_name)
        
        if ptr is None:
            self.violations.append(SafetyViolation(
                kind=SafetyViolationKind.USE_OF_UNINITIALIZED_MEMORY,
                message=f"Index into uninitialized buffer '{var_name}'",
                severity=ViolationSeverity.ERROR,
                location=location,
                variable=var_name
            ))
            return
        
        # Check use-after-free first
        self.track_use(var_name, location)
        
        if ptr.allocation_id is None:
            return
        
        bounds = self.buffers.get(ptr.allocation_id)
        if bounds is None:
            return
        
        # Calculate effective offset
        offset = ptr.offset + index
        
        if offset < 0:
            self.violations.append(SafetyViolation(
                kind=SafetyViolationKind.BUFFER_UNDERFLOW,
                message=f"Buffer underflow in '{var_name}' at index {index}",
                severity=ViolationSeverity.CRITICAL,
                location=location,
                variable=var_name,
                allocation_id=ptr.allocation_id
            ))
        elif offset >= bounds.size:
            self.violations.append(SafetyViolation(
                kind=SafetyViolationKind.BUFFER_OVERFLOW,
                message=f"Buffer overflow in '{var_name}' at index {index} (size: {bounds.size})",
                severity=ViolationSeverity.CRITICAL,
                location=location,
                variable=var_name,
                allocation_id=ptr.allocation_id
            ))
    
    def track_assignment(self, dest: str, source: str,
                        location: Optional[str] = None) -> None:
        """Track pointer assignment."""
        src_ptr = self.pointers.get(source)
        
        if src_ptr is None:
            # Source might be a literal or constant
            self.pointers[dest] = PointerState(
                name=dest,
                is_initialized=True
            )
            return
        
        # Create copy of pointer state
        self.pointers[dest] = PointerState(
            name=dest,
            allocation_id=src_ptr.allocation_id,
            is_null=src_ptr.is_null,
            is_valid=src_ptr.is_valid,
            is_freed=src_ptr.is_freed,
            is_initialized=True,
            offset=src_ptr.offset
        )
        
        # Check if source was freed (creates dangling pointer)
        if src_ptr.is_freed:
            self.violations.append(SafetyViolation(
                kind=SafetyViolationKind.DANGLING_POINTER,
                message=f"Assignment of dangling pointer '{source}' to '{dest}'",
                severity=ViolationSeverity.WARNING,
                location=location,
                variable=dest
            ))
    
    def track_null_assignment(self, var_name: str,
                             location: Optional[str] = None) -> None:
        """Track assignment of NULL to pointer."""
        self.pointers[var_name] = PointerState(
            name=var_name,
            is_null=True,
            is_initialized=True
        )
    
    def track_arithmetic(self, var_name: str, offset_change: int,
                        location: Optional[str] = None) -> None:
        """Track pointer arithmetic."""
        ptr = self.pointers.get(var_name)
        
        if ptr is None:
            return
        
        ptr.offset += offset_change
        
        # Check if pointer goes out of bounds
        if ptr.allocation_id is not None:
            bounds = self.buffers.get(ptr.allocation_id)
            if bounds and not bounds.is_valid_offset(ptr.offset):
                self.violations.append(SafetyViolation(
                    kind=SafetyViolationKind.BUFFER_OVERFLOW,
                    message=f"Pointer arithmetic moves '{var_name}' out of bounds",
                    severity=ViolationSeverity.WARNING,
                    location=location,
                    variable=var_name
                ))
    
    def check_leaks(self) -> List[SafetyViolation]:
        """Check for memory leaks at end of analysis."""
        leaks = []
        
        for alloc_id, alloc in self.allocations.items():
            if alloc.kind == AllocationKind.HEAP and not alloc.is_freed:
                leaks.append(SafetyViolation(
                    kind=SafetyViolationKind.MEMORY_LEAK,
                    message=f"Memory leak: allocation of {alloc.type_name} not freed",
                    severity=ViolationSeverity.WARNING,
                    location=alloc.location,
                    allocation_id=alloc_id,
                    suggestion="Ensure all heap allocations are freed"
                ))
        
        self.violations.extend(leaks)
        return leaks
    
    def analyze_ir(self, ir_data: Dict) -> List[SafetyViolation]:
        """Analyze IR for memory safety issues."""
        self.violations.clear()
        
        for func in ir_data.get('ir_functions', []):
            self._analyze_function(func)
        
        # Check for leaks
        self.check_leaks()
        
        return self.violations
    
    def _analyze_function(self, func: Dict) -> None:
        """Analyze a function for memory safety."""
        body = func.get('body', [])
        
        for stmt in body:
            self._analyze_statement(stmt)
    
    def _analyze_statement(self, stmt: Any) -> None:
        """Analyze a statement for memory safety."""
        if not isinstance(stmt, dict):
            return
        
        stmt_type = stmt.get('type', '')
        
        if stmt_type in ('var_decl', 'let'):
            self._analyze_var_decl(stmt)
        elif stmt_type == 'assign':
            self._analyze_assign(stmt)
        elif stmt_type == 'call':
            self._analyze_call(stmt)
        elif stmt_type == 'index':
            self._analyze_index(stmt)
        elif stmt_type == 'return':
            self._analyze_return(stmt)
        elif stmt_type in ('if', 'while', 'for'):
            self._analyze_control_flow(stmt)
    
    def _analyze_var_decl(self, stmt: Dict) -> None:
        """Analyze variable declaration."""
        var_name = stmt.get('var_name', stmt.get('name', ''))
        init = stmt.get('init', stmt.get('value'))
        
        if init is None:
            # Uninitialized pointer
            self.pointers[var_name] = PointerState(
                name=var_name,
                is_initialized=False
            )
        elif isinstance(init, dict):
            init_type = init.get('type', '')
            if init_type == 'call':
                func = init.get('func', '')
                if func in ('malloc', 'calloc', 'realloc', 'new'):
                    # Memory allocation
                    size = self._get_alloc_size(init)
                    self.track_allocation(var_name, size)
                elif func == 'NULL' or init.get('value') is None:
                    self.track_null_assignment(var_name)
    
    def _analyze_assign(self, stmt: Dict) -> None:
        """Analyze assignment."""
        target = stmt.get('target', '')
        value = stmt.get('value')
        
        if isinstance(value, dict):
            val_type = value.get('type', '')
            if val_type == 'call':
                func = value.get('func', '')
                if func in ('malloc', 'calloc', 'new'):
                    size = self._get_alloc_size(value)
                    self.track_allocation(target, size)
                    return
            elif val_type == 'var':
                self.track_assignment(target, value.get('name', ''))
                return
        elif value is None:
            self.track_null_assignment(target)
            return
        elif isinstance(value, str):
            self.track_assignment(target, value)
            return
    
    def _analyze_call(self, stmt: Dict) -> None:
        """Analyze function call."""
        func = stmt.get('func', '')
        args = stmt.get('args', [])
        
        if func == 'free':
            if args:
                arg = args[0]
                if isinstance(arg, str):
                    self.track_free(arg)
                elif isinstance(arg, dict) and arg.get('type') == 'var':
                    self.track_free(arg.get('name', ''))
        
        # Track usage of pointer arguments
        for arg in args:
            if isinstance(arg, str):
                self.track_use(arg)
            elif isinstance(arg, dict) and arg.get('type') == 'var':
                self.track_use(arg.get('name', ''))
    
    def _analyze_index(self, stmt: Dict) -> None:
        """Analyze array indexing."""
        base = stmt.get('base', '')
        index = stmt.get('index')
        
        if isinstance(base, dict) and base.get('type') == 'var':
            base = base.get('name', '')
        
        if isinstance(index, (int, float)):
            self.track_index(base, int(index))
        else:
            # Can't statically determine index
            self.track_use(base)
    
    def _analyze_return(self, stmt: Dict) -> None:
        """Analyze return statement."""
        value = stmt.get('value')
        
        # Check if returning a local pointer (dangling)
        if isinstance(value, dict) and value.get('type') == 'unary':
            op = value.get('op', '')
            if op == '&':  # Address-of
                operand = value.get('operand')
                if isinstance(operand, dict) and operand.get('type') == 'var':
                    var_name = operand.get('name', '')
                    ptr = self.pointers.get(var_name)
                    if ptr and ptr.allocation_id:
                        alloc = self.allocations.get(ptr.allocation_id)
                        if alloc and alloc.kind == AllocationKind.STACK:
                            self.violations.append(SafetyViolation(
                                kind=SafetyViolationKind.DANGLING_POINTER,
                                message=f"Returning address of local variable '{var_name}'",
                                severity=ViolationSeverity.CRITICAL,
                                variable=var_name
                            ))
    
    def _analyze_control_flow(self, stmt: Dict) -> None:
        """Analyze control flow statements."""
        # Analyze condition
        cond = stmt.get('cond')
        if isinstance(cond, dict):
            self._check_null_check(cond)
        
        # Analyze branches
        for s in stmt.get('then', []):
            self._analyze_statement(s)
        for s in stmt.get('else', []):
            self._analyze_statement(s)
        for s in stmt.get('body', []):
            self._analyze_statement(s)
    
    def _check_null_check(self, cond: Dict) -> None:
        """Check if condition is a null check."""
        cond_type = cond.get('type', '')
        
        if cond_type == 'binary':
            op = cond.get('op', '')
            left = cond.get('left')
            right = cond.get('right')
            
            # Check for ptr != NULL or ptr == NULL patterns
            if op in ('!=', '=='):
                var_name = None
                if isinstance(left, dict) and left.get('type') == 'var':
                    var_name = left.get('name')
                elif isinstance(right, dict) and right.get('type') == 'var':
                    var_name = right.get('name')
                
                # This is a null check, which is good practice
                # We could track this to improve analysis
    
    def _get_alloc_size(self, call: Dict) -> int:
        """Get allocation size from malloc/calloc call."""
        args = call.get('args', [])
        func = call.get('func', '')
        
        if func == 'malloc' and args:
            arg = args[0]
            if isinstance(arg, (int, float)):
                return int(arg)
        elif func == 'calloc' and len(args) >= 2:
            count = args[0]
            size = args[1]
            if isinstance(count, (int, float)) and isinstance(size, (int, float)):
                return int(count) * int(size)
        
        return 0  # Unknown size
    
    def get_errors(self) -> List[SafetyViolation]:
        """Get only error/critical violations."""
        return [v for v in self.violations 
                if v.severity in (ViolationSeverity.ERROR, ViolationSeverity.CRITICAL)]
    
    def get_warnings(self) -> List[SafetyViolation]:
        """Get only warning violations."""
        return [v for v in self.violations 
                if v.severity == ViolationSeverity.WARNING]
    
    def has_critical_issues(self) -> bool:
        """Check if there are any critical safety issues."""
        return any(v.severity == ViolationSeverity.CRITICAL for v in self.violations)
    
    def get_summary(self) -> Dict[str, int]:
        """Get summary of violations by kind."""
        summary: Dict[str, int] = {}
        for v in self.violations:
            kind_name = v.kind.name
            summary[kind_name] = summary.get(kind_name, 0) + 1
        return summary


__all__ = [
    'SafetyViolationKind', 'ViolationSeverity', 'SafetyViolation',
    'PointerState', 'BufferBounds', 'MemorySafetyAnalyzer'
]
