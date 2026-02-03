#!/usr/bin/env python3
"""STUNIR Enhancement Context Module.

This module provides the EnhancementContext class that packages all enhancement
data for use by target emitters during code generation.

The EnhancementContext serves as the unified interface between the enhancement
pipeline (control flow, type system, semantic, memory, optimization) and the
code emitters.

Part of Phase 1 (Foundation) of the STUNIR Enhancement Integration.

Example:
    >>> from tools.integration import EnhancementContext
    >>> context = EnhancementContext(original_ir=ir_data)
    >>> cfg = context.get_function_cfg('main')
    >>> var_info = context.lookup_variable('x')
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

# Type checking imports to avoid circular dependencies
if TYPE_CHECKING:
    from tools.ir.control_flow import ControlFlowGraph, LoopInfo, BranchInfo
    from tools.stunir_types import STUNIRType, TypeRegistry
    from tools.semantic import SymbolTable, VariableInfo, FunctionInfo, SemanticIssue
    from tools.memory import MemoryManager, SafetyViolation
    from tools.optimize import OptimizationStats, OptimizationLevel

logger = logging.getLogger(__name__)


class EnhancementStatus(Enum):
    """Status of an enhancement analysis."""
    NOT_RUN = auto()
    SUCCESS = auto()
    PARTIAL = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class ControlFlowData:
    """Container for control flow analysis results.
    
    Attributes:
        cfgs: Mapping of function names to their control flow graphs.
        loops: Mapping of function names to detected loops.
        branches: Mapping of function names to branch information.
        dominators: Mapping of function names to dominator trees.
        status: Status of the control flow analysis.
        error: Error message if analysis failed.
    """
    cfgs: Dict[str, Any] = field(default_factory=dict)
    loops: Dict[str, List[Any]] = field(default_factory=dict)
    branches: Dict[str, List[Any]] = field(default_factory=dict)
    dominators: Dict[str, Dict[int, Set[int]]] = field(default_factory=dict)
    status: EnhancementStatus = EnhancementStatus.NOT_RUN
    error: Optional[str] = None
    
    def get_cfg(self, func_name: str) -> Optional[Any]:
        """Get CFG for a specific function."""
        return self.cfgs.get(func_name)
    
    def get_loops(self, func_name: str) -> List[Any]:
        """Get loops for a specific function."""
        return self.loops.get(func_name, [])
    
    def get_branches(self, func_name: str) -> List[Any]:
        """Get branches for a specific function."""
        return self.branches.get(func_name, [])
    
    def is_available(self) -> bool:
        """Check if control flow data is available."""
        return self.status in (EnhancementStatus.SUCCESS, EnhancementStatus.PARTIAL)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            'cfgs': {k: self._serialize_cfg(v) for k, v in self.cfgs.items()},
            'loops': {k: [self._serialize_loop(l) for l in v] for k, v in self.loops.items()},
            'branches': {k: [self._serialize_branch(b) for b in v] for k, v in self.branches.items()},
            'status': self.status.name,
            'error': self.error
        }
    
    def _serialize_cfg(self, cfg: Any) -> Dict[str, Any]:
        """Serialize a CFG to dictionary."""
        if hasattr(cfg, 'to_dict'):
            return cfg.to_dict()
        return {'type': 'cfg', 'data': str(cfg)}
    
    def _serialize_loop(self, loop: Any) -> Dict[str, Any]:
        """Serialize loop info to dictionary."""
        if hasattr(loop, '__dict__'):
            return {k: v if not isinstance(v, set) else list(v) 
                    for k, v in loop.__dict__.items()}
        return {'data': str(loop)}
    
    def _serialize_branch(self, branch: Any) -> Dict[str, Any]:
        """Serialize branch info to dictionary."""
        if hasattr(branch, '__dict__'):
            return dict(branch.__dict__)
        return {'data': str(branch)}


@dataclass
class TypeSystemData:
    """Container for type system analysis results.
    
    Attributes:
        type_mappings: Mapping of expression IDs to inferred types.
        type_registry: Type registry with all known types.
        constraints: Type constraints discovered during inference.
        inference_results: Full type inference results per function.
        status: Status of the type analysis.
        error: Error message if analysis failed.
    """
    type_mappings: Dict[str, Any] = field(default_factory=dict)
    type_registry: Optional[Any] = None
    constraints: List[Any] = field(default_factory=list)
    inference_results: Dict[str, Any] = field(default_factory=dict)
    status: EnhancementStatus = EnhancementStatus.NOT_RUN
    error: Optional[str] = None
    
    def get_type(self, expr_id: str) -> Optional[Any]:
        """Get inferred type for an expression."""
        return self.type_mappings.get(expr_id)
    
    def get_function_types(self, func_name: str) -> Dict[str, Any]:
        """Get all type mappings for a function."""
        return self.inference_results.get(func_name, {})
    
    def is_available(self) -> bool:
        """Check if type data is available."""
        return self.status in (EnhancementStatus.SUCCESS, EnhancementStatus.PARTIAL)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            'type_mappings': {k: str(v) for k, v in self.type_mappings.items()},
            'constraints': [str(c) for c in self.constraints],
            'inference_results': self.inference_results,
            'status': self.status.name,
            'error': self.error
        }


@dataclass
class SemanticData:
    """Container for semantic analysis results.
    
    Attributes:
        symbol_table: Symbol table with variable and function info.
        call_graph: Function call graph.
        def_use_chains: Definition-use chains for variables.
        issues: Semantic issues and warnings discovered.
        status: Status of the semantic analysis.
        error: Error message if analysis failed.
    """
    symbol_table: Optional[Any] = None
    call_graph: Dict[str, Set[str]] = field(default_factory=dict)
    def_use_chains: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    issues: List[Any] = field(default_factory=list)
    status: EnhancementStatus = EnhancementStatus.NOT_RUN
    error: Optional[str] = None
    
    def lookup_variable(self, name: str, scope: Optional[str] = None) -> Optional[Any]:
        """Look up variable in symbol table."""
        if self.symbol_table is None:
            return None
        if hasattr(self.symbol_table, 'lookup_variable'):
            return self.symbol_table.lookup_variable(name)
        return None
    
    def lookup_function(self, name: str) -> Optional[Any]:
        """Look up function in symbol table."""
        if self.symbol_table is None:
            return None
        if hasattr(self.symbol_table, 'lookup_function'):
            return self.symbol_table.lookup_function(name)
        return None
    
    def get_callers(self, func_name: str) -> Set[str]:
        """Get functions that call the given function."""
        callers = set()
        for caller, callees in self.call_graph.items():
            if func_name in callees:
                callers.add(caller)
        return callers
    
    def get_callees(self, func_name: str) -> Set[str]:
        """Get functions called by the given function."""
        return self.call_graph.get(func_name, set())
    
    def is_available(self) -> bool:
        """Check if semantic data is available."""
        return self.status in (EnhancementStatus.SUCCESS, EnhancementStatus.PARTIAL)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            'call_graph': {k: list(v) for k, v in self.call_graph.items()},
            'def_use_chains': self.def_use_chains,
            'issues': [self._serialize_issue(i) for i in self.issues],
            'status': self.status.name,
            'error': self.error
        }
    
    def _serialize_issue(self, issue: Any) -> Dict[str, Any]:
        """Serialize a semantic issue to dictionary."""
        if hasattr(issue, '__dict__'):
            return {k: str(v) if isinstance(v, Enum) else v 
                    for k, v in issue.__dict__.items()}
        return {'message': str(issue)}


@dataclass
class MemoryData:
    """Container for memory analysis results.
    
    Attributes:
        patterns: Memory patterns detected (stack, heap, etc.).
        safety_violations: Memory safety violations found.
        allocation_sites: Mapping of variables to allocation sites.
        lifetime_info: Lifetime information for variables.
        memory_strategy: Recommended memory strategy for target.
        status: Status of the memory analysis.
        error: Error message if analysis failed.
    """
    patterns: Dict[str, str] = field(default_factory=dict)
    safety_violations: List[Any] = field(default_factory=list)
    allocation_sites: Dict[str, Any] = field(default_factory=dict)
    lifetime_info: Dict[str, Any] = field(default_factory=dict)
    memory_strategy: Optional[str] = None
    status: EnhancementStatus = EnhancementStatus.NOT_RUN
    error: Optional[str] = None
    
    def get_pattern(self, var_name: str) -> Optional[str]:
        """Get memory pattern for a variable."""
        return self.patterns.get(var_name)
    
    def get_allocation_site(self, var_name: str) -> Optional[Any]:
        """Get allocation site for a variable."""
        return self.allocation_sites.get(var_name)
    
    def has_violations(self) -> bool:
        """Check if any safety violations were found."""
        return len(self.safety_violations) > 0
    
    def is_available(self) -> bool:
        """Check if memory data is available."""
        return self.status in (EnhancementStatus.SUCCESS, EnhancementStatus.PARTIAL)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            'patterns': self.patterns,
            'safety_violations': [self._serialize_violation(v) for v in self.safety_violations],
            'allocation_sites': self.allocation_sites,
            'lifetime_info': self.lifetime_info,
            'memory_strategy': self.memory_strategy,
            'status': self.status.name,
            'error': self.error
        }
    
    def _serialize_violation(self, violation: Any) -> Dict[str, Any]:
        """Serialize a safety violation to dictionary."""
        if hasattr(violation, '__dict__'):
            return {k: str(v) if isinstance(v, Enum) else v 
                    for k, v in violation.__dict__.items()}
        return {'message': str(violation)}


@dataclass
class OptimizationData:
    """Container for optimization analysis results.
    
    Attributes:
        optimization_level: Level of optimization applied.
        passes_applied: List of optimization passes applied.
        optimization_hints: Hints for emitters.
        stats: Optimization statistics.
        optimized_ir: Optimized IR if available.
        status: Status of the optimization.
        error: Error message if optimization failed.
    """
    optimization_level: Optional[str] = None
    passes_applied: List[str] = field(default_factory=list)
    optimization_hints: Dict[str, Any] = field(default_factory=dict)
    stats: Optional[Any] = None
    optimized_ir: Optional[Dict[str, Any]] = None
    status: EnhancementStatus = EnhancementStatus.NOT_RUN
    error: Optional[str] = None
    
    def get_hint(self, key: str) -> Optional[Any]:
        """Get an optimization hint."""
        return self.optimization_hints.get(key)
    
    def was_pass_applied(self, pass_name: str) -> bool:
        """Check if a specific optimization pass was applied."""
        return pass_name in self.passes_applied
    
    def get_optimized_function(self, func_name: str) -> Optional[Dict[str, Any]]:
        """Get optimized version of a function."""
        if self.optimized_ir is None:
            return None
        for func in self.optimized_ir.get('ir_functions', []):
            if func.get('name') == func_name:
                return func
        return None
    
    def is_available(self) -> bool:
        """Check if optimization data is available."""
        return self.status in (EnhancementStatus.SUCCESS, EnhancementStatus.PARTIAL)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            'optimization_level': self.optimization_level,
            'passes_applied': self.passes_applied,
            'optimization_hints': self.optimization_hints,
            'stats': self._serialize_stats(self.stats) if self.stats else None,
            'status': self.status.name,
            'error': self.error
        }
    
    def _serialize_stats(self, stats: Any) -> Dict[str, Any]:
        """Serialize optimization stats to dictionary."""
        if hasattr(stats, '__dict__'):
            return dict(stats.__dict__)
        return {'data': str(stats)}


@dataclass
class EnhancementContext:
    """Unified context containing all enhancement data for emitters.
    
    This class packages all enhancement analysis results into a single context
    object that can be passed to target emitters. It provides safe accessors
    for each type of enhancement data and supports serialization.
    
    The EnhancementContext is the primary interface between the enhancement
    pipeline and the code emitters. It handles graceful degradation when
    some enhancements are not available.
    
    Attributes:
        original_ir: The original IR data before any processing.
        control_flow_data: Control flow analysis results.
        type_system_data: Type system analysis results.
        semantic_data: Semantic analysis results.
        memory_data: Memory analysis results.
        optimization_data: Optimization analysis results.
        target_language: Target language for code generation.
        options: Additional configuration options.
        created_at: Timestamp of context creation.
        metadata: Additional metadata for diagnostics.
    
    Example:
        >>> context = EnhancementContext(original_ir=ir_data)
        >>> context.target_language = 'rust'
        >>> 
        >>> # Access control flow data safely
        >>> if context.control_flow_data.is_available():
        ...     cfg = context.get_function_cfg('main')
        ...     loops = context.get_loops('main')
        >>> 
        >>> # Access semantic data safely
        >>> var_info = context.lookup_variable('x')
        >>> if var_info:
        ...     print(f"Variable x has type {var_info.type}")
    """
    
    # Original IR
    original_ir: Dict[str, Any] = field(default_factory=dict)
    
    # Enhancement data containers
    control_flow_data: ControlFlowData = field(default_factory=ControlFlowData)
    type_system_data: TypeSystemData = field(default_factory=TypeSystemData)
    semantic_data: SemanticData = field(default_factory=SemanticData)
    memory_data: MemoryData = field(default_factory=MemoryData)
    optimization_data: OptimizationData = field(default_factory=OptimizationData)
    
    # Configuration
    target_language: str = "python"
    options: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # -------------------------------------------------------------------------
    # Control Flow Accessors
    # -------------------------------------------------------------------------
    
    def get_function_cfg(self, func_name: str) -> Optional[Any]:
        """Get CFG for a specific function.
        
        Args:
            func_name: Name of the function.
            
        Returns:
            ControlFlowGraph for the function, or None if not available.
        """
        return self.control_flow_data.get_cfg(func_name)
    
    def get_loops(self, func_name: str) -> List[Any]:
        """Get detected loops for a function.
        
        Args:
            func_name: Name of the function.
            
        Returns:
            List of LoopInfo objects.
        """
        return self.control_flow_data.get_loops(func_name)
    
    def get_branches(self, func_name: str) -> List[Any]:
        """Get branch information for a function.
        
        Args:
            func_name: Name of the function.
            
        Returns:
            List of BranchInfo objects.
        """
        return self.control_flow_data.get_branches(func_name)
    
    # -------------------------------------------------------------------------
    # Type System Accessors
    # -------------------------------------------------------------------------
    
    def get_expression_type(self, expr_id: str) -> Optional[Any]:
        """Get inferred type of an expression.
        
        Args:
            expr_id: Identifier for the expression.
            
        Returns:
            STUNIRType for the expression, or None if not available.
        """
        return self.type_system_data.get_type(expr_id)
    
    def get_function_types(self, func_name: str) -> Dict[str, Any]:
        """Get all type mappings for a function.
        
        Args:
            func_name: Name of the function.
            
        Returns:
            Dictionary mapping expression IDs to types.
        """
        return self.type_system_data.get_function_types(func_name)
    
    # -------------------------------------------------------------------------
    # Semantic Accessors
    # -------------------------------------------------------------------------
    
    def lookup_variable(self, name: str, scope: Optional[str] = None) -> Optional[Any]:
        """Look up variable in symbol table.
        
        Args:
            name: Variable name.
            scope: Optional scope to search in.
            
        Returns:
            VariableInfo for the variable, or None if not found.
        """
        return self.semantic_data.lookup_variable(name, scope)
    
    def lookup_function(self, name: str) -> Optional[Any]:
        """Look up function in symbol table.
        
        Args:
            name: Function name.
            
        Returns:
            FunctionInfo for the function, or None if not found.
        """
        return self.semantic_data.lookup_function(name)
    
    def get_symbol_table(self) -> Optional[Any]:
        """Get the full symbol table.
        
        Returns:
            SymbolTable, or None if semantic analysis was not run.
        """
        return self.semantic_data.symbol_table
    
    def get_semantic_issues(self) -> List[Any]:
        """Get all semantic issues discovered.
        
        Returns:
            List of SemanticIssue objects.
        """
        return self.semantic_data.issues
    
    # -------------------------------------------------------------------------
    # Memory Accessors
    # -------------------------------------------------------------------------
    
    def get_memory_strategy(self) -> Optional[str]:
        """Get recommended memory strategy for target language.
        
        Returns:
            Memory strategy name (e.g., 'gc', 'manual', 'ownership').
        """
        return self.memory_data.memory_strategy
    
    def get_memory_pattern(self, var_name: str) -> Optional[str]:
        """Get memory pattern for a variable.
        
        Args:
            var_name: Variable name.
            
        Returns:
            Memory pattern (e.g., 'stack', 'heap', 'static').
        """
        return self.memory_data.get_pattern(var_name)
    
    def get_safety_violations(self) -> List[Any]:
        """Get memory safety violations.
        
        Returns:
            List of SafetyViolation objects.
        """
        return self.memory_data.safety_violations
    
    # -------------------------------------------------------------------------
    # Optimization Accessors
    # -------------------------------------------------------------------------
    
    def get_optimized_ir(self) -> Optional[Dict[str, Any]]:
        """Get optimized IR if available.
        
        Returns:
            Optimized IR dictionary, or None if not available.
        """
        return self.optimization_data.optimized_ir
    
    def get_optimized_function(self, func_name: str) -> Optional[Dict[str, Any]]:
        """Get optimized version of a function.
        
        Args:
            func_name: Name of the function.
            
        Returns:
            Optimized function dictionary, or None if not available.
        """
        return self.optimization_data.get_optimized_function(func_name)
    
    def get_optimization_hint(self, key: str) -> Optional[Any]:
        """Get an optimization hint.
        
        Args:
            key: Hint key.
            
        Returns:
            Hint value, or None if not available.
        """
        return self.optimization_data.get_hint(key)
    
    # -------------------------------------------------------------------------
    # IR Accessors
    # -------------------------------------------------------------------------
    
    def get_ir(self) -> Dict[str, Any]:
        """Get the best available IR (optimized if available, else original).
        
        Returns:
            IR dictionary.
        """
        if self.optimization_data.is_available() and self.optimization_data.optimized_ir:
            return self.optimization_data.optimized_ir
        return self.original_ir
    
    def get_functions(self) -> List[Dict[str, Any]]:
        """Get all functions from the best available IR.
        
        Returns:
            List of function dictionaries.
        """
        ir = self.get_ir()
        return ir.get('ir_functions', [])
    
    def get_function(self, func_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific function from the best available IR.
        
        Args:
            func_name: Name of the function.
            
        Returns:
            Function dictionary, or None if not found.
        """
        for func in self.get_functions():
            if func.get('name') == func_name:
                return func
        return None
    
    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate the enhancement context.
        
        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors = []
        
        # Check original IR
        if not self.original_ir:
            errors.append("original_ir is empty")
        elif 'ir_functions' not in self.original_ir:
            errors.append("original_ir missing 'ir_functions' key")
        
        # Check target language
        if not self.target_language:
            errors.append("target_language is not set")
        
        # Check for critical failures
        critical_failures = []
        if self.control_flow_data.status == EnhancementStatus.FAILED:
            critical_failures.append(f"control_flow: {self.control_flow_data.error}")
        if self.semantic_data.status == EnhancementStatus.FAILED:
            critical_failures.append(f"semantic: {self.semantic_data.error}")
        
        if critical_failures:
            errors.append(f"Critical enhancement failures: {critical_failures}")
        
        return len(errors) == 0, errors
    
    def is_complete(self) -> bool:
        """Check if all enhancements completed successfully.
        
        Returns:
            True if all enhancements succeeded.
        """
        return all([
            self.control_flow_data.status == EnhancementStatus.SUCCESS,
            self.type_system_data.status == EnhancementStatus.SUCCESS,
            self.semantic_data.status == EnhancementStatus.SUCCESS,
            self.memory_data.status == EnhancementStatus.SUCCESS,
            self.optimization_data.status == EnhancementStatus.SUCCESS
        ])
    
    def get_status_summary(self) -> Dict[str, str]:
        """Get a summary of enhancement statuses.
        
        Returns:
            Dictionary mapping enhancement names to status strings.
        """
        return {
            'control_flow': self.control_flow_data.status.name,
            'type_system': self.type_system_data.status.name,
            'semantic': self.semantic_data.status.name,
            'memory': self.memory_data.status.name,
            'optimization': self.optimization_data.status.name
        }
    
    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the context to a dictionary.
        
        Returns:
            Dictionary representation of the context.
        """
        return {
            'original_ir': self.original_ir,
            'control_flow_data': self.control_flow_data.to_dict(),
            'type_system_data': self.type_system_data.to_dict(),
            'semantic_data': self.semantic_data.to_dict(),
            'memory_data': self.memory_data.to_dict(),
            'optimization_data': self.optimization_data.to_dict(),
            'target_language': self.target_language,
            'options': self.options,
            'created_at': self.created_at,
            'metadata': self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize the context to JSON.
        
        Args:
            indent: JSON indentation level.
            
        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancementContext':
        """Deserialize a context from a dictionary.
        
        Args:
            data: Dictionary representation of the context.
            
        Returns:
            EnhancementContext instance.
        """
        context = cls(
            original_ir=data.get('original_ir', {}),
            target_language=data.get('target_language', 'python'),
            options=data.get('options', {}),
            created_at=data.get('created_at', datetime.utcnow().isoformat()),
            metadata=data.get('metadata', {})
        )
        
        # Restore control flow data
        cf_data = data.get('control_flow_data', {})
        context.control_flow_data.status = EnhancementStatus[cf_data.get('status', 'NOT_RUN')]
        context.control_flow_data.error = cf_data.get('error')
        context.control_flow_data.cfgs = cf_data.get('cfgs', {})
        context.control_flow_data.loops = cf_data.get('loops', {})
        context.control_flow_data.branches = cf_data.get('branches', {})
        
        # Restore type system data
        ts_data = data.get('type_system_data', {})
        context.type_system_data.status = EnhancementStatus[ts_data.get('status', 'NOT_RUN')]
        context.type_system_data.error = ts_data.get('error')
        context.type_system_data.type_mappings = ts_data.get('type_mappings', {})
        context.type_system_data.inference_results = ts_data.get('inference_results', {})
        
        # Restore semantic data
        sem_data = data.get('semantic_data', {})
        context.semantic_data.status = EnhancementStatus[sem_data.get('status', 'NOT_RUN')]
        context.semantic_data.error = sem_data.get('error')
        context.semantic_data.call_graph = {k: set(v) for k, v in sem_data.get('call_graph', {}).items()}
        context.semantic_data.def_use_chains = sem_data.get('def_use_chains', {})
        
        # Restore memory data
        mem_data = data.get('memory_data', {})
        context.memory_data.status = EnhancementStatus[mem_data.get('status', 'NOT_RUN')]
        context.memory_data.error = mem_data.get('error')
        context.memory_data.patterns = mem_data.get('patterns', {})
        context.memory_data.memory_strategy = mem_data.get('memory_strategy')
        
        # Restore optimization data
        opt_data = data.get('optimization_data', {})
        context.optimization_data.status = EnhancementStatus[opt_data.get('status', 'NOT_RUN')]
        context.optimization_data.error = opt_data.get('error')
        context.optimization_data.optimization_level = opt_data.get('optimization_level')
        context.optimization_data.passes_applied = opt_data.get('passes_applied', [])
        context.optimization_data.optimization_hints = opt_data.get('optimization_hints', {})
        
        return context
    
    @classmethod
    def from_json(cls, json_str: str) -> 'EnhancementContext':
        """Deserialize a context from JSON.
        
        Args:
            json_str: JSON string representation.
            
        Returns:
            EnhancementContext instance.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def __repr__(self) -> str:
        """String representation of the context."""
        status = self.get_status_summary()
        return (
            f"EnhancementContext("
            f"target={self.target_language}, "
            f"funcs={len(self.get_functions())}, "
            f"cf={status['control_flow']}, "
            f"ts={status['type_system']}, "
            f"sem={status['semantic']}, "
            f"mem={status['memory']}, "
            f"opt={status['optimization']})"
        )


# Convenience function to create a minimal context for testing
def create_minimal_context(
    ir_data: Dict[str, Any],
    target_language: str = "python"
) -> EnhancementContext:
    """Create a minimal EnhancementContext for testing or fallback.
    
    Args:
        ir_data: IR data dictionary.
        target_language: Target language.
        
    Returns:
        EnhancementContext with NOT_RUN status for all enhancements.
    """
    return EnhancementContext(
        original_ir=ir_data,
        target_language=target_language
    )
