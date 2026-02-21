#!/usr/bin/env python3
"""STUNIR Optimization Pass Framework.

Provides base classes and pass manager for optimization passes.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type
import copy
import time
import logging

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Optimization levels."""
    O0 = 0  # No optimization
    O1 = 1  # Basic optimizations (dead code, constant folding)
    O2 = 2  # Standard optimizations (O1 + CSE, loop invariant motion)
    O3 = 3  # Aggressive optimizations (O2 + inlining, unrolling)
    Os = 4  # Size optimization
    Oz = 5  # Minimum size


class PassKind(Enum):
    """Kinds of optimization passes."""
    ANALYSIS = auto()       # Analysis-only pass (no IR modification)
    TRANSFORM = auto()      # Transformation pass (modifies IR)
    CLEANUP = auto()        # Cleanup pass (final simplifications)


@dataclass
class PassStats:
    """Statistics from an optimization pass."""
    pass_name: str
    ir_changes: int = 0
    statements_removed: int = 0
    statements_modified: int = 0
    constants_folded: int = 0
    subexpressions_eliminated: int = 0
    functions_inlined: int = 0
    loops_optimized: int = 0
    dead_stores_removed: int = 0
    strength_reductions: int = 0
    copies_propagated: int = 0
    time_ms: float = 0.0
    
    def __str__(self) -> str:
        parts = [f"{self.pass_name}:"]
        if self.ir_changes > 0:
            parts.append(f"changes={self.ir_changes}")
        if self.statements_removed > 0:
            parts.append(f"removed={self.statements_removed}")
        if self.constants_folded > 0:
            parts.append(f"folded={self.constants_folded}")
        if self.subexpressions_eliminated > 0:
            parts.append(f"cse={self.subexpressions_eliminated}")
        if self.functions_inlined > 0:
            parts.append(f"inlined={self.functions_inlined}")
        parts.append(f"time={self.time_ms:.2f}ms")
        return " ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pass_name': self.pass_name,
            'ir_changes': self.ir_changes,
            'statements_removed': self.statements_removed,
            'statements_modified': self.statements_modified,
            'constants_folded': self.constants_folded,
            'subexpressions_eliminated': self.subexpressions_eliminated,
            'functions_inlined': self.functions_inlined,
            'loops_optimized': self.loops_optimized,
            'dead_stores_removed': self.dead_stores_removed,
            'strength_reductions': self.strength_reductions,
            'copies_propagated': self.copies_propagated,
            'time_ms': self.time_ms
        }


class OptimizationPass(ABC):
    """Abstract base class for optimization passes.
    
    Each pass must implement:
    - name: Return the pass name
    - analyze(): Analyze IR and return analysis results
    - transform(): Transform IR based on analysis
    - validate(): Validate transformation preserved semantics
    """
    
    def __init__(self):
        self.stats = PassStats(pass_name=self.name)
        self._analysis_cache: Dict[str, Any] = {}
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this pass."""
        pass
    
    @property
    def kind(self) -> PassKind:
        """Return the kind of this pass."""
        return PassKind.TRANSFORM
    
    @property
    def min_level(self) -> OptimizationLevel:
        """Minimum optimization level to run this pass."""
        return OptimizationLevel.O1
    
    def analyze(self, ir_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze IR and return analysis results.
        
        Args:
            ir_data: The IR data to analyze
            
        Returns:
            Dictionary of analysis results
        """
        return {}
    
    @abstractmethod
    def transform(self, ir_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform IR based on optimization.
        
        Args:
            ir_data: The IR data to transform
            
        Returns:
            Transformed IR data
        """
        pass
    
    def validate(self, original_ir: Dict[str, Any], 
                 transformed_ir: Dict[str, Any]) -> bool:
        """Validate transformation preserved semantics.
        
        Args:
            original_ir: Original IR before transformation
            transformed_ir: IR after transformation
            
        Returns:
            True if transformation is valid
        """
        # Basic structural validation
        if 'ir_functions' in original_ir:
            if 'ir_functions' not in transformed_ir:
                return False
            # All original functions should still exist
            orig_funcs = {f.get('name') for f in original_ir.get('ir_functions', [])}
            trans_funcs = {f.get('name') for f in transformed_ir.get('ir_functions', [])}
            if not trans_funcs >= orig_funcs - {'_inline_', '_cse_'}:  # Allow removed inline-only functions
                return False
        return True
    
    def run(self, ir_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the complete pass: analyze, transform, validate.
        
        Args:
            ir_data: The IR data to optimize
            
        Returns:
            Optimized IR data
        """
        start_time = time.time()
        
        # Analysis phase
        self._analysis_cache = self.analyze(ir_data)
        
        # Transform phase
        result = self.transform(copy.deepcopy(ir_data))
        
        # Validation phase
        if not self.validate(ir_data, result):
            logger.warning(f"Pass {self.name} failed validation, reverting")
            result = ir_data
        
        self.stats.time_ms = (time.time() - start_time) * 1000
        return result
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = PassStats(pass_name=self.name)
        self._analysis_cache.clear()


class PassManager:
    """Manages and runs optimization passes in sequence."""
    
    def __init__(self, level: OptimizationLevel = OptimizationLevel.O2):
        self.level = level
        self.passes: List[OptimizationPass] = []
        self.all_stats: List[PassStats] = []
        self.passes_run: List[str] = []
    
    def register_pass(self, pass_: OptimizationPass) -> None:
        """Register an optimization pass."""
        self.passes.append(pass_)
    
    def register_passes(self, passes: List[OptimizationPass]) -> None:
        """Register multiple optimization passes."""
        self.passes.extend(passes)
    
    def clear_passes(self) -> None:
        """Clear all registered passes."""
        self.passes.clear()
    
    def optimize(self, ir_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run all applicable optimization passes.
        
        Args:
            ir_data: The IR data to optimize
            
        Returns:
            Tuple of (optimized IR, optimization stats)
        """
        self.all_stats.clear()
        self.passes_run.clear()
        
        # O0 = no optimization
        if self.level == OptimizationLevel.O0:
            return ir_data, {'level': 'O0', 'passes': [], 'total_changes': 0}
        
        result = ir_data
        total_changes = 0
        
        for pass_ in self.passes:
            if pass_.min_level.value <= self.level.value:
                pass_.reset_stats()
                try:
                    result = pass_.run(result)
                    self.all_stats.append(pass_.stats)
                    self.passes_run.append(pass_.name)
                    total_changes += pass_.stats.ir_changes
                    logger.debug(f"Pass {pass_.name}: {pass_.stats.ir_changes} changes")
                except Exception as e:
                    logger.error(f"Pass {pass_.name} failed: {e}")
        
        stats = {
            'level': self.level.name,
            'passes': self.passes_run,
            'total_changes': total_changes,
            'pass_stats': [s.to_dict() for s in self.all_stats]
        }
        
        return result, stats
    
    def get_stats_summary(self) -> str:
        """Get summary of optimization statistics."""
        lines = [f"Optimization Level: {self.level.name}"]
        lines.append("-" * 40)
        
        for stat in self.all_stats:
            lines.append(str(stat))
        
        total_changes = sum(s.ir_changes for s in self.all_stats)
        total_time = sum(s.time_ms for s in self.all_stats)
        
        lines.append("-" * 40)
        lines.append(f"Total IR changes: {total_changes}")
        lines.append(f"Total time: {total_time:.2f}ms")
        
        return '\n'.join(lines)


def create_pass_manager(level: str = 'O2') -> PassManager:
    """Create a pass manager with default passes for the given level.
    
    Args:
        level: Optimization level string (O0, O1, O2, O3)
        
    Returns:
        Configured PassManager
    """
    from .o1_passes import get_o1_passes
    from .o2_passes import get_o2_passes
    from .o3_passes import get_o3_passes
    
    level_map = {
        'O0': OptimizationLevel.O0,
        'O1': OptimizationLevel.O1,
        'O2': OptimizationLevel.O2,
        'O3': OptimizationLevel.O3,
        'Os': OptimizationLevel.Os,
        'Oz': OptimizationLevel.Oz,
    }
    opt_level = level_map.get(level.upper(), OptimizationLevel.O2)
    
    pm = PassManager(level=opt_level)
    
    # Register passes based on level
    if opt_level.value >= OptimizationLevel.O1.value:
        pm.register_passes(get_o1_passes())
    
    if opt_level.value >= OptimizationLevel.O2.value:
        pm.register_passes(get_o2_passes())
    
    if opt_level.value >= OptimizationLevel.O3.value:
        pm.register_passes(get_o3_passes())
    
    return pm


__all__ = [
    'OptimizationLevel', 'PassKind', 'PassStats',
    'OptimizationPass', 'PassManager', 'create_pass_manager'
]
