"""STUNIR Optimization Module.

Provides a multi-pass optimization pipeline for STUNIR IR.
"""

from .optimizer import (
    OptimizationLevel, PassKind, OptimizationStats,
    OptimizationPass, DeadCodeEliminationPass, ConstantFoldingPass,
    CommonSubexpressionEliminationPass, FunctionInliningPass,
    LoopOptimizationPass, Optimizer, create_optimizer
)

__all__ = [
    'OptimizationLevel', 'PassKind', 'OptimizationStats',
    'OptimizationPass', 'DeadCodeEliminationPass', 'ConstantFoldingPass',
    'CommonSubexpressionEliminationPass', 'FunctionInliningPass',
    'LoopOptimizationPass', 'Optimizer', 'create_optimizer'
]
