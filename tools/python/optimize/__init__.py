"""STUNIR Optimization Module.

Provides a multi-pass optimization pipeline for STUNIR IR with
configurable optimization levels (O0, O1, O2, O3).

Optimization Levels:
- O0: No optimization (direct translation)
- O1: Basic optimizations (dead code elimination, constant folding)
- O2: Standard optimizations (O1 + CSE, loop invariant motion)
- O3: Aggressive optimizations (O2 + function inlining, loop unrolling)

Usage:
    from tools.optimize import create_pass_manager
    
    pm = create_pass_manager('O2')
    optimized_ir, stats = pm.optimize(ir_data)
"""

from .optimization_pass import (
    OptimizationLevel,
    PassKind,
    PassStats,
    OptimizationPass,
    PassManager,
    create_pass_manager,
)

from .o1_passes import (
    DeadCodeEliminationPass,
    ConstantFoldingPass,
    ConstantPropagationPass,
    AlgebraicSimplificationPass,
    get_o1_passes,
)

from .o2_passes import (
    CommonSubexpressionEliminationPass,
    LoopInvariantCodeMotionPass,
    CopyPropagationPass,
    StrengthReductionPass,
    get_o2_passes,
)

from .o3_passes import (
    FunctionInliningPass,
    LoopUnrollingPass,
    AggressiveConstantPropagationPass,
    DeadStoreEliminationPass,
    get_o3_passes,
)

from .validation import (
    OptimizationValidator,
    SemanticComparator,
    validate_optimization,
    compare_optimization,
)

# Legacy compatibility with old optimizer module
from .optimizer import (
    OptimizationStats,
    Optimizer,
    create_optimizer,
)

__all__ = [
    # Core framework
    'OptimizationLevel',
    'PassKind', 
    'PassStats',
    'OptimizationPass',
    'PassManager',
    'create_pass_manager',
    
    # O1 passes
    'DeadCodeEliminationPass',
    'ConstantFoldingPass',
    'ConstantPropagationPass',
    'AlgebraicSimplificationPass',
    'get_o1_passes',
    
    # O2 passes
    'CommonSubexpressionEliminationPass',
    'LoopInvariantCodeMotionPass',
    'CopyPropagationPass',
    'StrengthReductionPass',
    'get_o2_passes',
    
    # O3 passes
    'FunctionInliningPass',
    'LoopUnrollingPass',
    'AggressiveConstantPropagationPass',
    'DeadStoreEliminationPass',
    'get_o3_passes',
    
    # Validation
    'OptimizationValidator',
    'SemanticComparator',
    'validate_optimization',
    'compare_optimization',
    
    # Legacy
    'OptimizationStats',
    'Optimizer',
    'create_optimizer',
]
