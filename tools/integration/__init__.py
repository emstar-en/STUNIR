"""STUNIR Enhancement Integration Module.

This module provides the integration layer between STUNIR enhancement analyses
and target language emitters. It packages enhancement results into a unified
context that emitters can consume for intelligent code generation.

Phase 1 (Foundation) of the STUNIR Enhancement-to-Emitter Integration.

Components:
    EnhancementContext: Unified container for all enhancement data
    EnhancementPipeline: Orchestrates enhancement execution
    PipelineConfig: Configuration for the pipeline
    PipelineStats: Statistics from pipeline execution

Data Containers:
    ControlFlowData: CFG, loops, branches data
    TypeSystemData: Type mappings and inference results
    SemanticData: Symbol tables and call graphs
    MemoryData: Memory patterns and safety analysis
    OptimizationData: Optimization hints and results

Status:
    EnhancementStatus: Status of each enhancement analysis

Usage:
    Basic usage with defaults:
    
    >>> from tools.integration import EnhancementPipeline, EnhancementContext
    >>> pipeline = EnhancementPipeline(target_language='rust')
    >>> context = pipeline.run_all_enhancements(ir_data)
    >>> 
    >>> # Check status
    >>> print(context.get_status_summary())
    >>> 
    >>> # Access enhancement data safely
    >>> cfg = context.get_function_cfg('main')
    >>> var_info = context.lookup_variable('x')
    >>> memory_strategy = context.get_memory_strategy()
    
    Using the convenience function:
    
    >>> from tools.integration import create_pipeline
    >>> pipeline = create_pipeline('python', 'O2')
    >>> context = pipeline.run_all_enhancements(ir_data)
    
    Custom configuration:
    
    >>> from tools.integration import EnhancementPipeline, PipelineConfig
    >>> config = PipelineConfig(
    ...     enable_optimization=True,
    ...     optimization_level='O3',
    ...     enable_memory_analysis=False
    ... )
    >>> pipeline = EnhancementPipeline('c', config)
    >>> context = pipeline.run_all_enhancements(ir_data, config)
    
    Serialization:
    
    >>> # Save context
    >>> json_str = context.to_json()
    >>> 
    >>> # Load context
    >>> restored = EnhancementContext.from_json(json_str)

Architecture:
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                   ENHANCEMENT PIPELINE                           │
    └─────────────────────────────────────────────────────────────────┘
    
      Input IR ──┬──► SemanticAnalyzer ──► SymbolTable
                 │         │
                 │         ▼
                 ├──► ControlFlowAnalyzer ──► CFG (per function)
                 │         │
                 │         ▼
                 ├──► TypeInferenceEngine ──► TypeContext
                 │         │
                 │         ▼
                 ├──► MemoryAnalyzer ──► MemoryContext
                 │         │
                 │         ▼
                 └──► Optimizer (O0-O3) ──► OptimizedIR
                           │
                           ▼
                  EnhancementContext (packaged)
                           │
                           ▼
                      TargetEmitter
                           │
                           ▼
                   Generated Code (with bodies)

See Also:
    - docs/integration/ENHANCEMENT_INTEGRATION.md for detailed documentation
    - tools/ir/control_flow.py for control flow analysis
    - tools/stunir_types/ for type system
    - tools/semantic/ for semantic analysis
    - tools/memory/ for memory analysis
    - tools/optimize/ for optimization
"""

from .enhancement_context import (
    # Main context class
    EnhancementContext,
    
    # Status enum
    EnhancementStatus,
    
    # Data containers
    ControlFlowData,
    TypeSystemData,
    SemanticData,
    MemoryData,
    OptimizationData,
    
    # Convenience function
    create_minimal_context,
)

from .enhancement_pipeline import (
    # Main pipeline class
    EnhancementPipeline,
    
    # Configuration
    PipelineConfig,
    PipelineStats,
    
    # Convenience function
    create_pipeline,
)

__all__ = [
    # Context
    'EnhancementContext',
    'EnhancementStatus',
    
    # Data containers
    'ControlFlowData',
    'TypeSystemData',
    'SemanticData',
    'MemoryData',
    'OptimizationData',
    
    # Pipeline
    'EnhancementPipeline',
    'PipelineConfig',
    'PipelineStats',
    
    # Convenience functions
    'create_minimal_context',
    'create_pipeline',
]

__version__ = '1.0.0'
__author__ = 'STUNIR Team'
