#!/usr/bin/env python3
"""STUNIR Enhancement Pipeline Module.

This module provides the EnhancementPipeline class that orchestrates the
execution of all enhancement analyses (control flow, type system, semantic,
memory, optimization) and produces an EnhancementContext for emitters.

The pipeline supports:
- Configurable enhancement execution (enable/disable specific enhancements)
- Graceful degradation (continues if some enhancements fail)
- Comprehensive error handling and logging
- Performance tracking

Part of Phase 1 (Foundation) of the STUNIR Enhancement Integration.

Example:
    >>> from tools.integration import EnhancementPipeline, EnhancementContext
    >>> pipeline = EnhancementPipeline(target_language='rust')
    >>> context = pipeline.run_all_enhancements(ir_data)
    >>> print(context.get_status_summary())
"""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .enhancement_context import (
    EnhancementContext,
    EnhancementStatus,
    ControlFlowData,
    TypeSystemData,
    SemanticData,
    MemoryData,
    OptimizationData,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the enhancement pipeline.
    
    Attributes:
        enable_control_flow: Enable control flow analysis.
        enable_type_analysis: Enable type system analysis.
        enable_semantic_analysis: Enable semantic analysis.
        enable_memory_analysis: Enable memory analysis.
        enable_optimization: Enable optimization.
        optimization_level: Level of optimization (O0, O1, O2, O3).
        fail_fast: Stop on first enhancement failure.
        collect_diagnostics: Collect detailed diagnostics.
        timeout_seconds: Timeout for each enhancement (0 = no timeout).
    """
    enable_control_flow: bool = True
    enable_type_analysis: bool = True
    enable_semantic_analysis: bool = True
    enable_memory_analysis: bool = True
    enable_optimization: bool = True
    optimization_level: str = "O2"
    fail_fast: bool = False
    collect_diagnostics: bool = True
    timeout_seconds: float = 0.0  # 0 = no timeout


@dataclass
class PipelineStats:
    """Statistics from pipeline execution.
    
    Attributes:
        total_time_ms: Total execution time in milliseconds.
        control_flow_time_ms: Control flow analysis time.
        type_analysis_time_ms: Type analysis time.
        semantic_analysis_time_ms: Semantic analysis time.
        memory_analysis_time_ms: Memory analysis time.
        optimization_time_ms: Optimization time.
        enhancements_run: Number of enhancements run.
        enhancements_succeeded: Number of successful enhancements.
        enhancements_failed: Number of failed enhancements.
    """
    total_time_ms: float = 0.0
    control_flow_time_ms: float = 0.0
    type_analysis_time_ms: float = 0.0
    semantic_analysis_time_ms: float = 0.0
    memory_analysis_time_ms: float = 0.0
    optimization_time_ms: float = 0.0
    enhancements_run: int = 0
    enhancements_succeeded: int = 0
    enhancements_failed: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_time_ms': self.total_time_ms,
            'control_flow_time_ms': self.control_flow_time_ms,
            'type_analysis_time_ms': self.type_analysis_time_ms,
            'semantic_analysis_time_ms': self.semantic_analysis_time_ms,
            'memory_analysis_time_ms': self.memory_analysis_time_ms,
            'optimization_time_ms': self.optimization_time_ms,
            'enhancements_run': self.enhancements_run,
            'enhancements_succeeded': self.enhancements_succeeded,
            'enhancements_failed': self.enhancements_failed
        }


class EnhancementPipeline:
    """Orchestrates execution of all enhancements.
    
    The pipeline runs the following enhancements in order:
    1. Semantic Analysis - Variable/function tracking, symbol table
    2. Control Flow Analysis - CFG construction, loop detection
    3. Type Analysis - Type inference, type checking
    4. Memory Analysis - Memory patterns, safety analysis
    5. Optimization - Dead code elimination, constant folding, etc.
    
    Each enhancement phase has error handling for graceful degradation.
    If one phase fails, the pipeline continues with reduced functionality.
    
    Attributes:
        target_language: Target language for code generation.
        config: Pipeline configuration.
        stats: Execution statistics.
    
    Example:
        >>> pipeline = EnhancementPipeline('rust')
        >>> context = pipeline.run_all_enhancements(ir_data)
        >>> if context.semantic_data.is_available():
        ...     symbol_table = context.get_symbol_table()
    """
    
    def __init__(
        self,
        target_language: str = "python",
        config: Optional[PipelineConfig] = None
    ):
        """Initialize the enhancement pipeline.
        
        Args:
            target_language: Target language for code generation.
            config: Optional pipeline configuration.
        """
        self.target_language = target_language
        self.config = config or PipelineConfig()
        self.stats = PipelineStats()
        
        # Lazy-loaded enhancement modules
        self._semantic_analyzer = None
        self._control_flow_analyzer = None
        self._type_engine = None
        self._memory_analyzer = None
        self._optimizer = None
    
    # -------------------------------------------------------------------------
    # Lazy Loading of Enhancement Modules
    # -------------------------------------------------------------------------
    
    def _get_semantic_analyzer(self) -> Optional[Any]:
        """Lazily load the semantic analyzer."""
        if self._semantic_analyzer is None:
            try:
                from tools.semantic import SemanticAnalyzer
                self._semantic_analyzer = SemanticAnalyzer()
            except ImportError as e:
                logger.warning(f"Failed to import SemanticAnalyzer: {e}")
                return None
        return self._semantic_analyzer
    
    def _get_control_flow_analyzer(self) -> Optional[Any]:
        """Lazily load the control flow analyzer."""
        if self._control_flow_analyzer is None:
            try:
                from tools.ir.control_flow import ControlFlowAnalyzer
                self._control_flow_analyzer = ControlFlowAnalyzer()
            except ImportError as e:
                logger.warning(f"Failed to import ControlFlowAnalyzer: {e}")
                return None
        return self._control_flow_analyzer
    
    def _get_type_engine(self) -> Optional[Any]:
        """Lazily load the type inference engine."""
        if self._type_engine is None:
            try:
                from tools.stunir_types import TypeInferenceEngine, TypeRegistry
                registry = TypeRegistry()
                self._type_engine = TypeInferenceEngine(registry)
            except ImportError as e:
                logger.warning(f"Failed to import TypeInferenceEngine: {e}")
                return None
        return self._type_engine
    
    def _get_memory_analyzer(self) -> Optional[Any]:
        """Lazily load the memory safety analyzer."""
        if self._memory_analyzer is None:
            try:
                from tools.memory import MemorySafetyAnalyzer
                self._memory_analyzer = MemorySafetyAnalyzer()
            except ImportError as e:
                logger.warning(f"Failed to import MemorySafetyAnalyzer: {e}")
                return None
        return self._memory_analyzer
    
    def _get_optimizer(self) -> Optional[Any]:
        """Lazily load the optimizer (pass manager)."""
        if self._optimizer is None:
            try:
                from tools.optimize import create_pass_manager
                self._optimizer = create_pass_manager(self.config.optimization_level)
            except ImportError as e:
                logger.warning(f"Failed to import optimizer: {e}")
                return None
        return self._optimizer
    
    # -------------------------------------------------------------------------
    # Main Pipeline Entry Point
    # -------------------------------------------------------------------------
    
    def run_all_enhancements(
        self,
        ir_data: Dict[str, Any],
        config: Optional[PipelineConfig] = None
    ) -> EnhancementContext:
        """Run all enhancements on IR data.
        
        This is the main entry point for the pipeline. It runs all enabled
        enhancements in order and produces an EnhancementContext.
        
        Args:
            ir_data: The IR data to process.
            config: Optional config override for this run.
            
        Returns:
            EnhancementContext containing all enhancement results.
        """
        start_time = time.time()
        run_config = config or self.config
        
        # Reset stats
        self.stats = PipelineStats()
        
        # Create context
        context = EnhancementContext(
            original_ir=ir_data,
            target_language=self.target_language,
            options={
                'optimization_level': run_config.optimization_level,
                'config': {
                    'enable_control_flow': run_config.enable_control_flow,
                    'enable_type_analysis': run_config.enable_type_analysis,
                    'enable_semantic_analysis': run_config.enable_semantic_analysis,
                    'enable_memory_analysis': run_config.enable_memory_analysis,
                    'enable_optimization': run_config.enable_optimization,
                }
            }
        )
        
        logger.info(f"Starting enhancement pipeline for target={self.target_language}")
        
        # Phase 1: Semantic Analysis (foundation for other phases)
        if run_config.enable_semantic_analysis:
            context.semantic_data = self.run_semantic_analysis(ir_data)
            self.stats.enhancements_run += 1
            if context.semantic_data.status == EnhancementStatus.SUCCESS:
                self.stats.enhancements_succeeded += 1
            elif context.semantic_data.status == EnhancementStatus.FAILED:
                self.stats.enhancements_failed += 1
                if run_config.fail_fast:
                    return self._finalize_context(context, start_time)
        
        # Phase 2: Control Flow Analysis
        if run_config.enable_control_flow:
            context.control_flow_data = self.run_control_flow_analysis(
                ir_data, context.semantic_data.symbol_table
            )
            self.stats.enhancements_run += 1
            if context.control_flow_data.status == EnhancementStatus.SUCCESS:
                self.stats.enhancements_succeeded += 1
            elif context.control_flow_data.status == EnhancementStatus.FAILED:
                self.stats.enhancements_failed += 1
                if run_config.fail_fast:
                    return self._finalize_context(context, start_time)
        
        # Phase 3: Type Analysis
        if run_config.enable_type_analysis:
            context.type_system_data = self.run_type_analysis(
                ir_data, context.semantic_data.symbol_table
            )
            self.stats.enhancements_run += 1
            if context.type_system_data.status == EnhancementStatus.SUCCESS:
                self.stats.enhancements_succeeded += 1
            elif context.type_system_data.status == EnhancementStatus.FAILED:
                self.stats.enhancements_failed += 1
                if run_config.fail_fast:
                    return self._finalize_context(context, start_time)
        
        # Phase 4: Memory Analysis
        if run_config.enable_memory_analysis:
            context.memory_data = self.run_memory_analysis(
                ir_data, context.semantic_data.symbol_table
            )
            self.stats.enhancements_run += 1
            if context.memory_data.status == EnhancementStatus.SUCCESS:
                self.stats.enhancements_succeeded += 1
            elif context.memory_data.status == EnhancementStatus.FAILED:
                self.stats.enhancements_failed += 1
                if run_config.fail_fast:
                    return self._finalize_context(context, start_time)
        
        # Phase 5: Optimization
        if run_config.enable_optimization:
            context.optimization_data = self.run_optimization_analysis(
                ir_data, run_config.optimization_level
            )
            self.stats.enhancements_run += 1
            if context.optimization_data.status == EnhancementStatus.SUCCESS:
                self.stats.enhancements_succeeded += 1
            elif context.optimization_data.status == EnhancementStatus.FAILED:
                self.stats.enhancements_failed += 1
        
        return self._finalize_context(context, start_time)
    
    def _finalize_context(
        self,
        context: EnhancementContext,
        start_time: float
    ) -> EnhancementContext:
        """Finalize the context with metadata and stats."""
        self.stats.total_time_ms = (time.time() - start_time) * 1000
        
        context.metadata['pipeline_stats'] = self.stats.to_dict()
        context.metadata['target_language'] = self.target_language
        
        logger.info(
            f"Enhancement pipeline complete: "
            f"{self.stats.enhancements_succeeded}/{self.stats.enhancements_run} succeeded, "
            f"total_time={self.stats.total_time_ms:.2f}ms"
        )
        
        return context
    
    # -------------------------------------------------------------------------
    # Individual Enhancement Methods
    # -------------------------------------------------------------------------
    
    def run_semantic_analysis(
        self,
        ir_data: Dict[str, Any]
    ) -> SemanticData:
        """Run semantic analysis on IR data.
        
        Performs variable tracking, function analysis, and builds symbol table.
        
        Args:
            ir_data: The IR data to analyze.
            
        Returns:
            SemanticData containing analysis results.
        """
        start_time = time.time()
        result = SemanticData()
        
        try:
            analyzer = self._get_semantic_analyzer()
            if analyzer is None:
                result.status = EnhancementStatus.SKIPPED
                result.error = "Semantic analyzer not available"
                logger.warning("Semantic analysis skipped: analyzer not available")
                return result
            
            # Run analysis
            logger.debug("Running semantic analysis...")
            
            # Build symbol table from IR
            symbol_table = self._build_symbol_table(ir_data, analyzer)
            result.symbol_table = symbol_table
            
            # Build call graph
            result.call_graph = self._build_call_graph(ir_data)
            
            # Collect issues if analyzer supports it
            if hasattr(analyzer, 'issues'):
                result.issues = list(analyzer.issues)
            
            result.status = EnhancementStatus.SUCCESS
            logger.debug(f"Semantic analysis complete: {len(result.call_graph)} functions analyzed")
            
        except Exception as e:
            result.status = EnhancementStatus.FAILED
            result.error = f"Semantic analysis failed: {str(e)}"
            logger.error(f"Semantic analysis failed: {e}")
            if self.config.collect_diagnostics:
                logger.debug(traceback.format_exc())
        
        self.stats.semantic_analysis_time_ms = (time.time() - start_time) * 1000
        return result
    
    def run_control_flow_analysis(
        self,
        ir_data: Dict[str, Any],
        symbol_table: Optional[Any] = None
    ) -> ControlFlowData:
        """Run control flow analysis on IR data.
        
        Builds control flow graphs, detects loops, and analyzes branches.
        
        Args:
            ir_data: The IR data to analyze.
            symbol_table: Optional symbol table from semantic analysis.
            
        Returns:
            ControlFlowData containing analysis results.
        """
        start_time = time.time()
        result = ControlFlowData()
        
        try:
            analyzer = self._get_control_flow_analyzer()
            if analyzer is None:
                result.status = EnhancementStatus.SKIPPED
                result.error = "Control flow analyzer not available"
                logger.warning("Control flow analysis skipped: analyzer not available")
                return result
            
            logger.debug("Running control flow analysis...")
            
            # Analyze each function
            functions = ir_data.get('ir_functions', [])
            for func in functions:
                func_name = func.get('name', 'unnamed')
                try:
                    # Build CFG for function
                    if hasattr(analyzer, 'build_cfg'):
                        cfg = analyzer.build_cfg(func)
                        result.cfgs[func_name] = cfg
                        
                        # Detect loops
                        if hasattr(analyzer, 'detect_loops'):
                            loops = analyzer.detect_loops(cfg)
                            result.loops[func_name] = loops
                        
                        # Analyze branches
                        if hasattr(analyzer, 'analyze_branches'):
                            branches = analyzer.analyze_branches(cfg)
                            result.branches[func_name] = branches
                        
                        # Compute dominators
                        if hasattr(analyzer, 'compute_dominators'):
                            dominators = analyzer.compute_dominators(cfg)
                            result.dominators[func_name] = dominators
                    else:
                        # Fallback: create basic CFG from function body
                        cfg = self._build_basic_cfg(func)
                        result.cfgs[func_name] = cfg
                        
                except Exception as e:
                    logger.warning(f"Failed to analyze function {func_name}: {e}")
                    # Continue with other functions
            
            result.status = EnhancementStatus.SUCCESS if result.cfgs else EnhancementStatus.PARTIAL
            logger.debug(f"Control flow analysis complete: {len(result.cfgs)} CFGs built")
            
        except Exception as e:
            result.status = EnhancementStatus.FAILED
            result.error = f"Control flow analysis failed: {str(e)}"
            logger.error(f"Control flow analysis failed: {e}")
            if self.config.collect_diagnostics:
                logger.debug(traceback.format_exc())
        
        self.stats.control_flow_time_ms = (time.time() - start_time) * 1000
        return result
    
    def run_type_analysis(
        self,
        ir_data: Dict[str, Any],
        symbol_table: Optional[Any] = None
    ) -> TypeSystemData:
        """Run type analysis on IR data.
        
        Performs type inference and type checking.
        
        Args:
            ir_data: The IR data to analyze.
            symbol_table: Optional symbol table from semantic analysis.
            
        Returns:
            TypeSystemData containing analysis results.
        """
        start_time = time.time()
        result = TypeSystemData()
        
        try:
            engine = self._get_type_engine()
            if engine is None:
                result.status = EnhancementStatus.SKIPPED
                result.error = "Type engine not available"
                logger.warning("Type analysis skipped: engine not available")
                return result
            
            logger.debug("Running type analysis...")
            
            # Get type registry
            if hasattr(engine, 'registry'):
                result.type_registry = engine.registry
            
            # Analyze each function
            functions = ir_data.get('ir_functions', [])
            for func in functions:
                func_name = func.get('name', 'unnamed')
                try:
                    # Infer types for function
                    func_types = self._infer_function_types(func, engine)
                    result.inference_results[func_name] = func_types
                    
                    # Merge into global type mappings
                    for expr_id, inferred_type in func_types.items():
                        result.type_mappings[f"{func_name}.{expr_id}"] = inferred_type
                        
                except Exception as e:
                    logger.warning(f"Failed to infer types for function {func_name}: {e}")
            
            result.status = EnhancementStatus.SUCCESS if result.type_mappings else EnhancementStatus.PARTIAL
            logger.debug(f"Type analysis complete: {len(result.type_mappings)} types inferred")
            
        except Exception as e:
            result.status = EnhancementStatus.FAILED
            result.error = f"Type analysis failed: {str(e)}"
            logger.error(f"Type analysis failed: {e}")
            if self.config.collect_diagnostics:
                logger.debug(traceback.format_exc())
        
        self.stats.type_analysis_time_ms = (time.time() - start_time) * 1000
        return result
    
    def run_memory_analysis(
        self,
        ir_data: Dict[str, Any],
        symbol_table: Optional[Any] = None
    ) -> MemoryData:
        """Run memory analysis on IR data.
        
        Analyzes memory patterns and performs safety analysis.
        
        Args:
            ir_data: The IR data to analyze.
            symbol_table: Optional symbol table from semantic analysis.
            
        Returns:
            MemoryData containing analysis results.
        """
        start_time = time.time()
        result = MemoryData()
        
        try:
            analyzer = self._get_memory_analyzer()
            if analyzer is None:
                result.status = EnhancementStatus.SKIPPED
                result.error = "Memory analyzer not available"
                logger.warning("Memory analysis skipped: analyzer not available")
                return result
            
            logger.debug("Running memory analysis...")
            
            # Determine memory strategy for target
            result.memory_strategy = self._get_target_memory_strategy()
            
            # Analyze memory patterns in each function
            functions = ir_data.get('ir_functions', [])
            for func in functions:
                func_name = func.get('name', 'unnamed')
                try:
                    # Analyze function for memory patterns
                    patterns = self._analyze_memory_patterns(func)
                    for var_name, pattern in patterns.items():
                        result.patterns[f"{func_name}.{var_name}"] = pattern
                    
                    # Run safety analysis if available
                    if hasattr(analyzer, 'analyze'):
                        violations = analyzer.analyze(func)
                        if violations:
                            result.safety_violations.extend(violations)
                            
                except Exception as e:
                    logger.warning(f"Failed to analyze memory for function {func_name}: {e}")
            
            result.status = EnhancementStatus.SUCCESS
            logger.debug(f"Memory analysis complete: {len(result.patterns)} patterns, "
                        f"{len(result.safety_violations)} violations")
            
        except Exception as e:
            result.status = EnhancementStatus.FAILED
            result.error = f"Memory analysis failed: {str(e)}"
            logger.error(f"Memory analysis failed: {e}")
            if self.config.collect_diagnostics:
                logger.debug(traceback.format_exc())
        
        self.stats.memory_analysis_time_ms = (time.time() - start_time) * 1000
        return result
    
    def run_optimization_analysis(
        self,
        ir_data: Dict[str, Any],
        level: str = "O2"
    ) -> OptimizationData:
        """Run optimization analysis on IR data.
        
        Applies optimization passes based on the specified level.
        
        Args:
            ir_data: The IR data to optimize.
            level: Optimization level (O0, O1, O2, O3).
            
        Returns:
            OptimizationData containing optimization results.
        """
        start_time = time.time()
        result = OptimizationData()
        result.optimization_level = level
        
        # No optimization for O0
        if level == "O0":
            result.status = EnhancementStatus.SUCCESS
            result.optimized_ir = ir_data
            logger.debug("Optimization skipped: O0 level")
            return result
        
        try:
            # Use the new pass manager
            from tools.optimize import create_pass_manager
            pass_manager = create_pass_manager(level)
            
            logger.debug(f"Running optimization at level {level}...")
            
            # Run optimization passes
            optimized_ir, stats = pass_manager.optimize(ir_data)
            result.optimized_ir = optimized_ir
            result.stats = stats
            
            # Collect passes applied
            result.passes_applied = pass_manager.passes_run
            
            # Generate optimization hints
            result.optimization_hints = self._generate_optimization_hints(
                ir_data, result.optimized_ir
            )
            
            result.status = EnhancementStatus.SUCCESS
            logger.debug(f"Optimization complete: {len(result.passes_applied)} passes applied")
            
        except Exception as e:
            result.status = EnhancementStatus.FAILED
            result.error = f"Optimization failed: {str(e)}"
            result.optimized_ir = ir_data  # Fallback to original
            logger.error(f"Optimization failed: {e}")
            if self.config.collect_diagnostics:
                logger.debug(traceback.format_exc())
        
        self.stats.optimization_time_ms = (time.time() - start_time) * 1000
        return result
    
    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------
    
    def _build_symbol_table(
        self,
        ir_data: Dict[str, Any],
        analyzer: Any
    ) -> Any:
        """Build symbol table from IR data."""
        try:
            from tools.semantic import SymbolTable, VariableInfo, FunctionInfo
        except ImportError:
            # Create a minimal symbol table
            class MinimalSymbolTable:
                def __init__(self):
                    self.variables = {}
                    self.functions = {}
                
                def lookup_variable(self, name):
                    return self.variables.get(name)
                
                def lookup_function(self, name):
                    return self.functions.get(name)
            
            SymbolTable = MinimalSymbolTable
            VariableInfo = dict
            FunctionInfo = dict
        
        symbol_table = SymbolTable() if callable(SymbolTable) else SymbolTable
        
        # Add functions
        for func in ir_data.get('ir_functions', []):
            func_name = func.get('name', 'unnamed')
            params = func.get('params', [])
            returns = func.get('returns', 'void')
            
            if hasattr(symbol_table, 'define_function'):
                func_info = FunctionInfo(
                    name=func_name,
                    return_type=returns
                ) if callable(FunctionInfo) else {'name': func_name, 'return_type': returns}
                symbol_table.define_function(func_info)
            else:
                symbol_table.functions[func_name] = {'name': func_name, 'params': params, 'returns': returns}
        
        return symbol_table
    
    def _build_call_graph(self, ir_data: Dict[str, Any]) -> Dict[str, Set[str]]:
        """Build call graph from IR data."""
        call_graph = {}
        
        for func in ir_data.get('ir_functions', []):
            func_name = func.get('name', 'unnamed')
            callees = set()
            
            # Scan function body for calls
            body = func.get('body', [])
            self._find_calls(body, callees)
            
            call_graph[func_name] = callees
        
        return call_graph
    
    def _find_calls(self, statements: List[Any], callees: Set[str]) -> None:
        """Recursively find function calls in statements."""
        for stmt in statements:
            if isinstance(stmt, dict):
                stmt_type = stmt.get('type', stmt.get('kind', ''))
                
                if stmt_type == 'call':
                    callee = stmt.get('function', stmt.get('callee', ''))
                    if callee:
                        callees.add(callee)
                
                # Recurse into nested structures
                for key in ['body', 'then', 'else', 'statements', 'block']:
                    if key in stmt and isinstance(stmt[key], list):
                        self._find_calls(stmt[key], callees)
    
    def _build_basic_cfg(self, func: Dict[str, Any]) -> Dict[str, Any]:
        """Build a basic CFG from function body (fallback)."""
        body = func.get('body', [])
        
        return {
            'function': func.get('name', 'unnamed'),
            'entry': 0,
            'exit': len(body),
            'blocks': [
                {
                    'id': i,
                    'statements': [stmt],
                    'successors': [i + 1] if i < len(body) - 1 else []
                }
                for i, stmt in enumerate(body)
            ] if body else [{'id': 0, 'statements': [], 'successors': []}]
        }
    
    def _infer_function_types(
        self,
        func: Dict[str, Any],
        engine: Any
    ) -> Dict[str, Any]:
        """Infer types for a function's expressions."""
        types = {}
        
        # Add parameter types
        for i, param in enumerate(func.get('params', [])):
            param_name = param if isinstance(param, str) else param.get('name', f'param{i}')
            param_type = param.get('type', 'any') if isinstance(param, dict) else 'any'
            types[f'param.{param_name}'] = param_type
        
        # Add return type
        types['return'] = func.get('returns', 'void')
        
        # Scan body for variable declarations
        body = func.get('body', [])
        for i, stmt in enumerate(body):
            if isinstance(stmt, dict):
                stmt_type = stmt.get('type', stmt.get('kind', ''))
                
                if stmt_type == 'var_decl':
                    var_name = stmt.get('name', f'var{i}')
                    var_type = stmt.get('var_type', 'any')
                    types[f'var.{var_name}'] = var_type
        
        return types
    
    def _analyze_memory_patterns(self, func: Dict[str, Any]) -> Dict[str, str]:
        """Analyze memory patterns in a function."""
        patterns = {}
        
        body = func.get('body', [])
        for stmt in body:
            if isinstance(stmt, dict):
                stmt_type = stmt.get('type', stmt.get('kind', ''))
                
                if stmt_type == 'var_decl':
                    var_name = stmt.get('name', '')
                    # Simple heuristic: arrays on heap, primitives on stack
                    var_type = stmt.get('var_type', '')
                    if 'array' in str(var_type).lower() or 'vec' in str(var_type).lower():
                        patterns[var_name] = 'heap'
                    else:
                        patterns[var_name] = 'stack'
                
                elif stmt_type in ('malloc', 'new', 'alloc'):
                    var_name = stmt.get('target', stmt.get('name', ''))
                    patterns[var_name] = 'heap'
        
        return patterns
    
    def _get_target_memory_strategy(self) -> str:
        """Get memory strategy for target language."""
        strategies = {
            'python': 'gc',
            'java': 'gc',
            'go': 'gc',
            'rust': 'ownership',
            'c': 'manual',
            'cpp': 'raii',
            'c89': 'manual',
            'c99': 'manual',
            'haskell': 'gc',
            'node': 'gc',
            'wasm': 'linear',
        }
        return strategies.get(self.target_language.lower(), 'gc')
    
    def _generate_optimization_hints(
        self,
        original_ir: Dict[str, Any],
        optimized_ir: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate optimization hints for emitters."""
        hints = {}
        
        original_funcs = {f.get('name'): f for f in original_ir.get('ir_functions', [])}
        optimized_funcs = {f.get('name'): f for f in optimized_ir.get('ir_functions', [])}
        
        # Detect inlined functions
        removed_funcs = set(original_funcs.keys()) - set(optimized_funcs.keys())
        if removed_funcs:
            hints['inlined_functions'] = list(removed_funcs)
        
        # Detect size reduction
        original_size = sum(len(f.get('body', [])) for f in original_funcs.values())
        optimized_size = sum(len(f.get('body', [])) for f in optimized_funcs.values())
        if optimized_size < original_size:
            hints['size_reduction'] = original_size - optimized_size
            hints['reduction_percent'] = (1 - optimized_size / original_size) * 100 if original_size > 0 else 0
        
        return hints


# Convenience function to create a pipeline with default settings
def create_pipeline(
    target_language: str = "python",
    optimization_level: str = "O2"
) -> EnhancementPipeline:
    """Create an EnhancementPipeline with common settings.
    
    Args:
        target_language: Target language for code generation.
        optimization_level: Optimization level (O0, O1, O2, O3).
        
    Returns:
        Configured EnhancementPipeline instance.
    """
    config = PipelineConfig(optimization_level=optimization_level)
    return EnhancementPipeline(target_language, config)
