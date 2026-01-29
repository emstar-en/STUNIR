#!/usr/bin/env python3
"""STUNIR Enhancement Integration Tests.

Tests for Phase 1 (Foundation) of the STUNIR Enhancement-to-Emitter Integration.

Tests:
    - EnhancementContext creation and access
    - EnhancementPipeline execution
    - Error handling and graceful degradation
    - Emitter integration with enhancement context
    - Serialization/deserialization
    - Sample IR processing
"""

import json
import os
import sys
import unittest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.integration import (
    EnhancementContext,
    EnhancementStatus,
    ControlFlowData,
    TypeSystemData,
    SemanticData,
    MemoryData,
    OptimizationData,
    EnhancementPipeline,
    PipelineConfig,
    PipelineStats,
    create_pipeline,
    create_minimal_context,
)


# Sample IR data for testing
SAMPLE_IR = {
    'schema': 'stunir.ir.v1',
    'ir_module': 'test_module',
    'ir_epoch': 1706500000,
    'ir_functions': [
        {
            'name': 'add',
            'params': [
                {'name': 'a', 'type': 'i32'},
                {'name': 'b', 'type': 'i32'}
            ],
            'returns': 'i32',
            'body': [
                {'type': 'var_decl', 'name': 'result', 'var_type': 'i32'},
                {'type': 'assign', 'target': 'result', 'value': {'type': 'binary', 'op': '+', 'left': 'a', 'right': 'b'}},
                {'type': 'return', 'value': 'result'}
            ]
        },
        {
            'name': 'main',
            'params': [],
            'returns': 'i32',
            'body': [
                {'type': 'var_decl', 'name': 'x', 'var_type': 'i32', 'init': 10},
                {'type': 'var_decl', 'name': 'y', 'var_type': 'i32', 'init': 20},
                {'type': 'call', 'function': 'add', 'args': ['x', 'y'], 'result': 'z'},
                {'type': 'return', 'value': 'z'}
            ]
        }
    ],
    'ir_imports': [],
    'ir_exports': ['main']
}

SAMPLE_IR_WITH_LOOP = {
    'schema': 'stunir.ir.v1',
    'ir_module': 'loop_module',
    'ir_epoch': 1706500000,
    'ir_functions': [
        {
            'name': 'sum_to_n',
            'params': [{'name': 'n', 'type': 'i32'}],
            'returns': 'i32',
            'body': [
                {'type': 'var_decl', 'name': 'sum', 'var_type': 'i32', 'init': 0},
                {'type': 'var_decl', 'name': 'i', 'var_type': 'i32', 'init': 0},
                {
                    'type': 'while',
                    'condition': {'type': 'binary', 'op': '<', 'left': 'i', 'right': 'n'},
                    'body': [
                        {'type': 'assign', 'target': 'sum', 'value': {'type': 'binary', 'op': '+', 'left': 'sum', 'right': 'i'}},
                        {'type': 'assign', 'target': 'i', 'value': {'type': 'binary', 'op': '+', 'left': 'i', 'right': 1}}
                    ]
                },
                {'type': 'return', 'value': 'sum'}
            ]
        }
    ],
    'ir_imports': [],
    'ir_exports': ['sum_to_n']
}


class TestEnhancementContext(unittest.TestCase):
    """Tests for EnhancementContext class."""
    
    def test_create_minimal_context(self):
        """Test creating a minimal context."""
        context = create_minimal_context(SAMPLE_IR, 'python')
        
        self.assertEqual(context.original_ir, SAMPLE_IR)
        self.assertEqual(context.target_language, 'python')
        self.assertEqual(context.control_flow_data.status, EnhancementStatus.NOT_RUN)
    
    def test_context_initialization(self):
        """Test context initialization with defaults."""
        context = EnhancementContext(original_ir=SAMPLE_IR)
        
        self.assertIsNotNone(context.created_at)
        self.assertEqual(context.target_language, 'python')
        self.assertFalse(context.is_complete())
    
    def test_context_get_functions(self):
        """Test getting functions from context."""
        context = EnhancementContext(original_ir=SAMPLE_IR)
        
        functions = context.get_functions()
        self.assertEqual(len(functions), 2)
        
        add_func = context.get_function('add')
        self.assertIsNotNone(add_func)
        self.assertEqual(add_func['name'], 'add')
        
        missing_func = context.get_function('nonexistent')
        self.assertIsNone(missing_func)
    
    def test_context_get_ir(self):
        """Test getting IR from context."""
        context = EnhancementContext(original_ir=SAMPLE_IR)
        
        # Without optimization, should return original
        ir = context.get_ir()
        self.assertEqual(ir, SAMPLE_IR)
    
    def test_context_validation(self):
        """Test context validation."""
        context = EnhancementContext(original_ir=SAMPLE_IR)
        is_valid, errors = context.validate()
        
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_context_validation_empty_ir(self):
        """Test validation with empty IR."""
        context = EnhancementContext(original_ir={})
        is_valid, errors = context.validate()
        
        self.assertFalse(is_valid)
        # Should have error about empty or missing ir_functions
        self.assertTrue(len(errors) > 0)
    
    def test_context_status_summary(self):
        """Test getting status summary."""
        context = EnhancementContext(original_ir=SAMPLE_IR)
        
        summary = context.get_status_summary()
        self.assertIn('control_flow', summary)
        self.assertIn('type_system', summary)
        self.assertIn('semantic', summary)
        self.assertIn('memory', summary)
        self.assertIn('optimization', summary)


class TestEnhancementContextSerialization(unittest.TestCase):
    """Tests for EnhancementContext serialization."""
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        context = EnhancementContext(
            original_ir=SAMPLE_IR,
            target_language='rust'
        )
        context.control_flow_data.status = EnhancementStatus.SUCCESS
        
        data = context.to_dict()
        
        self.assertIn('original_ir', data)
        self.assertIn('control_flow_data', data)
        self.assertIn('target_language', data)
        self.assertEqual(data['target_language'], 'rust')
        self.assertEqual(data['control_flow_data']['status'], 'SUCCESS')
    
    def test_to_json(self):
        """Test serialization to JSON."""
        context = EnhancementContext(original_ir=SAMPLE_IR)
        
        json_str = context.to_json()
        
        self.assertIsInstance(json_str, str)
        data = json.loads(json_str)
        self.assertIn('original_ir', data)
    
    def test_from_dict(self):
        """Test deserialization from dictionary."""
        context = EnhancementContext(
            original_ir=SAMPLE_IR,
            target_language='go'
        )
        context.semantic_data.status = EnhancementStatus.PARTIAL
        
        data = context.to_dict()
        restored = EnhancementContext.from_dict(data)
        
        self.assertEqual(restored.target_language, 'go')
        self.assertEqual(restored.semantic_data.status, EnhancementStatus.PARTIAL)
    
    def test_from_json(self):
        """Test deserialization from JSON."""
        context = EnhancementContext(original_ir=SAMPLE_IR)
        json_str = context.to_json()
        
        restored = EnhancementContext.from_json(json_str)
        
        self.assertEqual(restored.original_ir, context.original_ir)


class TestControlFlowData(unittest.TestCase):
    """Tests for ControlFlowData container."""
    
    def test_default_status(self):
        """Test default status is NOT_RUN."""
        data = ControlFlowData()
        self.assertEqual(data.status, EnhancementStatus.NOT_RUN)
        self.assertFalse(data.is_available())
    
    def test_success_status(self):
        """Test SUCCESS status makes data available."""
        data = ControlFlowData()
        data.status = EnhancementStatus.SUCCESS
        self.assertTrue(data.is_available())
    
    def test_get_cfg(self):
        """Test getting CFG for function."""
        data = ControlFlowData()
        data.cfgs['main'] = {'entry': 0, 'blocks': []}
        
        cfg = data.get_cfg('main')
        self.assertIsNotNone(cfg)
        
        missing = data.get_cfg('nonexistent')
        self.assertIsNone(missing)
    
    def test_get_loops(self):
        """Test getting loops for function."""
        data = ControlFlowData()
        data.loops['main'] = [{'header_id': 1, 'depth': 1}]
        
        loops = data.get_loops('main')
        self.assertEqual(len(loops), 1)
        
        missing = data.get_loops('nonexistent')
        self.assertEqual(len(missing), 0)


class TestTypeSystemData(unittest.TestCase):
    """Tests for TypeSystemData container."""
    
    def test_get_type(self):
        """Test getting type for expression."""
        data = TypeSystemData()
        data.type_mappings['main.x'] = 'i32'
        data.status = EnhancementStatus.SUCCESS
        
        t = data.get_type('main.x')
        self.assertEqual(t, 'i32')
    
    def test_to_dict(self):
        """Test serialization."""
        data = TypeSystemData()
        data.type_mappings['x'] = 'i32'
        data.status = EnhancementStatus.SUCCESS
        
        d = data.to_dict()
        self.assertEqual(d['status'], 'SUCCESS')
        self.assertIn('x', d['type_mappings'])


class TestSemanticData(unittest.TestCase):
    """Tests for SemanticData container."""
    
    def test_call_graph(self):
        """Test call graph accessors."""
        data = SemanticData()
        data.call_graph = {
            'main': {'add', 'multiply'},
            'add': set()
        }
        
        callees = data.get_callees('main')
        self.assertIn('add', callees)
        
        callers = data.get_callers('add')
        self.assertIn('main', callers)


class TestMemoryData(unittest.TestCase):
    """Tests for MemoryData container."""
    
    def test_get_pattern(self):
        """Test getting memory pattern."""
        data = MemoryData()
        data.patterns['arr'] = 'heap'
        data.patterns['x'] = 'stack'
        
        self.assertEqual(data.get_pattern('arr'), 'heap')
        self.assertEqual(data.get_pattern('x'), 'stack')
        self.assertIsNone(data.get_pattern('nonexistent'))
    
    def test_has_violations(self):
        """Test checking for violations."""
        data = MemoryData()
        self.assertFalse(data.has_violations())
        
        data.safety_violations.append({'kind': 'use_after_free'})
        self.assertTrue(data.has_violations())


class TestOptimizationData(unittest.TestCase):
    """Tests for OptimizationData container."""
    
    def test_was_pass_applied(self):
        """Test checking if pass was applied."""
        data = OptimizationData()
        data.passes_applied = ['dce', 'constant_folding']
        
        self.assertTrue(data.was_pass_applied('dce'))
        self.assertFalse(data.was_pass_applied('inline'))


class TestEnhancementPipeline(unittest.TestCase):
    """Tests for EnhancementPipeline class."""
    
    def test_create_pipeline(self):
        """Test creating pipeline with defaults."""
        pipeline = create_pipeline('python', 'O2')
        
        self.assertEqual(pipeline.target_language, 'python')
        self.assertEqual(pipeline.config.optimization_level, 'O2')
    
    def test_pipeline_with_config(self):
        """Test creating pipeline with custom config."""
        config = PipelineConfig(
            enable_optimization=False,
            enable_memory_analysis=False
        )
        pipeline = EnhancementPipeline('rust', config)
        
        self.assertFalse(pipeline.config.enable_optimization)
        self.assertFalse(pipeline.config.enable_memory_analysis)
    
    def test_run_all_enhancements(self):
        """Test running all enhancements."""
        pipeline = create_pipeline('python', 'O2')
        context = pipeline.run_all_enhancements(SAMPLE_IR)
        
        self.assertIsInstance(context, EnhancementContext)
        self.assertEqual(context.target_language, 'python')
        
        # Check that enhancements were attempted
        summary = context.get_status_summary()
        for enhancement in summary:
            # All should be either SUCCESS, SKIPPED, or FAILED
            self.assertIn(summary[enhancement], 
                         ['SUCCESS', 'PARTIAL', 'SKIPPED', 'FAILED', 'NOT_RUN'])
    
    def test_run_with_disabled_enhancements(self):
        """Test running with some enhancements disabled."""
        config = PipelineConfig(
            enable_control_flow=True,
            enable_type_analysis=False,
            enable_semantic_analysis=True,
            enable_memory_analysis=False,
            enable_optimization=False
        )
        pipeline = EnhancementPipeline('c', config)
        context = pipeline.run_all_enhancements(SAMPLE_IR)
        
        # Type and memory should not have been run
        self.assertEqual(context.type_system_data.status, EnhancementStatus.NOT_RUN)
        self.assertEqual(context.memory_data.status, EnhancementStatus.NOT_RUN)
    
    def test_run_semantic_analysis(self):
        """Test running semantic analysis alone."""
        pipeline = create_pipeline('python')
        result = pipeline.run_semantic_analysis(SAMPLE_IR)
        
        self.assertIsInstance(result, SemanticData)
        # Should have attempted to build symbol table
        if result.status == EnhancementStatus.SUCCESS:
            self.assertIsNotNone(result.symbol_table)
    
    def test_run_control_flow_analysis(self):
        """Test running control flow analysis."""
        pipeline = create_pipeline('python')
        result = pipeline.run_control_flow_analysis(SAMPLE_IR)
        
        self.assertIsInstance(result, ControlFlowData)
    
    def test_run_type_analysis(self):
        """Test running type analysis."""
        pipeline = create_pipeline('rust')
        result = pipeline.run_type_analysis(SAMPLE_IR)
        
        self.assertIsInstance(result, TypeSystemData)
    
    def test_run_memory_analysis(self):
        """Test running memory analysis."""
        pipeline = create_pipeline('c')
        result = pipeline.run_memory_analysis(SAMPLE_IR)
        
        self.assertIsInstance(result, MemoryData)
        self.assertEqual(result.memory_strategy, 'manual')
    
    def test_run_optimization_O0(self):
        """Test running optimization at O0 (no optimization)."""
        pipeline = create_pipeline('python', 'O0')
        result = pipeline.run_optimization_analysis(SAMPLE_IR, 'O0')
        
        self.assertIsInstance(result, OptimizationData)
        self.assertEqual(result.status, EnhancementStatus.SUCCESS)
        self.assertEqual(result.optimized_ir, SAMPLE_IR)


class TestPipelineStats(unittest.TestCase):
    """Tests for PipelineStats."""
    
    def test_stats_after_pipeline(self):
        """Test that stats are collected after pipeline run."""
        pipeline = create_pipeline('python')
        pipeline.run_all_enhancements(SAMPLE_IR)
        
        stats = pipeline.stats
        self.assertGreater(stats.total_time_ms, 0)
        self.assertGreaterEqual(stats.enhancements_run, 0)
    
    def test_stats_to_dict(self):
        """Test stats serialization."""
        stats = PipelineStats()
        stats.total_time_ms = 100.5
        stats.enhancements_run = 5
        stats.enhancements_succeeded = 4
        
        d = stats.to_dict()
        self.assertEqual(d['total_time_ms'], 100.5)
        self.assertEqual(d['enhancements_run'], 5)


class TestGracefulDegradation(unittest.TestCase):
    """Tests for graceful degradation when enhancements fail."""
    
    def test_continues_on_failure(self):
        """Test that pipeline continues even when some enhancements fail."""
        # Run with config that doesn't fail fast
        config = PipelineConfig(fail_fast=False)
        pipeline = EnhancementPipeline('python', config)
        
        # Should complete even if some enhancements fail
        context = pipeline.run_all_enhancements(SAMPLE_IR)
        
        self.assertIsInstance(context, EnhancementContext)
        # Should still have original IR accessible
        self.assertEqual(context.original_ir, SAMPLE_IR)
    
    def test_fallback_to_original_ir(self):
        """Test that original IR is used when optimization fails."""
        config = PipelineConfig(enable_optimization=True)
        pipeline = EnhancementPipeline('python', config)
        context = pipeline.run_all_enhancements(SAMPLE_IR)
        
        # Even if optimization fails, should have IR available
        ir = context.get_ir()
        self.assertIsNotNone(ir)
        self.assertIn('ir_functions', ir)


class TestEmitterIntegration(unittest.TestCase):
    """Tests for emitter integration with enhancement context."""
    
    def test_emitter_base_import(self):
        """Test that BaseEmitter can be imported."""
        from tools.emitters.base_emitter import BaseEmitter, EmitterConfig
        
        self.assertTrue(hasattr(BaseEmitter, 'emit'))
        self.assertTrue(hasattr(BaseEmitter, 'get_function_cfg'))
    
    def test_emitter_config(self):
        """Test EmitterConfig defaults."""
        from tools.emitters.base_emitter import EmitterConfig
        
        config = EmitterConfig()
        self.assertTrue(config.use_enhancements)
        self.assertTrue(config.emit_comments)
        self.assertEqual(config.indent_size, 4)
    
    def test_emitter_with_context(self):
        """Test creating emitter with enhancement context."""
        from tools.emitters.base_emitter import BaseEmitter
        
        # Create context
        pipeline = create_pipeline('python')
        context = pipeline.run_all_enhancements(SAMPLE_IR)
        
        # Create a simple test emitter
        class TestEmitter(BaseEmitter):
            TARGET = 'test'
            
            def emit(self):
                return self.generate_manifest()
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            emitter = TestEmitter(SAMPLE_IR, tmpdir, context)
            
            self.assertTrue(emitter.has_enhancement_context())
            
            # Test accessors
            funcs = emitter.get_functions()
            self.assertEqual(len(funcs), 2)
    
    def test_emitter_without_context(self):
        """Test emitter works without context (backward compatible)."""
        from tools.emitters.base_emitter import BaseEmitter
        
        class TestEmitter(BaseEmitter):
            TARGET = 'test'
            
            def emit(self):
                return self.generate_manifest()
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            emitter = TestEmitter(SAMPLE_IR, tmpdir)  # No context
            
            self.assertFalse(emitter.has_enhancement_context())
            
            # Accessors should return safe defaults
            cfg = emitter.get_function_cfg('main')
            self.assertIsNone(cfg)
            
            loops = emitter.get_loops('main')
            self.assertEqual(loops, [])
            
            # Should still have access to IR
            funcs = emitter.get_functions()
            self.assertEqual(len(funcs), 2)


class TestWithLoopIR(unittest.TestCase):
    """Tests with IR containing loops."""
    
    def test_loop_ir_processing(self):
        """Test processing IR with loops."""
        pipeline = create_pipeline('python')
        context = pipeline.run_all_enhancements(SAMPLE_IR_WITH_LOOP)
        
        self.assertIsInstance(context, EnhancementContext)
        
        func = context.get_function('sum_to_n')
        self.assertIsNotNone(func)
        
        # Check body has while loop
        body = func.get('body', [])
        has_while = any(s.get('type') == 'while' for s in body if isinstance(s, dict))
        self.assertTrue(has_while)


if __name__ == '__main__':
    unittest.main(verbosity=2)
