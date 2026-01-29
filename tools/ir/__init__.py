"""STUNIR IR Analysis Module.

Provides control flow analysis, IR manipulation, and CFG construction.
"""

from .control_flow import (
    BasicBlock, BlockType, ControlFlowType,
    LoopInfo, BranchInfo,
    ControlFlowGraph, ControlFlowAnalyzer, ControlFlowTranslator
)

__all__ = [
    'BasicBlock', 'BlockType', 'ControlFlowType',
    'LoopInfo', 'BranchInfo',
    'ControlFlowGraph', 'ControlFlowAnalyzer', 'ControlFlowTranslator'
]
