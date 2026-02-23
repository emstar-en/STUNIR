"""STUNIR Semantic IR Emitters - Python Reference Implementation

This package provides Python reference implementations for all 24 STUNIR semantic IR
emitters, based on the DO-178C Level A compliant Ada SPARK implementations.

All emitters produce identical outputs to their SPARK counterparts (confluence).
"""

__version__ = "1.0.0"
__author__ = "STUNIR Team"

# Import all emitters for convenient access
from .base_emitter import BaseEmitter, EmitterConfig, EmitterResult, EmitterStatus
from .visitor import IRVisitor
from .codegen import CodeGenerator
from .types import (
    IRDataType,
    IRStatementType,
    Architecture,
    Endianness,
    ArchConfig,
    IRStatement,
    IRParameter,
    IRFunction,
    IRTypeField,
    IRType,
    IRModule,
    GeneratedFile,
)

__all__ = [
    "BaseEmitter",
    "EmitterConfig",
    "EmitterResult",
    "EmitterStatus",
    "IRVisitor",
    "CodeGenerator",
    "IRDataType",
    "IRStatementType",
    "Architecture",
    "Endianness",
    "ArchConfig",
    "IRStatement",
    "IRParameter",
    "IRFunction",
    "IRTypeField",
    "IRType",
    "IRModule",
    "GeneratedFile",
]
