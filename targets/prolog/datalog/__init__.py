#!/usr/bin/env python3
"""STUNIR Datalog Target Package.

Provides code generator for Datalog - a declarative query language
that is a syntactic subset of Prolog with guaranteed termination.

Key Datalog characteristics:
- Bottom-up evaluation (vs Prolog's top-down)
- No function symbols in rule heads
- Stratified negation only
- Set semantics (no duplicate results)
- No cut or side effects

Part of Phase 5C-4: Datalog Emitter.
"""

from .emitter import (
    DatalogEmitter,
    DatalogConfig,
    EmitterResult,
    StratificationResult,
    ValidationResult,
    ValidationLevel,
    DatalogRestrictionError,
    StratificationError,
)
from .types import (
    DatalogTypeMapper,
    DATALOG_TYPES,
    DATALOG_RESERVED,
    escape_atom,
    escape_string,
    format_variable,
)

__all__ = [
    # Emitter
    'DatalogEmitter',
    'DatalogConfig',
    'EmitterResult',
    'StratificationResult',
    'ValidationResult',
    'ValidationLevel',
    'DatalogRestrictionError',
    'StratificationError',
    # Types
    'DatalogTypeMapper',
    'DATALOG_TYPES',
    'DATALOG_RESERVED',
    'escape_atom',
    'escape_string',
    'format_variable',
]
