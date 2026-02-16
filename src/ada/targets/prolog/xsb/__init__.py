#!/usr/bin/env python3
"""STUNIR XSB Prolog Emitter Package.

Provides code generation for XSB Prolog with advanced tabling support.
XSB is renowned for its sophisticated tabling features including:

- Incremental tabling (`:- table pred/N as incremental`)
- Answer subsumption (`:- table pred/N as subsumptive`)
- Well-founded semantics (WFS) for sound negation handling
- Lattice tabling for aggregation operations
- Different module syntax from SWI/YAP

Part of Phase 5D-1: XSB with Advanced Tabling.
"""

from .emitter import XSBPrologEmitter, XSBPrologConfig, EmitterResult, TablingSpec
from .types import (
    XSBPrologTypeMapper,
    XSB_PROLOG_TYPES,
    XSB_TABLING_MODES,
    XSB_LATTICE_OPS,
    XSB_BUILTINS,
    TablingMode,
)

__all__ = [
    # Emitter
    'XSBPrologEmitter',
    'XSBPrologConfig',
    'EmitterResult',
    'TablingSpec',
    # Type mapper
    'XSBPrologTypeMapper',
    # Constants
    'XSB_PROLOG_TYPES',
    'XSB_TABLING_MODES',
    'XSB_LATTICE_OPS',
    'XSB_BUILTINS',
    'TablingMode',
]
