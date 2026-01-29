#!/usr/bin/env python3
"""STUNIR Prolog Targets Package.

Provides code generators for Prolog-family logic programming languages.

Supported targets:
- SWI-Prolog (swi_prolog): Full-featured, widely-used implementation
- GNU Prolog (gnu_prolog): Constraint Logic Programming with CLP(FD), CLP(B)
- YAP Prolog (yap): High-performance with tabling (memoization) support
- XSB Prolog (xsb): Advanced tabling with incremental evaluation, WFS
- Datalog (datalog): Declarative subset with stratified negation
- SICStus Prolog (sicstus): Future

Part of Phase 5C-5D: Logic Programming Targets.
"""

from .swi_prolog import SWIPrologEmitter, SWIPrologConfig
from .gnu_prolog import (
    GNUPrologEmitter, 
    GNUPrologConfig,
    GNUPrologTypeMapper,
    GNU_PROLOG_TYPES,
    CLPFD_OPERATORS,
    CLPB_OPERATORS,
    CLPFD_PREDICATES,
)
from .yap import (
    YAPPrologEmitter,
    YAPPrologConfig,
    YAPPrologTypeMapper,
    YAP_PROLOG_TYPES,
    TABLING_MODES,
)
from .xsb import (
    XSBPrologEmitter,
    XSBPrologConfig,
    XSBPrologTypeMapper,
    XSB_PROLOG_TYPES,
    XSB_TABLING_MODES,
    XSB_LATTICE_OPS,
    TablingMode,
)
from .datalog import (
    DatalogEmitter,
    DatalogConfig,
    DatalogTypeMapper,
    DATALOG_TYPES,
    DATALOG_RESERVED,
    ValidationLevel,
    DatalogRestrictionError,
    StratificationError,
    StratificationResult,
)

__all__ = [
    # SWI-Prolog
    'SWIPrologEmitter',
    'SWIPrologConfig',
    # GNU Prolog
    'GNUPrologEmitter',
    'GNUPrologConfig',
    'GNUPrologTypeMapper',
    'GNU_PROLOG_TYPES',
    'CLPFD_OPERATORS',
    'CLPB_OPERATORS',
    'CLPFD_PREDICATES',
    # YAP Prolog
    'YAPPrologEmitter',
    'YAPPrologConfig',
    'YAPPrologTypeMapper',
    'YAP_PROLOG_TYPES',
    'TABLING_MODES',
    # XSB Prolog
    'XSBPrologEmitter',
    'XSBPrologConfig',
    'XSBPrologTypeMapper',
    'XSB_PROLOG_TYPES',
    'XSB_TABLING_MODES',
    'XSB_LATTICE_OPS',
    'TablingMode',
    # Datalog
    'DatalogEmitter',
    'DatalogConfig',
    'DatalogTypeMapper',
    'DATALOG_TYPES',
    'DATALOG_RESERVED',
    'ValidationLevel',
    'DatalogRestrictionError',
    'StratificationError',
    'StratificationResult',
]
