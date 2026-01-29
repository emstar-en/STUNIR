#!/usr/bin/env python3
"""STUNIR Prolog Targets Package.

Provides code generators for Prolog-family logic programming languages.

Supported targets:
- SWI-Prolog (swi_prolog): Full-featured, widely-used implementation
- GNU Prolog (gnu_prolog): Constraint Logic Programming with CLP(FD), CLP(B)
- SICStus Prolog (sicstus): Future

Part of Phase 5C: Logic Programming Targets.
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
]
