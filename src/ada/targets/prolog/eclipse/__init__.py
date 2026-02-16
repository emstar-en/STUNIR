#!/usr/bin/env python3
"""ECLiPSe target package.

Provides ECLiPSe code generation from STUNIR Logic IR.
Supports constraint optimization, multiple CLP libraries (IC, FD, R, Q),
advanced search strategies, and branch-and-bound optimization.

Key ECLiPSe features:
- Constraint optimization (minimize/maximize, bb_min/bb_max)
- Multiple CLP libraries (IC, FD with different operator syntax)
- Global constraints (alldifferent, element, cumulative)
- Advanced search strategies (search/6)
- Branch-and-bound optimization

Key differences from other Prolog systems:
- Focus on optimization, not just satisfaction
- Different constraint operators (IC: $=, FD: #=)
- Uses .ecl file extension
- Rich library system with lib(ic), lib(ic_global), etc.

Part of Phase 5D-2: ECLiPSe with Constraint Optimization.
"""

from .emitter import (
    ECLiPSeEmitter,
    ECLiPSeConfig,
    EmitterResult,
    compute_sha256,
    canonical_json
)
from .types import (
    ECLiPSeTypeMapper,
    ECLIPSE_TYPES,
    IC_OPERATORS,
    FD_OPERATORS,
    ECLIPSE_GLOBALS,
    ECLIPSE_OPTIMIZATION,
    ECLIPSE_SEARCH,
    ECLIPSE_SELECT_METHODS,
    ECLIPSE_CHOICE_METHODS,
    ECLIPSE_LIBRARIES,
)

__all__ = [
    # Emitter
    'ECLiPSeEmitter',
    'ECLiPSeConfig',
    'EmitterResult',
    'compute_sha256',
    'canonical_json',
    # Types
    'ECLiPSeTypeMapper',
    'ECLIPSE_TYPES',
    'IC_OPERATORS',
    'FD_OPERATORS',
    'ECLIPSE_GLOBALS',
    'ECLIPSE_OPTIMIZATION',
    'ECLIPSE_SEARCH',
    'ECLIPSE_SELECT_METHODS',
    'ECLIPSE_CHOICE_METHODS',
    'ECLIPSE_LIBRARIES',
]
