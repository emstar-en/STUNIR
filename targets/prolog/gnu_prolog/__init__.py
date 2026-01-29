#!/usr/bin/env python3
"""GNU Prolog target package.

Provides GNU Prolog code generation from STUNIR Logic IR.
Supports file-based organization (no modules), CLP(FD), CLP(B), CLP(R).

Key differences from SWI-Prolog:
- No module system (uses :- public/1 instead)
- Built-in CLP(FD), CLP(B), CLP(R) support
- Different built-in predicate names (fd_domain, fd_labeling, etc.)

Part of Phase 5C-2: GNU Prolog with CLP support.
"""

from .emitter import (
    GNUPrologEmitter,
    GNUPrologConfig,
    EmitterResult,
    compute_sha256,
    canonical_json
)
from .types import (
    GNUPrologTypeMapper,
    GNU_PROLOG_TYPES,
    CLPFD_OPERATORS,
    CLPB_OPERATORS,
    CLPFD_PREDICATES
)

__all__ = [
    # Emitter
    'GNUPrologEmitter',
    'GNUPrologConfig',
    'EmitterResult',
    'compute_sha256',
    'canonical_json',
    # Types
    'GNUPrologTypeMapper',
    'GNU_PROLOG_TYPES',
    'CLPFD_OPERATORS',
    'CLPB_OPERATORS',
    'CLPFD_PREDICATES',
]
