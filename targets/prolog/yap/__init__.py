#!/usr/bin/env python3
"""STUNIR YAP Prolog Emitter Package.

Provides code generation for YAP Prolog with tabling support.
YAP (Yet Another Prolog) is a high-performance Prolog system known for
its powerful tabling (memoization) capabilities.

Key features:
- Module system (similar to SWI-Prolog)
- Tabling (memoization) support with :- table pred/N
- Indexing directives for performance
- Thread support
- Attributed variables

Part of Phase 5C-3: YAP with Tabling Support.
"""

from .emitter import YAPPrologEmitter, YAPPrologConfig, EmitterResult
from .types import (
    YAPPrologTypeMapper, 
    YAP_PROLOG_TYPES, 
    TABLING_MODES,
    MODE_DECLARATIONS
)

__all__ = [
    'YAPPrologEmitter',
    'YAPPrologConfig',
    'EmitterResult',
    'YAPPrologTypeMapper',
    'YAP_PROLOG_TYPES',
    'TABLING_MODES',
    'MODE_DECLARATIONS',
]
