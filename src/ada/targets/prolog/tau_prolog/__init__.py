#!/usr/bin/env python3
"""STUNIR Tau Prolog Emitter Package.

Generates Tau Prolog code from STUNIR Logic IR.
Tau Prolog is a JavaScript-based Prolog implementation that
runs in browsers and Node.js with DOM and JS interop support.

Part of Phase 5D-4: Extended Prolog Targets.
"""

from .emitter import (
    TauPrologEmitter,
    TauPrologConfig,
    TauPrologEmitterResult,
)
from .types import (
    TauPrologTypeMapper,
    TAU_PROLOG_TYPES,
    JS_TO_PROLOG_TYPES,
    TAU_LIBRARIES,
    DOM_PREDICATES,
    JS_PREDICATES,
    LISTS_PREDICATES,
)

__all__ = [
    # Emitter
    'TauPrologEmitter',
    'TauPrologConfig',
    'TauPrologEmitterResult',
    # Types
    'TauPrologTypeMapper',
    'TAU_PROLOG_TYPES',
    'JS_TO_PROLOG_TYPES',
    'TAU_LIBRARIES',
    'DOM_PREDICATES',
    'JS_PREDICATES',
    'LISTS_PREDICATES',
]
