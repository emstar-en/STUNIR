"""STUNIR Guile Target.

Emitter for GNU Guile (Scheme) code generation.
Part of Phase 5B: Extended Lisp Implementation.
"""

from .emitter import GuileEmitter, GuileConfig
from .types import GuileTypeMapper, GUILE_TYPES

__all__ = [
    'GuileEmitter',
    'GuileConfig',
    'GuileTypeMapper',
    'GUILE_TYPES',
]
