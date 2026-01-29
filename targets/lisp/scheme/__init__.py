#!/usr/bin/env python3
"""STUNIR Scheme Emitter Package.

Provides code emission for R7RS Scheme.
Part of Phase 5A: Core Lisp Implementation.
"""

from .emitter import SchemeEmitter, SchemeEmitterConfig
from .types import SCHEME_TYPES, SchemeTypeMapper

__all__ = [
    'SchemeEmitter',
    'SchemeEmitterConfig',
    'SCHEME_TYPES',
    'SchemeTypeMapper',
]
