#!/usr/bin/env python3
"""STUNIR Clojure Emitter Package.

Provides code emission for Clojure (JVM).
Part of Phase 5A: Core Lisp Implementation.
"""

from .emitter import ClojureEmitter, ClojureEmitterConfig
from .types import CLOJURE_TYPES, CLOJURE_TYPE_HINTS, ClojureTypeMapper

__all__ = [
    'ClojureEmitter',
    'ClojureEmitterConfig',
    'CLOJURE_TYPES',
    'CLOJURE_TYPE_HINTS',
    'ClojureTypeMapper',
]
