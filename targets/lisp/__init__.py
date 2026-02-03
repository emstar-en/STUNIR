#!/usr/bin/env python3
"""STUNIR Lisp Targets Package.

Provides code emitters for Lisp-family languages:

Phase 5A: Core Lisp
- Common Lisp (ANSI CL)
- Scheme (R7RS)
- Clojure (JVM)
- Racket

Phase 5B: Extended Lisp
- Emacs Lisp (GNU Emacs)
- Guile (GNU Scheme)
- Janet (Modern Lisp)
- Hy (Python Lisp)
"""

from .base import LispEmitterBase, LispEmitterConfig, EmitterResult

# Phase 5A: Core Lisp
from .common_lisp import CommonLispEmitter, CommonLispConfig
from .scheme import SchemeEmitter, SchemeEmitterConfig
from .clojure import ClojureEmitter, ClojureEmitterConfig
from .racket import RacketEmitter, RacketEmitterConfig

# Phase 5B: Extended Lisp
from .emacs_lisp import EmacsLispEmitter, EmacsLispConfig
from .guile import GuileEmitter, GuileConfig
from .janet import JanetEmitter, JanetConfig
from .hy import HyEmitter, HyConfig

__all__ = [
    # Base
    'LispEmitterBase',
    'LispEmitterConfig',
    'EmitterResult',
    # Phase 5A: Core Lisp
    'CommonLispEmitter',
    'CommonLispConfig',
    'SchemeEmitter',
    'SchemeEmitterConfig',
    'ClojureEmitter',
    'ClojureEmitterConfig',
    'RacketEmitter',
    'RacketEmitterConfig',
    # Phase 5B: Extended Lisp
    'EmacsLispEmitter',
    'EmacsLispConfig',
    'GuileEmitter',
    'GuileConfig',
    'JanetEmitter',
    'JanetConfig',
    'HyEmitter',
    'HyConfig',
]
