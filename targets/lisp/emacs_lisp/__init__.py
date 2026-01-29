"""STUNIR Emacs Lisp Target.

Emitter for GNU Emacs Lisp code generation.
Part of Phase 5B: Extended Lisp Implementation.
"""

from .emitter import EmacsLispEmitter, EmacsLispConfig
from .types import EmacsLispTypeMapper, EMACS_LISP_TYPES

__all__ = [
    'EmacsLispEmitter',
    'EmacsLispConfig',
    'EmacsLispTypeMapper',
    'EMACS_LISP_TYPES',
]
