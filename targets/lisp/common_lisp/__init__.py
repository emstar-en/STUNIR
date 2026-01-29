#!/usr/bin/env python3
"""STUNIR Common Lisp Emitter Package.

Provides code emission for ANSI Common Lisp.
Part of Phase 5A: Core Lisp Implementation.
"""

from .emitter import CommonLispEmitter, CommonLispConfig
from .types import COMMON_LISP_TYPES, CommonLispTypeMapper

__all__ = [
    'CommonLispEmitter',
    'CommonLispConfig',
    'COMMON_LISP_TYPES',
    'CommonLispTypeMapper',
]
