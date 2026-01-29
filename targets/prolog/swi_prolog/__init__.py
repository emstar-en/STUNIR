#!/usr/bin/env python3
"""STUNIR SWI-Prolog Emitter Package.

Generates SWI-Prolog code from STUNIR Logic IR.
"""

from .emitter import SWIPrologEmitter, SWIPrologConfig, EmitterResult
from .types import SWIPrologTypeMapper, SWI_PROLOG_TYPES

__all__ = [
    'SWIPrologEmitter',
    'SWIPrologConfig',
    'EmitterResult',
    'SWIPrologTypeMapper',
    'SWI_PROLOG_TYPES',
]
