#!/usr/bin/env python3
"""STUNIR Prolog Targets Package.

Provides code generators for Prolog-family languages.

Supported targets:
- SWI-Prolog (swi_prolog): Full-featured, widely-used implementation
- GNU Prolog (gnu_prolog): Future
- SICStus Prolog (sicstus): Future
"""

from .swi_prolog import SWIPrologEmitter, SWIPrologConfig

__all__ = [
    'SWIPrologEmitter',
    'SWIPrologConfig',
]
