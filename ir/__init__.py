#!/usr/bin/env python3
"""STUNIR Intermediate Representation Package.

Provides intermediate representation modules for various language families.
"""

# Import available IR modules
try:
    from . import grammar
except ImportError:
    grammar = None

__all__ = [
    'grammar',
]
