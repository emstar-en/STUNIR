#!/usr/bin/env python3
"""STUNIR Targets Package.

Provides code generators for various target languages and platforms.

Available target families:
- lisp: Lisp family languages (Common Lisp, Scheme, Clojure, Racket, etc.)
- prolog: Logic programming languages (SWI-Prolog, etc.)
- polyglot: General-purpose languages (C, C++, Rust, Go, Python, etc.)
- assembly: Low-level assembly targets (x86, ARM)
- wasm: WebAssembly targets
- gpu: GPU compute targets (CUDA, OpenCL)
"""

# Import available target families
try:
    from . import lisp
except ImportError:
    lisp = None

try:
    from . import prolog
except ImportError:
    prolog = None

try:
    from . import polyglot
except ImportError:
    polyglot = None

try:
    from . import assembly
except ImportError:
    assembly = None

try:
    from . import grammar
except ImportError:
    grammar = None

__all__ = [
    'lisp',
    'prolog',
    'polyglot',
    'assembly',
    'grammar',
]
