#!/usr/bin/env python3
"""STUNIR Functional Language Emitters Package.

This package provides code emitters for functional programming languages,
including Haskell and OCaml.

Supported Languages:
    - Haskell: Pure functional, lazy evaluation, type classes, monads
    - OCaml: Multi-paradigm, strict evaluation, modules, functors

Usage:
    from targets.functional import HaskellEmitter, OCamlEmitter
    from ir.functional import Module, FunctionDef
    
    # Emit Haskell code
    haskell_emitter = HaskellEmitter()
    haskell_code = haskell_emitter.emit_module(module)
    
    # Emit OCaml code
    ocaml_emitter = OCamlEmitter()
    ocaml_code = ocaml_emitter.emit_module(module)
"""

from targets.functional.base import (
    FunctionalEmitterBase,
    EmitterResult,
    canonical_json,
    compute_sha256,
)

from targets.functional.haskell_emitter import HaskellEmitter
from targets.functional.ocaml_emitter import OCamlEmitter

__all__ = [
    # Base
    'FunctionalEmitterBase',
    'EmitterResult',
    'canonical_json',
    'compute_sha256',
    # Emitters
    'HaskellEmitter',
    'OCamlEmitter',
]
