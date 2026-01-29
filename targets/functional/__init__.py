#!/usr/bin/env python3
"""STUNIR Functional Language Emitters Package.

This package provides code emitters for functional programming languages,
including Haskell, OCaml, and F#.

Supported Languages:
    - Haskell: Pure functional, lazy evaluation, type classes, monads
    - OCaml: Multi-paradigm, strict evaluation, modules, functors
    - F#: .NET functional, computation expressions, units of measure

Usage:
    from targets.functional import HaskellEmitter, OCamlEmitter, FSharpEmitter
    from ir.functional import Module, FunctionDef
    
    # Emit Haskell code
    haskell_emitter = HaskellEmitter()
    haskell_code = haskell_emitter.emit_module(module)
    
    # Emit OCaml code
    ocaml_emitter = OCamlEmitter()
    ocaml_code = ocaml_emitter.emit_module(module)
    
    # Emit F# code
    fsharp_emitter = FSharpEmitter()
    fsharp_code = fsharp_emitter.emit_module(module)
"""

from targets.functional.base import (
    FunctionalEmitterBase,
    EmitterResult,
    canonical_json,
    compute_sha256,
)

from targets.functional.haskell_emitter import HaskellEmitter
from targets.functional.ocaml_emitter import OCamlEmitter
from targets.functional.fsharp_emitter import FSharpEmitter

__all__ = [
    # Base
    'FunctionalEmitterBase',
    'EmitterResult',
    'canonical_json',
    'compute_sha256',
    # Emitters
    'HaskellEmitter',
    'OCamlEmitter',
    'FSharpEmitter',
]
