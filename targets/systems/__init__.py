#!/usr/bin/env python3
"""STUNIR Systems Languages Targets - Package exports.

This package provides code emitters for systems programming languages
including Ada (with SPARK formal verification) and D.

Usage:
    from targets.systems import AdaEmitter, DEmitter
    from ir.systems import Package
    
    # Ada emission
    ada_emitter = AdaEmitter()
    spec, body = ada_emitter.emit_package(package)
    
    # D emission
    d_emitter = DEmitter()
    code = d_emitter.emit_module(package)
"""

from targets.systems.ada_emitter import AdaEmitter, EmitterResult as AdaEmitterResult
from targets.systems.d_emitter import DEmitter, EmitterResult as DEmitterResult

__all__ = [
    'AdaEmitter',
    'DEmitter',
    'AdaEmitterResult',
    'DEmitterResult',
]
