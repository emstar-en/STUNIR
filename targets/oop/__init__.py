#!/usr/bin/env python3
"""STUNIR OOP Emitters - Smalltalk and ALGOL code generation.

This package provides code emitters for object-oriented and historical
programming languages:

- Smalltalk: Pure OOP with message passing, blocks, and classes
- ALGOL: Block-structured language with call-by-name semantics

Usage:
    from targets.oop import SmalltalkEmitter, ALGOLEmitter
    
    # Generate Smalltalk code
    st_emitter = SmalltalkEmitter()
    result = st_emitter.emit(ir)
    
    # Generate ALGOL code
    algol_emitter = ALGOLEmitter()
    result = algol_emitter.emit(ir)
"""

from targets.oop.smalltalk_emitter import (
    SmalltalkEmitter,
    EmitterResult as SmalltalkEmitterResult,
    SmalltalkEmitterError,
    UnsupportedFeatureError as SmalltalkUnsupportedFeatureError,
)

from targets.oop.algol_emitter import (
    ALGOLEmitter,
    EmitterResult as ALGOLEmitterResult,
    ALGOLEmitterError,
    UnsupportedFeatureError as ALGOLUnsupportedFeatureError,
)

__all__ = [
    # Smalltalk
    'SmalltalkEmitter',
    'SmalltalkEmitterResult',
    'SmalltalkEmitterError',
    'SmalltalkUnsupportedFeatureError',
    
    # ALGOL
    'ALGOLEmitter',
    'ALGOLEmitterResult',
    'ALGOLEmitterError',
    'ALGOLUnsupportedFeatureError',
]
