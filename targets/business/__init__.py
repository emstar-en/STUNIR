#!/usr/bin/env python3
"""STUNIR Business Language Emitters Package.

Provides code emitters for business-oriented programming languages:
- COBOL: Enterprise business data processing
- BASIC: Simple programming with line numbers

Usage:
    from targets.business import COBOLEmitter, BASICEmitter
    
    # Generate COBOL code
    cobol_emitter = COBOLEmitter()
    result = cobol_emitter.emit(ir_dict)
    print(result.code)
    
    # Generate BASIC code
    basic_emitter = BASICEmitter()
    result = basic_emitter.emit(ir_dict)
    print(result.code)
"""

from .cobol_emitter import COBOLEmitter, EmitterResult
from .basic_emitter import BASICEmitter

__all__ = [
    'COBOLEmitter',
    'BASICEmitter',
    'EmitterResult',
]
