"""STUNIR Scientific Language Emitters.

This package provides code emitters for scientific and legacy
programming languages.

Supported languages:
- Fortran (2003/2008/2018) with parallel constructs
- Pascal (standard Pascal, Object Pascal/Delphi/FPC)

Usage:
    from targets.scientific import FortranEmitter, PascalEmitter
    
    # Fortran
    fortran = FortranEmitter()
    result = fortran.emit(ir_dict)
    print(result.code)
    
    # Pascal
    pascal = PascalEmitter()
    result = pascal.emit(ir_dict)
    print(result.code)
"""

from targets.scientific.fortran_emitter import FortranEmitter, EmitterResult
from targets.scientific.pascal_emitter import PascalEmitter

__all__ = [
    'FortranEmitter',
    'PascalEmitter',
    'EmitterResult',
]
