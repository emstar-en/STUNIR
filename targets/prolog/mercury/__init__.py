#!/usr/bin/env python3
"""STUNIR Mercury Target Package.

Provides Mercury code generation from STUNIR Logic IR with full
type, mode, and determinism declarations.

Mercury is a pure declarative logic/functional programming language
designed for reliable software with compile-time type checking.

Key features supported:
- Strong static typing with inference
- Mode declarations (in, out, in_out, di, uo)
- Determinism declarations (det, semidet, multi, nondet, etc.)
- Module system with interface/implementation sections
- Functions (in addition to predicates)
- Type declarations (enum, algebraic, record)

Part of Phase 5D-3: Advanced Logic Programming Targets.

Example usage:
    from targets.prolog.mercury import MercuryEmitter, MercuryConfig
    
    config = MercuryConfig(
        module_prefix="my",
        emit_comments=True,
        infer_determinism=True
    )
    emitter = MercuryEmitter(config)
    result = emitter.emit(logic_ir)
    print(result.code)
"""

from .emitter import (
    MercuryEmitter,
    MercuryConfig,
    EmitterResult,
    compute_sha256,
    canonical_json,
)

from .types import (
    MercuryTypeMapper,
    MercuryMode,
    Determinism,
    Purity,
    MERCURY_TYPES,
    MODE_MAPPING,
    MERCURY_IMPORTS,
    MERCURY_RESERVED,
)


__all__ = [
    # Emitter
    'MercuryEmitter',
    'MercuryConfig',
    'EmitterResult',
    'compute_sha256',
    'canonical_json',
    # Types
    'MercuryTypeMapper',
    'MercuryMode',
    'Determinism',
    'Purity',
    'MERCURY_TYPES',
    'MODE_MAPPING',
    'MERCURY_IMPORTS',
    'MERCURY_RESERVED',
]
