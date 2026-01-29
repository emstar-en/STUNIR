"""Expert System Emitters for STUNIR.

This package provides emitters for expert system languages:
- CLIPS: C Language Integrated Production System
- Jess: Java Expert System Shell

Phase 7A: Expert Systems Foundation
"""

from .base import (
    BaseExpertSystemEmitter,
    canonical_json,
    compute_sha256,
)

from .clips_emitter import CLIPSEmitter
from .jess_emitter import JessEmitter

__all__ = [
    # Base
    'BaseExpertSystemEmitter',
    'canonical_json',
    'compute_sha256',
    
    # Emitters
    'CLIPSEmitter',
    'JessEmitter',
]
