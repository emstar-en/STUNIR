"""STUNIR Constraints Emitter Package"""

from .emitter import ConstraintsEmitter

# Aliases for backward compatibility
MiniZincEmitter = ConstraintsEmitter
CHREmitter = ConstraintsEmitter

__all__ = ["ConstraintsEmitter", "MiniZincEmitter", "CHREmitter"]
