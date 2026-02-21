"""STUNIR Beam Emitter Package"""

from .emitter import BeamEmitter

# Aliases for backward compatibility
ErlangEmitter = BeamEmitter
ElixirEmitter = BeamEmitter

__all__ = ["BeamEmitter", "ErlangEmitter", "ElixirEmitter"]
