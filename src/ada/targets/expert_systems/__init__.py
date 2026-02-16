"""STUNIR Expert_Systems Emitter Package"""

from .emitter import ExpertSystemsEmitter

# Aliases for backward compatibility
CLIPSEmitter = ExpertSystemsEmitter
JessEmitter = ExpertSystemsEmitter

__all__ = ["ExpertSystemsEmitter", "CLIPSEmitter", "JessEmitter"]
