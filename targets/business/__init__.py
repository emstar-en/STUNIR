"""STUNIR Business Emitter Package"""

from .emitter import BusinessEmitter

# Aliases for backward compatibility
COBOLEmitter = BusinessEmitter
BASICEmitter = BusinessEmitter

class EmitterResult:
    """Result from emitter."""
    def __init__(self, code="", manifest=None):
        self.code = code
        self.manifest = manifest or {}

__all__ = ["BusinessEmitter", "COBOLEmitter", "BASICEmitter", "EmitterResult"]
