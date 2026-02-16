"""STUNIR Asp Emitter Package"""

from .emitter import AspEmitter

# Aliases for backward compatibility
ClingoEmitter = AspEmitter
DLVEmitter = AspEmitter

# Placeholder config classes
class ClingoConfig:
    """Configuration for Clingo emitter."""
    pass

class DLVConfig:
    """Configuration for DLV emitter."""
    pass

class EmitterResult:
    """Result from emitter."""
    def __init__(self, code="", manifest=None):
        self.code = code
        self.manifest = manifest or {}

# Placeholder helper functions
def emit_clingo(program):
    """Emit Clingo code from program."""
    emitter = ClingoEmitter()
    return emitter.emit(program)

def emit_dlv(program):
    """Emit DLV code from program."""
    emitter = DLVEmitter()
    return emitter.emit(program)

def emit_clingo_to_file(program, path):
    """Emit Clingo code to file."""
    result = emit_clingo(program)
    with open(path, 'w') as f:
        f.write(result.code)
    return result

def emit_dlv_to_file(program, path):
    """Emit DLV code to file."""
    result = emit_dlv(program)
    with open(path, 'w') as f:
        f.write(result.code)
    return result

__all__ = [
    "AspEmitter", "ClingoEmitter", "DLVEmitter",
    "ClingoConfig", "DLVConfig", "EmitterResult",
    "emit_clingo", "emit_dlv", "emit_clingo_to_file", "emit_dlv_to_file"
]
