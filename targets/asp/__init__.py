"""ASP Emitters package.

This package provides emitters for Answer Set Programming systems,
including Clingo and DLV.

Part of Phase 7D: Answer Set Programming

Example usage:
    from ir.asp import ASPProgram, atom, pos, neg, var
    from targets.asp import ClingoEmitter, DLVEmitter, emit_clingo, emit_dlv
    
    # Create a program
    program = ASPProgram("graph_coloring")
    program.add_fact(atom("node", "a"))
    program.add_fact(atom("node", "b"))
    program.add_fact(atom("edge", "a", "b"))
    
    # Emit to Clingo
    result = emit_clingo(program)
    print(result.code)
    
    # Emit to DLV
    result = emit_dlv(program)
    print(result.code)
"""

from .clingo_emitter import (
    ClingoEmitter,
    ClingoConfig,
    EmitterResult,
    emit_clingo,
    emit_clingo_to_file,
)

from .dlv_emitter import (
    DLVEmitter,
    DLVConfig,
    emit_dlv,
    emit_dlv_to_file,
)

__all__ = [
    # Clingo
    "ClingoEmitter",
    "ClingoConfig",
    "emit_clingo",
    "emit_clingo_to_file",
    # DLV
    "DLVEmitter",
    "DLVConfig",
    "emit_dlv",
    "emit_dlv_to_file",
    # Common
    "EmitterResult",
]
