"""Constraint Programming emitters for STUNIR.

This package provides emitters for constraint programming languages:
- MiniZinc: High-level constraint modeling language
- CHR: Constraint Handling Rules for Prolog

Example:
    from ir.constraints import ConstraintModel, Domain, VariableType, Objective
    from targets.constraints import MiniZincEmitter
    
    model = ConstraintModel("example")
    model.add_int_variable("x", 1, 10)
    model.add_int_variable("y", 1, 10)
    model.set_objective(Objective.satisfy())
    
    emitter = MiniZincEmitter()
    result = emitter.emit(model)
    print(result.code)
"""

from .base import BaseConstraintEmitter, canonical_json, compute_sha256
from .minizinc_emitter import MiniZincEmitter
from .chr_emitter import (
    CHREmitter, 
    emit_simplification_rule, 
    emit_propagation_rule,
    emit_simpagation_rule
)

__all__ = [
    # Base
    'BaseConstraintEmitter',
    'canonical_json',
    'compute_sha256',
    # Emitters
    'MiniZincEmitter',
    'CHREmitter',
    # CHR helpers
    'emit_simplification_rule',
    'emit_propagation_rule',
    'emit_simpagation_rule',
]
