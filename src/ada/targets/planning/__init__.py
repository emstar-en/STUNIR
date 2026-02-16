"""STUNIR Planning Emitter Package"""

from .emitter import PlanningEmitter

# Alias for backward compatibility
PDDLEmitter = PlanningEmitter

__all__ = ["PlanningEmitter", "PDDLEmitter"]
