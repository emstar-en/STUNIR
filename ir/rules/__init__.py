"""Rule-based IR for Expert Systems.

This package provides a rule-based intermediate representation
for expert systems with CLIPS and Jess emitter support.

Phase 7A: Expert Systems Foundation
"""

from .rule_ir import (
    PatternType,
    ConditionType,
    ActionType,
    ConflictResolutionStrategy,
    EmitterResult,
    FunctionDef,
)

from .pattern import (
    PatternElement,
    LiteralPattern,
    VariablePattern,
    WildcardPattern,
    MultifieldPattern,
    AnyPatternElement,
    PatternMatcher,
)

from .fact import (
    Fact,
    FactTemplate,
)

from .rule import (
    Condition,
    PatternCondition,
    TestCondition,
    CompositeCondition,
    Action,
    AssertAction,
    RetractAction,
    ModifyAction,
    BindAction,
    CallAction,
    PrintoutAction,
    HaltAction,
    Rule,
    RuleBase,
)

from .working_memory import WorkingMemory

from .forward_chaining import (
    Activation,
    Agenda,
    ForwardChainingEngine,
)

__all__ = [
    # Enums
    'PatternType',
    'ConditionType',
    'ActionType',
    'ConflictResolutionStrategy',
    
    # Core classes
    'EmitterResult',
    'FunctionDef',
    
    # Pattern classes
    'PatternElement',
    'LiteralPattern',
    'VariablePattern',
    'WildcardPattern',
    'MultifieldPattern',
    'AnyPatternElement',
    'PatternMatcher',
    
    # Fact classes
    'Fact',
    'FactTemplate',
    
    # Condition classes
    'Condition',
    'PatternCondition',
    'TestCondition',
    'CompositeCondition',
    
    # Action classes
    'Action',
    'AssertAction',
    'RetractAction',
    'ModifyAction',
    'BindAction',
    'CallAction',
    'PrintoutAction',
    'HaltAction',
    
    # Rule classes
    'Rule',
    'RuleBase',
    
    # Working Memory
    'WorkingMemory',
    
    # Forward Chaining
    'Activation',
    'Agenda',
    'ForwardChainingEngine',
]
