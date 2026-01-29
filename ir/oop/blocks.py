#!/usr/bin/env python3
"""STUNIR OOP IR - Block and closure constructs.

This module provides block and closure constructs for Smalltalk-style
programming, supporting closures, non-local returns, and block evaluation.

Usage:
    from ir.oop.blocks import FullBlock, BlockParameter, BlockReturn
    
    # Create a full block with parameters and temporaries
    block = FullBlock(
        parameters=[BlockParameter(name='x'), BlockParameter(name='y')],
        temporaries=[BlockTemporary(name='sum')],
        statements=[...]
    )
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any
from ir.oop.oop_ir import OOPNode


# =============================================================================
# Block Components
# =============================================================================

@dataclass
class BlockParameter(OOPNode):
    """Block parameter declaration.
    
    In Smalltalk: [ :x :y | ... ]
    The :x and :y are block parameters.
    """
    kind: str = 'block_param'
    name: str = ''
    type_hint: Optional[str] = None  # Optional type annotation


@dataclass
class BlockTemporary(OOPNode):
    """Block temporary variable.
    
    In Smalltalk: [ | temp1 temp2 | ... ]
    """
    kind: str = 'block_temp'
    name: str = ''
    initial_value: Optional['OOPNode'] = None


@dataclass
class BlockReturn(OOPNode):
    """Block return (non-local return from method).
    
    In Smalltalk: [ ... ^value ... ]
    The ^ causes return from the enclosing method, not just the block.
    """
    kind: str = 'block_return'
    value: Optional['OOPNode'] = None
    is_non_local: bool = True  # True for ^, False for implicit return


# =============================================================================
# Block Types
# =============================================================================

@dataclass
class FullBlock(OOPNode):
    """Full closure block with parameters, temporaries, and statements.
    
    Syntax: [ :param1 :param2 | | temp1 temp2 | statement1. statement2. ... ]
    
    Full blocks are true closures that capture their lexical environment.
    """
    kind: str = 'full_block'
    parameters: List[BlockParameter] = field(default_factory=list)
    temporaries: List[BlockTemporary] = field(default_factory=list)
    statements: List['OOPNode'] = field(default_factory=list)
    captured_variables: List[str] = field(default_factory=list)  # Closed-over variables
    
    @property
    def arity(self) -> int:
        """Number of parameters."""
        return len(self.parameters)
    
    @property 
    def param_names(self) -> List[str]:
        """Get parameter names."""
        return [p.name for p in self.parameters]
    
    @property
    def temp_names(self) -> List[str]:
        """Get temporary names."""
        return [t.name for t in self.temporaries]


@dataclass
class InlinedBlock(OOPNode):
    """Inlined block for control structures.
    
    Inlined blocks are optimized blocks that don't create a closure object.
    Used for ifTrue:ifFalse:, whileTrue:, etc.
    """
    kind: str = 'inlined_block'
    statements: List['OOPNode'] = field(default_factory=list)
    optimization_hint: str = ''  # 'condition', 'loop_body', etc.


@dataclass
class ConstantBlock(OOPNode):
    """Block that always returns the same constant.
    
    Example: [42] - always returns 42
    These can be heavily optimized.
    """
    kind: str = 'constant_block'
    value: Optional['OOPNode'] = None


# =============================================================================
# Block Evaluation
# =============================================================================

@dataclass
class BlockEvaluation(OOPNode):
    """Block evaluation with arguments.
    
    Variants:
    - value      (no args)
    - value:     (1 arg)
    - value:value:  (2 args)
    - valueWithArguments:  (array of args)
    """
    kind: str = 'block_eval'
    block: Optional['OOPNode'] = None
    arguments: List['OOPNode'] = field(default_factory=list)
    selector: str = 'value'  # value, value:, value:value:, etc.
    
    @classmethod
    def with_args(cls, block: 'OOPNode', *args) -> 'BlockEvaluation':
        """Create block evaluation with given arguments."""
        arg_list = list(args)
        if len(arg_list) == 0:
            selector = 'value'
        elif len(arg_list) == 1:
            selector = 'value:'
        else:
            selector = 'value:' * len(arg_list)
        return cls(block=block, arguments=arg_list, selector=selector)


@dataclass
class BlockOnDo(OOPNode):
    """Exception handling with blocks.
    
    Syntax: [ ... ] on: ExceptionClass do: [ :ex | ... ]
    """
    kind: str = 'block_on_do'
    protected_block: Optional['OOPNode'] = None
    exception_class: str = 'Error'
    handler_block: Optional['OOPNode'] = None


@dataclass
class BlockEnsure(OOPNode):
    """Ensure block execution.
    
    Syntax: [ ... ] ensure: [ cleanup ]
    The ensure block runs regardless of how the protected block exits.
    """
    kind: str = 'block_ensure'
    protected_block: Optional['OOPNode'] = None
    ensure_block: Optional['OOPNode'] = None


@dataclass
class BlockIfCurtailed(OOPNode):
    """If-curtailed block (runs on abnormal exit).
    
    Syntax: [ ... ] ifCurtailed: [ cleanup ]
    The cleanup block only runs if the protected block is terminated abnormally.
    """
    kind: str = 'block_if_curtailed'
    protected_block: Optional['OOPNode'] = None
    curtailment_block: Optional['OOPNode'] = None


# =============================================================================
# Control Flow Blocks
# =============================================================================

@dataclass
class IfTrueBlock(OOPNode):
    """ifTrue: control structure.
    
    condition ifTrue: [ statements ]
    """
    kind: str = 'if_true'
    condition: Optional['OOPNode'] = None
    true_block: Optional['OOPNode'] = None


@dataclass
class IfFalseBlock(OOPNode):
    """ifFalse: control structure.
    
    condition ifFalse: [ statements ]
    """
    kind: str = 'if_false'
    condition: Optional['OOPNode'] = None
    false_block: Optional['OOPNode'] = None


@dataclass
class IfTrueIfFalseBlock(OOPNode):
    """ifTrue:ifFalse: control structure.
    
    condition ifTrue: [ trueStatements ] ifFalse: [ falseStatements ]
    """
    kind: str = 'if_true_if_false'
    condition: Optional['OOPNode'] = None
    true_block: Optional['OOPNode'] = None
    false_block: Optional['OOPNode'] = None


@dataclass
class WhileTrueBlock(OOPNode):
    """whileTrue: loop.
    
    [ condition ] whileTrue: [ body ]
    """
    kind: str = 'while_true'
    condition_block: Optional['OOPNode'] = None
    body_block: Optional['OOPNode'] = None


@dataclass
class WhileFalseBlock(OOPNode):
    """whileFalse: loop.
    
    [ condition ] whileFalse: [ body ]
    """
    kind: str = 'while_false'
    condition_block: Optional['OOPNode'] = None
    body_block: Optional['OOPNode'] = None


@dataclass
class TimesRepeatBlock(OOPNode):
    """timesRepeat: loop.
    
    n timesRepeat: [ body ]
    """
    kind: str = 'times_repeat'
    count: Optional['OOPNode'] = None
    body_block: Optional['OOPNode'] = None


@dataclass
class ToDoBlock(OOPNode):
    """to:do: loop.
    
    start to: end do: [ :i | body ]
    """
    kind: str = 'to_do'
    start: Optional['OOPNode'] = None
    end: Optional['OOPNode'] = None
    step: Optional['OOPNode'] = None  # For to:by:do:
    body_block: Optional['OOPNode'] = None


# =============================================================================
# Collection Iteration Blocks
# =============================================================================

@dataclass
class DoBlock(OOPNode):
    """do: iteration.
    
    collection do: [ :each | body ]
    """
    kind: str = 'do_block'
    collection: Optional['OOPNode'] = None
    body_block: Optional['OOPNode'] = None


@dataclass
class CollectBlock(OOPNode):
    """collect: transformation.
    
    collection collect: [ :each | transformed ]
    Returns new collection with transformed elements.
    """
    kind: str = 'collect_block'
    collection: Optional['OOPNode'] = None
    transform_block: Optional['OOPNode'] = None


@dataclass
class SelectBlock(OOPNode):
    """select: filtering.
    
    collection select: [ :each | condition ]
    Returns elements for which condition is true.
    """
    kind: str = 'select_block'
    collection: Optional['OOPNode'] = None
    condition_block: Optional['OOPNode'] = None


@dataclass
class RejectBlock(OOPNode):
    """reject: filtering.
    
    collection reject: [ :each | condition ]
    Returns elements for which condition is false.
    """
    kind: str = 'reject_block'
    collection: Optional['OOPNode'] = None
    condition_block: Optional['OOPNode'] = None


@dataclass
class DetectBlock(OOPNode):
    """detect: search.
    
    collection detect: [ :each | condition ]
    Returns first element for which condition is true.
    """
    kind: str = 'detect_block'
    collection: Optional['OOPNode'] = None
    condition_block: Optional['OOPNode'] = None
    if_none_block: Optional['OOPNode'] = None  # For detect:ifNone:


@dataclass
class InjectIntoBlock(OOPNode):
    """inject:into: reduction.
    
    collection inject: initial into: [ :acc :each | newAcc ]
    Reduces collection to single value.
    """
    kind: str = 'inject_into'
    collection: Optional['OOPNode'] = None
    initial_value: Optional['OOPNode'] = None
    reduce_block: Optional['OOPNode'] = None


# =============================================================================
# Block Utilities
# =============================================================================

def make_simple_block(statements: List['OOPNode']) -> FullBlock:
    """Create a simple block without parameters."""
    return FullBlock(statements=statements)


def make_param_block(param_names: List[str], statements: List['OOPNode']) -> FullBlock:
    """Create a block with parameters."""
    params = [BlockParameter(name=n) for n in param_names]
    return FullBlock(parameters=params, statements=statements)


def make_single_param_block(param: str, body: 'OOPNode') -> FullBlock:
    """Create a single-parameter block."""
    return FullBlock(
        parameters=[BlockParameter(name=param)],
        statements=[body]
    )


def make_dual_param_block(param1: str, param2: str, body: 'OOPNode') -> FullBlock:
    """Create a two-parameter block."""
    return FullBlock(
        parameters=[BlockParameter(name=param1), BlockParameter(name=param2)],
        statements=[body]
    )


def is_simple_block(block: 'OOPNode') -> bool:
    """Check if block is simple (no params, no temps, single statement)."""
    if not isinstance(block, (FullBlock,)) and block.kind != 'full_block':
        return False
    if hasattr(block, 'parameters') and block.parameters:
        return False
    if hasattr(block, 'temporaries') and block.temporaries:
        return False
    stmts = getattr(block, 'statements', [])
    return len(stmts) <= 1


def is_constant_block(block: 'OOPNode') -> bool:
    """Check if block always returns a constant."""
    if isinstance(block, ConstantBlock) or block.kind == 'constant_block':
        return True
    if not is_simple_block(block):
        return False
    stmts = getattr(block, 'statements', [])
    if not stmts:
        return True  # Empty block returns nil
    # Check if single statement is a literal
    return stmts[0].kind == 'literal'


def get_block_return_value(block: 'OOPNode') -> Optional['OOPNode']:
    """Get the return value of a block (last statement)."""
    stmts = getattr(block, 'statements', [])
    if not stmts:
        return None
    return stmts[-1]
