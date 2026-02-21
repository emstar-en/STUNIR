#!/usr/bin/env python3
"""STUNIR OOP IR - Core OOP and historical language constructs.

This module defines IR nodes for object-oriented and historical programming
languages like Smalltalk and ALGOL, including message passing, blocks,
classes, and call-by-name parameters.

Usage:
    from ir.oop.oop_ir import ClassDefinition, Message, Block
    from ir.oop import MessageType, ParameterMode
    
    # Create a Smalltalk class
    cls = ClassDefinition(
        name='Point',
        superclass='Object',
        instance_variables=['x', 'y']
    )
    
    # Create a message send
    msg = Message(
        receiver=Variable(name='point'),
        selector='x',
        message_type=MessageType.UNARY
    )
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Tuple
from abc import ABC
from enum import Enum


# =============================================================================
# Enumerations
# =============================================================================

class MessageType(Enum):
    """Smalltalk message types."""
    UNARY = 'unary'           # receiver message
    BINARY = 'binary'         # receiver + argument
    KEYWORD = 'keyword'       # receiver message: arg1 with: arg2


class ParameterMode(Enum):
    """ALGOL parameter passing modes."""
    BY_VALUE = 'value'
    BY_NAME = 'name'
    BY_RESULT = 'result'


class CollectionType(Enum):
    """Smalltalk collection types."""
    ARRAY = 'Array'
    ORDERED_COLLECTION = 'OrderedCollection'
    SET = 'Set'
    BAG = 'Bag'
    DICTIONARY = 'Dictionary'
    SORTED_COLLECTION = 'SortedCollection'
    LINKED_LIST = 'LinkedList'
    INTERVAL = 'Interval'
    STRING = 'String'
    BYTE_ARRAY = 'ByteArray'


class VariableScope(Enum):
    """Variable scope types."""
    LOCAL = 'local'
    INSTANCE = 'instance'
    CLASS = 'class'
    GLOBAL = 'global'
    TEMPORARY = 'temp'
    POOL = 'pool'


# =============================================================================
# Base Class
# =============================================================================

@dataclass
class OOPNode(ABC):
    """Base class for all OOP IR nodes."""
    kind: str = 'oop_node'
    
    def to_dict(self) -> dict:
        """Convert node to dictionary representation."""
        result = {'kind': self.kind}
        for key, value in self.__dict__.items():
            if key != 'kind' and value is not None:
                if isinstance(value, OOPNode):
                    result[key] = value.to_dict()
                elif isinstance(value, list):
                    result[key] = [
                        v.to_dict() if isinstance(v, OOPNode) else 
                        (v.value if isinstance(v, Enum) else v)
                        for v in value
                    ]
                elif isinstance(value, tuple):
                    result[key] = tuple(
                        v.to_dict() if isinstance(v, OOPNode) else 
                        (v.value if isinstance(v, Enum) else v)
                        for v in value
                    )
                elif isinstance(value, Enum):
                    result[key] = value.value
                else:
                    result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> 'OOPNode':
        """Create node from dictionary representation."""
        # This is a simple implementation - subclasses may override
        return cls(**{k: v for k, v in data.items() if k != 'kind'})


# =============================================================================
# Expression Nodes
# =============================================================================

@dataclass
class Literal(OOPNode):
    """Literal value."""
    kind: str = 'literal'
    value: Any = None
    literal_type: str = 'object'  # integer, float, string, symbol, array, character


@dataclass
class Variable(OOPNode):
    """Variable reference."""
    kind: str = 'variable'
    name: str = ''
    scope: VariableScope = VariableScope.LOCAL


@dataclass
class Assignment(OOPNode):
    """Assignment expression."""
    kind: str = 'assignment'
    target: str = ''
    value: Optional['OOPNode'] = None


@dataclass
class Return(OOPNode):
    """Return statement (^ in Smalltalk, := funcname in ALGOL)."""
    kind: str = 'return'
    value: Optional['OOPNode'] = None


@dataclass
class SelfReference(OOPNode):
    """Self reference (self in Smalltalk)."""
    kind: str = 'self'


@dataclass
class SuperReference(OOPNode):
    """Super reference for method lookup."""
    kind: str = 'super'


@dataclass
class ArrayLiteral(OOPNode):
    """Array literal #(1 2 3) or {expr1. expr2}."""
    kind: str = 'array_literal'
    elements: List['OOPNode'] = field(default_factory=list)
    is_dynamic: bool = False  # True for {expr1. expr2} style


@dataclass
class SymbolLiteral(OOPNode):
    """Symbol literal #symbol."""
    kind: str = 'symbol'
    value: str = ''


@dataclass
class CharacterLiteral(OOPNode):
    """Character literal $c."""
    kind: str = 'character'
    value: str = ''


@dataclass
class BinaryOperation(OOPNode):
    """Binary operation (for ALGOL primarily)."""
    kind: str = 'binary_op'
    operator: str = ''
    left: Optional['OOPNode'] = None
    right: Optional['OOPNode'] = None


@dataclass
class UnaryOperation(OOPNode):
    """Unary operation (for ALGOL)."""
    kind: str = 'unary_op'
    operator: str = ''  # not, -, +
    operand: Optional['OOPNode'] = None


# =============================================================================
# Message Passing (Smalltalk)
# =============================================================================

@dataclass
class Message(OOPNode):
    """Message send expression."""
    kind: str = 'message'
    receiver: Optional['OOPNode'] = None
    selector: str = ''
    message_type: MessageType = MessageType.UNARY
    arguments: List['OOPNode'] = field(default_factory=list)


@dataclass
class CascadedMessage(OOPNode):
    """Cascaded message sends (receiver msg1; msg2; msg3)."""
    kind: str = 'cascade'
    receiver: Optional['OOPNode'] = None
    messages: List[Message] = field(default_factory=list)


@dataclass
class SuperSend(OOPNode):
    """Super send (super message)."""
    kind: str = 'super_send'
    selector: str = ''
    message_type: MessageType = MessageType.UNARY
    arguments: List['OOPNode'] = field(default_factory=list)


# =============================================================================
# Blocks and Closures (Smalltalk)
# =============================================================================

@dataclass
class Block(OOPNode):
    """Block/closure expression [ :args | | temps | statements ]."""
    kind: str = 'block'
    parameters: List[str] = field(default_factory=list)
    temporaries: List[str] = field(default_factory=list)
    statements: List['OOPNode'] = field(default_factory=list)
    return_value: Optional['OOPNode'] = None


@dataclass
class BlockValue(OOPNode):
    """Block evaluation (block value, block value: arg)."""
    kind: str = 'block_value'
    block: Optional[Block] = None
    arguments: List['OOPNode'] = field(default_factory=list)


@dataclass
class NonLocalReturn(OOPNode):
    """Non-local return from block (^value inside block)."""
    kind: str = 'non_local_return'
    value: Optional['OOPNode'] = None


# =============================================================================
# Class Definitions (Smalltalk)
# =============================================================================

@dataclass
class ClassDefinition(OOPNode):
    """Class definition."""
    kind: str = 'class_def'
    name: str = ''
    superclass: str = 'Object'
    instance_variables: List[str] = field(default_factory=list)
    class_variables: List[str] = field(default_factory=list)
    pool_dictionaries: List[str] = field(default_factory=list)
    category: str = ''


@dataclass
class MethodDefinition(OOPNode):
    """Method definition."""
    kind: str = 'method_def'
    selector: str = ''
    message_type: MessageType = MessageType.UNARY
    parameters: List[str] = field(default_factory=list)
    temporaries: List[str] = field(default_factory=list)
    statements: List['OOPNode'] = field(default_factory=list)
    is_class_method: bool = False
    category: str = ''
    primitive: Optional[int] = None  # Primitive number if any


@dataclass
class Metaclass(OOPNode):
    """Metaclass representation."""
    kind: str = 'metaclass'
    base_class: str = ''
    class_instance_variables: List[str] = field(default_factory=list)
    class_methods: List[MethodDefinition] = field(default_factory=list)


@dataclass
class TraitDefinition(OOPNode):
    """Trait definition (for Pharo-style Smalltalk)."""
    kind: str = 'trait_def'
    name: str = ''
    methods: List[MethodDefinition] = field(default_factory=list)
    required_methods: List[str] = field(default_factory=list)


# =============================================================================
# Collections (Smalltalk)
# =============================================================================

@dataclass
class CollectionNew(OOPNode):
    """Collection creation."""
    kind: str = 'collection_new'
    collection_type: CollectionType = CollectionType.ARRAY
    initial_size: Optional[int] = None
    initial_elements: List['OOPNode'] = field(default_factory=list)


@dataclass
class CollectionAccess(OOPNode):
    """Collection access (at:, at:put:, etc.)."""
    kind: str = 'collection_access'
    collection: Optional['OOPNode'] = None
    operation: str = 'at:'  # at:, at:put:, first, last, size, includes:
    arguments: List['OOPNode'] = field(default_factory=list)


@dataclass
class DictionaryLiteral(OOPNode):
    """Dictionary literal for Smalltalk."""
    kind: str = 'dict_literal'
    entries: List[Tuple['OOPNode', 'OOPNode']] = field(default_factory=list)


# =============================================================================
# Control Structures as Messages (Smalltalk)
# =============================================================================

@dataclass
class ConditionalMessage(OOPNode):
    """Conditional as message (ifTrue:, ifFalse:, ifTrue:ifFalse:)."""
    kind: str = 'conditional'
    condition: Optional['OOPNode'] = None
    true_block: Optional[Block] = None
    false_block: Optional[Block] = None


@dataclass
class LoopMessage(OOPNode):
    """Loop as message (whileTrue:, whileFalse:, timesRepeat:)."""
    kind: str = 'loop'
    loop_type: str = 'whileTrue:'  # whileTrue:, whileFalse:, timesRepeat:, to:do:
    condition_or_count: Optional['OOPNode'] = None
    body: Optional[Block] = None
    end_value: Optional['OOPNode'] = None  # For to:do:


@dataclass
class IterationMessage(OOPNode):
    """Collection iteration (do:, collect:, select:, reject:)."""
    kind: str = 'iteration'
    collection: Optional['OOPNode'] = None
    iterator: str = 'do:'  # do:, collect:, select:, reject:, detect:, inject:into:
    block: Optional[Block] = None
    initial_value: Optional['OOPNode'] = None  # For inject:into:


# =============================================================================
# ALGOL-Specific Constructs
# =============================================================================

@dataclass
class AlgolBlock(OOPNode):
    """ALGOL block (begin...end)."""
    kind: str = 'algol_block'
    declarations: List['OOPNode'] = field(default_factory=list)
    statements: List['OOPNode'] = field(default_factory=list)
    label: Optional[str] = None


@dataclass
class AlgolProcedure(OOPNode):
    """ALGOL procedure declaration."""
    kind: str = 'algol_procedure'
    name: str = ''
    parameters: List['AlgolParameter'] = field(default_factory=list)
    result_type: Optional[str] = None  # None for procedures, type for functions
    body: Optional[AlgolBlock] = None
    own_variables: List['OwnVariable'] = field(default_factory=list)


@dataclass
class AlgolParameter(OOPNode):
    """ALGOL parameter with mode specification."""
    kind: str = 'algol_param'
    name: str = ''
    param_type: str = 'integer'
    mode: ParameterMode = ParameterMode.BY_VALUE


@dataclass
class OwnVariable(OOPNode):
    """ALGOL own variable (static)."""
    kind: str = 'own_variable'
    name: str = ''
    var_type: str = 'integer'
    initial_value: Optional['OOPNode'] = None


@dataclass
class AlgolArray(OOPNode):
    """ALGOL array with dynamic bounds."""
    kind: str = 'algol_array'
    name: str = ''
    element_type: str = 'real'
    bounds: List[Tuple['OOPNode', 'OOPNode']] = field(default_factory=list)


@dataclass
class AlgolFor(OOPNode):
    """ALGOL for loop with step clause."""
    kind: str = 'algol_for'
    variable: str = ''
    init_value: Optional['OOPNode'] = None
    step: Optional['OOPNode'] = None
    until_value: Optional['OOPNode'] = None
    body: Optional['OOPNode'] = None
    while_condition: Optional['OOPNode'] = None  # for-while variant


@dataclass
class AlgolSwitch(OOPNode):
    """ALGOL switch (computed goto)."""
    kind: str = 'algol_switch'
    name: str = ''
    labels: List[str] = field(default_factory=list)


@dataclass
class AlgolGoto(OOPNode):
    """ALGOL goto statement."""
    kind: str = 'algol_goto'
    target: str = ''
    switch_name: Optional[str] = None  # For switch-based goto
    switch_index: Optional['OOPNode'] = None


@dataclass
class AlgolIf(OOPNode):
    """ALGOL if statement."""
    kind: str = 'algol_if'
    condition: Optional['OOPNode'] = None
    then_branch: Optional['OOPNode'] = None
    else_branch: Optional['OOPNode'] = None


@dataclass
class AlgolVarDecl(OOPNode):
    """ALGOL variable declaration."""
    kind: str = 'algol_var_decl'
    name: str = ''
    var_type: str = 'integer'
    initial_value: Optional['OOPNode'] = None


@dataclass
class AlgolProcedureCall(OOPNode):
    """ALGOL procedure call."""
    kind: str = 'algol_call'
    name: str = ''
    arguments: List['OOPNode'] = field(default_factory=list)


@dataclass
class AlgolComment(OOPNode):
    """ALGOL comment."""
    kind: str = 'algol_comment'
    text: str = ''


# =============================================================================
# Program Structure
# =============================================================================

@dataclass
class SmalltalkProgram(OOPNode):
    """Smalltalk program (collection of classes and methods)."""
    kind: str = 'smalltalk_program'
    classes: List[ClassDefinition] = field(default_factory=list)
    metaclasses: List[Metaclass] = field(default_factory=list)
    traits: List[TraitDefinition] = field(default_factory=list)
    methods: List[MethodDefinition] = field(default_factory=list)
    workspace_code: List['OOPNode'] = field(default_factory=list)  # Workspace expressions


@dataclass
class AlgolProgram(OOPNode):
    """ALGOL program."""
    kind: str = 'algol_program'
    name: str = ''
    main_block: Optional[AlgolBlock] = None
    procedures: List[AlgolProcedure] = field(default_factory=list)


# =============================================================================
# Helper Functions
# =============================================================================

def make_unary_message(receiver: OOPNode, selector: str) -> Message:
    """Create a unary message send."""
    return Message(
        receiver=receiver,
        selector=selector,
        message_type=MessageType.UNARY
    )


def make_binary_message(receiver: OOPNode, operator: str, argument: OOPNode) -> Message:
    """Create a binary message send."""
    return Message(
        receiver=receiver,
        selector=operator,
        message_type=MessageType.BINARY,
        arguments=[argument]
    )


def make_keyword_message(receiver: OOPNode, selector: str, arguments: List[OOPNode]) -> Message:
    """Create a keyword message send."""
    return Message(
        receiver=receiver,
        selector=selector,
        message_type=MessageType.KEYWORD,
        arguments=arguments
    )


def make_block(statements: List[OOPNode], params: Optional[List[str]] = None) -> Block:
    """Create a block with optional parameters."""
    return Block(
        parameters=params or [],
        statements=statements
    )


def make_conditional(condition: OOPNode, true_stmts: List[OOPNode], 
                     false_stmts: Optional[List[OOPNode]] = None) -> ConditionalMessage:
    """Create a conditional message."""
    return ConditionalMessage(
        condition=condition,
        true_block=Block(statements=true_stmts) if true_stmts else None,
        false_block=Block(statements=false_stmts) if false_stmts else None
    )


def make_while_loop(condition: OOPNode, body_stmts: List[OOPNode], 
                    while_true: bool = True) -> LoopMessage:
    """Create a while loop."""
    return LoopMessage(
        loop_type='whileTrue:' if while_true else 'whileFalse:',
        condition_or_count=condition,
        body=Block(statements=body_stmts)
    )


def make_times_repeat(count: OOPNode, body_stmts: List[OOPNode]) -> LoopMessage:
    """Create a timesRepeat: loop."""
    return LoopMessage(
        loop_type='timesRepeat:',
        condition_or_count=count,
        body=Block(statements=body_stmts)
    )
