#!/usr/bin/env python3
"""STUNIR OOP IR - Object-oriented and historical language constructs.

This package provides IR nodes for object-oriented languages (Smalltalk)
and historical languages (ALGOL), including message passing, blocks,
classes, and call-by-name parameters.

Usage:
    from ir.oop import ClassDefinition, Message, Block, MessageType
    from ir.oop import ALGOLProcedure, ParameterMode
    
    # Create a Smalltalk class
    cls = ClassDefinition(
        name='Point',
        superclass='Object',
        instance_variables=['x', 'y']
    )
    
    # Create an ALGOL procedure
    proc = AlgolProcedure(
        name='swap',
        parameters=[
            AlgolParameter(name='a', mode=ParameterMode.BY_NAME),
            AlgolParameter(name='b', mode=ParameterMode.BY_NAME)
        ]
    )
"""

# Core OOP IR classes
from ir.oop.oop_ir import (
    # Base
    OOPNode,
    
    # Enumerations
    MessageType,
    ParameterMode,
    CollectionType,
    VariableScope,
    
    # Expressions
    Literal,
    Variable,
    Assignment,
    Return,
    SelfReference,
    SuperReference,
    ArrayLiteral,
    SymbolLiteral,
    CharacterLiteral,
    BinaryOperation,
    UnaryOperation,
    
    # Messages
    Message,
    CascadedMessage,
    SuperSend,
    
    # Blocks
    Block,
    BlockValue,
    NonLocalReturn,
    
    # Classes
    ClassDefinition,
    MethodDefinition,
    Metaclass,
    TraitDefinition,
    
    # Collections
    CollectionNew,
    CollectionAccess,
    DictionaryLiteral,
    
    # Control structures
    ConditionalMessage,
    LoopMessage,
    IterationMessage,
    
    # ALGOL constructs
    AlgolBlock,
    AlgolProcedure,
    AlgolParameter,
    OwnVariable,
    AlgolArray,
    AlgolFor,
    AlgolSwitch,
    AlgolGoto,
    AlgolIf,
    AlgolVarDecl,
    AlgolProcedureCall,
    AlgolComment,
    
    # Programs
    SmalltalkProgram,
    AlgolProgram,
    
    # Helper functions
    make_unary_message,
    make_binary_message,
    make_keyword_message,
    make_block,
    make_conditional,
    make_while_loop,
    make_times_repeat,
)

# Message passing utilities
from ir.oop.messages import (
    UnaryMessage,
    BinaryMessage,
    KeywordMessage,
    MessageChain,
    SelfSend,
    MessagePattern,
    PrimitiveCall,
    
    # Utilities
    parse_selector,
    selector_from_keywords,
    is_binary_selector,
    is_keyword_selector,
    is_unary_selector,
    count_arguments,
    
    # Common selectors
    ARITHMETIC_SELECTORS,
    COMPARISON_SELECTORS,
    LOGICAL_SELECTORS,
    COMMON_UNARY_SELECTORS,
    COMMON_KEYWORD_SELECTORS,
)

# Block constructs
from ir.oop.blocks import (
    BlockParameter,
    BlockTemporary,
    BlockReturn,
    FullBlock,
    InlinedBlock,
    ConstantBlock,
    BlockEvaluation,
    BlockOnDo,
    BlockEnsure,
    BlockIfCurtailed,
    
    # Control flow blocks
    IfTrueBlock,
    IfFalseBlock,
    IfTrueIfFalseBlock,
    WhileTrueBlock,
    WhileFalseBlock,
    TimesRepeatBlock,
    ToDoBlock,
    
    # Collection iteration
    DoBlock,
    CollectBlock,
    SelectBlock,
    RejectBlock,
    DetectBlock,
    InjectIntoBlock,
    
    # Utilities
    make_simple_block,
    make_param_block,
    make_single_param_block,
    make_dual_param_block,
    is_simple_block,
    is_constant_block,
    get_block_return_value,
)

__all__ = [
    # Base
    'OOPNode',
    
    # Enumerations
    'MessageType',
    'ParameterMode', 
    'CollectionType',
    'VariableScope',
    
    # Expressions
    'Literal',
    'Variable',
    'Assignment',
    'Return',
    'SelfReference',
    'SuperReference',
    'ArrayLiteral',
    'SymbolLiteral',
    'CharacterLiteral',
    'BinaryOperation',
    'UnaryOperation',
    
    # Messages
    'Message',
    'CascadedMessage',
    'SuperSend',
    'UnaryMessage',
    'BinaryMessage',
    'KeywordMessage',
    'MessageChain',
    'SelfSend',
    'MessagePattern',
    'PrimitiveCall',
    
    # Blocks
    'Block',
    'BlockValue',
    'NonLocalReturn',
    'BlockParameter',
    'BlockTemporary',
    'BlockReturn',
    'FullBlock',
    'InlinedBlock',
    'ConstantBlock',
    'BlockEvaluation',
    'BlockOnDo',
    'BlockEnsure',
    'BlockIfCurtailed',
    
    # Control flow
    'IfTrueBlock',
    'IfFalseBlock',
    'IfTrueIfFalseBlock',
    'WhileTrueBlock',
    'WhileFalseBlock',
    'TimesRepeatBlock',
    'ToDoBlock',
    
    # Collection iteration
    'DoBlock',
    'CollectBlock',
    'SelectBlock',
    'RejectBlock',
    'DetectBlock',
    'InjectIntoBlock',
    
    # Classes
    'ClassDefinition',
    'MethodDefinition',
    'Metaclass',
    'TraitDefinition',
    
    # Collections
    'CollectionNew',
    'CollectionAccess',
    'DictionaryLiteral',
    
    # Control structures
    'ConditionalMessage',
    'LoopMessage',
    'IterationMessage',
    
    # ALGOL
    'AlgolBlock',
    'AlgolProcedure',
    'AlgolParameter',
    'OwnVariable',
    'AlgolArray',
    'AlgolFor',
    'AlgolSwitch',
    'AlgolGoto',
    'AlgolIf',
    'AlgolVarDecl',
    'AlgolProcedureCall',
    'AlgolComment',
    
    # Programs
    'SmalltalkProgram',
    'AlgolProgram',
    
    # Helper functions
    'make_unary_message',
    'make_binary_message',
    'make_keyword_message',
    'make_block',
    'make_conditional',
    'make_while_loop',
    'make_times_repeat',
    'make_simple_block',
    'make_param_block',
    'make_single_param_block',
    'make_dual_param_block',
    'is_simple_block',
    'is_constant_block',
    'get_block_return_value',
    'parse_selector',
    'selector_from_keywords',
    'is_binary_selector',
    'is_keyword_selector',
    'is_unary_selector',
    'count_arguments',
    
    # Common selectors
    'ARITHMETIC_SELECTORS',
    'COMPARISON_SELECTORS',
    'LOGICAL_SELECTORS',
    'COMMON_UNARY_SELECTORS',
    'COMMON_KEYWORD_SELECTORS',
]
