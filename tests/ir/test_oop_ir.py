#!/usr/bin/env python3
"""Tests for STUNIR OOP IR.

Tests cover:
- Message passing constructs
- Block and closure constructs
- Class definitions
- ALGOL-specific constructs
- Helper functions
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ir.oop import (
    # Core
    OOPNode, MessageType, ParameterMode, CollectionType, VariableScope,
    # Expressions
    Literal, Variable, Assignment, Return, SelfReference, SuperReference,
    ArrayLiteral, SymbolLiteral, CharacterLiteral, BinaryOperation,
    # Messages
    Message, CascadedMessage, SuperSend,
    UnaryMessage, BinaryMessage, KeywordMessage, MessageChain,
    # Blocks
    Block, BlockValue, NonLocalReturn,
    BlockParameter, BlockTemporary, FullBlock,
    # Classes
    ClassDefinition, MethodDefinition, Metaclass,
    # Collections
    CollectionNew, CollectionAccess,
    # Control
    ConditionalMessage, LoopMessage, IterationMessage,
    # ALGOL
    AlgolBlock, AlgolProcedure, AlgolParameter, OwnVariable,
    AlgolArray, AlgolFor, AlgolSwitch, AlgolIf, AlgolVarDecl,
    # Programs
    SmalltalkProgram, AlgolProgram,
    # Helpers
    make_unary_message, make_binary_message, make_keyword_message,
    make_block, make_conditional, make_while_loop, make_times_repeat,
    parse_selector, is_binary_selector, is_keyword_selector, is_unary_selector,
    count_arguments,
)


# =============================================================================
# Message Passing Tests
# =============================================================================

class TestMessagePassing:
    """Test message passing constructs."""
    
    def test_unary_message(self):
        """TC-OOP-001: Unary message."""
        msg = Message(
            receiver=Variable(name='anObject'),
            selector='size',
            message_type=MessageType.UNARY
        )
        d = msg.to_dict()
        assert d['kind'] == 'message'
        assert d['selector'] == 'size'
        assert d['message_type'] == 'unary'
    
    def test_binary_message(self):
        """TC-OOP-002: Binary message."""
        msg = Message(
            receiver=Literal(value=3, literal_type='integer'),
            selector='+',
            message_type=MessageType.BINARY,
            arguments=[Literal(value=4, literal_type='integer')]
        )
        d = msg.to_dict()
        assert d['message_type'] == 'binary'
        assert d['selector'] == '+'
        assert len(d['arguments']) == 1
    
    def test_keyword_message(self):
        """TC-OOP-003: Keyword message."""
        msg = Message(
            receiver=Variable(name='dict'),
            selector='at:put:',
            message_type=MessageType.KEYWORD,
            arguments=[
                SymbolLiteral(value='key'),
                Literal(value='value', literal_type='string')
            ]
        )
        d = msg.to_dict()
        assert d['selector'] == 'at:put:'
        assert d['message_type'] == 'keyword'
        assert len(d['arguments']) == 2
    
    def test_cascaded_message(self):
        """TC-OOP-004: Cascaded messages."""
        cascade = CascadedMessage(
            receiver=Variable(name='stream'),
            messages=[
                Message(selector='nextPutAll:', message_type=MessageType.KEYWORD,
                       arguments=[Literal(value='Hello', literal_type='string')]),
                Message(selector='cr', message_type=MessageType.UNARY),
                Message(selector='flush', message_type=MessageType.UNARY),
            ]
        )
        d = cascade.to_dict()
        assert d['kind'] == 'cascade'
        assert len(d['messages']) == 3
    
    def test_super_send(self):
        """TC-OOP-005: Super message send."""
        send = SuperSend(
            selector='initialize',
            message_type=MessageType.UNARY
        )
        d = send.to_dict()
        assert d['kind'] == 'super_send'
        assert d['selector'] == 'initialize'


class TestMessageUtilities:
    """Test message utility functions."""
    
    def test_parse_unary_selector(self):
        """Parse unary selector."""
        msg_type, keywords = parse_selector('size')
        assert msg_type == MessageType.UNARY
        assert keywords == ['size']
    
    def test_parse_binary_selector(self):
        """Parse binary selector."""
        msg_type, keywords = parse_selector('+')
        assert msg_type == MessageType.BINARY
        assert keywords == ['+']
    
    def test_parse_keyword_selector(self):
        """Parse keyword selector."""
        msg_type, keywords = parse_selector('at:put:')
        assert msg_type == MessageType.KEYWORD
        assert keywords == ['at:', 'put:']
    
    def test_is_binary_selector(self):
        """Check binary selector detection."""
        assert is_binary_selector('+')
        assert is_binary_selector('<=')
        assert not is_binary_selector('size')
        assert not is_binary_selector('at:')
    
    def test_is_keyword_selector(self):
        """Check keyword selector detection."""
        assert is_keyword_selector('at:')
        assert is_keyword_selector('at:put:')
        assert not is_keyword_selector('size')
        assert not is_keyword_selector('+')
    
    def test_is_unary_selector(self):
        """Check unary selector detection."""
        assert is_unary_selector('size')
        assert is_unary_selector('isEmpty')
        assert not is_unary_selector('+')
        assert not is_unary_selector('at:')
    
    def test_count_arguments(self):
        """Count selector arguments."""
        assert count_arguments('size') == 0
        assert count_arguments('+') == 1
        assert count_arguments('at:') == 1
        assert count_arguments('at:put:') == 2
        assert count_arguments('copyFrom:to:') == 2


# =============================================================================
# Block Tests
# =============================================================================

class TestBlocks:
    """Test block constructs."""
    
    def test_simple_block(self):
        """TC-OOP-006: Simple block."""
        block = Block(
            statements=[Literal(value=42, literal_type='integer')]
        )
        d = block.to_dict()
        assert d['kind'] == 'block'
        assert len(d['statements']) == 1
    
    def test_block_with_params(self):
        """TC-OOP-007: Block with parameters."""
        block = Block(
            parameters=['x', 'y'],
            statements=[
                Message(
                    receiver=Variable(name='x'),
                    selector='+',
                    message_type=MessageType.BINARY,
                    arguments=[Variable(name='y')]
                )
            ]
        )
        d = block.to_dict()
        assert d['parameters'] == ['x', 'y']
    
    def test_block_with_temporaries(self):
        """TC-OOP-008: Block with temporaries."""
        block = Block(
            parameters=['x'],
            temporaries=['temp'],
            statements=[
                Assignment(target='temp', value=Variable(name='x')),
                Return(value=Variable(name='temp'))
            ]
        )
        d = block.to_dict()
        assert d['temporaries'] == ['temp']
        assert len(d['statements']) == 2
    
    def test_full_block(self):
        """TC-OOP-009: Full block with parameters and temps."""
        block = FullBlock(
            parameters=[BlockParameter(name='each')],
            temporaries=[BlockTemporary(name='sum')],
            statements=[
                Assignment(target='sum', value=Literal(value=0))
            ]
        )
        d = block.to_dict()
        assert d['kind'] == 'full_block'
        assert len(d['parameters']) == 1
        assert len(d['temporaries']) == 1
    
    def test_block_value(self):
        """TC-OOP-010: Block evaluation."""
        bv = BlockValue(
            block=Block(parameters=['x'], statements=[Variable(name='x')]),
            arguments=[Literal(value=5)]
        )
        d = bv.to_dict()
        assert d['kind'] == 'block_value'
        assert len(d['arguments']) == 1


# =============================================================================
# Class Definition Tests
# =============================================================================

class TestClassDefinition:
    """Test class definition."""
    
    def test_class_with_variables(self):
        """TC-OOP-011: Class with instance variables."""
        cls = ClassDefinition(
            name='Point',
            superclass='Object',
            instance_variables=['x', 'y'],
            class_variables=['Origin'],
            category='Graphics-Primitives'
        )
        d = cls.to_dict()
        assert d['name'] == 'Point'
        assert d['superclass'] == 'Object'
        assert d['instance_variables'] == ['x', 'y']
        assert d['class_variables'] == ['Origin']
    
    def test_method_definition(self):
        """TC-OOP-012: Method definition."""
        method = MethodDefinition(
            selector='x:',
            message_type=MessageType.KEYWORD,
            parameters=['aNumber'],
            statements=[
                Assignment(target='x', value=Variable(name='aNumber'))
            ]
        )
        d = method.to_dict()
        assert d['selector'] == 'x:'
        assert d['parameters'] == ['aNumber']
    
    def test_class_method(self):
        """TC-OOP-013: Class method definition."""
        method = MethodDefinition(
            selector='new',
            message_type=MessageType.UNARY,
            is_class_method=True,
            statements=[
                Return(value=Message(
                    receiver=SuperReference(),
                    selector='new',
                    message_type=MessageType.UNARY
                ))
            ]
        )
        d = method.to_dict()
        assert d['is_class_method'] is True
    
    def test_metaclass(self):
        """TC-OOP-014: Metaclass."""
        meta = Metaclass(
            base_class='Point',
            class_instance_variables=['defaultOrigin'],
            class_methods=[
                MethodDefinition(selector='origin', message_type=MessageType.UNARY)
            ]
        )
        d = meta.to_dict()
        assert d['base_class'] == 'Point'
        assert len(d['class_methods']) == 1


# =============================================================================
# ALGOL Tests
# =============================================================================

class TestALGOL:
    """Test ALGOL-specific constructs."""
    
    def test_call_by_name_parameter(self):
        """TC-OOP-015: Call-by-name parameter."""
        param = AlgolParameter(
            name='expr',
            param_type='real',
            mode=ParameterMode.BY_NAME
        )
        d = param.to_dict()
        assert d['mode'] == 'name'
    
    def test_call_by_value_parameter(self):
        """TC-OOP-016: Call-by-value parameter."""
        param = AlgolParameter(
            name='x',
            param_type='integer',
            mode=ParameterMode.BY_VALUE
        )
        d = param.to_dict()
        assert d['mode'] == 'value'
    
    def test_own_variable(self):
        """TC-OOP-017: Own variable."""
        own = OwnVariable(
            name='counter',
            var_type='integer',
            initial_value=Literal(value=0, literal_type='integer')
        )
        d = own.to_dict()
        assert d['kind'] == 'own_variable'
        assert d['name'] == 'counter'
    
    def test_for_loop_with_step(self):
        """TC-OOP-018: For loop with step."""
        loop = AlgolFor(
            variable='i',
            init_value=Literal(value=1, literal_type='integer'),
            step=Literal(value=2, literal_type='integer'),
            until_value=Literal(value=10, literal_type='integer'),
            body=Variable(name='stmt')
        )
        d = loop.to_dict()
        assert d['variable'] == 'i'
        assert d['step']['value'] == 2
    
    def test_algol_block(self):
        """TC-OOP-019: ALGOL block structure."""
        block = AlgolBlock(
            declarations=[
                AlgolVarDecl(name='x', var_type='integer'),
                AlgolVarDecl(name='y', var_type='real')
            ],
            statements=[
                Assignment(target='x', value=Literal(value=10))
            ],
            label='L1'
        )
        d = block.to_dict()
        assert d['kind'] == 'algol_block'
        assert d['label'] == 'L1'
        assert len(d['declarations']) == 2
    
    def test_algol_procedure(self):
        """TC-OOP-020: ALGOL procedure."""
        proc = AlgolProcedure(
            name='swap',
            parameters=[
                AlgolParameter(name='a', param_type='integer', mode=ParameterMode.BY_NAME),
                AlgolParameter(name='b', param_type='integer', mode=ParameterMode.BY_NAME)
            ],
            body=AlgolBlock(
                declarations=[AlgolVarDecl(name='temp', var_type='integer')],
                statements=[]
            )
        )
        d = proc.to_dict()
        assert d['name'] == 'swap'
        assert len(d['parameters']) == 2
    
    def test_algol_function(self):
        """TC-OOP-021: ALGOL function (typed procedure)."""
        func = AlgolProcedure(
            name='factorial',
            result_type='integer',
            parameters=[
                AlgolParameter(name='n', param_type='integer', mode=ParameterMode.BY_VALUE)
            ],
            body=AlgolBlock(statements=[])
        )
        d = func.to_dict()
        assert d['result_type'] == 'integer'
    
    def test_algol_array(self):
        """TC-OOP-022: ALGOL array with dynamic bounds."""
        arr = AlgolArray(
            name='matrix',
            element_type='real',
            bounds=[
                (Literal(value=1), Variable(name='n')),
                (Literal(value=1), Variable(name='m'))
            ]
        )
        d = arr.to_dict()
        assert d['name'] == 'matrix'
        assert len(d['bounds']) == 2
    
    def test_algol_switch(self):
        """TC-OOP-023: ALGOL switch declaration."""
        switch = AlgolSwitch(
            name='S',
            labels=['L1', 'L2', 'L3', 'L4']
        )
        d = switch.to_dict()
        assert d['name'] == 'S'
        assert len(d['labels']) == 4
    
    def test_algol_if(self):
        """TC-OOP-024: ALGOL if statement."""
        if_stmt = AlgolIf(
            condition=BinaryOperation(
                operator='<',
                left=Variable(name='x'),
                right=Literal(value=0)
            ),
            then_branch=Assignment(target='x', value=Literal(value=0)),
            else_branch=None
        )
        d = if_stmt.to_dict()
        assert d['kind'] == 'algol_if'


# =============================================================================
# Control Structure Tests
# =============================================================================

class TestControlStructures:
    """Test control structure constructs."""
    
    def test_conditional_message(self):
        """TC-OOP-025: Conditional message."""
        cond = ConditionalMessage(
            condition=Variable(name='flag'),
            true_block=Block(statements=[Literal(value='yes')]),
            false_block=Block(statements=[Literal(value='no')])
        )
        d = cond.to_dict()
        assert d['kind'] == 'conditional'
    
    def test_while_loop(self):
        """TC-OOP-026: While loop message."""
        loop = LoopMessage(
            loop_type='whileTrue:',
            condition_or_count=BinaryOperation(
                operator='<',
                left=Variable(name='i'),
                right=Literal(value=10)
            ),
            body=Block(statements=[])
        )
        d = loop.to_dict()
        assert d['loop_type'] == 'whileTrue:'
    
    def test_times_repeat(self):
        """TC-OOP-027: Times repeat loop."""
        loop = LoopMessage(
            loop_type='timesRepeat:',
            condition_or_count=Literal(value=5),
            body=Block(statements=[])
        )
        d = loop.to_dict()
        assert d['loop_type'] == 'timesRepeat:'
    
    def test_iteration(self):
        """TC-OOP-028: Collection iteration."""
        iter_msg = IterationMessage(
            collection=Variable(name='array'),
            iterator='do:',
            block=Block(
                parameters=['each'],
                statements=[Message(
                    receiver=Variable(name='each'),
                    selector='printString',
                    message_type=MessageType.UNARY
                )]
            )
        )
        d = iter_msg.to_dict()
        assert d['iterator'] == 'do:'


# =============================================================================
# Collection Tests
# =============================================================================

class TestCollections:
    """Test collection constructs."""
    
    def test_collection_new(self):
        """TC-OOP-029: Collection creation."""
        coll = CollectionNew(
            collection_type=CollectionType.ORDERED_COLLECTION,
            initial_size=10
        )
        d = coll.to_dict()
        assert d['collection_type'] == 'OrderedCollection'
    
    def test_collection_access(self):
        """TC-OOP-030: Collection access."""
        access = CollectionAccess(
            collection=Variable(name='dict'),
            operation='at:put:',
            arguments=[
                SymbolLiteral(value='key'),
                Literal(value='value', literal_type='string')
            ]
        )
        d = access.to_dict()
        assert d['operation'] == 'at:put:'
    
    def test_array_literal(self):
        """TC-OOP-031: Array literal."""
        arr = ArrayLiteral(
            elements=[
                Literal(value=1, literal_type='integer'),
                Literal(value=2, literal_type='integer'),
                Literal(value=3, literal_type='integer')
            ]
        )
        d = arr.to_dict()
        assert len(d['elements']) == 3


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestHelperFunctions:
    """Test helper functions."""
    
    def test_make_unary_message(self):
        """Make unary message helper."""
        msg = make_unary_message(Variable(name='obj'), 'size')
        d = msg.to_dict()
        assert d['selector'] == 'size'
        assert d['message_type'] == 'unary'
    
    def test_make_binary_message(self):
        """Make binary message helper."""
        msg = make_binary_message(
            Literal(value=3),
            '+',
            Literal(value=4)
        )
        d = msg.to_dict()
        assert d['selector'] == '+'
        assert d['message_type'] == 'binary'
    
    def test_make_keyword_message(self):
        """Make keyword message helper."""
        msg = make_keyword_message(
            Variable(name='dict'),
            'at:put:',
            [SymbolLiteral(value='key'), Literal(value='value')]
        )
        d = msg.to_dict()
        assert d['selector'] == 'at:put:'
    
    def test_make_block(self):
        """Make block helper."""
        block = make_block(
            [Literal(value=42)],
            params=['x']
        )
        d = block.to_dict()
        assert d['parameters'] == ['x']
    
    def test_make_conditional(self):
        """Make conditional helper."""
        cond = make_conditional(
            Variable(name='flag'),
            [Literal(value='yes')],
            [Literal(value='no')]
        )
        d = cond.to_dict()
        assert d['true_block'] is not None
        assert d['false_block'] is not None
    
    def test_make_while_loop(self):
        """Make while loop helper."""
        loop = make_while_loop(
            Variable(name='condition'),
            [Literal(value='body')]
        )
        d = loop.to_dict()
        assert d['loop_type'] == 'whileTrue:'
    
    def test_make_times_repeat(self):
        """Make times repeat helper."""
        loop = make_times_repeat(
            Literal(value=5),
            [Literal(value='body')]
        )
        d = loop.to_dict()
        assert d['loop_type'] == 'timesRepeat:'


# =============================================================================
# Program Tests
# =============================================================================

class TestPrograms:
    """Test program structures."""
    
    def test_smalltalk_program(self):
        """TC-OOP-032: Smalltalk program."""
        prog = SmalltalkProgram(
            classes=[
                ClassDefinition(name='MyClass', superclass='Object')
            ],
            methods=[
                MethodDefinition(selector='test', message_type=MessageType.UNARY)
            ]
        )
        d = prog.to_dict()
        assert d['kind'] == 'smalltalk_program'
        assert len(d['classes']) == 1
    
    def test_algol_program(self):
        """TC-OOP-033: ALGOL program."""
        prog = AlgolProgram(
            name='test',
            main_block=AlgolBlock(
                declarations=[],
                statements=[
                    Assignment(target='x', value=Literal(value=1))
                ]
            )
        )
        d = prog.to_dict()
        assert d['kind'] == 'algol_program'
        assert d['name'] == 'test'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
