#!/usr/bin/env python3
"""STUNIR OOP IR - Message passing constructs.

This module provides specialized message passing constructs for Smalltalk-style
object-oriented programming.

Usage:
    from ir.oop.messages import UnaryMessage, KeywordMessage, MessageChain
    
    # Create a message chain: obj foo bar: 1 baz: 2
    chain = MessageChain(
        receiver=Variable(name='obj'),
        messages=[
            UnaryMessage(selector='foo'),
            KeywordMessage(selector='bar:baz:', args=[...])
        ]
    )
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any
from ir.oop.oop_ir import OOPNode, MessageType


# =============================================================================
# Specialized Message Types
# =============================================================================

@dataclass
class UnaryMessage(OOPNode):
    """Unary message (no arguments).
    
    Examples: size, isEmpty, yourself
    """
    kind: str = 'unary_message'
    selector: str = ''


@dataclass
class BinaryMessage(OOPNode):
    """Binary message (one argument, operator-style selector).
    
    Examples: + 3, >= threshold, @ y
    """
    kind: str = 'binary_message'
    selector: str = ''  # +, -, *, /, //, \\, @, <, >, <=, >=, =, ~=, ==, ~~
    argument: Optional['OOPNode'] = None


@dataclass
class KeywordMessage(OOPNode):
    """Keyword message (one or more keyword:argument pairs).
    
    Examples: at: 1, at: 1 put: 'value', ifTrue: [yes] ifFalse: [no]
    """
    kind: str = 'keyword_message'
    selector: str = ''  # Full selector including colons: 'at:put:'
    arguments: List['OOPNode'] = field(default_factory=list)
    
    @property
    def keywords(self) -> List[str]:
        """Get individual keywords from selector."""
        if not self.selector:
            return []
        return [k + ':' for k in self.selector.rstrip(':').split(':')]
    
    @property
    def arity(self) -> int:
        """Number of arguments."""
        return len(self.keywords)


@dataclass
class MessageChain(OOPNode):
    """Chain of messages with precedence handling.
    
    In Smalltalk: unary > binary > keyword
    
    Example: 3 factorial + 4 factorial between: 10 and: 100
    Parses as: ((3 factorial) + (4 factorial)) between: 10 and: 100
    """
    kind: str = 'message_chain'
    receiver: Optional['OOPNode'] = None
    messages: List['OOPNode'] = field(default_factory=list)  # UnaryMessage, BinaryMessage, KeywordMessage
    
    def get_message_type(self, msg: 'OOPNode') -> MessageType:
        """Determine message type for precedence."""
        if isinstance(msg, UnaryMessage) or msg.kind == 'unary_message':
            return MessageType.UNARY
        elif isinstance(msg, BinaryMessage) or msg.kind == 'binary_message':
            return MessageType.BINARY
        elif isinstance(msg, KeywordMessage) or msg.kind == 'keyword_message':
            return MessageType.KEYWORD
        return MessageType.UNARY


@dataclass
class SuperSend(OOPNode):
    """Message sent to super.
    
    Example: super initialize
    """
    kind: str = 'super_send'
    message: Optional['OOPNode'] = None  # UnaryMessage, BinaryMessage, or KeywordMessage


@dataclass
class SelfSend(OOPNode):
    """Explicit message sent to self.
    
    Example: self initialize
    """
    kind: str = 'self_send'
    message: Optional['OOPNode'] = None


# =============================================================================
# Message Pattern Matching
# =============================================================================

@dataclass
class MessagePattern(OOPNode):
    """Message pattern for method definition.
    
    Unary: methodName
    Binary: + anObject
    Keyword: at: anIndex put: aValue
    """
    kind: str = 'message_pattern'
    selector: str = ''
    message_type: MessageType = MessageType.UNARY
    parameter_names: List[str] = field(default_factory=list)
    
    @classmethod
    def unary(cls, selector: str) -> 'MessagePattern':
        """Create unary message pattern."""
        return cls(
            selector=selector,
            message_type=MessageType.UNARY
        )
    
    @classmethod
    def binary(cls, selector: str, param: str) -> 'MessagePattern':
        """Create binary message pattern."""
        return cls(
            selector=selector,
            message_type=MessageType.BINARY,
            parameter_names=[param]
        )
    
    @classmethod
    def keyword(cls, selector: str, params: List[str]) -> 'MessagePattern':
        """Create keyword message pattern."""
        return cls(
            selector=selector,
            message_type=MessageType.KEYWORD,
            parameter_names=params
        )


# =============================================================================
# Primitive Messages
# =============================================================================

@dataclass
class PrimitiveCall(OOPNode):
    """Primitive operation call.
    
    Primitives are low-level operations implemented by the VM.
    Example: <primitive: 1> for SmallInteger +
    """
    kind: str = 'primitive'
    primitive_number: int = 0
    module_name: Optional[str] = None  # For named primitives
    primitive_name: Optional[str] = None
    error_code: Optional[str] = None  # Variable to receive error code


# =============================================================================
# Message Utilities
# =============================================================================

def parse_selector(selector: str) -> tuple:
    """Parse a selector into type and keywords.
    
    Returns:
        Tuple of (MessageType, list of keywords)
    
    Examples:
        'size' -> (UNARY, ['size'])
        '+' -> (BINARY, ['+'])
        'at:put:' -> (KEYWORD, ['at:', 'put:'])
    """
    if not selector:
        return (MessageType.UNARY, [])
    
    # Check for binary operators
    binary_chars = set('+-*/\\<>=@%|&?!~,')
    if all(c in binary_chars for c in selector):
        return (MessageType.BINARY, [selector])
    
    # Check for keyword message
    if ':' in selector:
        keywords = [k + ':' for k in selector.rstrip(':').split(':') if k]
        return (MessageType.KEYWORD, keywords)
    
    # Must be unary
    return (MessageType.UNARY, [selector])


def selector_from_keywords(keywords: List[str]) -> str:
    """Construct selector from keyword list.
    
    Example: ['at:', 'put:'] -> 'at:put:'
    """
    return ''.join(keywords)


def is_binary_selector(selector: str) -> bool:
    """Check if selector is a binary operator."""
    binary_chars = set('+-*/\\<>=@%|&?!~,')
    return bool(selector) and all(c in binary_chars for c in selector)


def is_keyword_selector(selector: str) -> bool:
    """Check if selector is a keyword selector."""
    return ':' in selector


def is_unary_selector(selector: str) -> bool:
    """Check if selector is a unary selector."""
    return bool(selector) and not is_binary_selector(selector) and not is_keyword_selector(selector)


def count_arguments(selector: str) -> int:
    """Count number of arguments for a selector.
    
    Unary: 0
    Binary: 1
    Keyword: number of colons
    """
    if not selector:
        return 0
    if is_binary_selector(selector):
        return 1
    if is_keyword_selector(selector):
        return selector.count(':')
    return 0


# =============================================================================
# Common Message Patterns
# =============================================================================

# Arithmetic selectors (binary)
ARITHMETIC_SELECTORS = ['+', '-', '*', '/', '//', '\\\\', '**']

# Comparison selectors (binary)
COMPARISON_SELECTORS = ['<', '>', '<=', '>=', '=', '~=', '==', '~~']

# Logical selectors (binary)
LOGICAL_SELECTORS = ['&', '|']

# Common unary selectors
COMMON_UNARY_SELECTORS = [
    'yourself', 'class', 'copy', 'deepCopy', 'hash',
    'isNil', 'notNil', 'isEmpty', 'notEmpty', 'size',
    'first', 'last', 'printString', 'asString', 'asSymbol',
    'negated', 'abs', 'squared', 'sqrt', 'sin', 'cos',
    'new', 'initialize', 'basicNew', 'basicSize',
]

# Common keyword selectors
COMMON_KEYWORD_SELECTORS = [
    'at:', 'at:put:', 'ifTrue:', 'ifFalse:', 'ifTrue:ifFalse:',
    'whileTrue:', 'whileFalse:', 'timesRepeat:', 'to:do:', 'to:by:do:',
    'do:', 'collect:', 'select:', 'reject:', 'detect:', 'detect:ifNone:',
    'inject:into:', 'includes:', 'add:', 'remove:', 'remove:ifAbsent:',
    'value', 'value:', 'value:value:', 'valueWithArguments:',
    'perform:', 'perform:with:', 'perform:withArguments:',
    'respondsTo:', 'isKindOf:', 'isMemberOf:',
]
