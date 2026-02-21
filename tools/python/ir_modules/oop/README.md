# STUNIR OOP IR

**Version:** 1.0  
**Phase:** 10B (OOP & Historical Languages)

## Overview

The OOP IR module provides intermediate representation nodes for object-oriented
and historical programming languages, specifically Smalltalk and ALGOL.

## Architecture

```
ir/oop/
├── __init__.py      # Package exports
├── oop_ir.py        # Core OOP IR classes
├── messages.py      # Message passing constructs
├── blocks.py        # Block/closure support
└── README.md        # This file
```

## Core Concepts

### Message Passing (Smalltalk)

Smalltalk uses message passing as its primary computation mechanism:

```python
from ir.oop import Message, MessageType, Variable, Literal

# Unary message: collection size
msg = Message(
    receiver=Variable(name='collection'),
    selector='size',
    message_type=MessageType.UNARY
)

# Binary message: 3 + 4
msg = Message(
    receiver=Literal(value=3),
    selector='+',
    message_type=MessageType.BINARY,
    arguments=[Literal(value=4)]
)

# Keyword message: dict at: key put: value
msg = Message(
    receiver=Variable(name='dict'),
    selector='at:put:',
    message_type=MessageType.KEYWORD,
    arguments=[SymbolLiteral(value='key'), Literal(value='value')]
)
```

### Blocks and Closures

Blocks are first-class closures in Smalltalk:

```python
from ir.oop import Block, FullBlock, BlockParameter

# Simple block: [ 42 ]
block = Block(statements=[Literal(value=42)])

# Block with parameter: [ :x | x squared ]
block = Block(
    parameters=['x'],
    statements=[Message(
        receiver=Variable(name='x'),
        selector='squared',
        message_type=MessageType.UNARY
    )]
)

# Full block with temporaries
block = FullBlock(
    parameters=[BlockParameter(name='each')],
    temporaries=[BlockTemporary(name='sum')],
    statements=[...]
)
```

### Class Definitions

```python
from ir.oop import ClassDefinition, MethodDefinition

# Define a class
cls = ClassDefinition(
    name='Point',
    superclass='Object',
    instance_variables=['x', 'y'],
    class_variables=['Origin'],
    category='Graphics-Primitives'
)

# Define a method
method = MethodDefinition(
    selector='x:',
    message_type=MessageType.KEYWORD,
    parameters=['aNumber'],
    statements=[Assignment(target='x', value=Variable(name='aNumber'))]
)
```

### ALGOL Constructs

```python
from ir.oop import AlgolProcedure, AlgolParameter, ParameterMode, AlgolFor

# Procedure with call-by-name
proc = AlgolProcedure(
    name='sum',
    result_type='real',
    parameters=[
        AlgolParameter(name='i', mode=ParameterMode.BY_NAME),
        AlgolParameter(name='term', mode=ParameterMode.BY_NAME)
    ],
    body=AlgolBlock(...)
)

# For loop with step
loop = AlgolFor(
    variable='i',
    init_value=Literal(value=1),
    step=Literal(value=2),
    until_value=Literal(value=10),
    body=...
)
```

## Enumerations

### MessageType
- `UNARY` - No arguments (e.g., `size`, `isEmpty`)
- `BINARY` - One argument, operator-style (e.g., `+`, `<`)
- `KEYWORD` - Named arguments (e.g., `at:put:`)

### ParameterMode
- `BY_VALUE` - Call-by-value (evaluated once)
- `BY_NAME` - Call-by-name (thunk, re-evaluated)
- `BY_RESULT` - Result returned via parameter

### CollectionType
- `ARRAY`, `ORDERED_COLLECTION`, `SET`, `BAG`
- `DICTIONARY`, `SORTED_COLLECTION`, `LINKED_LIST`

## Helper Functions

```python
from ir.oop import (
    make_unary_message,
    make_binary_message,
    make_keyword_message,
    make_block,
    make_conditional,
    make_while_loop,
    parse_selector,
    is_binary_selector,
    count_arguments
)

# Quick message creation
msg = make_unary_message(Variable(name='obj'), 'size')
msg = make_binary_message(Literal(value=3), '+', Literal(value=4))

# Parse selector type
msg_type, keywords = parse_selector('at:put:')  # (KEYWORD, ['at:', 'put:'])

# Count arguments
count_arguments('at:put:')  # 2
```

## Control Structures

### Smalltalk (as messages)

```python
from ir.oop import ConditionalMessage, LoopMessage, IterationMessage

# condition ifTrue: [yes] ifFalse: [no]
cond = ConditionalMessage(
    condition=...,
    true_block=Block(...),
    false_block=Block(...)
)

# [condition] whileTrue: [body]
loop = LoopMessage(
    loop_type='whileTrue:',
    condition_or_count=...,
    body=Block(...)
)

# n timesRepeat: [body]
loop = LoopMessage(
    loop_type='timesRepeat:',
    condition_or_count=Literal(value=5),
    body=Block(...)
)
```

### ALGOL

```python
from ir.oop import AlgolIf, AlgolFor, AlgolSwitch

# if condition then stmt1 else stmt2
if_stmt = AlgolIf(
    condition=...,
    then_branch=...,
    else_branch=...
)

# for i := 1 step 1 until n do stmt
for_loop = AlgolFor(
    variable='i',
    init_value=Literal(value=1),
    step=Literal(value=1),
    until_value=Variable(name='n'),
    body=...
)

# switch S := L1, L2, L3
switch = AlgolSwitch(
    name='S',
    labels=['L1', 'L2', 'L3']
)
```

## Related Modules

- `targets/oop/smalltalk_emitter.py` - Smalltalk code generation
- `targets/oop/algol_emitter.py` - ALGOL code generation
- `tests/ir/test_oop_ir.py` - IR tests
- `tests/codegen/test_oop_emitters.py` - Emitter tests

## References

- Smalltalk-80: The Language (Goldberg & Robson)
- Revised Report on ALGOL 60
- STUNIR HLI Phase 10B
