# Actor Model IR

**Package:** `ir.actor`  
**Version:** 1.0.0  
**Phase:** 8C (BEAM VM - Erlang/Elixir)

## Overview

The Actor Model IR provides intermediate representation constructs for actor-based concurrency, as used in BEAM VM languages (Erlang, Elixir). This IR captures the essential elements of the actor model:

- **Processes**: Lightweight concurrent execution units
- **Message Passing**: Asynchronous communication via send/receive
- **Pattern Matching**: Matching messages in receive blocks
- **OTP Behaviors**: GenServer, Supervisor, Application patterns
- **Fault Tolerance**: Supervision trees and restart strategies

## Architecture

```
ir/actor/
├── __init__.py         # Package exports
├── actor_ir.py         # Core classes and expressions
├── process.py          # Process definitions and spawning
├── message.py          # Message passing and patterns
├── otp.py              # OTP behavior implementations
└── README.md           # This file
```

## Core Concepts

### Actor Module

The top-level container for actor-based code:

```python
from ir.actor import ActorModule, FunctionDef, FunctionClause, AtomExpr

module = ActorModule(
    name='my_server',
    exports=['start/0', 'handle/1'],
    functions=[
        FunctionDef(
            name='start',
            arity=0,
            exported=True,
            clauses=[...]
        )
    ],
    behaviors=[...]  # OTP behaviors
)
```

### Processes

Process spawning and management:

```python
from ir.actor import SpawnExpr, SpawnStrategy, SelfExpr, LinkExpr

# Spawn a new process
spawn = SpawnExpr(
    strategy=SpawnStrategy.SPAWN_LINK,
    module='my_module',
    function='loop',
    args=[LiteralExpr(value=0)]
)

# Get current process PID
self_pid = SelfExpr()

# Link to another process
link = LinkExpr(target=VarExpr(name='pid'))
```

### Message Passing

Sending and receiving messages:

```python
from ir.actor import SendExpr, ReceiveExpr, ReceiveClause
from ir.actor import TuplePattern, AtomPattern, VarPattern

# Send a message
send = SendExpr(
    target=VarExpr(name='pid'),
    message=TupleExpr(elements=[
        AtomExpr(value='hello'),
        VarExpr(name='data')
    ])
)

# Receive messages
receive = ReceiveExpr(
    clauses=[
        ReceiveClause(
            pattern=TuplePattern(elements=[
                AtomPattern(value='request'),
                VarPattern(name='from'),
                VarPattern(name='data')
            ]),
            body=[...]
        )
    ],
    timeout=LiteralExpr(value=5000),
    timeout_body=[AtomExpr(value='timeout')]
)
```

### Pattern Matching

Patterns for matching in receive and function clauses:

```python
from ir.actor import (
    WildcardPattern,    # _
    VarPattern,         # Variable binding
    LiteralPattern,     # Literal values
    TuplePattern,       # {A, B, C}
    ListPattern,        # [H|T] or [A, B, C]
    MapPattern,         # #{key := Value}
    BinaryPattern,      # <<A:8, B:16>>
)

# Tuple pattern with nested patterns
pattern = TuplePattern(elements=[
    AtomPattern(value='ok'),
    VarPattern(name='result'),
    WildcardPattern()
])

# List pattern with tail
list_pattern = ListPattern(
    elements=[VarPattern(name='head')],
    tail=VarPattern(name='tail')
)
```

### OTP Behaviors

#### GenServer

```python
from ir.actor import (
    GenServerImpl, GenServerState, Callback,
    BehaviorImpl, BehaviorType
)

gen_server = GenServerImpl(
    name='counter',
    state=GenServerState(
        initial_state=LiteralExpr(value=0)
    ),
    handle_call_callbacks=[
        Callback(
            name='handle_call',
            args=['{get}', '_From', 'State'],
            body=[TupleExpr(elements=[
                AtomExpr(value='reply'),
                VarExpr(name='State'),
                VarExpr(name='State')
            ])]
        )
    ]
)

behavior = BehaviorImpl(
    behavior_type=BehaviorType.GEN_SERVER,
    implementation=gen_server
)
```

#### Supervisor

```python
from ir.actor import (
    SupervisorImpl, SupervisorFlags, SupervisorStrategy,
    ChildSpec, ChildRestart, ChildType
)

supervisor = SupervisorImpl(
    name='my_sup',
    flags=SupervisorFlags(
        strategy=SupervisorStrategy.ONE_FOR_ONE,
        intensity=3,
        period=5
    ),
    children=[
        ChildSpec(
            id='worker1',
            start=('worker_module', 'start_link', []),
            restart=ChildRestart.PERMANENT,
            type=ChildType.WORKER
        )
    ]
)
```

## Expressions

Common expression types:

| Expression | Description | Example |
|------------|-------------|--------|
| `LiteralExpr` | Literal values | `42`, `3.14`, `"hello"` |
| `AtomExpr` | Atom values | `ok`, `error` |
| `VarExpr` | Variable reference | `State`, `Pid` |
| `TupleExpr` | Tuple | `{ok, Result}` |
| `ListExpr` | List | `[1, 2, 3]` |
| `MapExpr` | Map | `#{key => value}` |
| `CallExpr` | Function call | `module:function(args)` |
| `CaseExpr` | Case expression | `case X of ... end` |
| `FunExpr` | Anonymous function | `fun(X) -> X end` |
| `TryExpr` | Try-catch | `try ... catch ... end` |

## Serialization

All IR nodes support serialization to dictionaries:

```python
module = ActorModule(name='test', ...)
d = module.to_dict()
# Can be serialized to JSON
```

## Related Components

- **targets/beam/erlang_emitter.py**: Generates Erlang code from Actor IR
- **targets/beam/elixir_emitter.py**: Generates Elixir code from Actor IR

## Testing

```bash
python -m pytest tests/ir/test_actor_ir.py -v
```

## See Also

- [BEAM Emitters README](../../targets/beam/README.md)
- [HLI Document](../../../stunir_implementation_framework/phase8/HLI_BEAM_VM.md)
