# BEAM VM Emitters

**Package:** `targets.beam`  
**Version:** 1.0.0  
**Phase:** 8C (BEAM VM - Erlang/Elixir)

## Overview

The BEAM VM Emitters package provides code generation for BEAM virtual machine languages, specifically Erlang and Elixir. Both languages share the same underlying runtime (BEAM) and follow the actor model for concurrent programming.

## Architecture

```
targets/beam/
├── __init__.py           # Package exports
├── base.py               # Shared base emitter class
├── erlang_emitter.py     # Erlang code generator
├── elixir_emitter.py     # Elixir code generator
└── README.md             # This file
```

## Quick Start

### Erlang

```python
from ir.actor import ActorModule, FunctionDef, FunctionClause, AtomExpr
from targets.beam import ErlangEmitter

# Create module
module = ActorModule(
    name='hello',
    exports=['world/0'],
    functions=[
        FunctionDef(
            name='world',
            arity=0,
            exported=True,
            clauses=[
                FunctionClause(
                    args=[],
                    body=[AtomExpr(value='hello_world')]
                )
            ]
        )
    ]
)

# Generate Erlang code
emitter = ErlangEmitter()
result = emitter.emit_module(module)

print(result.code)
# Output:
# -module(hello).
# 
# -export([world/0]).
# 
# world() ->
#     hello_world.
```

### Elixir

```python
from ir.actor import ActorModule, FunctionDef, FunctionClause, AtomExpr
from targets.beam import ElixirEmitter

# Same module as above
emitter = ElixirEmitter()
result = emitter.emit_module(module)

print(result.code)
# Output:
# defmodule Hello do
#     def world, do: :hello_world
# end
```

## Features

### Erlang Emitter

| Feature | Support | Example |
|---------|---------|--------|
| Module definitions | ✓ | `-module(name).` |
| Exports | ✓ | `-export([fun/arity]).` |
| Functions | ✓ | `fun(Args) -> Body.` |
| Pattern matching | ✓ | `{ok, Result}` |
| Guards | ✓ | `when is_integer(X)` |
| Process spawning | ✓ | `spawn`, `spawn_link` |
| Message passing | ✓ | `Pid ! Msg`, `receive` |
| GenServer | ✓ | `-behaviour(gen_server).` |
| Supervisor | ✓ | `-behaviour(supervisor).` |
| Binary patterns | ✓ | `<<A:8, B:16>>` |
| List comprehensions | ✓ | `[X || X <- List]` |

### Elixir Emitter

| Feature | Support | Example |
|---------|---------|--------|
| Module definitions | ✓ | `defmodule Name do` |
| Public functions | ✓ | `def fun(args)` |
| Private functions | ✓ | `defp fun(args)` |
| Pattern matching | ✓ | `{:ok, result}` |
| Guards | ✓ | `when is_integer(x)` |
| Process spawning | ✓ | `spawn`, `spawn_link` |
| Message passing | ✓ | `send(pid, msg)`, `receive` |
| GenServer | ✓ | `use GenServer` |
| Supervisor | ✓ | `use Supervisor` |
| Pipe operator | ✓ | `data \|> transform()` |
| For comprehensions | ✓ | `for x <- list, do: x` |

## OTP Behaviors

### GenServer

```python
from ir.actor import (
    ActorModule, GenServerImpl, GenServerState,
    BehaviorImpl, BehaviorType, Callback,
    LiteralExpr, TupleExpr, AtomExpr, VarExpr
)
from targets.beam import ErlangEmitter, ElixirEmitter

gen_server = GenServerImpl(
    name='counter',
    state=GenServerState(
        initial_state=LiteralExpr(value=0)
    ),
    handle_call_callbacks=[
        Callback(
            name='handle_call',
            args=['get', '_From', 'State'],
            body=[TupleExpr(elements=[
                AtomExpr(value='reply'),
                VarExpr(name='State'),
                VarExpr(name='State')
            ])]
        )
    ]
)

module = ActorModule(
    name='counter_server',
    behaviors=[
        BehaviorImpl(
            behavior_type=BehaviorType.GEN_SERVER,
            implementation=gen_server
        )
    ]
)

# Erlang output includes:
# -behaviour(gen_server).
# -export([init/1, handle_call/3, ...]).
# init(_Args) -> {ok, 0}.
# handle_call(get, _From, State) -> {reply, State, State}.

# Elixir output includes:
# use GenServer
# @impl true
# def init(_args), do: {:ok, 0}
# def handle_call(:get, _from, state), do: {:reply, state, state}
```

### Supervisor

```python
from ir.actor import (
    SupervisorImpl, SupervisorFlags, SupervisorStrategy,
    ChildSpec, ChildRestart, ChildType
)

supervisor = SupervisorImpl(
    flags=SupervisorFlags(
        strategy=SupervisorStrategy.ONE_FOR_ONE,
        intensity=3,
        period=5
    ),
    children=[
        ChildSpec(
            id='worker',
            start=('MyWorker', 'start_link', []),
            restart=ChildRestart.PERMANENT,
            type=ChildType.WORKER
        )
    ]
)
```

## Manifest Generation

Both emitters generate manifests with metadata about the generated code:

```python
result = emitter.emit_module(module)
print(result.manifest)
# {
#     'schema': 'stunir.manifest.erlang.v1',
#     'module': 'hello',
#     'language': 'erlang',
#     'emitter_version': '1.0.0',
#     'code_hash': 'sha256...',
#     'code_size': 123,
#     'exports': ['world/0'],
#     'behaviors': []
# }
```

## Language Differences

| Aspect | Erlang | Elixir |
|--------|--------|--------|
| Variables | `Uppercase` | `lowercase` |
| Atoms | `atom` | `:atom` |
| Module names | `module_name` | `ModuleName` |
| Functions | `fun_name/arity` | `fun_name/arity` |
| Send | `Pid ! Msg` | `send(pid, msg)` |
| Maps | `#{key => val}` | `%{key => val}` |
| Strings | `"list"` | `"binary"` |

## Elixir-Specific Features

### Pipe Operator

```python
from targets.beam.elixir_emitter import PipeExpr

pipe = PipeExpr(
    left=VarExpr(name='data'),
    right=CallExpr(function='transform', args=[])
)
# Emits: data |> transform()
```

### With Expression

```python
from targets.beam.elixir_emitter import WithExpr

with_expr = WithExpr(
    clauses=[(pattern1, expr1), (pattern2, expr2)],
    body=[...],
    else_clauses=[...]
)
```

## Testing

```bash
# Run BEAM emitter tests
python -m pytest tests/codegen/test_beam_emitters.py -v

# Run all tests
python -m pytest tests/ir/test_actor_ir.py tests/codegen/test_beam_emitters.py -v
```

## Example: Chat Server

### Erlang

```erlang
%% Generated by STUNIR Erlang Emitter
-module(chat_server).
-export([start/0, join/1]).

start() ->
    spawn(fun() -> loop([]) end).

loop(Clients) ->
    receive
        {join, Pid} ->
            loop([Pid|Clients]);
        {message, From, Msg} ->
            [C ! {chat, From, Msg} || C <- Clients],
            loop(Clients)
    end.
```

### Elixir

```elixir
# Generated by STUNIR Elixir Emitter
defmodule ChatServer do
  def start do
    spawn(fn -> loop([]) end)
  end

  defp loop(clients) do
    receive do
      {:join, pid} ->
        loop([pid | clients])
      {:message, from, msg} ->
        for c <- clients, do: send(c, {:chat, from, msg})
        loop(clients)
    end
  end
end
```

## Related Components

- **ir/actor/**: Actor Model IR definitions
- **ir/actor/README.md**: Actor IR documentation

## See Also

- [HLI Document](../../../stunir_implementation_framework/phase8/HLI_BEAM_VM.md)
- [Erlang Documentation](https://www.erlang.org/doc/)
- [Elixir Documentation](https://hexdocs.pm/elixir/)
