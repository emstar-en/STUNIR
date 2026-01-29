#!/usr/bin/env python3
"""STUNIR BEAM VM Emitters Package.

This package provides code emitters for BEAM VM languages,
including Erlang and Elixir.

The BEAM (Bogdan/Björn's Erlang Abstract Machine) is a virtual machine
designed for building highly concurrent, fault-tolerant systems using
the actor model.

Supported Languages:
    - Erlang: Original BEAM language, OTP, telecom heritage
    - Elixir: Modern syntax, macros, pipe operator, Phoenix framework

Usage:
    from targets.beam import ErlangEmitter, ElixirEmitter
    from ir.actor import ActorModule, FunctionDef
    
    # Emit Erlang code
    erlang_emitter = ErlangEmitter()
    erlang_code = erlang_emitter.emit_module(module)
    
    # Emit Elixir code
    elixir_emitter = ElixirEmitter()
    elixir_code = elixir_emitter.emit_module(module)
"""

from targets.beam.base import (
    BEAMEmitterBase,
    EmitterResult,
    canonical_json,
    compute_sha256,
)

from targets.beam.erlang_emitter import ErlangEmitter
from targets.beam.elixir_emitter import (
    ElixirEmitter,
    PipeExpr,
    WithExpr,
    StructExpr,
    SigilExpr,
)

__all__ = [
    # Base
    'BEAMEmitterBase',
    'EmitterResult',
    'canonical_json',
    'compute_sha256',
    # Emitters
    'ErlangEmitter',
    'ElixirEmitter',
    # Elixir-specific expressions
    'PipeExpr',
    'WithExpr',
    'StructExpr',
    'SigilExpr',
]
