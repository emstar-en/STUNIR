"""STUNIR Core Emitters"""

from .embedded import EmbeddedEmitter
from .gpu import GPUEmitter
from .wasm import WebAssemblyEmitter
from .assembly import AssemblyEmitter
from .polyglot import PolyglotEmitter

__all__ = ["EmbeddedEmitter", "GPUEmitter", "WebAssemblyEmitter", "AssemblyEmitter", "PolyglotEmitter"]
