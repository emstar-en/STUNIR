#!/usr/bin/env python3
"""STUNIR Code Generation Package.

Provides code generators for multiple target languages.

Languages supported:
- Common Lisp (ANSI CL)
- Scheme (R7RS)  
- Clojure (JVM)
- Racket
- C99
- C++
- Go
- Java
- JavaScript
- Python
- Rust
- TypeScript
"""

# Existing generators
from .c99_generator import C99Generator
from .cpp_generator import CppGenerator
from .go_generator import GoGenerator
from .java_generator import JavaGenerator
from .javascript_generator import JavaScriptGenerator
from .python_generator import PythonGenerator
from .rust_generator import RustGenerator
from .typescript_generator import TypeScriptGenerator

# Lisp family emitters (Phase 5A)
try:
    from targets.lisp.common_lisp.emitter import CommonLispEmitter, CommonLispConfig
    from targets.lisp.scheme.emitter import SchemeEmitter, SchemeEmitterConfig
    from targets.lisp.clojure.emitter import ClojureEmitter, ClojureEmitterConfig
    from targets.lisp.racket.emitter import RacketEmitter, RacketEmitterConfig
    LISP_EMITTERS_AVAILABLE = True
except ImportError:
    LISP_EMITTERS_AVAILABLE = False

__all__ = [
    # Existing generators
    'C99Generator',
    'CppGenerator',
    'GoGenerator',
    'JavaGenerator',
    'JavaScriptGenerator',
    'PythonGenerator',
    'RustGenerator',
    'TypeScriptGenerator',
]

# Add Lisp emitters if available
if LISP_EMITTERS_AVAILABLE:
    __all__.extend([
        'CommonLispEmitter',
        'CommonLispConfig',
        'SchemeEmitter',
        'SchemeEmitterConfig',
        'ClojureEmitter',
        'ClojureEmitterConfig',
        'RacketEmitter',
        'RacketEmitterConfig',
    ])
