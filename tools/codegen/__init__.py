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
from .c99_generator import C99CodeGenerator
from .cpp_generator import CppCodeGenerator
from .go_generator import GoCodeGenerator
from .java_generator import JavaCodeGenerator
from .javascript_generator import JavaScriptCodeGenerator
from .python_generator import PythonCodeGenerator
from .rust_generator import RustCodeGenerator
from .typescript_generator import TypeScriptCodeGenerator

# Aliases for backward compatibility
C99Generator = C99CodeGenerator
CppGenerator = CppCodeGenerator
GoGenerator = GoCodeGenerator
JavaGenerator = JavaCodeGenerator
JavaScriptGenerator = JavaScriptCodeGenerator
PythonGenerator = PythonCodeGenerator
RustGenerator = RustCodeGenerator
TypeScriptGenerator = TypeScriptCodeGenerator

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
    # Code generators (preferred names)
    'C99CodeGenerator',
    'CppCodeGenerator',
    'GoCodeGenerator',
    'JavaCodeGenerator',
    'JavaScriptCodeGenerator',
    'PythonCodeGenerator',
    'RustCodeGenerator',
    'TypeScriptCodeGenerator',
    # Backward compatibility aliases
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


# Helper functions
def get_supported_targets():
    """Get list of supported target languages."""
    return ['python', 'rust', 'go', 'c99', 'javascript', 'typescript', 'java', 'cpp']


def get_generator(target):
    """Get generator instance for a specific target language.
    
    Args:
        target: Target language name (e.g., 'python', 'rust', 'c99')
        
    Returns:
        Generator instance for the specified target
        
    Raises:
        ValueError: If target is not supported
    """
    generators = {
        'python': PythonCodeGenerator,
        'rust': RustCodeGenerator,
        'go': GoCodeGenerator,
        'c99': C99CodeGenerator,
        'javascript': JavaScriptCodeGenerator,
        'typescript': TypeScriptCodeGenerator,
        'java': JavaCodeGenerator,
        'cpp': CppCodeGenerator,
    }
    
    if target.lower() not in generators:
        raise ValueError(f"Unsupported target: {target}. Supported targets: {list(generators.keys())}")
    
    return generators[target.lower()]()


# Export helper functions
__all__.extend(['get_generator', 'get_supported_targets'])


# Export expression translators for testing
from .python_generator import PythonExpressionTranslator
from .rust_generator import RustExpressionTranslator
from .c99_generator import C99ExpressionTranslator
from .cpp_generator import CppExpressionTranslator
from .go_generator import GoExpressionTranslator
from .java_generator import JavaExpressionTranslator
from .javascript_generator import JavaScriptExpressionTranslator
from .typescript_generator import TypeScriptExpressionTranslator

__all__.extend([
    'PythonExpressionTranslator',
    'RustExpressionTranslator',
    'C99ExpressionTranslator',
    'CppExpressionTranslator',
    'GoExpressionTranslator',
    'JavaExpressionTranslator',
    'JavaScriptExpressionTranslator',
    'TypeScriptExpressionTranslator',
])
