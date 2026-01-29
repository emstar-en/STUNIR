#!/usr/bin/env python3
"""STUNIR Code Generation Package.

This package provides code generation capabilities for translating STUNIR IR
to multiple target languages. Part of Phase 3 (Advanced Code Generation) of the
STUNIR Enhancement Integration.

Supported Languages:
- Python (with type hints, match statements)
- Rust (with ownership/borrowing, match expressions)
- Go (with idiomatic patterns, switch statements)
- C99 (with fixed-width types, do-while loops)
- JavaScript (ES6+, arrow functions)
- TypeScript (full type system)
- Java (lambdas, generics)
- C++ (C++17/20, smart pointers, templates)

Architecture:
- StatementTranslator: Base class for statement translation with control flow
- ExpressionTranslator: Base class for expression translation with complex expressions
- Language-specific generators extend both translators

Usage:
    from tools.codegen import (
        PythonCodeGenerator,
        RustCodeGenerator,
        GoCodeGenerator,
        C99CodeGenerator,
        JavaScriptCodeGenerator,
        TypeScriptCodeGenerator,
        JavaCodeGenerator,
        CppCodeGenerator,
        get_generator
    )
    
    # Create generator with optional enhancement context
    generator = get_generator('python', enhancement_context=ctx)
    
    # Generate function code
    code = generator.generate_function(func_ir)
    
    # Generate module code
    module_code = generator.generate_module(module_ir)

Example IR Structure with Control Flow:
    func_ir = {
        "name": "factorial",
        "params": [{"name": "n", "type": "i32"}],
        "return_type": "i32",
        "body": [
            {"type": "if", "condition": {"type": "binary", "op": "<=", 
             "left": {"type": "var", "name": "n"}, "right": 1},
             "then": [{"type": "return", "value": 1}],
             "else": [{"type": "return", "value": {
                 "type": "binary", "op": "*",
                 "left": {"type": "var", "name": "n"},
                 "right": {"type": "call", "func": "factorial", 
                          "args": [{"type": "binary", "op": "-",
                                   "left": {"type": "var", "name": "n"},
                                   "right": 1}]}
             }}]}
        ]
    }
"""

from .statement_translator import StatementTranslator
from .expression_translator import ExpressionTranslator, ARITHMETIC_OPS, COMPARISON_OPS, LOGICAL_OPS
from .python_generator import (
    PythonCodeGenerator,
    PythonStatementTranslator,
    PythonExpressionTranslator
)
from .rust_generator import (
    RustCodeGenerator,
    RustStatementTranslator,
    RustExpressionTranslator
)
from .go_generator import (
    GoCodeGenerator,
    GoStatementTranslator,
    GoExpressionTranslator
)
from .c99_generator import (
    C99CodeGenerator,
    C99StatementTranslator,
    C99ExpressionTranslator
)
from .javascript_generator import (
    JavaScriptCodeGenerator,
    JavaScriptStatementTranslator,
    JavaScriptExpressionTranslator
)
from .typescript_generator import (
    TypeScriptCodeGenerator,
    TypeScriptStatementTranslator,
    TypeScriptExpressionTranslator
)
from .java_generator import (
    JavaCodeGenerator,
    JavaStatementTranslator,
    JavaExpressionTranslator
)
from .cpp_generator import (
    CppCodeGenerator,
    CppStatementTranslator,
    CppExpressionTranslator
)

__all__ = [
    # Base classes
    'StatementTranslator',
    'ExpressionTranslator',
    'ARITHMETIC_OPS',
    'COMPARISON_OPS',
    'LOGICAL_OPS',
    
    # Python
    'PythonCodeGenerator',
    'PythonStatementTranslator',
    'PythonExpressionTranslator',
    
    # Rust
    'RustCodeGenerator',
    'RustStatementTranslator',
    'RustExpressionTranslator',
    
    # Go
    'GoCodeGenerator',
    'GoStatementTranslator',
    'GoExpressionTranslator',
    
    # C99
    'C99CodeGenerator',
    'C99StatementTranslator',
    'C99ExpressionTranslator',
    
    # JavaScript
    'JavaScriptCodeGenerator',
    'JavaScriptStatementTranslator',
    'JavaScriptExpressionTranslator',
    
    # TypeScript
    'TypeScriptCodeGenerator',
    'TypeScriptStatementTranslator',
    'TypeScriptExpressionTranslator',
    
    # Java
    'JavaCodeGenerator',
    'JavaStatementTranslator',
    'JavaExpressionTranslator',
    
    # C++
    'CppCodeGenerator',
    'CppStatementTranslator',
    'CppExpressionTranslator',
    
    # Helper functions
    'get_generator',
    'get_supported_targets',
]

# Version info
__version__ = '0.3.0'
__phase__ = 'Phase 3: Advanced Code Generation'


def get_generator(target: str, enhancement_context=None, **kwargs):
    """Get a code generator for the specified target language.
    
    Args:
        target: Target language ('python', 'rust', 'go', 'c99', 'javascript', 
                'typescript', 'java', 'cpp').
        enhancement_context: Optional EnhancementContext for type info.
        **kwargs: Additional generator-specific options.
        
    Returns:
        Code generator instance for the target language.
        
    Raises:
        ValueError: If target language is not supported.
    """
    generators = {
        'python': PythonCodeGenerator,
        'py': PythonCodeGenerator,
        'rust': RustCodeGenerator,
        'rs': RustCodeGenerator,
        'go': GoCodeGenerator,
        'golang': GoCodeGenerator,
        'c99': C99CodeGenerator,
        'c': C99CodeGenerator,
        'c89': C99CodeGenerator,
        'javascript': JavaScriptCodeGenerator,
        'js': JavaScriptCodeGenerator,
        'typescript': TypeScriptCodeGenerator,
        'ts': TypeScriptCodeGenerator,
        'java': JavaCodeGenerator,
        'cpp': CppCodeGenerator,
        'c++': CppCodeGenerator,
        'cxx': CppCodeGenerator,
    }
    
    target_lower = target.lower()
    if target_lower not in generators:
        supported = sorted(set(generators.keys()))
        raise ValueError(f"Unsupported target: {target}. Supported: {supported}")
    
    return generators[target_lower](enhancement_context=enhancement_context, **kwargs)


def get_supported_targets() -> list:
    """Get list of supported target languages.
    
    Returns:
        List of supported target language names.
    """
    return ['python', 'rust', 'go', 'c99', 'javascript', 'typescript', 'java', 'cpp']
