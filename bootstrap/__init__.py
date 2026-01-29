"""
STUNIR Bootstrap Compilation Package.

Phase 6D: STUNIR Self-Specification and Bootstrap Compilation

This package provides:
- STUNIRGrammarBuilder: STUNIR grammar specification using Grammar IR
- STUNIRLexerBuilder: STUNIR lexer specification using LexerSpec
- ToolGenerator: Generate parser and lexer from specifications
- BootstrapCompiler: Bootstrap compiler for STUNIR source
- SelfHostValidator: Validate self-hosting capability

The bootstrap system enables STUNIR to be self-hosting by:
1. Defining STUNIR grammar using Grammar IR (Phase 6A)
2. Generating STUNIR parser using Parser Generator (Phase 6B)
3. Generating STUNIR lexer using Lexer Generator (Phase 6C)
4. Parsing STUNIR source files with generated tools

Usage:
    from bootstrap import (
        STUNIRGrammarBuilder,
        STUNIRLexerBuilder,
        ToolGenerator,
        BootstrapCompiler,
        SelfHostValidator
    )
    
    # Build specifications
    grammar = STUNIRGrammarBuilder().build()
    lexer_spec = STUNIRLexerBuilder().build()
    
    # Generate tools
    generator = ToolGenerator()
    generator.generate_all(grammar, lexer_spec)
    
    # Use bootstrap compiler
    compiler = BootstrapCompiler()
    result = compiler.parse(source)
    
    # Validate self-hosting
    validator = SelfHostValidator()
    result = validator.validate()
"""

from .stunir_lexer import (
    STUNIRLexerBuilder,
    STUNIR_KEYWORDS,
    STUNIR_TOKENS,
)

from .stunir_grammar import (
    STUNIRGrammarBuilder,
)

from .generate_tools import (
    ToolGenerator,
    GenerationResult,
)

from .bootstrap_compiler import (
    BootstrapCompiler,
    BootstrapResult,
    STUNIRASTNode,
    STUNIRToken,
)

from .self_host_validator import (
    SelfHostValidator,
    ValidationResult,
)

__all__ = [
    # Lexer
    'STUNIRLexerBuilder',
    'STUNIR_KEYWORDS',
    'STUNIR_TOKENS',
    # Grammar
    'STUNIRGrammarBuilder',
    # Generation
    'ToolGenerator',
    'GenerationResult',
    # Compiler
    'BootstrapCompiler',
    'BootstrapResult',
    'STUNIRASTNode',
    'STUNIRToken',
    # Validation
    'SelfHostValidator',
    'ValidationResult',
]
