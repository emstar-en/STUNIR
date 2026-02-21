"""
STUNIR Lexer Generator Package.

Provides lexer generation infrastructure using finite automata theory:
- Token specification
- Regular expression parsing
- NFA construction (Thompson's algorithm)
- DFA construction (Subset construction)
- DFA minimization (Hopcroft's algorithm)
- Lexer simulation

Usage:
    from ir.lexer import LexerSpec, TokenSpec, LexerGenerator
    
    spec = LexerSpec("MyLexer", [
        TokenSpec("INT", "[0-9]+"),
        TokenSpec("ID", "[a-z]+"),
        TokenSpec("WS", "[ \\t\\n]+", skip=True)
    ])
    
    gen = LexerGenerator(spec)
    dfa = gen.generate()
    
    # Use emitter to generate code
    from targets.lexer import PythonLexerEmitter
    emitter = PythonLexerEmitter()
    code = gen.emit(emitter)
"""

# Token specification
from .token_spec import (
    TokenType,
    TokenSpec,
    LexerSpec,
    Token,
    LexerError,
    COMMON_TOKENS,
)

# Regex parsing
from .regex import (
    RegexNode,
    EpsilonNode,
    CharNode,
    AnyCharNode,
    CharClassNode,
    ConcatNode,
    UnionNode,
    StarNode,
    PlusNode,
    OptionalNode,
    RegexParser,
    RegexError,
    parse_regex,
)

# NFA
from .nfa import (
    EPSILON,
    NFAState,
    NFA,
    NFABuilder,
    combine_nfas,
    build_nfa_from_pattern,
)

# DFA
from .dfa import (
    DFAState,
    DFA,
    MinimizedDFA,
    TransitionTable,
    SubsetConstruction,
    HopcroftMinimizer,
    nfa_to_dfa,
    minimize_dfa,
)

# Lexer generator
from .lexer_generator import (
    LexerGenerator,
    LexerSimulator,
    create_lexer,
    tokenize,
)

__all__ = [
    # Token specification
    'TokenType',
    'TokenSpec',
    'LexerSpec',
    'Token',
    'LexerError',
    'COMMON_TOKENS',
    
    # Regex
    'RegexNode',
    'EpsilonNode',
    'CharNode',
    'AnyCharNode',
    'CharClassNode',
    'ConcatNode',
    'UnionNode',
    'StarNode',
    'PlusNode',
    'OptionalNode',
    'RegexParser',
    'RegexError',
    'parse_regex',
    
    # NFA
    'EPSILON',
    'NFAState',
    'NFA',
    'NFABuilder',
    'combine_nfas',
    'build_nfa_from_pattern',
    
    # DFA
    'DFAState',
    'DFA',
    'MinimizedDFA',
    'TransitionTable',
    'SubsetConstruction',
    'HopcroftMinimizer',
    'nfa_to_dfa',
    'minimize_dfa',
    
    # Lexer generator
    'LexerGenerator',
    'LexerSimulator',
    'create_lexer',
    'tokenize',
]
