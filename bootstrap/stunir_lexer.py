"""
STUNIR Lexer Specification.

Defines the complete lexer specification for the STUNIR language using
the LexerSpec and TokenSpec classes from Phase 6C.

The STUNIR lexer recognizes:
- 37 keywords (module, type, function, ir, target, etc.)
- 30 operators (+, -, *, /, ==, !=, &&, ||, ->, =>, etc.)
- 12 punctuation symbols ((, ), {, }, [, ], ;, etc.)
- Identifiers, integer/float/string literals
- Comments (line and block)
- Whitespace (skipped)
"""

from typing import List, Dict

from ir.lexer.token_spec import (
    LexerSpec,
    TokenSpec,
    TokenType,
)


# STUNIR Keywords (37 total)
STUNIR_KEYWORDS: Dict[str, str] = {
    # Module keywords
    'module': 'KW_MODULE',
    'import': 'KW_IMPORT',
    'from': 'KW_FROM',
    'export': 'KW_EXPORT',
    'as': 'KW_AS',
    
    # Declaration keywords
    'type': 'KW_TYPE',
    'function': 'KW_FUNCTION',
    'ir': 'KW_IR',
    'target': 'KW_TARGET',
    'const': 'KW_CONST',
    
    # Type keywords (primitive types)
    'i8': 'KW_I8',
    'i16': 'KW_I16',
    'i32': 'KW_I32',
    'i64': 'KW_I64',
    'u8': 'KW_U8',
    'u16': 'KW_U16',
    'u32': 'KW_U32',
    'u64': 'KW_U64',
    'f32': 'KW_F32',
    'f64': 'KW_F64',
    'bool': 'KW_BOOL',
    'string': 'KW_STRING',
    'void': 'KW_VOID',
    'any': 'KW_ANY',
    
    # Statement keywords
    'let': 'KW_LET',
    'var': 'KW_VAR',
    'if': 'KW_IF',
    'else': 'KW_ELSE',
    'while': 'KW_WHILE',
    'for': 'KW_FOR',
    'in': 'KW_IN',
    'match': 'KW_MATCH',
    'return': 'KW_RETURN',
    'emit': 'KW_EMIT',
    
    # IR keywords
    'child': 'KW_CHILD',
    'op': 'KW_OP',
    
    # Literals
    'true': 'KW_TRUE',
    'false': 'KW_FALSE',
    'null': 'KW_NULL',
}


# STUNIR Token Specifications
STUNIR_TOKENS: List[TokenSpec] = [
    # ===========================================
    # Skip tokens (highest priority - processed first, then skipped)
    # ===========================================
    TokenSpec(
        name='COMMENT_LINE',
        pattern=r'//[^\n]*',
        priority=100,
        token_type=TokenType.COMMENT,
        skip=True
    ),
    TokenSpec(
        name='COMMENT_BLOCK',
        pattern=r'/\*([^*]|\*[^/])*\*/',
        priority=100,
        token_type=TokenType.COMMENT,
        skip=True
    ),
    TokenSpec(
        name='WHITESPACE',
        pattern=r'[ \t\r\n]+',
        priority=99,
        token_type=TokenType.WHITESPACE,
        skip=True
    ),
    
    # ===========================================
    # Literals (high priority)
    # ===========================================
    TokenSpec(
        name='FLOAT_LITERAL',
        pattern=r'[0-9]+\.[0-9]+([eE][+-]?[0-9]+)?',
        priority=80,
        token_type=TokenType.LITERAL
    ),
    TokenSpec(
        name='INTEGER_LITERAL',
        pattern=r'[0-9]+',
        priority=79,
        token_type=TokenType.LITERAL
    ),
    TokenSpec(
        name='STRING_LITERAL',
        pattern=r'"([^"\\]|\\.)*"',
        priority=78,
        token_type=TokenType.LITERAL
    ),
    
    # ===========================================
    # Identifiers (keywords are handled in post-processing)
    # ===========================================
    TokenSpec(
        name='IDENTIFIER',
        pattern=r'[a-zA-Z_][a-zA-Z0-9_]*',
        priority=50,
        token_type=TokenType.IDENTIFIER
    ),
    
    # ===========================================
    # Multi-character operators (before single-char)
    # ===========================================
    TokenSpec(
        name='ARROW',
        pattern=r'->',
        priority=45,
        token_type=TokenType.OPERATOR
    ),
    TokenSpec(
        name='FAT_ARROW',
        pattern=r'=>',
        priority=45,
        token_type=TokenType.OPERATOR
    ),
    TokenSpec(
        name='LE',
        pattern=r'<=',
        priority=45,
        token_type=TokenType.OPERATOR
    ),
    TokenSpec(
        name='GE',
        pattern=r'>=',
        priority=45,
        token_type=TokenType.OPERATOR
    ),
    TokenSpec(
        name='EQ',
        pattern=r'==',
        priority=45,
        token_type=TokenType.OPERATOR
    ),
    TokenSpec(
        name='NE',
        pattern=r'!=',
        priority=45,
        token_type=TokenType.OPERATOR
    ),
    TokenSpec(
        name='AND',
        pattern=r'&&',
        priority=45,
        token_type=TokenType.OPERATOR
    ),
    TokenSpec(
        name='OR',
        pattern=r'\|\|',
        priority=45,
        token_type=TokenType.OPERATOR
    ),
    TokenSpec(
        name='LSHIFT',
        pattern=r'<<',
        priority=45,
        token_type=TokenType.OPERATOR
    ),
    TokenSpec(
        name='RSHIFT',
        pattern=r'>>',
        priority=45,
        token_type=TokenType.OPERATOR
    ),
    TokenSpec(
        name='PLUS_EQ',
        pattern=r'\+=',
        priority=44,
        token_type=TokenType.OPERATOR
    ),
    TokenSpec(
        name='MINUS_EQ',
        pattern=r'-=',
        priority=44,
        token_type=TokenType.OPERATOR
    ),
    TokenSpec(
        name='STAR_EQ',
        pattern=r'\*=',
        priority=44,
        token_type=TokenType.OPERATOR
    ),
    TokenSpec(
        name='SLASH_EQ',
        pattern=r'/=',
        priority=44,
        token_type=TokenType.OPERATOR
    ),
    TokenSpec(
        name='PERCENT_EQ',
        pattern=r'%=',
        priority=44,
        token_type=TokenType.OPERATOR
    ),
    
    # ===========================================
    # Single-character operators
    # ===========================================
    TokenSpec(
        name='PLUS',
        pattern=r'\+',
        priority=30,
        token_type=TokenType.OPERATOR
    ),
    TokenSpec(
        name='MINUS',
        pattern=r'-',
        priority=30,
        token_type=TokenType.OPERATOR
    ),
    TokenSpec(
        name='STAR',
        pattern=r'\*',
        priority=30,
        token_type=TokenType.OPERATOR
    ),
    TokenSpec(
        name='SLASH',
        pattern=r'/',
        priority=30,
        token_type=TokenType.OPERATOR
    ),
    TokenSpec(
        name='PERCENT',
        pattern=r'%',
        priority=30,
        token_type=TokenType.OPERATOR
    ),
    TokenSpec(
        name='LT',
        pattern=r'<',
        priority=30,
        token_type=TokenType.OPERATOR
    ),
    TokenSpec(
        name='GT',
        pattern=r'>',
        priority=30,
        token_type=TokenType.OPERATOR
    ),
    TokenSpec(
        name='NOT',
        pattern=r'!',
        priority=30,
        token_type=TokenType.OPERATOR
    ),
    TokenSpec(
        name='AMPERSAND',
        pattern=r'&',
        priority=30,
        token_type=TokenType.OPERATOR
    ),
    TokenSpec(
        name='PIPE',
        pattern=r'\|',
        priority=30,
        token_type=TokenType.OPERATOR
    ),
    TokenSpec(
        name='CARET',
        pattern=r'\^',
        priority=30,
        token_type=TokenType.OPERATOR
    ),
    TokenSpec(
        name='TILDE',
        pattern=r'~',
        priority=30,
        token_type=TokenType.OPERATOR
    ),
    TokenSpec(
        name='QUESTION',
        pattern=r'\?',
        priority=30,
        token_type=TokenType.OPERATOR
    ),
    TokenSpec(
        name='EQUALS',
        pattern=r'=',
        priority=30,
        token_type=TokenType.OPERATOR
    ),
    
    # ===========================================
    # Punctuation
    # ===========================================
    TokenSpec(
        name='LPAREN',
        pattern=r'\(',
        priority=20,
        token_type=TokenType.PUNCTUATION
    ),
    TokenSpec(
        name='RPAREN',
        pattern=r'\)',
        priority=20,
        token_type=TokenType.PUNCTUATION
    ),
    TokenSpec(
        name='LBRACE',
        pattern=r'\{',
        priority=20,
        token_type=TokenType.PUNCTUATION
    ),
    TokenSpec(
        name='RBRACE',
        pattern=r'\}',
        priority=20,
        token_type=TokenType.PUNCTUATION
    ),
    TokenSpec(
        name='LBRACKET',
        pattern=r'\[',
        priority=20,
        token_type=TokenType.PUNCTUATION
    ),
    TokenSpec(
        name='RBRACKET',
        pattern=r'\]',
        priority=20,
        token_type=TokenType.PUNCTUATION
    ),
    TokenSpec(
        name='COMMA',
        pattern=r',',
        priority=20,
        token_type=TokenType.PUNCTUATION
    ),
    TokenSpec(
        name='SEMICOLON',
        pattern=r';',
        priority=20,
        token_type=TokenType.PUNCTUATION
    ),
    TokenSpec(
        name='DOT',
        pattern=r'\.',
        priority=20,
        token_type=TokenType.PUNCTUATION
    ),
    TokenSpec(
        name='COLON',
        pattern=r':',
        priority=20,
        token_type=TokenType.PUNCTUATION
    ),
]


class STUNIRLexerBuilder:
    """
    Builder for STUNIR lexer specification.
    
    Creates a complete LexerSpec for the STUNIR language using
    the token specifications defined above.
    
    Usage:
        builder = STUNIRLexerBuilder()
        spec = builder.build()
    """
    
    def __init__(self):
        """Initialize the builder."""
        self._tokens = STUNIR_TOKENS.copy()
        self._keywords = STUNIR_KEYWORDS.copy()
    
    def build(self) -> LexerSpec:
        """
        Build complete STUNIR lexer specification.
        
        Returns:
            LexerSpec configured for STUNIR language
        """
        return LexerSpec(
            name='STUNIRLexer',
            tokens=self._tokens,
            keywords=self._keywords,
            case_sensitive=True
        )
    
    def get_keywords(self) -> Dict[str, str]:
        """Get STUNIR keywords dictionary."""
        return self._keywords.copy()
    
    def get_tokens(self) -> List[TokenSpec]:
        """Get STUNIR token specifications."""
        return self._tokens.copy()
    
    def validate(self) -> List[str]:
        """
        Validate the lexer specification.
        
        Returns:
            List of error messages (empty if valid)
        """
        spec = self.build()
        return spec.validate()


def create_stunir_lexer_spec() -> LexerSpec:
    """
    Create STUNIR lexer specification.
    
    Convenience function for creating the STUNIR lexer spec.
    
    Returns:
        LexerSpec for STUNIR language
    """
    return STUNIRLexerBuilder().build()
