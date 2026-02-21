"""
Token Specification Module for STUNIR Lexer Generator.

Provides data structures for defining token specifications including:
- TokenType enumeration
- TokenSpec for individual token definitions
- LexerSpec for complete lexer specifications
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set


class TokenType(Enum):
    """Standard token type categories."""
    KEYWORD = auto()       # Language keywords
    IDENTIFIER = auto()    # User identifiers
    LITERAL = auto()       # Literals (numbers, strings)
    OPERATOR = auto()      # Operators
    PUNCTUATION = auto()   # Punctuation marks
    COMMENT = auto()       # Comments (usually skipped)
    WHITESPACE = auto()    # Whitespace (usually skipped)
    ERROR = auto()         # Error token
    EOF = auto()           # End of file


@dataclass
class TokenSpec:
    """
    Specification for a single token type.
    
    Attributes:
        name: Token name (e.g., "INTEGER", "PLUS")
        pattern: Regular expression pattern
        priority: Higher priority wins on equal length match
        token_type: Category of token
        skip: Whether to skip this token in output
        action: Optional action code to execute
    """
    name: str
    pattern: str
    priority: int = 0
    token_type: TokenType = TokenType.IDENTIFIER
    skip: bool = False
    action: Optional[str] = None
    
    def __post_init__(self):
        """Validate token specification."""
        if not self.name:
            raise ValueError("Token name cannot be empty")
        if not self.pattern:
            raise ValueError("Token pattern cannot be empty")
        if not self.name.isidentifier() or not self.name[0].isupper():
            # Allow names that start with uppercase for convention
            pass  # Relaxed validation for flexibility
    
    def __hash__(self):
        return hash((self.name, self.pattern))
    
    def __eq__(self, other):
        if not isinstance(other, TokenSpec):
            return False
        return self.name == other.name and self.pattern == other.pattern


@dataclass
class LexerSpec:
    """
    Complete lexer specification.
    
    Attributes:
        name: Lexer name (used for generated class names)
        tokens: List of token specifications
        keywords: Mapping of keyword strings to token names
        case_sensitive: Whether matching is case-sensitive
    """
    name: str
    tokens: List[TokenSpec]
    keywords: Dict[str, str] = field(default_factory=dict)
    case_sensitive: bool = True
    
    def __post_init__(self):
        """Initialize and validate."""
        if not self.name:
            raise ValueError("Lexer name cannot be empty")
        if not self.tokens:
            raise ValueError("Lexer must have at least one token")
    
    def validate(self) -> List[str]:
        """
        Validate lexer specification.
        
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        names: Set[str] = set()
        
        for token in self.tokens:
            if token.name in names:
                errors.append(f"Duplicate token name: {token.name}")
            names.add(token.name)
            
            # Validate pattern is not empty
            if not token.pattern.strip():
                errors.append(f"Empty pattern for token: {token.name}")
        
        # Validate keywords reference existing tokens
        for keyword, token_name in self.keywords.items():
            if token_name not in names:
                errors.append(f"Keyword '{keyword}' references undefined token: {token_name}")
        
        return errors
    
    def get_skip_tokens(self) -> Set[str]:
        """Get names of tokens that should be skipped."""
        return {t.name for t in self.tokens if t.skip}
    
    def get_token_by_name(self, name: str) -> Optional[TokenSpec]:
        """Get token specification by name."""
        for token in self.tokens:
            if token.name == name:
                return token
        return None


@dataclass
class Token:
    """
    A token produced by the lexer.
    
    Attributes:
        type: Token type name
        value: Matched text (lexeme)
        line: Line number (1-based)
        column: Column number (1-based)
    """
    type: str
    value: str
    line: int
    column: int
    
    def __repr__(self):
        return f"Token({self.type}, {self.value!r}, {self.line}:{self.column})"
    
    def __eq__(self, other):
        if not isinstance(other, Token):
            return False
        return (self.type == other.type and 
                self.value == other.value and
                self.line == other.line and
                self.column == other.column)


class LexerError(Exception):
    """Exception raised for lexer errors."""
    
    def __init__(self, message: str, line: int = 0, column: int = 0):
        super().__init__(message)
        self.line = line
        self.column = column
    
    def __str__(self):
        if self.line and self.column:
            return f"{self.args[0]} at line {self.line}, column {self.column}"
        return self.args[0]


# Common token specifications for convenience
COMMON_TOKENS = {
    'INTEGER': TokenSpec('INTEGER', r'[0-9]+', token_type=TokenType.LITERAL),
    'FLOAT': TokenSpec('FLOAT', r'[0-9]+\.[0-9]+', token_type=TokenType.LITERAL),
    'STRING': TokenSpec('STRING', r'"[^"]*"', token_type=TokenType.LITERAL),
    'IDENTIFIER': TokenSpec('IDENTIFIER', r'[a-zA-Z_][a-zA-Z0-9_]*', token_type=TokenType.IDENTIFIER),
    'WHITESPACE': TokenSpec('WHITESPACE', r'[ \t\n\r]+', token_type=TokenType.WHITESPACE, skip=True),
    'COMMENT_LINE': TokenSpec('COMMENT_LINE', r'//[^\n]*', token_type=TokenType.COMMENT, skip=True),
    'COMMENT_BLOCK': TokenSpec('COMMENT_BLOCK', r'/\*[^*]*\*/', token_type=TokenType.COMMENT, skip=True),
}
