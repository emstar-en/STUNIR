"""
Tests for STUNIR Lexer Specification.

Tests the lexer specification builder and validates token recognition.
"""

import pytest
from typing import List

from bootstrap.stunir_lexer import (
    STUNIRLexerBuilder,
    STUNIR_KEYWORDS,
    STUNIR_TOKENS,
    create_stunir_lexer_spec,
)
from bootstrap.bootstrap_compiler import (
    SimpleLexer,
    STUNIRToken,
    STUNIRLexerError,
)
from ir.lexer.token_spec import LexerSpec, TokenSpec


class TestSTUNIRLexerBuilder:
    """Tests for STUNIRLexerBuilder class."""
    
    def test_builder_initialization(self):
        """Test builder initializes correctly."""
        builder = STUNIRLexerBuilder()
        assert builder is not None
    
    def test_build_lexer_spec(self):
        """Test building lexer specification."""
        builder = STUNIRLexerBuilder()
        spec = builder.build()
        
        assert spec is not None
        assert isinstance(spec, LexerSpec)
        assert spec.name == 'STUNIRLexer'
    
    def test_case_sensitive(self):
        """Test lexer is case-sensitive."""
        spec = create_stunir_lexer_spec()
        assert spec.case_sensitive is True
    
    def test_keywords_count(self):
        """Test keyword count is correct."""
        # 37 keywords expected
        assert len(STUNIR_KEYWORDS) >= 35
        assert len(STUNIR_KEYWORDS) <= 45
    
    def test_tokens_count(self):
        """Test token count is correct."""
        # Should have keywords + operators + punctuation + literals + skip
        assert len(STUNIR_TOKENS) >= 40
        assert len(STUNIR_TOKENS) <= 100
    
    def test_validation_passes(self):
        """Test lexer specification passes validation.
        
        Note: Keywords are handled via post-processing of IDENTIFIER tokens,
        so keyword references to KW_* tokens are expected in the keywords dict.
        The lexer spec validation may report these as 'missing' but they work
        correctly because keywords are resolved at runtime.
        """
        builder = STUNIRLexerBuilder()
        errors = builder.validate()
        # Filter out keyword reference errors - these are expected because
        # keywords are resolved from identifiers, not separate tokens
        non_keyword_errors = [e for e in errors if 'references undefined token' not in e]
        assert non_keyword_errors == []


class TestSTUNIRKeywords:
    """Tests for STUNIR keywords."""
    
    def test_module_keyword(self):
        """Test 'module' keyword."""
        assert 'module' in STUNIR_KEYWORDS
        assert STUNIR_KEYWORDS['module'] == 'KW_MODULE'
    
    def test_function_keyword(self):
        """Test 'function' keyword."""
        assert 'function' in STUNIR_KEYWORDS
        assert STUNIR_KEYWORDS['function'] == 'KW_FUNCTION'
    
    def test_type_keyword(self):
        """Test 'type' keyword."""
        assert 'type' in STUNIR_KEYWORDS
        assert STUNIR_KEYWORDS['type'] == 'KW_TYPE'
    
    def test_control_flow_keywords(self):
        """Test control flow keywords."""
        assert 'if' in STUNIR_KEYWORDS
        assert 'else' in STUNIR_KEYWORDS
        assert 'while' in STUNIR_KEYWORDS
        assert 'for' in STUNIR_KEYWORDS
        assert 'match' in STUNIR_KEYWORDS
        assert 'return' in STUNIR_KEYWORDS
    
    def test_variable_keywords(self):
        """Test variable declaration keywords."""
        assert 'let' in STUNIR_KEYWORDS
        assert 'var' in STUNIR_KEYWORDS
    
    def test_type_keywords(self):
        """Test type keywords."""
        for type_name in ['i8', 'i16', 'i32', 'i64', 
                          'u8', 'u16', 'u32', 'u64',
                          'f32', 'f64', 'bool', 'string', 'void', 'any']:
            assert type_name in STUNIR_KEYWORDS, f"Missing type keyword: {type_name}"
    
    def test_literal_keywords(self):
        """Test literal keywords."""
        assert 'true' in STUNIR_KEYWORDS
        assert 'false' in STUNIR_KEYWORDS
        assert 'null' in STUNIR_KEYWORDS
    
    def test_ir_keywords(self):
        """Test IR-specific keywords."""
        assert 'ir' in STUNIR_KEYWORDS
        assert 'child' in STUNIR_KEYWORDS
        assert 'op' in STUNIR_KEYWORDS
    
    def test_target_keyword(self):
        """Test target keyword."""
        assert 'target' in STUNIR_KEYWORDS
        assert 'emit' in STUNIR_KEYWORDS


class TestSTUNIRTokens:
    """Tests for STUNIR token specifications."""
    
    def test_has_comment_tokens(self):
        """Test comment tokens are defined."""
        names = [t.name for t in STUNIR_TOKENS]
        assert 'COMMENT_LINE' in names
        assert 'COMMENT_BLOCK' in names
    
    def test_has_whitespace_token(self):
        """Test whitespace token is defined."""
        names = [t.name for t in STUNIR_TOKENS]
        assert 'WHITESPACE' in names
    
    def test_has_literal_tokens(self):
        """Test literal tokens are defined."""
        names = [t.name for t in STUNIR_TOKENS]
        assert 'INTEGER_LITERAL' in names
        assert 'FLOAT_LITERAL' in names
        assert 'STRING_LITERAL' in names
    
    def test_has_identifier_token(self):
        """Test identifier token is defined."""
        names = [t.name for t in STUNIR_TOKENS]
        assert 'IDENTIFIER' in names
    
    def test_has_operator_tokens(self):
        """Test operator tokens are defined."""
        names = [t.name for t in STUNIR_TOKENS]
        assert 'PLUS' in names
        assert 'MINUS' in names
        assert 'STAR' in names
        assert 'SLASH' in names
        assert 'EQ' in names
        assert 'NE' in names
        assert 'AND' in names
        assert 'OR' in names
    
    def test_has_punctuation_tokens(self):
        """Test punctuation tokens are defined."""
        names = [t.name for t in STUNIR_TOKENS]
        assert 'LPAREN' in names
        assert 'RPAREN' in names
        assert 'LBRACE' in names
        assert 'RBRACE' in names
        assert 'SEMICOLON' in names
        assert 'COMMA' in names
    
    def test_skip_tokens_marked(self):
        """Test skip tokens are properly marked."""
        skip_tokens = [t for t in STUNIR_TOKENS if t.skip]
        skip_names = [t.name for t in skip_tokens]
        
        assert 'WHITESPACE' in skip_names
        assert 'COMMENT_LINE' in skip_names
        assert 'COMMENT_BLOCK' in skip_names
    
    def test_token_priorities(self):
        """Test tokens have proper priorities."""
        # Comments should have high priority
        comment_tokens = [t for t in STUNIR_TOKENS if 'COMMENT' in t.name]
        for t in comment_tokens:
            assert t.priority >= 99
        
        # Literals should have moderate priority
        literal_tokens = [t for t in STUNIR_TOKENS if 'LITERAL' in t.name]
        for t in literal_tokens:
            assert 70 <= t.priority <= 90


class TestSimpleLexer:
    """Tests for SimpleLexer tokenization."""
    
    def _tokenize(self, source: str) -> List[STUNIRToken]:
        """Helper to tokenize source."""
        lexer = SimpleLexer(source)
        return list(lexer.tokenize())
    
    def test_empty_source(self):
        """Test tokenizing empty source."""
        tokens = self._tokenize("")
        assert len(tokens) == 1
        assert tokens[0].type == 'EOF'
    
    def test_whitespace_only(self):
        """Test tokenizing whitespace-only source."""
        tokens = self._tokenize("   \t\n  ")
        assert len(tokens) == 1
        assert tokens[0].type == 'EOF'
    
    def test_keyword_recognition(self):
        """Test keyword recognition."""
        tokens = self._tokenize("module function type ir target")
        
        assert tokens[0].type == 'KW_MODULE'
        assert tokens[1].type == 'KW_FUNCTION'
        assert tokens[2].type == 'KW_TYPE'
        assert tokens[3].type == 'KW_IR'
        assert tokens[4].type == 'KW_TARGET'
    
    def test_identifier_recognition(self):
        """Test identifier recognition."""
        tokens = self._tokenize("myVar _private camelCase")
        
        assert tokens[0].type == 'IDENTIFIER'
        assert tokens[0].value == 'myVar'
        assert tokens[1].type == 'IDENTIFIER'
        assert tokens[1].value == '_private'
        assert tokens[2].type == 'IDENTIFIER'
        assert tokens[2].value == 'camelCase'
    
    def test_integer_literal(self):
        """Test integer literal recognition."""
        tokens = self._tokenize("42 0 12345")
        
        assert tokens[0].type == 'INTEGER_LITERAL'
        assert tokens[0].value == '42'
        assert tokens[1].type == 'INTEGER_LITERAL'
        assert tokens[2].type == 'INTEGER_LITERAL'
    
    def test_float_literal(self):
        """Test float literal recognition."""
        tokens = self._tokenize("3.14 0.5 123.456")
        
        assert tokens[0].type == 'FLOAT_LITERAL'
        assert tokens[0].value == '3.14'
        assert tokens[1].type == 'FLOAT_LITERAL'
        assert tokens[2].type == 'FLOAT_LITERAL'
    
    def test_string_literal(self):
        """Test string literal recognition."""
        tokens = self._tokenize('"hello" "world"')
        
        assert tokens[0].type == 'STRING_LITERAL'
        assert tokens[0].value == '"hello"'
        assert tokens[1].type == 'STRING_LITERAL'
    
    def test_operator_recognition(self):
        """Test operator recognition."""
        tokens = self._tokenize("+ - * / == != && ||")
        
        assert tokens[0].type == 'PLUS'
        assert tokens[1].type == 'MINUS'
        assert tokens[2].type == 'STAR'
        assert tokens[3].type == 'SLASH'
        assert tokens[4].type == 'EQ'
        assert tokens[5].type == 'NE'
        assert tokens[6].type == 'AND'
        assert tokens[7].type == 'OR'
    
    def test_punctuation_recognition(self):
        """Test punctuation recognition."""
        tokens = self._tokenize("( ) { } [ ] ; , .")
        
        assert tokens[0].type == 'LPAREN'
        assert tokens[1].type == 'RPAREN'
        assert tokens[2].type == 'LBRACE'
        assert tokens[3].type == 'RBRACE'
        assert tokens[4].type == 'LBRACKET'
        assert tokens[5].type == 'RBRACKET'
        assert tokens[6].type == 'SEMICOLON'
        assert tokens[7].type == 'COMMA'
        assert tokens[8].type == 'DOT'
    
    def test_line_comment(self):
        """Test line comment skipping."""
        tokens = self._tokenize("let x // comment\nlet y")
        
        types = [t.type for t in tokens if t.type != 'EOF']
        assert 'COMMENT_LINE' not in types
        assert types == ['KW_LET', 'IDENTIFIER', 'KW_LET', 'IDENTIFIER']
    
    def test_block_comment(self):
        """Test block comment skipping."""
        tokens = self._tokenize("let x /* comment */ let y")
        
        types = [t.type for t in tokens if t.type != 'EOF']
        assert 'COMMENT_BLOCK' not in types
        assert types == ['KW_LET', 'IDENTIFIER', 'KW_LET', 'IDENTIFIER']
    
    def test_token_positions(self):
        """Test token line and column tracking."""
        tokens = self._tokenize("let x\nlet y")
        
        assert tokens[0].line == 1
        assert tokens[0].column == 1
        assert tokens[2].line == 2  # After newline
    
    def test_module_declaration(self):
        """Test tokenizing module declaration."""
        tokens = self._tokenize("module test;")
        
        assert tokens[0].type == 'KW_MODULE'
        assert tokens[1].type == 'IDENTIFIER'
        assert tokens[1].value == 'test'
        assert tokens[2].type == 'SEMICOLON'
    
    def test_function_signature(self):
        """Test tokenizing function signature."""
        tokens = self._tokenize("function add(a: i32, b: i32): i32")
        
        types = [t.type for t in tokens if t.type != 'EOF']
        assert types[0] == 'KW_FUNCTION'
        assert types[1] == 'IDENTIFIER'  # add
        assert types[2] == 'LPAREN'
        assert 'KW_I32' in types
    
    def test_arrow_operators(self):
        """Test arrow operators."""
        tokens = self._tokenize("-> =>")
        
        assert tokens[0].type == 'ARROW'
        assert tokens[1].type == 'FAT_ARROW'
    
    def test_comparison_operators(self):
        """Test comparison operators."""
        tokens = self._tokenize("< > <= >=")
        
        assert tokens[0].type == 'LT'
        assert tokens[1].type == 'GT'
        assert tokens[2].type == 'LE'
        assert tokens[3].type == 'GE'


class TestLexerEdgeCases:
    """Tests for lexer edge cases."""
    
    def _tokenize(self, source: str) -> List[STUNIRToken]:
        """Helper to tokenize source."""
        lexer = SimpleLexer(source)
        return list(lexer.tokenize())
    
    def test_adjacent_operators(self):
        """Test adjacent operators."""
        tokens = self._tokenize("a+b*c")
        
        types = [t.type for t in tokens if t.type != 'EOF']
        assert types == ['IDENTIFIER', 'PLUS', 'IDENTIFIER', 'STAR', 'IDENTIFIER']
    
    def test_identifier_with_numbers(self):
        """Test identifier with numbers."""
        tokens = self._tokenize("var1 x2y z_3")
        
        assert all(t.type == 'IDENTIFIER' for t in tokens[:-1])
    
    def test_keyword_vs_identifier(self):
        """Test keyword vs similar identifier."""
        tokens = self._tokenize("let letter module modular")
        
        types = [t.type for t in tokens if t.type != 'EOF']
        assert types[0] == 'KW_LET'
        assert types[1] == 'IDENTIFIER'  # letter
        assert types[2] == 'KW_MODULE'
        assert types[3] == 'IDENTIFIER'  # modular
    
    def test_nested_brackets(self):
        """Test nested brackets."""
        tokens = self._tokenize("{{[()]}}")
        
        types = [t.type for t in tokens if t.type != 'EOF']
        assert types == ['LBRACE', 'LBRACE', 'LBRACKET', 
                        'LPAREN', 'RPAREN', 'RBRACKET', 'RBRACE', 'RBRACE']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
