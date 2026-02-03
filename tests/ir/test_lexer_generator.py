"""
Tests for Lexer Generator.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ir.lexer.token_spec import TokenSpec, LexerSpec, Token, LexerError, TokenType
from ir.lexer.lexer_generator import (
    LexerGenerator, LexerSimulator, create_lexer, tokenize
)


class TestTokenSpec:
    """Test TokenSpec class."""
    
    def test_basic_creation(self):
        """Create basic token spec."""
        spec = TokenSpec("INT", "[0-9]+")
        assert spec.name == "INT"
        assert spec.pattern == "[0-9]+"
        assert spec.priority == 0
        assert not spec.skip
    
    def test_with_priority(self):
        """Create token with priority."""
        spec = TokenSpec("IF", "if", priority=10)
        assert spec.priority == 10
    
    def test_skip_token(self):
        """Create skip token."""
        spec = TokenSpec("WS", "[ \\t]+", skip=True)
        assert spec.skip
    
    def test_empty_name_error(self):
        """Empty name raises error."""
        with pytest.raises(ValueError):
            TokenSpec("", "[0-9]+")
    
    def test_empty_pattern_error(self):
        """Empty pattern raises error."""
        with pytest.raises(ValueError):
            TokenSpec("INT", "")


class TestLexerSpec:
    """Test LexerSpec class."""
    
    def test_basic_creation(self):
        """Create basic lexer spec."""
        spec = LexerSpec("Test", [
            TokenSpec("INT", "[0-9]+"),
            TokenSpec("ID", "[a-z]+")
        ])
        assert spec.name == "Test"
        assert len(spec.tokens) == 2
    
    def test_validation_duplicate_names(self):
        """Validation catches duplicate token names."""
        spec = LexerSpec("Test", [
            TokenSpec("INT", "[0-9]+"),
            TokenSpec("INT", "[0-9]+\\.[0-9]+")
        ])
        errors = spec.validate()
        assert any("Duplicate" in e for e in errors)
    
    def test_get_skip_tokens(self):
        """get_skip_tokens returns correct set."""
        spec = LexerSpec("Test", [
            TokenSpec("INT", "[0-9]+"),
            TokenSpec("WS", "[ \\t]+", skip=True),
            TokenSpec("COMMENT", "//[^\\n]*", skip=True)
        ])
        skip = spec.get_skip_tokens()
        assert skip == {"WS", "COMMENT"}
    
    def test_get_token_by_name(self):
        """get_token_by_name works."""
        spec = LexerSpec("Test", [
            TokenSpec("INT", "[0-9]+"),
            TokenSpec("ID", "[a-z]+")
        ])
        token = spec.get_token_by_name("INT")
        assert token is not None
        assert token.pattern == "[0-9]+"
        
        assert spec.get_token_by_name("NONEXISTENT") is None


class TestLexerGenerator:
    """Test LexerGenerator class."""
    
    def test_simple_generation(self):
        """Generate simple lexer."""
        spec = LexerSpec("Simple", [
            TokenSpec("INT", "[0-9]+"),
            TokenSpec("ID", "[a-z]+")
        ])
        gen = LexerGenerator(spec)
        dfa = gen.generate()
        
        assert dfa is not None
        assert dfa.num_states > 0
    
    def test_validation(self):
        """Validation catches invalid patterns."""
        spec = LexerSpec("Bad", [
            TokenSpec("BAD", "[unclosed")
        ])
        gen = LexerGenerator(spec)
        errors = gen.validate()
        assert len(errors) > 0
    
    def test_generate_with_skip_tokens(self):
        """Generation handles skip tokens."""
        spec = LexerSpec("WithSkip", [
            TokenSpec("INT", "[0-9]+"),
            TokenSpec("WS", "[ \\t\\n]+", skip=True)
        ])
        gen = LexerGenerator(spec)
        gen.generate()
        
        assert gen.get_skip_tokens() == {"WS"}
    
    def test_statistics(self):
        """Statistics are computed correctly."""
        spec = LexerSpec("Stats", [
            TokenSpec("INT", "[0-9]+"),
            TokenSpec("ID", "[a-z]+"),
            TokenSpec("WS", "[ ]+", skip=True)
        ])
        gen = LexerGenerator(spec)
        gen.generate()
        
        stats = gen.get_statistics()
        assert stats["lexer_name"] == "Stats"
        assert stats["num_tokens"] == 3
        assert stats["num_skip_tokens"] == 1
        assert stats["minimized_states"] > 0


class TestLexerSimulator:
    """Test LexerSimulator class."""
    
    @pytest.fixture
    def simple_lexer(self):
        """Create simple lexer for tests."""
        spec = LexerSpec("Simple", [
            TokenSpec("INT", "[0-9]+"),
            TokenSpec("PLUS", "\\+"),
            TokenSpec("MINUS", "-"),
            TokenSpec("ID", "[a-zA-Z_][a-zA-Z0-9_]*"),
            TokenSpec("WS", "[ \\t\\n]+", skip=True)
        ])
        gen = LexerGenerator(spec)
        gen.generate()
        return LexerSimulator(gen.table, gen.get_skip_tokens())
    
    def test_tokenize_integers(self, simple_lexer):
        """Tokenize integer literals."""
        tokens = simple_lexer.tokenize("123 456 789")
        
        assert len(tokens) == 3
        assert all(t.type == "INT" for t in tokens)
        assert tokens[0].value == "123"
        assert tokens[1].value == "456"
        assert tokens[2].value == "789"
    
    def test_tokenize_mixed(self, simple_lexer):
        """Tokenize mixed input."""
        tokens = simple_lexer.tokenize("abc + 123 - def")
        
        assert len(tokens) == 5
        assert tokens[0].type == "ID"
        assert tokens[1].type == "PLUS"
        assert tokens[2].type == "INT"
        assert tokens[3].type == "MINUS"
        assert tokens[4].type == "ID"
    
    def test_skip_whitespace(self, simple_lexer):
        """Whitespace is skipped."""
        tokens = simple_lexer.tokenize("a   b\n\tc")
        
        assert len(tokens) == 3
        assert all(t.type == "ID" for t in tokens)
    
    def test_line_column_tracking(self, simple_lexer):
        """Line and column are tracked correctly."""
        tokens = simple_lexer.tokenize("a\nb\nc")
        
        assert tokens[0].line == 1
        assert tokens[0].column == 1
        assert tokens[1].line == 2
        assert tokens[1].column == 1
        assert tokens[2].line == 3
        assert tokens[2].column == 1
    
    def test_error_on_invalid(self, simple_lexer):
        """Error on invalid character."""
        with pytest.raises(LexerError) as exc_info:
            simple_lexer.tokenize("abc @ def")
        
        assert "Unexpected character" in str(exc_info.value)
    
    def test_longest_match(self):
        """Longest match rule works."""
        spec = LexerSpec("LongestMatch", [
            TokenSpec("IF", "if", priority=10),
            TokenSpec("ID", "[a-z]+", priority=0),
            TokenSpec("WS", "[ ]+", skip=True)
        ])
        gen = LexerGenerator(spec)
        gen.generate()
        lexer = LexerSimulator(gen.table, gen.get_skip_tokens())
        
        # "if" should match IF
        tokens = lexer.tokenize("if")
        assert len(tokens) == 1
        assert tokens[0].type == "IF"
        
        # "iff" should match ID (longest match)
        tokens = lexer.tokenize("iff")
        assert len(tokens) == 1
        assert tokens[0].type == "ID"
        assert tokens[0].value == "iff"
    
    def test_priority_resolution(self):
        """Higher priority wins on same length."""
        spec = LexerSpec("Priority", [
            TokenSpec("WHILE", "while", priority=10),
            TokenSpec("ID", "[a-z]+", priority=0),
            TokenSpec("WS", "[ ]+", skip=True)
        ])
        gen = LexerGenerator(spec)
        gen.generate()
        lexer = LexerSimulator(gen.table, gen.get_skip_tokens())
        
        tokens = lexer.tokenize("while whiles")
        
        assert len(tokens) == 2
        assert tokens[0].type == "WHILE"
        assert tokens[1].type == "ID"


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_lexer(self):
        """create_lexer function works."""
        spec = LexerSpec("Test", [
            TokenSpec("INT", "[0-9]+"),
            TokenSpec("WS", "[ ]+", skip=True)
        ])
        lexer = create_lexer(spec)
        
        tokens = lexer.tokenize("1 2 3")
        assert len(tokens) == 3
    
    def test_tokenize_function(self):
        """tokenize function works."""
        spec = LexerSpec("Test", [
            TokenSpec("INT", "[0-9]+"),
            TokenSpec("WS", "[ ]+", skip=True)
        ])
        tokens = tokenize(spec, "1 2 3")
        
        assert len(tokens) == 3


class TestComplexLexer:
    """Test complex lexer scenarios."""
    
    def test_programming_language_tokens(self):
        """Tokenize programming language constructs."""
        spec = LexerSpec("Lang", [
            # Keywords
            TokenSpec("IF", "if", priority=10),
            TokenSpec("ELSE", "else", priority=10),
            TokenSpec("WHILE", "while", priority=10),
            TokenSpec("RETURN", "return", priority=10),
            
            # Literals
            TokenSpec("INT", "[0-9]+"),
            TokenSpec("FLOAT", "[0-9]+\\.[0-9]+", priority=5),
            
            # Identifiers
            TokenSpec("ID", "[a-zA-Z_][a-zA-Z0-9_]*"),
            
            # Operators
            TokenSpec("PLUS", "\\+"),
            TokenSpec("MINUS", "-"),
            TokenSpec("STAR", "\\*"),
            TokenSpec("SLASH", "/"),
            TokenSpec("EQ", "="),
            TokenSpec("EQEQ", "==", priority=5),
            
            # Punctuation
            TokenSpec("LPAREN", "\\("),
            TokenSpec("RPAREN", "\\)"),
            TokenSpec("LBRACE", "\\{"),
            TokenSpec("RBRACE", "\\}"),
            TokenSpec("SEMI", ";"),
            
            # Skip
            TokenSpec("WS", "[ \\t\\n]+", skip=True)
        ])
        
        code = "if (x == 10) { return x + 1; }"
        lexer = create_lexer(spec)
        tokens = lexer.tokenize(code)
        
        # Verify token sequence
        types = [t.type for t in tokens]
        assert types[0] == "IF"
        assert types[1] == "LPAREN"
        assert types[2] == "ID"
        assert types[3] == "EQEQ"
        assert types[4] == "INT"
        assert types[5] == "RPAREN"
    
    def test_multiline_input(self):
        """Tokenize multiline input."""
        spec = LexerSpec("Multi", [
            TokenSpec("INT", "[0-9]+"),
            TokenSpec("ID", "[a-z]+"),
            TokenSpec("WS", "[ \\t\\n]+", skip=True)
        ])
        
        code = """abc
123
def
456"""
        
        lexer = create_lexer(spec)
        tokens = lexer.tokenize(code)
        
        assert len(tokens) == 4
        assert tokens[0].line == 1
        assert tokens[1].line == 2
        assert tokens[2].line == 3
        assert tokens[3].line == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
