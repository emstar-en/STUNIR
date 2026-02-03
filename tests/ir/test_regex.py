"""
Tests for Regular Expression Parser.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ir.lexer.regex import (
    RegexParser, RegexError, parse_regex,
    CharNode, CharClassNode, ConcatNode, UnionNode,
    StarNode, PlusNode, OptionalNode, AnyCharNode, EpsilonNode,
    DIGITS, WORD_CHARS, WHITESPACE
)


class TestRegexParserBasics:
    """Test basic regex parsing."""
    
    def test_empty_pattern(self):
        """Empty pattern produces epsilon node."""
        ast = parse_regex("")
        assert isinstance(ast, EpsilonNode)
    
    def test_single_char(self):
        """Single character produces CharNode."""
        ast = parse_regex("a")
        assert isinstance(ast, CharNode)
        assert ast.char == "a"
    
    def test_two_chars_concat(self):
        """Two characters produce concatenation."""
        ast = parse_regex("ab")
        assert isinstance(ast, ConcatNode)
        assert isinstance(ast.left, CharNode)
        assert ast.left.char == "a"
        assert isinstance(ast.right, CharNode)
        assert ast.right.char == "b"
    
    def test_multiple_chars_concat(self):
        """Multiple characters produce nested concatenation."""
        ast = parse_regex("abc")
        assert isinstance(ast, ConcatNode)
        # Left-associative: ((a . b) . c)
        assert isinstance(ast.left, ConcatNode)
    
    def test_any_char(self):
        """Dot produces AnyCharNode."""
        ast = parse_regex(".")
        assert isinstance(ast, AnyCharNode)


class TestRegexParserAlternation:
    """Test alternation (|) parsing."""
    
    def test_simple_alternation(self):
        """a|b produces UnionNode."""
        ast = parse_regex("a|b")
        assert isinstance(ast, UnionNode)
        assert isinstance(ast.left, CharNode)
        assert isinstance(ast.right, CharNode)
    
    def test_multiple_alternation(self):
        """a|b|c produces nested UnionNode."""
        ast = parse_regex("a|b|c")
        assert isinstance(ast, UnionNode)
        # Left-associative: ((a | b) | c)
        assert isinstance(ast.left, UnionNode)
    
    def test_alternation_precedence(self):
        """Concatenation binds tighter than alternation."""
        ast = parse_regex("ab|cd")
        assert isinstance(ast, UnionNode)
        assert isinstance(ast.left, ConcatNode)
        assert isinstance(ast.right, ConcatNode)


class TestRegexParserQuantifiers:
    """Test quantifier parsing."""
    
    def test_star(self):
        """a* produces StarNode."""
        ast = parse_regex("a*")
        assert isinstance(ast, StarNode)
        assert isinstance(ast.child, CharNode)
    
    def test_plus(self):
        """a+ produces PlusNode."""
        ast = parse_regex("a+")
        assert isinstance(ast, PlusNode)
        assert isinstance(ast.child, CharNode)
    
    def test_optional(self):
        """a? produces OptionalNode."""
        ast = parse_regex("a?")
        assert isinstance(ast, OptionalNode)
        assert isinstance(ast.child, CharNode)
    
    def test_quantifier_precedence(self):
        """Quantifiers bind tighter than concatenation."""
        ast = parse_regex("ab*")
        assert isinstance(ast, ConcatNode)
        assert isinstance(ast.left, CharNode)
        assert isinstance(ast.right, StarNode)


class TestRegexParserGrouping:
    """Test grouping with parentheses."""
    
    def test_simple_group(self):
        """(a) is equivalent to a."""
        ast = parse_regex("(a)")
        assert isinstance(ast, CharNode)
        assert ast.char == "a"
    
    def test_group_quantifier(self):
        """(ab)* quantifies the group."""
        ast = parse_regex("(ab)*")
        assert isinstance(ast, StarNode)
        assert isinstance(ast.child, ConcatNode)
    
    def test_nested_groups(self):
        """Nested groups work correctly."""
        ast = parse_regex("((a))")
        assert isinstance(ast, CharNode)
    
    def test_group_alternation(self):
        """(a|b) groups alternation."""
        ast = parse_regex("(a|b)c")
        assert isinstance(ast, ConcatNode)
        assert isinstance(ast.left, UnionNode)


class TestRegexParserCharClass:
    """Test character class parsing."""
    
    def test_simple_char_class(self):
        """[abc] produces CharClassNode."""
        ast = parse_regex("[abc]")
        assert isinstance(ast, CharClassNode)
        assert ast.chars == frozenset("abc")
    
    def test_char_range(self):
        """[a-z] produces CharClassNode with range."""
        ast = parse_regex("[a-z]")
        assert isinstance(ast, CharClassNode)
        assert 'a' in ast.chars
        assert 'm' in ast.chars
        assert 'z' in ast.chars
        assert len(ast.chars) == 26
    
    def test_negated_char_class(self):
        """[^abc] produces negated CharClassNode."""
        ast = parse_regex("[^abc]")
        assert isinstance(ast, CharClassNode)
        assert ast.negated
        assert ast.chars == frozenset("abc")
    
    def test_mixed_char_class(self):
        """[a-z0-9_] works correctly."""
        ast = parse_regex("[a-z0-9_]")
        assert isinstance(ast, CharClassNode)
        assert 'a' in ast.chars
        assert 'z' in ast.chars
        assert '0' in ast.chars
        assert '9' in ast.chars
        assert '_' in ast.chars


class TestRegexParserEscapes:
    """Test escape sequence parsing."""
    
    def test_digit_escape(self):
        """\d produces digit character class."""
        ast = parse_regex("\\d")
        assert isinstance(ast, CharClassNode)
        assert ast.chars == DIGITS
    
    def test_word_escape(self):
        """\w produces word character class."""
        ast = parse_regex("\\w")
        assert isinstance(ast, CharClassNode)
        assert ast.chars == WORD_CHARS
    
    def test_whitespace_escape(self):
        """\s produces whitespace character class."""
        ast = parse_regex("\\s")
        assert isinstance(ast, CharClassNode)
        assert ast.chars == WHITESPACE
    
    def test_literal_escape(self):
        """\\. produces literal dot."""
        ast = parse_regex("\\.")
        assert isinstance(ast, CharNode)
        assert ast.char == "."
    
    def test_backslash_escape(self):
        """\\\\ produces literal backslash."""
        ast = parse_regex("\\\\")
        assert isinstance(ast, CharNode)
        assert ast.char == "\\"
    
    def test_special_escapes(self):
        """\\n, \\r, \\t work correctly."""
        ast = parse_regex("\\n")
        assert isinstance(ast, CharNode)
        assert ast.char == "\n"
        
        ast = parse_regex("\\t")
        assert isinstance(ast, CharNode)
        assert ast.char == "\t"


class TestRegexParserErrors:
    """Test error handling."""
    
    def test_unclosed_group(self):
        """Unclosed group raises error."""
        with pytest.raises(RegexError):
            parse_regex("(abc")
    
    def test_unclosed_char_class(self):
        """Unclosed character class raises error."""
        with pytest.raises(RegexError):
            parse_regex("[abc")
    
    def test_unexpected_close_paren(self):
        """Unexpected ) raises error."""
        with pytest.raises(RegexError):
            parse_regex("abc)")
    
    def test_invalid_range(self):
        """Invalid range [z-a] raises error."""
        with pytest.raises(RegexError):
            parse_regex("[z-a]")
    
    def test_unexpected_quantifier(self):
        """Quantifier without preceding expression raises error."""
        with pytest.raises(RegexError):
            parse_regex("*abc")


class TestRegexParserComplex:
    """Test complex patterns."""
    
    def test_identifier_pattern(self):
        """[a-zA-Z_][a-zA-Z0-9_]* parses correctly."""
        ast = parse_regex("[a-zA-Z_][a-zA-Z0-9_]*")
        assert isinstance(ast, ConcatNode)
        assert isinstance(ast.left, CharClassNode)
        assert isinstance(ast.right, StarNode)
    
    def test_integer_pattern(self):
        """[0-9]+ parses correctly."""
        ast = parse_regex("[0-9]+")
        assert isinstance(ast, PlusNode)
        assert isinstance(ast.child, CharClassNode)
    
    def test_float_pattern(self):
        """[0-9]+\\.[0-9]+ parses correctly."""
        ast = parse_regex("[0-9]+\\.[0-9]+")
        assert isinstance(ast, ConcatNode)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
