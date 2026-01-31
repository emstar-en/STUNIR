"""Tests for Lisp family parser."""

import pytest
from tools.semantic_ir.parser import SpecParser
from tools.semantic_ir.types import ParserOptions
from tools.semantic_ir.categories import LispParser


class TestLispParser:
    """Test Lisp family parser."""

    def test_lisp_parser_init(self):
        """Test Lisp parser initialization."""
        parser = LispParser()
        assert parser.category == "lisp"

    @pytest.mark.parametrize("dialect", [
        "common-lisp", "scheme", "clojure", "racket",
        "emacs-lisp", "guile", "hy", "janet"
    ])
    def test_parse_lisp_dialects(self, dialect):
        """Test parsing all Lisp dialects."""
        spec = {
            "metadata": {
                "category": "lisp",
                "dialect": dialect,
            },
            "forms": [
                {"name": "add", "params": ["a", "b"], "body": "(+ a b)"}
            ],
            "functions": []
        }
        
        options = ParserOptions(category="lisp")
        parser = SpecParser(options)
        ir = parser.parse_dict(spec)
        
        assert ir is not None
        assert ir.metadata.category == "lisp"

    def test_validate_lisp_spec_missing_dialect(self):
        """Test validation error for missing dialect."""
        spec = {
            "metadata": {}
        }
        
        parser = LispParser()
        result = parser.validate_spec(spec)
        
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_lisp_macro_support(self):
        """Test Lisp macro support."""
        spec = {
            "metadata": {
                "dialect": "common-lisp"
            },
            "macros": [
                {"name": "when", "params": ["condition", "body"]}
            ]
        }
        
        parser = LispParser()
        result = parser.validate_spec(spec)
        
        assert result.is_valid
