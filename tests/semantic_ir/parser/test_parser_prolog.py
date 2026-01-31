"""Tests for Prolog family parser."""

import pytest
from tools.semantic_ir.parser import SpecParser
from tools.semantic_ir.types import ParserOptions
from tools.semantic_ir.categories import PrologParser


class TestPrologParser:
    """Test Prolog family parser."""

    def test_prolog_parser_init(self):
        """Test Prolog parser initialization."""
        parser = PrologParser()
        assert parser.category == "prolog"

    @pytest.mark.parametrize("dialect", [
        "swi-prolog", "gnu-prolog", "sicstus", "yap",
        "xsb", "ciao", "b-prolog", "eclipse"
    ])
    def test_parse_prolog_dialects(self, dialect):
        """Test parsing all Prolog dialects."""
        spec = {
            "metadata": {
                "category": "prolog",
                "dialect": dialect,
            },
            "facts": [
                {"predicate": "parent", "args": ["john", "mary"]}
            ],
            "rules": [
                {"head": "grandparent(X, Z)", "body": "parent(X, Y), parent(Y, Z)"}
            ],
            "functions": []
        }
        
        options = ParserOptions(category="prolog")
        parser = SpecParser(options)
        ir = parser.parse_dict(spec)
        
        assert ir is not None
        assert ir.metadata.category == "prolog"

    def test_validate_prolog_spec_missing_dialect(self):
        """Test validation error for missing dialect."""
        spec = {
            "metadata": {}
        }
        
        parser = PrologParser()
        result = parser.validate_spec(spec)
        
        assert not result.is_valid
        assert len(result.errors) > 0
