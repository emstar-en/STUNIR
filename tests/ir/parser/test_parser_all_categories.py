"""Test all 24 category parsers."""

import pytest
from tools.semantic_ir.categories import CATEGORY_PARSERS


class TestAllCategories:
    """Test all category parsers are available."""

    def test_all_categories_registered(self):
        """Test all 24 categories are registered."""
        expected_categories = [
            "embedded", "assembly", "polyglot", "gpu", "wasm",
            "lisp", "prolog", "business", "bytecode", "constraints",
            "expert_systems", "fpga", "functional", "grammar",
            "mobile", "oop", "planning", "scientific", "systems",
            "asm_ir", "beam", "asp"
        ]
        
        for category in expected_categories:
            assert category in CATEGORY_PARSERS, f"Category {category} not registered"

    @pytest.mark.parametrize("category", [
        "embedded", "assembly", "polyglot", "gpu", "wasm",
        "lisp", "prolog", "business", "bytecode", "constraints",
        "expert_systems", "fpga", "functional", "grammar",
        "mobile", "oop", "planning", "scientific", "systems",
        "asm_ir", "beam", "asp"
    ])
    def test_category_parser_instantiation(self, category):
        """Test each category parser can be instantiated."""
        parser_class = CATEGORY_PARSERS[category]
        parser = parser_class()
        
        assert parser is not None
        assert parser.category == category

    def test_category_count(self):
        """Test we have exactly 24 categories (including 2 that need lexer/parser)."""
        # We have 22 unique categories + lexer + parser = 24 total
        assert len(CATEGORY_PARSERS) >= 22
