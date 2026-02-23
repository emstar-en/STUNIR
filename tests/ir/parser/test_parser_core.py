"""Core parser functionality tests."""

import pytest
import json
from tools.semantic_ir.parser import SpecParser
from tools.semantic_ir.types import ParserOptions


class TestSpecParser:
    """Test SpecParser main functionality."""

    def test_parser_initialization(self):
        """Test parser can be initialized."""
        options = ParserOptions(category="embedded")
        parser = SpecParser(options)
        assert parser is not None
        assert parser.options.category == "embedded"

    def test_parse_simple_spec(self):
        """Test parsing a simple specification."""
        spec = {
            "schema": "https://stunir.dev/schemas/semantic_ir_v1_schema.json",
            "metadata": {
                "category": "embedded",
                "version": "1.0.0",
            },
            "functions": [
                {
                    "name": "add",
                    "parameters": [
                        {"name": "a", "type": "i32"},
                        {"name": "b", "type": "i32"},
                    ],
                    "return_type": "i32",
                    "body": []
                }
            ]
        }
        
        options = ParserOptions(category="embedded")
        parser = SpecParser(options)
        ir = parser.parse_dict(spec)
        
        assert ir is not None
        assert ir.metadata.category == "embedded"
        assert len(ir.functions) == 1
        assert ir.functions[0].name == "add"

    def test_parse_string_json(self):
        """Test parsing from JSON string."""
        spec_str = json.dumps({
            "metadata": {"category": "embedded"},
            "functions": []
        })
        
        options = ParserOptions(category="embedded")
        parser = SpecParser(options)
        ir = parser.parse_string(spec_str, format="json")
        
        assert ir is not None

    def test_category_mismatch_error(self):
        """Test error on category mismatch."""
        spec = {
            "metadata": {"category": "wasm"},
            "functions": []
        }
        
        options = ParserOptions(category="embedded")
        parser = SpecParser(options)
        ir = parser.parse_dict(spec)
        
        # Should have category mismatch error
        errors = parser.get_errors()
        assert len(errors) > 0

    def test_validate_ir(self):
        """Test IR validation."""
        spec = {
            "metadata": {"category": "embedded"},
            "functions": [{"name": "test", "parameters": [], "return_type": "void", "body": []}]
        }
        
        options = ParserOptions(category="embedded")
        parser = SpecParser(options)
        ir = parser.parse_dict(spec)
        
        result = parser.validate_ir(ir)
        assert result.is_valid
