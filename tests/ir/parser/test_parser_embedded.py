"""Tests for embedded category parser."""

import pytest
from tools.semantic_ir.parser import SpecParser
from tools.semantic_ir.types import ParserOptions
from tools.semantic_ir.categories import EmbeddedParser


class TestEmbeddedParser:
    """Test embedded category parser."""

    def test_embedded_parser_init(self):
        """Test embedded parser initialization."""
        parser = EmbeddedParser()
        assert parser.category == "embedded"

    def test_parse_embedded_spec(self):
        """Test parsing embedded specification."""
        spec = {
            "metadata": {
                "category": "embedded",
                "target_arch": "arm",
                "memory": {
                    "ram_size": 65536,
                    "flash_size": 262144
                }
            },
            "interrupts": [
                {"name": "Timer0_IRQ", "priority": 1}
            ],
            "peripherals": [
                {"name": "UART0", "base_address": "0x40000000"}
            ],
            "functions": []
        }
        
        options = ParserOptions(category="embedded")
        parser = SpecParser(options)
        ir = parser.parse_dict(spec)
        
        assert ir is not None
        assert ir.metadata.category == "embedded"

    def test_validate_embedded_spec(self):
        """Test embedded spec validation."""
        spec = {
            "metadata": {
                "target_arch": "arm"
            }
        }
        
        parser = EmbeddedParser()
        result = parser.validate_spec(spec)
        
        assert result.is_valid

    def test_embedded_low_memory_warning(self):
        """Test warning for low memory."""
        spec = {
            "metadata": {
                "target_arch": "avr",
                "memory": {
                    "ram_size": 512  # Very low RAM
                }
            }
        }
        
        parser = EmbeddedParser()
        result = parser.validate_spec(spec)
        
        assert len(result.warnings) > 0
