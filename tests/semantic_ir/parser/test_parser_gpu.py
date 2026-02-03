"""Tests for GPU category parser."""

import pytest
from tools.semantic_ir.parser import SpecParser
from tools.semantic_ir.types import ParserOptions
from tools.semantic_ir.categories import GPUParser


class TestGPUParser:
    """Test GPU category parser."""

    def test_gpu_parser_init(self):
        """Test GPU parser initialization."""
        parser = GPUParser()
        assert parser.category == "gpu"

    def test_parse_gpu_spec(self):
        """Test parsing GPU specification."""
        spec = {
            "metadata": {
                "category": "gpu",
                "gpu_platform": "cuda",
            },
            "kernels": [
                {
                    "name": "vector_add",
                    "grid_size": [256, 1, 1],
                    "block_size": [256, 1, 1]
                }
            ],
            "functions": []
        }
        
        options = ParserOptions(category="gpu")
        parser = SpecParser(options)
        ir = parser.parse_dict(spec)
        
        assert ir is not None
        assert ir.metadata.category == "gpu"

    def test_validate_gpu_spec_missing_platform(self):
        """Test validation error for missing platform."""
        spec = {
            "metadata": {}
        }
        
        parser = GPUParser()
        result = parser.validate_spec(spec)
        
        assert not result.is_valid
        assert len(result.errors) > 0
