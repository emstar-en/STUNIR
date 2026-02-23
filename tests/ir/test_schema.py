"""Test JSON Schema validation for Semantic IR.

Tests:
- Schema file loading
- Schema validation
- Required fields
- Type constraints
"""

import json
import pytest
from pathlib import Path

# Schema directory
SCHEMA_DIR = Path(__file__).parent.parent.parent / "schemas" / "semantic_ir"


class TestSchemas:
    """Test JSON schemas."""

    def test_schema_files_exist(self):
        """Test that all schema files exist."""
        expected_schemas = [
            "ir_schema.json",
            "node_types.json",
            "type_system.json",
            "expressions.json",
            "statements.json",
            "declarations.json",
            "modules.json",
            "target_extensions.json",
        ]
        
        for schema_file in expected_schemas:
            schema_path = SCHEMA_DIR / schema_file
            assert schema_path.exists(), f"Schema file missing: {schema_file}"
    
    def test_schemas_are_valid_json(self):
        """Test that all schemas are valid JSON."""
        for schema_file in SCHEMA_DIR.glob("*.json"):
            with open(schema_file, 'r') as f:
                try:
                    json.load(f)
                except json.JSONDecodeError as e:
                    pytest.fail(f"Invalid JSON in {schema_file.name}: {e}")
    
    def test_ir_schema_structure(self):
        """Test main IR schema structure."""
        with open(SCHEMA_DIR / "ir_schema.json", 'r') as f:
            schema = json.load(f)
        
        assert "$schema" in schema
        assert "title" in schema
        assert schema["title"] == "STUNIR Semantic IR Schema"
        assert "properties" in schema
        assert "root" in schema["properties"]
    
    def test_node_types_enumeration(self):
        """Test node types enumeration."""
        with open(SCHEMA_DIR / "node_types.json", 'r') as f:
            schema = json.load(f)
        
        assert "definitions" in schema
        assert "IRNodeKind" in schema["definitions"]
        
        node_kinds = schema["definitions"]["IRNodeKind"]
        assert "enum" in node_kinds
        
        # Check essential node kinds
        enum_values = node_kinds["enum"]
        assert "module" in enum_values
        assert "function_decl" in enum_values
        assert "binary_expr" in enum_values
        assert "return_stmt" in enum_values
