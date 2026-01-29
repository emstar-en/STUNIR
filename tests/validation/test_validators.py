"""STUNIR Validation Framework Tests.

Tests for the validation utilities.
"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.validation import (
    ValidationResult,
    validate_output,
    validate_file_output,
    validate_state,
    validate_config,
    validate_json_schema,
    validate_manifest,
    OutputValidator,
    StateValidator,
    ConfigValidator,
)


class TestValidationResult(unittest.TestCase):
    """Test ValidationResult class."""
    
    def test_valid_result(self):
        """Test creating valid result."""
        result = ValidationResult(valid=True)
        self.assertTrue(result.valid)
        self.assertEqual(len(result.issues), 0)
    
    def test_add_error(self):
        """Test adding errors."""
        result = ValidationResult(valid=True)
        result.add_error("Test error")
        self.assertFalse(result.valid)
        self.assertEqual(len(result.errors), 1)
    
    def test_add_warning(self):
        """Test adding warnings."""
        result = ValidationResult(valid=True)
        result.add_warning("Test warning")
        self.assertTrue(result.valid)  # Warnings don't invalidate
        self.assertEqual(len(result.warnings), 1)
    
    def test_merge_results(self):
        """Test merging validation results."""
        result1 = ValidationResult(valid=True)
        result1.add_warning("Warning 1")
        
        result2 = ValidationResult(valid=True)
        result2.add_error("Error 1")
        
        result1.merge(result2)
        self.assertFalse(result1.valid)
        self.assertEqual(len(result1.issues), 2)


class TestOutputValidation(unittest.TestCase):
    """Test output validation functions."""
    
    def test_validate_output_type(self):
        """Test type validation."""
        result = validate_output({"key": "value"}, expected_type=dict)
        self.assertTrue(result.valid)
        
        result = validate_output("string", expected_type=dict)
        self.assertFalse(result.valid)
    
    def test_validate_output_required_fields(self):
        """Test required fields validation."""
        data = {"name": "test", "version": "1.0"}
        
        result = validate_output(data, required_fields=["name", "version"])
        self.assertTrue(result.valid)
        
        result = validate_output(data, required_fields=["name", "missing"])
        self.assertFalse(result.valid)
    
    def test_validate_file_output(self):
        """Test file output validation."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            f.write(b'{"test": true}')
            filepath = f.name
        
        try:
            result = validate_file_output(filepath, expected_extension='.json')
            self.assertTrue(result.valid)
            
            result = validate_file_output(filepath, min_size=1)
            self.assertTrue(result.valid)
            
            result = validate_file_output(filepath, max_size=1)  # Too small
            self.assertFalse(result.valid)
        finally:
            os.unlink(filepath)
    
    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        result = validate_file_output('/nonexistent/path/file.json')
        self.assertFalse(result.valid)


class TestStateValidation(unittest.TestCase):
    """Test state validation functions."""
    
    def test_validate_state_required_keys(self):
        """Test required keys validation."""
        state = {"key1": "value1", "key2": "value2"}
        
        result = validate_state(state, required_keys=["key1"])
        self.assertTrue(result.valid)
        
        result = validate_state(state, required_keys=["missing"])
        self.assertFalse(result.valid)
    
    def test_validate_state_consistency(self):
        """Test consistency checks."""
        state = {"count": 5, "items": [1, 2, 3, 4, 5]}
        
        # Valid consistency check
        check = lambda s: len(s["items"]) == s["count"]
        result = validate_state(state, consistency_checks=[check])
        self.assertTrue(result.valid)
        
        # Invalid consistency
        state["count"] = 10
        result = validate_state(state, consistency_checks=[check])
        self.assertFalse(result.valid)


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation."""
    
    def test_validate_config_required(self):
        """Test required option validation."""
        config = {"name": "test"}
        
        result = validate_config(config, required=["name"])
        self.assertTrue(result.valid)
        
        result = validate_config(config, required=["missing"])
        self.assertFalse(result.valid)
    
    def test_validate_config_types(self):
        """Test type validation."""
        config = {"count": 5, "name": "test"}
        
        result = validate_config(config, types={"count": int})
        self.assertTrue(result.valid)
        
        result = validate_config(config, types={"count": str})
        self.assertFalse(result.valid)
    
    def test_validate_config_choices(self):
        """Test choices validation."""
        config = {"level": "debug"}
        
        result = validate_config(config, choices={"level": ["debug", "info", "error"]})
        self.assertTrue(result.valid)
        
        result = validate_config(config, choices={"level": ["info", "error"]})
        self.assertFalse(result.valid)
    
    def test_config_validator_class(self):
        """Test ConfigValidator class."""
        validator = ConfigValidator()
        validator.add_option("name", required=True, option_type=str)
        validator.add_option("count", option_type=int, choices=[1, 2, 3])
        
        result = validator.validate({"name": "test", "count": 2})
        self.assertTrue(result.valid)
        
        result = validator.validate({"count": 2})  # Missing required
        self.assertFalse(result.valid)


class TestJsonSchemaValidation(unittest.TestCase):
    """Test JSON Schema validation."""
    
    def test_validate_type(self):
        """Test type validation."""
        schema = {"type": "object"}
        
        result = validate_json_schema({"key": "value"}, schema)
        self.assertTrue(result.valid)
        
        result = validate_json_schema("string", schema)
        self.assertFalse(result.valid)
    
    def test_validate_required(self):
        """Test required fields."""
        schema = {
            "type": "object",
            "required": ["name", "version"],
            "properties": {
                "name": {"type": "string"},
                "version": {"type": "string"}
            }
        }
        
        result = validate_json_schema({"name": "test", "version": "1.0"}, schema)
        self.assertTrue(result.valid)
        
        result = validate_json_schema({"name": "test"}, schema)
        self.assertFalse(result.valid)
    
    def test_validate_enum(self):
        """Test enum validation."""
        schema = {"type": "string", "enum": ["red", "green", "blue"]}
        
        result = validate_json_schema("red", schema)
        self.assertTrue(result.valid)
        
        result = validate_json_schema("purple", schema)
        self.assertFalse(result.valid)
    
    def test_validate_string_constraints(self):
        """Test string constraints."""
        schema = {"type": "string", "minLength": 3, "maxLength": 10}
        
        result = validate_json_schema("hello", schema)
        self.assertTrue(result.valid)
        
        result = validate_json_schema("hi", schema)  # Too short
        self.assertFalse(result.valid)
    
    def test_validate_number_constraints(self):
        """Test number constraints."""
        schema = {"type": "integer", "minimum": 0, "maximum": 100}
        
        result = validate_json_schema(50, schema)
        self.assertTrue(result.valid)
        
        result = validate_json_schema(150, schema)  # Too large
        self.assertFalse(result.valid)


class TestManifestValidation(unittest.TestCase):
    """Test manifest validation."""
    
    def test_valid_manifest(self):
        """Test valid manifest."""
        manifest = {
            "schema": "stunir.manifest.test.v1",
            "epoch": 1706400000,
            "entries": [
                {"name": "file1.json", "hash": "abc123"}
            ],
            "manifest_hash": ""  # Will fail hash check
        }
        
        result = validate_manifest(manifest)
        # Should have required fields but fail hash check
        self.assertEqual(len([i for i in result.errors if 'hash' not in i.message.lower()]), 0)
    
    def test_missing_required_fields(self):
        """Test manifest with missing fields."""
        manifest = {"schema": "test"}
        
        result = validate_manifest(manifest)
        self.assertFalse(result.valid)
        # Should have errors for missing epoch, manifest_hash, entries
        self.assertGreater(len(result.errors), 0)


if __name__ == '__main__':
    unittest.main()
