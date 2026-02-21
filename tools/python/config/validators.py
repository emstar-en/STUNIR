"""Configuration validators for STUNIR.

Provides schema-based validation for configuration data.
"""

import re
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
from enum import Enum


class ValidationError(Exception):
    """Configuration validation error."""
    
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Validation failed: {', '.join(errors)}")


class ConfigValidator:
    """Configuration validator using schema definitions."""
    
    def __init__(self, schema: 'ConfigSchema'):
        self.schema = schema
    
    def validate(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration against schema.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors: List[str] = []
        self._validate_dict(config, self.schema.fields, '', errors)
        return errors
    
    def validate_or_raise(self, config: Dict[str, Any]) -> None:
        """Validate configuration and raise on errors.
        
        Raises:
            ValidationError: If validation fails
        """
        errors = self.validate(config)
        if errors:
            raise ValidationError(errors)
    
    def _validate_dict(
        self,
        data: Dict[str, Any],
        fields: Dict[str, 'Field'],
        prefix: str,
        errors: List[str],
    ) -> None:
        """Validate dictionary against field definitions."""
        # Check required fields
        for name, field in fields.items():
            key = f"{prefix}{name}" if prefix else name
            
            if name not in data:
                if field.required:
                    errors.append(f"Missing required field: {key}")
                continue
            
            value = data[name]
            self._validate_field(value, field, key, errors)
        
        # Check for unknown fields
        if self.schema.strict:
            for name in data:
                if name not in fields:
                    key = f"{prefix}{name}" if prefix else name
                    errors.append(f"Unknown field: {key}")
    
    def _validate_field(
        self,
        value: Any,
        field: 'Field',
        key: str,
        errors: List[str],
    ) -> None:
        """Validate a single field value."""
        # Type check
        if field.field_type and not self._check_type(value, field.field_type):
            errors.append(f"Invalid type for {key}: expected {field.field_type.name}, got {type(value).__name__}")
            return
        
        # Enum check
        if field.choices and value not in field.choices:
            errors.append(f"Invalid value for {key}: must be one of {field.choices}")
        
        # Range check
        if field.min_value is not None and value < field.min_value:
            errors.append(f"Value for {key} too small: minimum is {field.min_value}")
        if field.max_value is not None and value > field.max_value:
            errors.append(f"Value for {key} too large: maximum is {field.max_value}")
        
        # Length check
        if hasattr(value, '__len__'):
            if field.min_length is not None and len(value) < field.min_length:
                errors.append(f"Value for {key} too short: minimum length is {field.min_length}")
            if field.max_length is not None and len(value) > field.max_length:
                errors.append(f"Value for {key} too long: maximum length is {field.max_length}")
        
        # Pattern check
        if field.pattern and isinstance(value, str):
            if not re.match(field.pattern, value):
                errors.append(f"Value for {key} doesn't match pattern: {field.pattern}")
        
        # Custom validator
        if field.validator:
            try:
                result = field.validator(value)
                if result is False:
                    errors.append(f"Custom validation failed for {key}")
                elif isinstance(result, str):
                    errors.append(f"Validation error for {key}: {result}")
            except Exception as e:
                errors.append(f"Validator error for {key}: {str(e)}")
        
        # Nested object
        if field.nested and isinstance(value, dict):
            self._validate_dict(value, field.nested, f"{key}.", errors)
        
        # List items
        if field.items and isinstance(value, list):
            for i, item in enumerate(value):
                self._validate_field(item, field.items, f"{key}[{i}]", errors)
    
    def _check_type(self, value: Any, field_type: 'FieldType') -> bool:
        """Check if value matches the expected type."""
        from .schema import FieldType
        
        type_map = {
            FieldType.STRING: str,
            FieldType.INTEGER: int,
            FieldType.FLOAT: (int, float),
            FieldType.BOOLEAN: bool,
            FieldType.LIST: list,
            FieldType.DICT: dict,
            FieldType.ANY: object,
        }
        
        expected = type_map.get(field_type, object)
        return isinstance(value, expected)


def validate_config(config: Dict[str, Any], schema: 'ConfigSchema') -> List[str]:
    """Validate configuration against a schema.
    
    Args:
        config: Configuration dictionary
        schema: Schema definition
    
    Returns:
        List of validation errors
    """
    validator = ConfigValidator(schema)
    return validator.validate(config)


# Common validators
def is_port(value: int) -> bool:
    """Validate port number."""
    return 0 < value < 65536


def is_url(value: str) -> Union[bool, str]:
    """Validate URL format."""
    pattern = r'^https?://[\w\-.]+(:\d+)?(/.*)?$'
    if not re.match(pattern, value):
        return "Invalid URL format"
    return True


def is_email(value: str) -> Union[bool, str]:
    """Validate email format."""
    pattern = r'^[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, value):
        return "Invalid email format"
    return True


def is_path(value: str) -> bool:
    """Validate that value looks like a file path."""
    return bool(value) and not value.startswith(' ')
