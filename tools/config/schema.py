"""Configuration schema definitions for STUNIR.

Provides schema classes for defining configuration structure.
"""

from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field


class FieldType(Enum):
    """Supported field types."""
    STRING = auto()
    INTEGER = auto()
    FLOAT = auto()
    BOOLEAN = auto()
    LIST = auto()
    DICT = auto()
    ANY = auto()


@dataclass
class Field:
    """Configuration field definition.
    
    Attributes:
        field_type: Expected type of the field
        required: Whether the field is required
        default: Default value if not provided
        description: Human-readable description
        choices: Allowed values (for enum-like fields)
        min_value: Minimum value (for numbers)
        max_value: Maximum value (for numbers)
        min_length: Minimum length (for strings/lists)
        max_length: Maximum length (for strings/lists)
        pattern: Regex pattern (for strings)
        validator: Custom validation function
        nested: Nested field definitions (for dicts)
        items: Item field definition (for lists)
    """
    field_type: Optional[FieldType] = None
    required: bool = False
    default: Any = None
    description: str = ''
    choices: Optional[Set[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    validator: Optional[Callable[[Any], Union[bool, str]]] = None
    nested: Optional[Dict[str, 'Field']] = None
    items: Optional['Field'] = None
    
    @classmethod
    def string(
        cls,
        required: bool = False,
        default: str = '',
        **kwargs,
    ) -> 'Field':
        """Create a string field."""
        return cls(field_type=FieldType.STRING, required=required, default=default, **kwargs)
    
    @classmethod
    def integer(
        cls,
        required: bool = False,
        default: int = 0,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
        **kwargs,
    ) -> 'Field':
        """Create an integer field."""
        return cls(
            field_type=FieldType.INTEGER,
            required=required,
            default=default,
            min_value=min_value,
            max_value=max_value,
            **kwargs,
        )
    
    @classmethod
    def number(
        cls,
        required: bool = False,
        default: float = 0.0,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        **kwargs,
    ) -> 'Field':
        """Create a float field."""
        return cls(
            field_type=FieldType.FLOAT,
            required=required,
            default=default,
            min_value=min_value,
            max_value=max_value,
            **kwargs,
        )
    
    @classmethod
    def boolean(
        cls,
        required: bool = False,
        default: bool = False,
        **kwargs,
    ) -> 'Field':
        """Create a boolean field."""
        return cls(field_type=FieldType.BOOLEAN, required=required, default=default, **kwargs)
    
    @classmethod
    def list_of(
        cls,
        items: 'Field',
        required: bool = False,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        **kwargs,
    ) -> 'Field':
        """Create a list field."""
        return cls(
            field_type=FieldType.LIST,
            required=required,
            items=items,
            min_length=min_length,
            max_length=max_length,
            **kwargs,
        )
    
    @classmethod
    def object(
        cls,
        fields: Dict[str, 'Field'],
        required: bool = False,
        **kwargs,
    ) -> 'Field':
        """Create a nested object field."""
        return cls(
            field_type=FieldType.DICT,
            required=required,
            nested=fields,
            **kwargs,
        )
    
    @classmethod
    def enum(
        cls,
        choices: Set[Any],
        required: bool = False,
        default: Any = None,
        **kwargs,
    ) -> 'Field':
        """Create an enum field."""
        return cls(
            required=required,
            default=default,
            choices=choices,
            **kwargs,
        )


class ConfigSchema:
    """Configuration schema definition.
    
    Example:
        schema = ConfigSchema(
            fields={
                'database': Field.object({
                    'host': Field.string(required=True),
                    'port': Field.integer(default=5432, min_value=1, max_value=65535),
                }),
                'debug': Field.boolean(default=False),
            },
            strict=True,
        )
    """
    
    def __init__(
        self,
        fields: Dict[str, Field],
        strict: bool = False,
        description: str = '',
    ):
        self.fields = fields
        self.strict = strict
        self.description = description
    
    def get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values.
        
        Returns:
            Dictionary with default values for all fields
        """
        return self._get_defaults_recursive(self.fields)
    
    def _get_defaults_recursive(self, fields: Dict[str, Field]) -> Dict[str, Any]:
        """Recursively build defaults dictionary."""
        result = {}
        for name, field in fields.items():
            if field.nested:
                result[name] = self._get_defaults_recursive(field.nested)
            elif field.default is not None:
                result[name] = field.default
        return result
    
    def merge_with_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration with default values.
        
        Args:
            config: User-provided configuration
        
        Returns:
            Configuration with defaults filled in
        """
        defaults = self.get_defaults()
        return self._deep_merge(defaults, config)
    
    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Recursively merge dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Export schema as dictionary.
        
        Returns:
            Dictionary representation of the schema
        """
        return {
            'fields': self._fields_to_dict(self.fields),
            'strict': self.strict,
            'description': self.description,
        }
    
    def _fields_to_dict(self, fields: Dict[str, Field]) -> Dict[str, Any]:
        """Convert fields to dictionary representation."""
        result = {}
        for name, field in fields.items():
            field_dict = {
                'type': field.field_type.name if field.field_type else 'ANY',
                'required': field.required,
            }
            if field.default is not None:
                field_dict['default'] = field.default
            if field.description:
                field_dict['description'] = field.description
            if field.choices:
                field_dict['choices'] = list(field.choices)
            if field.nested:
                field_dict['nested'] = self._fields_to_dict(field.nested)
            result[name] = field_dict
        return result


# Pre-defined schemas for STUNIR
STUNIR_CONFIG_SCHEMA = ConfigSchema(
    fields={
        'log_level': Field.enum(
            choices={'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'},
            default='INFO',
            description='Logging level',
        ),
        'output_dir': Field.string(
            default='./output',
            description='Output directory for generated files',
        ),
        'cache': Field.object({
            'enabled': Field.boolean(default=True),
            'ttl': Field.integer(default=3600, min_value=0),
            'max_size': Field.integer(default=1000, min_value=1),
        }),
        'retry': Field.object({
            'max_attempts': Field.integer(default=3, min_value=1, max_value=100),
            'backoff_multiplier': Field.number(default=2.0, min_value=1.0),
        }),
    },
    description='STUNIR main configuration schema',
)
