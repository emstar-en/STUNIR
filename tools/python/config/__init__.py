"""STUNIR Configuration Management.

Flexible configuration system supporting multiple sources:
- Configuration files (YAML, JSON, TOML)
- Environment variables
- Command-line arguments
- Defaults
"""

from .config import (
    Config,
    ConfigManager,
    get_config,
    load_config,
)
from .loaders import (
    ConfigLoader,
    YamlLoader,
    JsonLoader,
    TomlLoader,
    EnvLoader,
)
from .validators import (
    ConfigValidator,
    validate_config,
    ValidationError,
)
from .schema import (
    ConfigSchema,
    Field,
    FieldType,
)

__all__ = [
    # Config
    'Config',
    'ConfigManager',
    'get_config',
    'load_config',
    # Loaders
    'ConfigLoader',
    'YamlLoader',
    'JsonLoader',
    'TomlLoader',
    'EnvLoader',
    # Validators
    'ConfigValidator',
    'validate_config',
    'ValidationError',
    # Schema
    'ConfigSchema',
    'Field',
    'FieldType',
]
