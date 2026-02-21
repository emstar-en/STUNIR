"""STUNIR Validation Framework.

Provides comprehensive validation utilities:
- Output validation (verify generated files)
- State validation (check internal consistency)
- Configuration validation (validate config options)
- API response validation (check external APIs)
- Schema validation (JSON Schema support)

Usage:
    from tools.validation import validate_output, validate_state
    from tools.validation import validate_config, ValidationResult
"""

from .validators import (
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

__all__ = [
    'ValidationResult',
    'validate_output',
    'validate_file_output',
    'validate_state',
    'validate_config',
    'validate_json_schema',
    'validate_manifest',
    'OutputValidator',
    'StateValidator',
    'ConfigValidator',
]
