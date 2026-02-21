"""STUNIR Validators - Comprehensive Validation Framework.

Provides validators for outputs, state, configurations, and API responses.
"""

import os
import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable, Type
from enum import Enum

# Import common utilities
try:
    from tools.common.hash_utils import compute_sha256, compute_file_hash
    from tools.common.json_utils import canonical_json, parse_json
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from tools.common.hash_utils import compute_sha256, compute_file_hash
    from tools.common.json_utils import canonical_json, parse_json


class ValidationLevel(Enum):
    """Severity level for validation issues."""
    ERROR = 'error'
    WARNING = 'warning'
    INFO = 'info'


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    level: ValidationLevel
    message: str
    field: Optional[str] = None
    expected: Any = None
    actual: Any = None
    
    def __str__(self) -> str:
        result = f"[{self.level.value.upper()}] {self.message}"
        if self.field:
            result += f" (field: {self.field})"
        if self.expected is not None:
            result += f" expected: {self.expected}"
        if self.actual is not None:
            result += f" actual: {self.actual}"
        return result


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    data: Optional[Any] = None
    
    @property
    def errors(self) -> List[ValidationIssue]:
        """Get only error-level issues."""
        return [i for i in self.issues if i.level == ValidationLevel.ERROR]
    
    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get only warning-level issues."""
        return [i for i in self.issues if i.level == ValidationLevel.WARNING]
    
    def add_error(self, message: str, **kwargs) -> 'ValidationResult':
        """Add an error issue."""
        self.issues.append(ValidationIssue(ValidationLevel.ERROR, message, **kwargs))
        self.valid = False
        return self
    
    def add_warning(self, message: str, **kwargs) -> 'ValidationResult':
        """Add a warning issue."""
        self.issues.append(ValidationIssue(ValidationLevel.WARNING, message, **kwargs))
        return self
    
    def add_info(self, message: str, **kwargs) -> 'ValidationResult':
        """Add an info issue."""
        self.issues.append(ValidationIssue(ValidationLevel.INFO, message, **kwargs))
        return self
    
    def merge(self, other: 'ValidationResult') -> 'ValidationResult':
        """Merge another validation result into this one."""
        self.issues.extend(other.issues)
        if not other.valid:
            self.valid = False
        return self
    
    def __str__(self) -> str:
        status = 'VALID' if self.valid else 'INVALID'
        lines = [f"Validation Result: {status}"]
        for issue in self.issues:
            lines.append(f"  {issue}")
        return '\n'.join(lines)


# ============================================================================
# Output Validation
# ============================================================================

class OutputValidator:
    """Validates generated outputs for correctness."""
    
    def __init__(self):
        self._rules: List[Callable[[Any], ValidationResult]] = []
    
    def add_rule(self, rule: Callable[[Any], ValidationResult]) -> 'OutputValidator':
        """Add a validation rule."""
        self._rules.append(rule)
        return self
    
    def validate(self, output: Any) -> ValidationResult:
        """Validate output against all rules."""
        result = ValidationResult(valid=True)
        for rule in self._rules:
            rule_result = rule(output)
            result.merge(rule_result)
        return result


def validate_output(output: Any, expected_type: Optional[Type] = None,
                    required_fields: Optional[List[str]] = None,
                    hash_field: Optional[str] = None) -> ValidationResult:
    """Validate an output object.
    
    Args:
        output: Object to validate
        expected_type: Expected Python type
        required_fields: List of required field names (for dicts)
        hash_field: Field containing hash to verify
        
    Returns:
        ValidationResult with any issues found
    """
    result = ValidationResult(valid=True, data=output)
    
    # Type check
    if expected_type and not isinstance(output, expected_type):
        result.add_error(
            f"Type mismatch",
            expected=expected_type.__name__,
            actual=type(output).__name__
        )
        return result
    
    # Required fields (for dicts)
    if required_fields and isinstance(output, dict):
        for field in required_fields:
            if field not in output:
                result.add_error(f"Missing required field: {field}", field=field)
    
    # Hash verification
    if hash_field and isinstance(output, dict) and hash_field in output:
        # Compute hash excluding the hash field itself
        output_copy = {k: v for k, v in output.items() if k != hash_field}
        computed_hash = compute_sha256(canonical_json(output_copy))
        if computed_hash != output.get(hash_field):
            result.add_error(
                "Hash mismatch",
                field=hash_field,
                expected=output.get(hash_field),
                actual=computed_hash
            )
    
    return result


def validate_file_output(filepath: str, expected_hash: Optional[str] = None,
                         min_size: int = 0, max_size: Optional[int] = None,
                         expected_extension: Optional[str] = None) -> ValidationResult:
    """Validate a file output.
    
    Args:
        filepath: Path to file to validate
        expected_hash: Expected SHA-256 hash
        min_size: Minimum file size in bytes
        max_size: Maximum file size in bytes
        expected_extension: Expected file extension
        
    Returns:
        ValidationResult with any issues found
    """
    result = ValidationResult(valid=True)
    path = Path(filepath)
    
    # Existence check
    if not path.exists():
        result.add_error(f"File does not exist: {filepath}")
        return result
    
    if not path.is_file():
        result.add_error(f"Path is not a file: {filepath}")
        return result
    
    # Extension check
    if expected_extension:
        if not expected_extension.startswith('.'):
            expected_extension = '.' + expected_extension
        if path.suffix.lower() != expected_extension.lower():
            result.add_error(
                "Wrong file extension",
                expected=expected_extension,
                actual=path.suffix
            )
    
    # Size checks
    size = path.stat().st_size
    if size < min_size:
        result.add_error(
            f"File too small",
            expected=f">= {min_size} bytes",
            actual=f"{size} bytes"
        )
    if max_size is not None and size > max_size:
        result.add_error(
            f"File too large",
            expected=f"<= {max_size} bytes",
            actual=f"{size} bytes"
        )
    
    # Hash check
    if expected_hash:
        actual_hash = compute_file_hash(filepath)
        if actual_hash.lower() != expected_hash.lower():
            result.add_error(
                "Hash mismatch",
                expected=expected_hash,
                actual=actual_hash
            )
    
    result.data = {'path': filepath, 'size': size}
    return result


# ============================================================================
# State Validation
# ============================================================================

class StateValidator:
    """Validates internal state consistency."""
    
    def __init__(self):
        self._invariants: List[Callable[[Any], bool]] = []
        self._descriptions: List[str] = []
    
    def add_invariant(self, check: Callable[[Any], bool], 
                      description: str) -> 'StateValidator':
        """Add a state invariant to check."""
        self._invariants.append(check)
        self._descriptions.append(description)
        return self
    
    def validate(self, state: Any) -> ValidationResult:
        """Validate state against all invariants."""
        result = ValidationResult(valid=True, data=state)
        for check, desc in zip(self._invariants, self._descriptions):
            try:
                if not check(state):
                    result.add_error(f"Invariant violated: {desc}")
            except Exception as e:
                result.add_error(f"Invariant check failed: {desc} ({e})")
        return result


def validate_state(state: Dict[str, Any], 
                   required_keys: Optional[List[str]] = None,
                   consistency_checks: Optional[List[Callable[[Dict], bool]]] = None) -> ValidationResult:
    """Validate internal state for consistency.
    
    Args:
        state: State dictionary to validate
        required_keys: Keys that must be present
        consistency_checks: Functions that return True if state is consistent
        
    Returns:
        ValidationResult with any issues found
    """
    result = ValidationResult(valid=True, data=state)
    
    if not isinstance(state, dict):
        result.add_error("State must be a dictionary")
        return result
    
    # Required keys
    if required_keys:
        for key in required_keys:
            if key not in state:
                result.add_error(f"Missing required state key: {key}", field=key)
    
    # Consistency checks
    if consistency_checks:
        for i, check in enumerate(consistency_checks):
            try:
                if not check(state):
                    result.add_error(f"Consistency check {i+1} failed")
            except Exception as e:
                result.add_error(f"Consistency check {i+1} raised exception: {e}")
    
    return result


# ============================================================================
# Configuration Validation
# ============================================================================

class ConfigValidator:
    """Validates configuration options."""
    
    def __init__(self):
        self._schema: Dict[str, Dict[str, Any]] = {}
    
    def add_option(self, name: str, 
                   required: bool = False,
                   option_type: Optional[Type] = None,
                   choices: Optional[List[Any]] = None,
                   default: Any = None,
                   validator: Optional[Callable[[Any], bool]] = None) -> 'ConfigValidator':
        """Add a configuration option definition."""
        self._schema[name] = {
            'required': required,
            'type': option_type,
            'choices': choices,
            'default': default,
            'validator': validator,
        }
        return self
    
    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate configuration against schema."""
        result = ValidationResult(valid=True, data=config)
        
        # Check required options
        for name, schema in self._schema.items():
            if schema['required'] and name not in config:
                result.add_error(f"Missing required config option: {name}", field=name)
                continue
            
            if name not in config:
                continue
            
            value = config[name]
            
            # Type check
            if schema['type'] and not isinstance(value, schema['type']):
                result.add_error(
                    f"Invalid type for config option",
                    field=name,
                    expected=schema['type'].__name__,
                    actual=type(value).__name__
                )
            
            # Choices check
            if schema['choices'] and value not in schema['choices']:
                result.add_error(
                    f"Invalid value for config option",
                    field=name,
                    expected=f"one of {schema['choices']}",
                    actual=value
                )
            
            # Custom validator
            if schema['validator']:
                try:
                    if not schema['validator'](value):
                        result.add_error(
                            f"Custom validation failed for config option",
                            field=name,
                            actual=value
                        )
                except Exception as e:
                    result.add_error(
                        f"Validation error for config option: {e}",
                        field=name
                    )
        
        # Check for unknown options
        for name in config:
            if name not in self._schema:
                result.add_warning(f"Unknown config option: {name}", field=name)
        
        return result


def validate_config(config: Dict[str, Any],
                    required: Optional[List[str]] = None,
                    types: Optional[Dict[str, Type]] = None,
                    choices: Optional[Dict[str, List[Any]]] = None) -> ValidationResult:
    """Validate a configuration dictionary.
    
    Args:
        config: Configuration to validate
        required: List of required option names
        types: Dict mapping option names to expected types
        choices: Dict mapping option names to valid choices
        
    Returns:
        ValidationResult with any issues found
    """
    result = ValidationResult(valid=True, data=config)
    
    if not isinstance(config, dict):
        result.add_error("Configuration must be a dictionary")
        return result
    
    # Required checks
    if required:
        for opt in required:
            if opt not in config:
                result.add_error(f"Missing required option: {opt}", field=opt)
    
    # Type checks
    if types:
        for opt, expected_type in types.items():
            if opt in config and not isinstance(config[opt], expected_type):
                result.add_error(
                    "Invalid type",
                    field=opt,
                    expected=expected_type.__name__,
                    actual=type(config[opt]).__name__
                )
    
    # Choice checks
    if choices:
        for opt, valid_choices in choices.items():
            if opt in config and config[opt] not in valid_choices:
                result.add_error(
                    "Invalid value",
                    field=opt,
                    expected=f"one of {valid_choices}",
                    actual=config[opt]
                )
    
    return result


# ============================================================================
# JSON Schema Validation
# ============================================================================

def validate_json_schema(data: Any, schema: Dict[str, Any]) -> ValidationResult:
    """Validate data against a JSON Schema (simplified implementation).
    
    Supports basic JSON Schema features:
    - type (string, number, integer, boolean, array, object, null)
    - required (for objects)
    - properties (for objects)
    - items (for arrays)
    - enum (allowed values)
    - minimum/maximum (for numbers)
    - minLength/maxLength (for strings)
    - pattern (regex for strings)
    
    Args:
        data: Data to validate
        schema: JSON Schema definition
        
    Returns:
        ValidationResult with any issues found
    """
    result = ValidationResult(valid=True, data=data)
    
    def validate_type(value: Any, expected: str) -> bool:
        type_map = {
            'string': str,
            'number': (int, float),
            'integer': int,
            'boolean': bool,
            'array': list,
            'object': dict,
            'null': type(None),
        }
        expected_type = type_map.get(expected)
        if expected_type is None:
            return True
        return isinstance(value, expected_type)
    
    def validate_value(value: Any, sch: Dict, path: str = '') -> None:
        # Type check
        if 'type' in sch:
            expected_types = sch['type'] if isinstance(sch['type'], list) else [sch['type']]
            if not any(validate_type(value, t) for t in expected_types):
                result.add_error(
                    f"Type mismatch",
                    field=path or 'root',
                    expected=sch['type'],
                    actual=type(value).__name__
                )
                return
        
        # Enum check
        if 'enum' in sch and value not in sch['enum']:
            result.add_error(
                "Value not in enum",
                field=path or 'root',
                expected=sch['enum'],
                actual=value
            )
        
        # String checks
        if isinstance(value, str):
            if 'minLength' in sch and len(value) < sch['minLength']:
                result.add_error(f"String too short", field=path)
            if 'maxLength' in sch and len(value) > sch['maxLength']:
                result.add_error(f"String too long", field=path)
            if 'pattern' in sch and not re.match(sch['pattern'], value):
                result.add_error(f"Pattern mismatch", field=path)
        
        # Number checks
        if isinstance(value, (int, float)):
            if 'minimum' in sch and value < sch['minimum']:
                result.add_error(f"Value below minimum", field=path)
            if 'maximum' in sch and value > sch['maximum']:
                result.add_error(f"Value above maximum", field=path)
        
        # Object checks
        if isinstance(value, dict):
            # Required fields
            for req in sch.get('required', []):
                if req not in value:
                    result.add_error(f"Missing required field: {req}", field=f"{path}.{req}" if path else req)
            
            # Properties
            props = sch.get('properties', {})
            for prop, prop_schema in props.items():
                if prop in value:
                    validate_value(value[prop], prop_schema, f"{path}.{prop}" if path else prop)
        
        # Array checks
        if isinstance(value, list):
            if 'items' in sch:
                for i, item in enumerate(value):
                    validate_value(item, sch['items'], f"{path}[{i}]" if path else f"[{i}]")
    
    validate_value(data, schema)
    return result


# ============================================================================
# Manifest Validation
# ============================================================================

def validate_manifest(manifest: Dict[str, Any]) -> ValidationResult:
    """Validate a STUNIR manifest.
    
    Checks:
    - Required fields (schema, epoch, manifest_hash)
    - Hash integrity
    - Entry format
    
    Args:
        manifest: Manifest dictionary to validate
        
    Returns:
        ValidationResult with any issues found
    """
    result = ValidationResult(valid=True, data=manifest)
    
    # Required fields
    required = ['schema', 'epoch', 'manifest_hash', 'entries']
    for field in required:
        if field not in manifest:
            result.add_error(f"Missing required field: {field}", field=field)
    
    if not result.valid:
        return result
    
    # Schema format
    schema = manifest.get('schema', '')
    if not schema.startswith('stunir.manifest.'):
        result.add_warning(
            "Non-standard schema format",
            field='schema',
            expected='stunir.manifest.<type>.v<n>',
            actual=schema
        )
    
    # Verify manifest hash
    manifest_copy = {k: v for k, v in manifest.items() if k != 'manifest_hash'}
    computed_hash = compute_sha256(canonical_json(manifest_copy))
    if computed_hash != manifest.get('manifest_hash'):
        result.add_error(
            "Manifest hash mismatch",
            field='manifest_hash',
            expected=manifest.get('manifest_hash'),
            actual=computed_hash
        )
    
    # Validate entries
    entries = manifest.get('entries', [])
    if not isinstance(entries, list):
        result.add_error("Entries must be a list", field='entries')
        return result
    
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            result.add_error(f"Entry {i} must be a dictionary", field=f'entries[{i}]')
            continue
        
        # Each entry should have at least name and hash
        if 'name' not in entry and 'path' not in entry:
            result.add_warning(f"Entry {i} missing name/path", field=f'entries[{i}]')
        if 'hash' not in entry:
            result.add_warning(f"Entry {i} missing hash", field=f'entries[{i}]')
    
    return result
