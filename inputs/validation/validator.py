#!/usr/bin/env python3
"""STUNIR Input Validator

Pipeline Stage: inputs -> validation
Issue: #1081

Comprehensive input validation framework for STUNIR specs.
"""

import json
import hashlib
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field


def canonical_json(data: Any) -> str:
    """Generate RFC 8785 canonical JSON."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def compute_sha256(data: str) -> str:
    """Compute SHA-256 hash."""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


@dataclass
class ValidationError:
    """Validation error with context."""
    path: str
    message: str
    severity: str = 'error'  # error, warning, info
    code: str = ''


@dataclass
class ValidationResult:
    """Validation result container."""
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    hash: str = ''
    
    def add_error(self, path: str, message: str, code: str = ''):
        self.errors.append(ValidationError(path, message, 'error', code))
        self.valid = False
    
    def add_warning(self, path: str, message: str, code: str = ''):
        self.warnings.append(ValidationError(path, message, 'warning', code))


class Validator:
    """STUNIR Input Validator.
    
    Validates specs against schema, type constraints, and semantic rules.
    """
    
    def __init__(self, rules_path: Optional[Path] = None):
        """Initialize validator.
        
        Args:
            rules_path: Optional path to custom rules JSON
        """
        self.rules_path = rules_path or Path(__file__).parent / 'rules.json'
        self._rules = None
        self._custom_validators: Dict[str, Callable] = {}
    
    @property
    def rules(self) -> Dict[str, Any]:
        """Lazy load validation rules."""
        if self._rules is None:
            if self.rules_path.exists():
                with open(self.rules_path, 'r') as f:
                    data = json.load(f)
                    # Handle nested 'rules' structure
                    self._rules = data.get('rules', data)
            else:
                self._rules = self._default_rules()
        return self._rules
    
    def _default_rules(self) -> Dict[str, Any]:
        """Default validation rules."""
        return {
            'required_fields': ['schema', 'id', 'name'],
            'schema_pattern': r'^stunir\.(spec|input|module)\.v[0-9]+$',
            'id_pattern': r'^[a-zA-Z][a-zA-Z0-9_-]*$',
            'version_pattern': r'^[0-9]+\.[0-9]+\.[0-9]+$',
            'valid_profiles': ['profile1', 'profile2', 'profile3', 'profile4'],
            'valid_stages': ['STANDARDIZATION', 'UNIQUE_NORMALS', 'IR', 'BINARY', 'RECEIPT'],
            'max_name_length': 256,
            'max_description_length': 4096,
            'max_functions': 1000,
            'max_types': 500
        }
    
    def register_validator(self, name: str, validator: Callable[[Any, ValidationResult, str], None]):
        """Register a custom validator function.
        
        Args:
            name: Validator name
            validator: Function(value, result, path) -> None
        """
        self._custom_validators[name] = validator
    
    def validate(self, spec: Dict[str, Any]) -> ValidationResult:
        """Validate a spec.
        
        Args:
            spec: Spec dictionary to validate
            
        Returns:
            ValidationResult with errors and warnings
        """
        result = ValidationResult(valid=True)
        result.hash = compute_sha256(canonical_json(spec))
        
        # Required fields
        self._validate_required(spec, result)
        
        # Field-specific validation
        if 'schema' in spec:
            self._validate_schema(spec['schema'], result)
        
        if 'id' in spec:
            self._validate_id(spec['id'], result)
        
        if 'name' in spec:
            self._validate_name(spec['name'], result)
        
        if 'version' in spec:
            self._validate_version(spec['version'], result)
        
        if 'profile' in spec:
            self._validate_profile(spec['profile'], result)
        
        if 'stages' in spec:
            self._validate_stages(spec['stages'], result)
        
        if 'targets' in spec:
            self._validate_targets(spec['targets'], result)
        
        if 'module' in spec:
            self._validate_module(spec['module'], result)
        
        # Run custom validators
        for name, validator in self._custom_validators.items():
            try:
                validator(spec, result, '')
            except Exception as e:
                result.add_warning('', f"Custom validator '{name}' failed: {e}")
        
        return result
    
    def _validate_required(self, spec: Dict[str, Any], result: ValidationResult):
        """Validate required fields."""
        for field in self.rules.get('required_fields', []):
            if field not in spec:
                result.add_error('', f"Missing required field: {field}", 'MISSING_REQUIRED')
    
    def _validate_schema(self, schema: str, result: ValidationResult):
        """Validate schema field."""
        pattern = self.rules.get('schema_pattern', r'^stunir\.')
        if not re.match(pattern, schema):
            result.add_error('schema', f"Invalid schema format: {schema}", 'INVALID_SCHEMA')
    
    def _validate_id(self, id_val: str, result: ValidationResult):
        """Validate id field."""
        pattern = self.rules.get('id_pattern', r'^[a-zA-Z]')
        if not re.match(pattern, id_val):
            result.add_error('id', f"Invalid id format: {id_val}", 'INVALID_ID')
    
    def _validate_name(self, name: str, result: ValidationResult):
        """Validate name field."""
        max_len = self.rules.get('max_name_length', 256)
        if len(name) > max_len:
            result.add_error('name', f"Name too long (max {max_len})", 'NAME_TOO_LONG')
        if len(name) == 0:
            result.add_error('name', "Name cannot be empty", 'EMPTY_NAME')
    
    def _validate_version(self, version: str, result: ValidationResult):
        """Validate version field (semver)."""
        pattern = self.rules.get('version_pattern', r'^[0-9]+\.[0-9]+\.[0-9]+$')
        if not re.match(pattern, version):
            result.add_error('version', f"Invalid version format: {version}", 'INVALID_VERSION')
    
    def _validate_profile(self, profile: str, result: ValidationResult):
        """Validate profile field."""
        valid = self.rules.get('valid_profiles', [])
        if profile not in valid:
            result.add_error('profile', f"Invalid profile: {profile}", 'INVALID_PROFILE')
    
    def _validate_stages(self, stages: List[str], result: ValidationResult):
        """Validate stages array."""
        if not isinstance(stages, list):
            result.add_error('stages', "stages must be an array", 'INVALID_STAGES')
            return
        
        valid = self.rules.get('valid_stages', [])
        for i, stage in enumerate(stages):
            if stage not in valid:
                result.add_error(f'stages[{i}]', f"Invalid stage: {stage}", 'INVALID_STAGE')
    
    def _validate_targets(self, targets: Any, result: ValidationResult):
        """Validate targets array."""
        if not isinstance(targets, list):
            result.add_error('targets', "targets must be an array", 'INVALID_TARGETS')
            return
        
        if len(targets) == 0:
            result.add_error('targets', "targets cannot be empty", 'EMPTY_TARGETS')
    
    def _validate_module(self, module: Dict[str, Any], result: ValidationResult):
        """Validate module definition."""
        if 'name' not in module:
            result.add_error('module', "Module missing 'name' field", 'MISSING_MODULE_NAME')
        
        # Validate functions
        if 'functions' in module:
            max_funcs = self.rules.get('max_functions', 1000)
            if len(module['functions']) > max_funcs:
                result.add_error('module.functions', f"Too many functions (max {max_funcs})", 'TOO_MANY_FUNCTIONS')
            
            for i, func in enumerate(module['functions']):
                if 'name' not in func:
                    result.add_error(f'module.functions[{i}]', "Function missing 'name'", 'MISSING_FUNC_NAME')
        
        # Validate types
        if 'types' in module:
            max_types = self.rules.get('max_types', 500)
            if len(module['types']) > max_types:
                result.add_error('module.types', f"Too many types (max {max_types})", 'TOO_MANY_TYPES')
            
            for i, typ in enumerate(module['types']):
                if 'name' not in typ:
                    result.add_error(f'module.types[{i}]', "Type missing 'name'", 'MISSING_TYPE_NAME')
                if 'kind' not in typ:
                    result.add_error(f'module.types[{i}]', "Type missing 'kind'", 'MISSING_TYPE_KIND')
    
    def validate_file(self, path: Path) -> ValidationResult:
        """Validate a spec file."""
        try:
            with open(path, 'r') as f:
                spec = json.load(f)
        except json.JSONDecodeError as e:
            result = ValidationResult(valid=False)
            result.add_error('', f"JSON parse error: {e}", 'JSON_PARSE_ERROR')
            return result
        except Exception as e:
            result = ValidationResult(valid=False)
            result.add_error('', f"File read error: {e}", 'FILE_ERROR')
            return result
        
        return self.validate(spec)


def main():
    """CLI interface for validator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='STUNIR Input Validator')
    parser.add_argument('files', nargs='+', help='Spec files to validate')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    args = parser.parse_args()
    
    validator = Validator()
    all_valid = True
    results = []
    
    for file_path in args.files:
        result = validator.validate_file(Path(file_path))
        
        if not result.valid:
            all_valid = False
        
        if args.json:
            results.append({
                'file': file_path,
                'valid': result.valid,
                'hash': result.hash,
                'errors': [{'path': e.path, 'message': e.message, 'code': e.code} for e in result.errors],
                'warnings': [{'path': w.path, 'message': w.message, 'code': w.code} for w in result.warnings]
            })
        elif not args.quiet:
            status = "\u2713 PASS" if result.valid else "\u2717 FAIL"
            print(f"{status}: {file_path}")
            
            for err in result.errors:
                path_str = f" ({err.path})" if err.path else ""
                print(f"  ERROR{path_str}: {err.message}")
            
            for warn in result.warnings:
                path_str = f" ({warn.path})" if warn.path else ""
                print(f"  WARNING{path_str}: {warn.message}")
            
            if result.valid:
                print(f"  Hash: {result.hash[:16]}...")
    
    if args.json:
        print(json.dumps(results, indent=2))
    
    # Count passed files
    passed_count = len(args.files)
    for r in results:
        if isinstance(r, dict) and not r.get('valid', True):
            passed_count -= 1
    
    print(f"\n{'='*40}")
    print(f"Results: {passed_count}/{len(args.files)} passed")
    
    sys.exit(0 if all_valid else 1)


if __name__ == '__main__':
    main()
