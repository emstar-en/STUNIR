#!/usr/bin/env python3
"""STUNIR Input Parser

Pipeline Stage: spec -> inputs
Issue: #1048

Parses and validates STUNIR spec input files.
"""

import json
import hashlib
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


def canonical_json(data: Any) -> str:
    """Generate RFC 8785 canonical JSON."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def compute_sha256(data: str) -> str:
    """Compute SHA-256 hash."""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


class InputParser:
    """STUNIR Input Parser.
    
    Parses spec files and validates against the input schema.
    """
    
    def __init__(self):
        """Initialize parser with default schema."""
        self.schema_path = Path(__file__).parent / 'input_schema.json'
        self._schema = None
    
    @property
    def schema(self) -> Dict[str, Any]:
        """Lazy load the schema."""
        if self._schema is None:
            if self.schema_path.exists():
                with open(self.schema_path, 'r') as f:
                    self._schema = json.load(f)
            else:
                self._schema = {}
        return self._schema
    
    def parse_file(self, path: Path) -> Tuple[Dict[str, Any], str]:
        """Parse a spec file.
        
        Args:
            path: Path to JSON spec file
            
        Returns:
            Tuple of (parsed_spec, hash)
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
        """
        with open(path, 'r') as f:
            spec = json.load(f)
        
        hash_value = compute_sha256(canonical_json(spec))
        return spec, hash_value
    
    def parse_string(self, content: str) -> Tuple[Dict[str, Any], str]:
        """Parse spec from string.
        
        Args:
            content: JSON string
            
        Returns:
            Tuple of (parsed_spec, hash)
        """
        spec = json.loads(content)
        hash_value = compute_sha256(canonical_json(spec))
        return spec, hash_value
    
    def validate(self, spec: Dict[str, Any]) -> List[str]:
        """Validate a spec against the input schema.
        
        Args:
            spec: Parsed spec dictionary
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Required fields
        required = self.schema.get('required', ['schema', 'id', 'name'])
        for field in required:
            if field not in spec:
                errors.append(f"Missing required field: {field}")
        
        # Schema version validation
        if 'schema' in spec:
            schema_val = spec['schema']
            if not schema_val.startswith('stunir.'):
                errors.append(f"Invalid schema: must start with 'stunir.'")
        
        # ID validation
        if 'id' in spec:
            import re
            id_pattern = r'^[a-zA-Z][a-zA-Z0-9_-]*$'
            if not re.match(id_pattern, spec['id']):
                errors.append(f"Invalid id format: {spec['id']}")
        
        # Version validation (semver)
        if 'version' in spec:
            version = spec['version']
            if not self._is_valid_semver(version):
                errors.append(f"Invalid version format: {version}")
        
        # Profile validation
        if 'profile' in spec:
            valid_profiles = ['profile1', 'profile2', 'profile3', 'profile4']
            if spec['profile'] not in valid_profiles:
                errors.append(f"Invalid profile: {spec['profile']}")
        
        # Targets validation
        if 'targets' in spec:
            if not isinstance(spec['targets'], list):
                errors.append("targets must be an array")
            elif len(spec['targets']) == 0:
                errors.append("targets array cannot be empty")
        
        return errors
    
    def _is_valid_semver(self, version: str) -> bool:
        """Check if version is valid semver."""
        parts = version.split('.')
        if len(parts) != 3:
            return False
        try:
            return all(int(p) >= 0 for p in parts)
        except ValueError:
            return False
    
    def normalize(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a spec with default values.
        
        Args:
            spec: Raw spec dictionary
            
        Returns:
            Normalized spec with defaults applied
        """
        normalized = dict(spec)
        
        # Apply defaults
        defaults = {
            'version': '1.0.0',
            'profile': 'profile3',
            'stages': ['STANDARDIZATION', 'UNIQUE_NORMALS', 'IR', 'BINARY', 'RECEIPT'],
            'dependencies': [],
            'metadata': {}
        }
        
        for key, default in defaults.items():
            if key not in normalized:
                normalized[key] = default
        
        return normalized
    
    def get_input_hash(self, spec: Dict[str, Any]) -> str:
        """Get deterministic hash of normalized input."""
        normalized = self.normalize(spec)
        return compute_sha256(canonical_json(normalized))


def main():
    """CLI interface for input parser."""
    import argparse
    
    parser = argparse.ArgumentParser(description='STUNIR Input Parser')
    parser.add_argument('input', help='Input spec file path')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('--validate', action='store_true', help='Validate only')
    parser.add_argument('--normalize', action='store_true', help='Apply defaults')
    
    args = parser.parse_args()
    
    input_parser = InputParser()
    
    # Parse input
    try:
        spec, input_hash = input_parser.parse_file(Path(args.input))
    except FileNotFoundError:
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Validate
    errors = input_parser.validate(spec)
    
    if args.validate:
        if errors:
            print("Validation FAILED:")
            for err in errors:
                print(f"  - {err}")
            sys.exit(1)
        print("Validation PASSED")
        print(f"Hash: {input_hash}")
        sys.exit(0)
    
    if errors:
        print("Validation errors:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
    
    # Normalize if requested
    if args.normalize:
        spec = input_parser.normalize(spec)
    
    # Output
    output = canonical_json(spec)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
            f.write('\n')
        print(f"Output written to: {args.output}", file=sys.stderr)
    else:
        print(json.dumps(spec, indent=2, sort_keys=True))
    
    print(f"Hash: {input_parser.get_input_hash(spec)}", file=sys.stderr)


if __name__ == '__main__':
    main()
