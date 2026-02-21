#!/usr/bin/env python3
"""STUNIR Spec Examples Validator

Pipeline Stage: spec -> examples
Issue: #1080

Validates example spec files against the STUNIR spec schema.
"""

import json
import hashlib
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple


def canonical_json(data: Any) -> str:
    """Generate RFC 8785 canonical JSON."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def compute_sha256(data: str) -> str:
    """Compute SHA-256 hash."""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def validate_spec(spec: Dict[str, Any]) -> List[str]:
    """Validate a spec against STUNIR requirements.
    
    Args:
        spec: Spec dictionary to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Required top-level fields
    required = ['schema', 'id', 'name']
    for field in required:
        if field not in spec:
            errors.append(f"Missing required field: {field}")
    
    # Schema validation
    if 'schema' in spec:
        if not spec['schema'].startswith('stunir.'):
            errors.append(f"Invalid schema: {spec['schema']}")
    
    # Version format (semver)
    if 'version' in spec:
        version = spec['version']
        parts = version.split('.')
        if len(parts) != 3:
            errors.append(f"Invalid version format: {version}")
    
    # Profile validation
    valid_profiles = ['profile1', 'profile2', 'profile3', 'profile4']
    if 'profile' in spec and spec['profile'] not in valid_profiles:
        errors.append(f"Invalid profile: {spec['profile']}")
    
    # Targets validation
    if 'targets' in spec:
        if not isinstance(spec['targets'], list):
            errors.append("targets must be an array")
        elif len(spec['targets']) == 0:
            errors.append("targets cannot be empty")
    
    # Stages validation
    if 'stages' in spec:
        valid_stages = ['STANDARDIZATION', 'UNIQUE_NORMALS', 'IR', 'BINARY', 'RECEIPT']
        for stage in spec['stages']:
            if stage not in valid_stages:
                errors.append(f"Invalid stage: {stage}")
    
    # Module validation
    if 'module' in spec:
        module = spec['module']
        if 'name' not in module:
            errors.append("Module missing 'name' field")
        
        # Function validation
        if 'functions' in module:
            for i, func in enumerate(module['functions']):
                if 'name' not in func:
                    errors.append(f"Function {i} missing 'name'")
                if 'params' in func and not isinstance(func['params'], list):
                    errors.append(f"Function {func.get('name', i)} params must be array")
        
        # Type validation
        if 'types' in module:
            for i, typ in enumerate(module['types']):
                if 'name' not in typ:
                    errors.append(f"Type {i} missing 'name'")
                if 'kind' not in typ:
                    errors.append(f"Type {typ.get('name', i)} missing 'kind'")
    
    return errors


def validate_file(path: Path) -> Tuple[bool, List[str], str]:
    """Validate a spec file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Tuple of (is_valid, errors, hash)
    """
    try:
        with open(path, 'r') as f:
            spec = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"JSON parse error: {e}"], ""
    except Exception as e:
        return False, [f"Read error: {e}"], ""
    
    errors = validate_spec(spec)
    hash_value = compute_sha256(canonical_json(spec))
    
    return len(errors) == 0, errors, hash_value


def main():
    """Validate all example specs."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate STUNIR example specs')
    parser.add_argument('files', nargs='*', help='Specific files to validate')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode')
    
    args = parser.parse_args()
    
    # Determine files to validate
    examples_dir = Path(__file__).parent
    if args.files:
        files = [Path(f) for f in args.files]
    else:
        files = list(examples_dir.glob('*.json'))
    
    total = 0
    passed = 0
    results = []
    
    for file_path in sorted(files):
        total += 1
        is_valid, errors, hash_value = validate_file(file_path)
        
        if is_valid:
            passed += 1
            status = "\u2713 PASS"
        else:
            status = "\u2717 FAIL"
        
        results.append({
            'file': file_path.name,
            'valid': is_valid,
            'errors': errors,
            'hash': hash_value
        })
        
        if not args.quiet:
            print(f"{status}: {file_path.name}")
            if errors:
                for err in errors:
                    print(f"       - {err}")
            if is_valid and hash_value:
                print(f"       Hash: {hash_value[:16]}...")
    
    # Summary
    print(f"\n{'='*40}")
    print(f"Results: {passed}/{total} passed")
    
    # Return appropriate exit code
    sys.exit(0 if passed == total else 1)


if __name__ == '__main__':
    main()
