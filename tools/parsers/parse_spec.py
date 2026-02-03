#!/usr/bin/env python3
"""STUNIR Spec Parser - Parses and validates spec JSON files.

This tool is part of the tools → parsers pipeline stage.
It reads spec files and validates their structure.

Usage:
    parse_spec.py <spec.json> [--strict]
"""

import json
import sys

# Required fields for spec files
SPEC_REQUIRED_FIELDS = [
    "module"  # or "name"
]

SPEC_OPTIONAL_FIELDS = [
    "version",
    "functions",
    "ops",
    "types",
    "imports",
    "exports",
    "description"
]

def parse_spec(spec_data, strict=False):
    """Parse and validate spec data.
    
    Returns:
        tuple: (is_valid, errors, warnings, normalized_spec)
    """
    errors = []
    warnings = []
    normalized = {}
    
    # Check for module name (accept 'module' or 'name')
    module = spec_data.get('module') or spec_data.get('name')
    if not module:
        errors.append("Missing 'module' or 'name' field")
    else:
        normalized['module'] = module
    
    # Check version
    version = spec_data.get('version', '1.0.0')
    normalized['version'] = version
    
    # Extract functions (accept 'functions' or 'ops')
    functions = spec_data.get('functions') or spec_data.get('ops', [])
    if not isinstance(functions, list):
        errors.append("'functions' must be a list")
    else:
        normalized['functions'] = functions
        for i, func in enumerate(functions):
            if isinstance(func, dict):
                if 'name' not in func:
                    warnings.append(f"functions[{i}] missing 'name'")
    
    # Extract types
    types = spec_data.get('types', [])
    if not isinstance(types, list):
        errors.append("'types' must be a list")
    else:
        normalized['types'] = types
    
    # Extract imports/exports
    normalized['imports'] = spec_data.get('imports', [])
    normalized['exports'] = spec_data.get('exports', [module] if module else [])
    
    # Strict mode checks
    if strict:
        for field in SPEC_OPTIONAL_FIELDS:
            if field not in spec_data:
                warnings.append(f"Missing optional field: {field}")
    
    is_valid = len(errors) == 0
    return is_valid, errors, warnings, normalized

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <spec.json> [--strict]", file=sys.stderr)
        print("\nSTUNIR Spec Parser - Parses and validates spec JSON files.", file=sys.stderr)
        sys.exit(1)
    
    input_path = sys.argv[1]
    strict = '--strict' in sys.argv
    
    try:
        # Read spec file
        with open(input_path, 'r') as f:
            spec_data = json.load(f)
        
        # Parse and validate
        is_valid, errors, warnings, normalized = parse_spec(spec_data, strict)
        
        # Output results
        print(f"File: {input_path}")
        print(f"Module: {normalized.get('module', 'unknown')}")
        print(f"Version: {normalized.get('version', 'unknown')}")
        print(f"Functions: {len(normalized.get('functions', []))}")
        print(f"Types: {len(normalized.get('types', []))}")
        print(f"Valid: {is_valid}")
        
        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  - {w}")
        
        if errors:
            print("\nErrors:")
            for e in errors:
                print(f"  - {e}")
            sys.exit(1)
        
        print("\n✓ Spec parsed successfully")
        
    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
