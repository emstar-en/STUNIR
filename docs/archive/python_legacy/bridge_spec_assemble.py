#!/usr/bin/env python3
"""
STUNIR Spec Assembly Bridge (Phase 1)

Converts extraction.json to spec.json format.
Replaces broken stunir_spec_assemble_main.exe

Usage:
    python bridge_spec_assemble.py --input extraction.json --output spec.json
    python bridge_spec_assemble.py -i extraction.json -o spec.json -v
"""

import json
import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional


def validate_extraction_json(data: Dict[str, Any]) -> bool:
    """Validate extraction.json against expected schema."""
    # Support both old format and new stunir.extraction.v1 format
    if 'kind' in data and data['kind'] == 'stunir.extraction.v1':
        # New format validation
        if 'meta' not in data:
            raise ValueError("Missing required field: meta")
        if 'extractions' not in data:
            raise ValueError("Missing required field: extractions")
        if not isinstance(data['extractions'], list):
            raise ValueError("'extractions' must be a list")
        return True
    else:
        # Old format validation (for backward compatibility)
        required_fields = ['source_files', 'total_functions', 'functions']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        if not isinstance(data['functions'], list):
            raise ValueError("'functions' must be a list")

        if len(data['functions']) != data['total_functions']:
            raise ValueError(f"Function count mismatch: {len(data['functions'])} vs {data['total_functions']}")

        for i, func in enumerate(data['functions']):
            if 'name' not in func:
                raise ValueError(f"Function {i} missing 'name' field")
            if 'return_type' not in func:
                raise ValueError(f"Function {i} missing 'return_type' field")
            if 'parameters' not in func:
                raise ValueError(f"Function {i} missing 'parameters' field")

        return True


def convert_function_to_spec(func: Dict[str, Any]) -> Dict[str, Any]:
    """Convert extraction function format to spec function format (old format)."""
    # Convert parameters to spec args format
    args = []
    for param in func.get('parameters', []):
        args.append({
            'name': param['name'],
            'type': param['type']
        })

    return {
        'name': func['name'],
        'signature': {
            'args': args,
            'return_type': func['return_type']
        }
    }


def convert_function_to_spec_new(func: Dict[str, Any]) -> Dict[str, Any]:
    """Convert extraction function format to spec function format (new format)."""
    # New format has signature nested
    signature = func.get('signature', {})
    args = signature.get('args', [])

    return {
        'name': func['name'],
        'signature': {
            'args': args,
            'return_type': signature.get('return_type', 'void')
        }
    }


def generate_spec_hash(extraction_data: Dict[str, Any]) -> str:
    """Generate a hash for the spec based on extraction data."""
    # Handle both old and new formats
    is_new_format = 'kind' in extraction_data and extraction_data['kind'] == 'stunir.extraction.v1'

    if is_new_format:
        # New format: collect all functions from all extractions
        all_functions = []
        for extraction in extraction_data.get('extractions', []):
            all_functions.extend(extraction.get('functions', []))
        content = json.dumps(all_functions, sort_keys=True)
    else:
        # Old format: direct function list
        content = json.dumps(extraction_data.get('functions', []), sort_keys=True)

    return hashlib.sha256(content.encode()).hexdigest()[:16]


def assemble_spec(extraction_data: Dict[str, Any], module_name: Optional[str] = None) -> Dict[str, Any]:
    """Convert extraction.json to spec.json format."""
    # Validate input
    validate_extraction_json(extraction_data)

    # Determine if this is new or old format
    is_new_format = 'kind' in extraction_data and extraction_data['kind'] == 'stunir.extraction.v1'

    # Determine module name
    if module_name is None:
        if is_new_format:
            # New format: use first extraction's source file
            if extraction_data['extractions']:
                first_source = Path(extraction_data['extractions'][0]['source_file'])
                module_name = first_source.stem
            else:
                module_name = 'unnamed_module'
        else:
            # Old format: use first source file
            if extraction_data['source_files']:
                first_source = Path(extraction_data['source_files'][0])
                module_name = first_source.stem
            else:
                module_name = 'unnamed_module'

    # Convert functions
    spec_functions = []
    if is_new_format:
        # New format: iterate through extractions
        for extraction in extraction_data['extractions']:
            for func in extraction.get('functions', []):
                spec_functions.append(convert_function_to_spec_new(func))
        source_files = [e['source_file'] for e in extraction_data['extractions']]
    else:
        # Old format: direct function list
        for func in extraction_data['functions']:
            spec_functions.append(convert_function_to_spec(func))
        source_files = extraction_data['source_files']

    # Build spec.json structure
    spec = {
        'kind': 'stunir.spec.v1',
        'meta': {
            'origin': 'bridge_spec_assemble',
            'spec_hash': generate_spec_hash(extraction_data),
            'source_index': ','.join(source_files)
        },
        'modules': [
            {
                'name': module_name,
                'functions': spec_functions,
                'types': []  # Types not yet extracted
            }
        ]
    }

    return spec


def main():
    parser = argparse.ArgumentParser(
        description='STUNIR Spec Assembly Bridge - Convert extraction.json to spec.json'
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Path to extraction.json input file'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Path to spec.json output file'
    )
    parser.add_argument(
        '-m', '--module-name',
        help='Module name (default: derived from source file)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Read extraction.json
    try:
        with open(args.input, 'r') as f:
            extraction_data = json.load(f)
        if args.verbose:
            print(f"Loaded {args.input}")
            print(f"  Source files: {extraction_data.get('source_files', [])}")
            print(f"  Total functions: {extraction_data.get('total_functions', 0)}")
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Convert to spec.json
    try:
        spec = assemble_spec(extraction_data, args.module_name)
        if args.verbose:
            print(f"\nConverted to spec.json format:")
            print(f"  Kind: {spec['kind']}")
            print(f"  Module: {spec['modules'][0]['name']}")
            print(f"  Functions: {len(spec['modules'][0]['functions'])}")
    except ValueError as e:
        print(f"Error: Validation failed: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Write spec.json
    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(spec, f, indent=2)
        print(f"\nCreated {args.output}")
        print(f"  Functions: {len(spec['modules'][0]['functions'])}")
    except Exception as e:
        print(f"Error: Failed to write output: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
