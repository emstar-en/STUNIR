#!/usr/bin/env python3
"""
STUNIR Spec to IR Bridge (Phase 2)

Converts spec.json to ir.json format.
Replaces broken stunir_spec_to_ir_main.exe

Usage:
    python bridge_spec_to_ir.py --input spec.json --output ir.json
    python bridge_spec_to_ir.py -i spec.json -o ir.json -m module_name -v
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional


def validate_spec_json(data: Dict[str, Any]) -> bool:
    """Validate spec.json against expected schema."""
    if 'kind' not in data or data['kind'] != 'stunir.spec.v1':
        raise ValueError("Missing or invalid 'kind' field (expected 'stunir.spec.v1')")
    
    if 'modules' not in data or not isinstance(data['modules'], list):
        raise ValueError("Missing or invalid 'modules' field")
    
    if len(data['modules']) == 0:
        raise ValueError("No modules found in spec")
    
    for i, module in enumerate(data['modules']):
        if 'name' not in module:
            raise ValueError(f"Module {i} missing 'name' field")
        if 'functions' not in module:
            raise ValueError(f"Module {i} missing 'functions' field")
        if not isinstance(module['functions'], list):
            raise ValueError(f"Module {i} 'functions' must be a list")
        
        for j, func in enumerate(module['functions']):
            if 'name' not in func:
                raise ValueError(f"Function {j} in module {i} missing 'name' field")
            if 'signature' not in func:
                raise ValueError(f"Function {j} in module {i} missing 'signature' field")
            sig = func['signature']
            if 'args' not in sig:
                raise ValueError(f"Function {func['name']} missing 'args' in signature")
            if 'return_type' not in sig:
                raise ValueError(f"Function {func['name']} missing 'return_type' in signature")
    
    return True


def convert_arg_to_ir(arg: Dict[str, Any]) -> Dict[str, Any]:
    """Convert spec arg format to IR arg format."""
    return {
        'name': arg['name'],
        'type': arg['type']
    }


def convert_function_to_ir(func: Dict[str, Any]) -> Dict[str, Any]:
    """Convert spec function format to IR function format."""
    sig = func['signature']
    
    # Convert args
    ir_args = [convert_arg_to_ir(arg) for arg in sig.get('args', [])]
    
    # Generate minimal steps (noop for now since we're only doing signatures)
    steps = [{'op': 'noop'}]
    
    return {
        'name': func['name'],
        'args': ir_args,
        'return_type': sig['return_type'],
        'steps': steps
    }


def convert_spec_to_ir(spec_data: Dict[str, Any], module_name: Optional[str] = None) -> Dict[str, Any]:
    """Convert spec.json to ir.json format."""
    # Validate input
    validate_spec_json(spec_data)
    
    # Get the first module (or specified module)
    module = spec_data['modules'][0]
    
    # Determine module name
    if module_name is None:
        module_name = module.get('name', 'unnamed_module')
    
    # Convert functions
    ir_functions = []
    for func in module['functions']:
        ir_functions.append(convert_function_to_ir(func))
    
    # Build ir.json structure
    ir = {
        'schema': 'stunir_flat_ir_v1',
        'ir_version': 'v1',
        'module_name': module_name,
        'docstring': '',
        'types': [],
        'functions': ir_functions
    }
    
    return ir


def main():
    parser = argparse.ArgumentParser(
        description='STUNIR Spec to IR Bridge - Convert spec.json to ir.json'
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Path to spec.json input file'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Path to ir.json output file'
    )
    parser.add_argument(
        '-m', '--module-name',
        help='Module name override (default: from spec)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Read spec.json
    try:
        with open(args.input, 'r') as f:
            spec_data = json.load(f)
        if args.verbose:
            print(f"Loaded {args.input}")
            print(f"  Kind: {spec_data.get('kind', 'unknown')}")
            print(f"  Modules: {len(spec_data.get('modules', []))}")
            if spec_data.get('modules'):
                print(f"  Functions: {len(spec_data['modules'][0].get('functions', []))}")
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Convert to ir.json
    try:
        ir = convert_spec_to_ir(spec_data, args.module_name)
        if args.verbose:
            print(f"\nConverted to ir.json format:")
            print(f"  Schema: {ir['schema']}")
            print(f"  IR Version: {ir['ir_version']}")
            print(f"  Module: {ir['module_name']}")
            print(f"  Functions: {len(ir['functions'])}")
    except ValueError as e:
        print(f"Error: Validation failed: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Write ir.json
    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(ir, f)
        print(f"\nCreated {args.output}")
        print(f"  Functions: {len(ir['functions'])}")
    except Exception as e:
        print(f"Error: Failed to write output: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
