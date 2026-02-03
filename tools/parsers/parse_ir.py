#!/usr/bin/env python3
"""STUNIR IR Parser - Parses and validates IR JSON files.

This tool is part of the tools → parsers pipeline stage.
It reads IR files and validates their structure.

Usage:
    parse_ir.py <ir.json> [--strict]
"""

import json
import sys
import hashlib

# Required fields for stunir.ir.v1 schema
IR_V1_REQUIRED_FIELDS = [
    "schema",
    "ir_module"
]

IR_V1_OPTIONAL_FIELDS = [
    "ir_epoch",
    "ir_spec_hash",
    "ir_functions",
    "ir_types",
    "ir_imports",
    "ir_exports"
]

def compute_sha256(data):
    """Compute SHA-256 hash of bytes."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()

def parse_ir(ir_data, strict=False):
    """Parse and validate IR data.
    
    Returns:
        tuple: (is_valid, errors, warnings)
    """
    errors = []
    warnings = []
    
    # Check schema
    schema = ir_data.get('schema', '')
    if not schema:
        errors.append("Missing 'schema' field")
    elif not schema.startswith('stunir.ir.'):
        errors.append(f"Invalid schema: {schema}")
    
    # Check required fields
    for field in IR_V1_REQUIRED_FIELDS:
        if field not in ir_data:
            errors.append(f"Missing required field: {field}")
    
    # Check optional fields in strict mode
    if strict:
        for field in IR_V1_OPTIONAL_FIELDS:
            if field not in ir_data:
                warnings.append(f"Missing optional field: {field}")
    
    # Validate ir_module
    module = ir_data.get('ir_module', '')
    if not module:
        errors.append("Empty ir_module")
    elif not isinstance(module, str):
        errors.append(f"ir_module must be string, got {type(module).__name__}")
    
    # Validate ir_functions if present
    functions = ir_data.get('ir_functions', [])
    if not isinstance(functions, list):
        errors.append("ir_functions must be a list")
    else:
        for i, func in enumerate(functions):
            if not isinstance(func, dict):
                errors.append(f"ir_functions[{i}] must be an object")
            elif 'name' not in func:
                warnings.append(f"ir_functions[{i}] missing 'name'")
    
    # Validate ir_types if present
    types = ir_data.get('ir_types', [])
    if not isinstance(types, list):
        errors.append("ir_types must be a list")
    
    is_valid = len(errors) == 0
    return is_valid, errors, warnings

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <ir.json> [--strict]", file=sys.stderr)
        print("\nSTUNIR IR Parser - Parses and validates IR JSON files.", file=sys.stderr)
        sys.exit(1)
    
    input_path = sys.argv[1]
    strict = '--strict' in sys.argv
    
    try:
        # Read IR file
        with open(input_path, 'r') as f:
            ir_data = json.load(f)
        
        # Parse and validate
        is_valid, errors, warnings = parse_ir(ir_data, strict)
        
        # Output results
        print(f"File: {input_path}")
        print(f"Schema: {ir_data.get('schema', 'unknown')}")
        print(f"Module: {ir_data.get('ir_module', 'unknown')}")
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
        
        print("\n✓ IR parsed successfully")
        
    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
