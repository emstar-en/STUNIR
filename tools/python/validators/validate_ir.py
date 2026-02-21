#!/usr/bin/env python3
"""STUNIR IR Validator - Validates IR files against schema.

This tool is part of the tools → validators pipeline stage.
It validates IR files and checks for determinism issues.

Usage:
    validate_ir.py <ir.json> [--strict] [--hash]
"""

import json
import sys
import hashlib
import os
from typing import Any, Dict, List, Tuple

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# IR Schema definitions
IR_SCHEMAS = {
    "stunir.ir.v1": {
        "required": ["schema", "ir_module"],
        "optional": ["ir_epoch", "ir_spec_hash", "ir_functions", "ir_types", "ir_imports", "ir_exports"]
    }
}

def canonical_json(data: Any) -> str:
    """Generate canonical JSON output."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False)

def compute_sha256(data: Any) -> str:
    """Compute SHA-256 hash."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()

def validate_ir(ir_data: Dict[str, Any], strict: bool = False) -> Tuple[bool, List[str], List[str], Dict[str, Any]]:
    """Validate IR data against schema.

    Returns:
        tuple: (is_valid, errors, warnings, metadata)
    """
    errors = []
    warnings = []
    metadata = {}
    
    # Check schema field
    schema = ir_data.get('schema', '')
    if not schema:
        errors.append("Missing 'schema' field")
        return False, errors, warnings, metadata
    
    metadata['schema'] = schema
    
    # Get schema definition
    schema_def = IR_SCHEMAS.get(schema)
    if not schema_def:
        if strict:
            errors.append(f"Unknown schema: {schema}")
        else:
            warnings.append(f"Unknown schema: {schema}")
    
    # Validate required fields
    if schema_def:
        for field in schema_def['required']:
            if field not in ir_data:
                errors.append(f"Missing required field: {field}")
        
        # Check optional fields in strict mode
        if strict:
            for field in schema_def['optional']:
                if field not in ir_data:
                    warnings.append(f"Missing optional field: {field}")
    
    # Validate module name
    module = ir_data.get('ir_module', '')
    if module:
        metadata['module'] = module
        if not isinstance(module, str):
            errors.append(f"ir_module must be string, got {type(module).__name__}")
    
    # Validate functions
    functions = ir_data.get('ir_functions', [])
    if functions:
        if not isinstance(functions, list):
            errors.append("ir_functions must be a list")
        else:
            metadata['function_count'] = len(functions)
            for i, func in enumerate(functions):
                if not isinstance(func, dict):
                    errors.append(f"ir_functions[{i}] must be an object")
                elif 'name' not in func:
                    warnings.append(f"ir_functions[{i}] missing 'name'")
    
    # Compute content hash
    content_hash = compute_sha256(canonical_json(ir_data))
    metadata['content_hash'] = content_hash
    
    is_valid = len(errors) == 0
    return is_valid, errors, warnings, metadata

def main() -> None:
    """Validate an IR JSON file via CLI and print results."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <ir.json> [--strict] [--hash]", file=sys.stderr)
        print("\nSTUNIR IR Validator - Validates IR files against schema.", file=sys.stderr)
        sys.exit(1)
    
    input_path = sys.argv[1]
    strict = '--strict' in sys.argv
    show_hash = '--hash' in sys.argv
    
    try:
        # Read IR file
        with open(input_path, 'r') as f:
            ir_data = json.load(f)
        
        # Validate
        is_valid, errors, warnings, metadata = validate_ir(ir_data, strict)
        
        # Output results
        print(f"File: {input_path}")
        print(f"Schema: {metadata.get('schema', 'unknown')}")
        print(f"Module: {metadata.get('module', 'unknown')}")
        print(f"Functions: {metadata.get('function_count', 0)}")
        
        if show_hash:
            print(f"Content Hash: {metadata.get('content_hash', 'unknown')}")
        
        print(f"Valid: {is_valid}")
        
        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  ⚠ {w}")
        
        if errors:
            print("\nErrors:")
            for e in errors:
                print(f"  ✗ {e}")
            sys.exit(1)
        
        print("\n✓ IR validation passed")
        
    except json.JSONDecodeError as e:
        print(f"JSON Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
